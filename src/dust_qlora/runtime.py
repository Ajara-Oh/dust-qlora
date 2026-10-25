"""Frozen local models, bounded loss computation, and verified decoder suffix replay.

The replay path reuses canonical decoder modules and the exact kwargs captured on
a clean forward. It never reuses KV/recurrent state from a perturbed pass.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import importlib.metadata
import os
import re
from types import SimpleNamespace
from typing import Any
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from .config import TrainConfig
from .adapters import LowRankLinear, inject_lora


def _layer_index(name: str) -> int | None:
    match = re.search(r"(?:^|\.)(?:layers|h)\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def select_adapter_targets(model: nn.Module, config: TrainConfig) -> list[str]:
    modules = dict(model.named_modules())
    if config.adapter.explicit_targets:
        names = list(config.adapter.explicit_targets)
        missing = set(names) - set(modules)
        if missing:
            raise ValueError(f"Adapter target modules not found: {sorted(missing)}")
    else:
        candidates = [name for name, module in modules.items()
                      if any(name == suffix or name.endswith("." + suffix)
                             for suffix in config.adapter.target_modules)
                      and hasattr(module, "in_features") and hasattr(module, "out_features")
                      and _layer_index(name) is not None
                      and ".visual." not in name]
        layers = sorted({_layer_index(name) for name in candidates})
        keep = set(layers[-config.adapter.last_n_layers:])
        names = [name for name in candidates if _layer_index(name) in keep]
    if not names:
        raise ValueError("No adapter targets matched. Inspect the model and configure explicit_targets.")
    if len(set(names)) != len(names):
        raise ValueError("Adapter targets contain duplicates")
    for name in names:
        module = modules[name]
        if not hasattr(module, "in_features") or not hasattr(module, "out_features"):
            raise ValueError(f"Adapter target {name!r} is not a supported linear module")
        if _layer_index(name) is None or ".visual." in name:
            raise ValueError(f"Adapter target {name!r} must be a text decoder-layer projection; output heads and vision modules are unsupported")
    return sorted(names, key=lambda name: (_layer_index(name) or 0, name))


class TinyMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.up_proj = nn.Linear(width, width * 2, bias=False)
        self.down_proj = nn.Linear(width * 2, width, bias=False)

    def forward(self, x):
        return self.down_proj(torch.tanh(self.up_proj(x)))


class TinyDecoderBlock(nn.Module):
    """Small causal mixer for offline algorithm tests; not a Qwen approximation."""
    def __init__(self, width: int):
        super().__init__()
        self.mix = nn.Linear(width, width, bias=False)
        self.mlp = TinyMLP(width)
        self.norm = nn.LayerNorm(width)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        mask = (attention_mask if attention_mask is not None else
                torch.ones(hidden_states.shape[:2], device=hidden_states.device)).unsqueeze(-1)
        history = (hidden_states * mask).cumsum(1) / mask.cumsum(1).clamp_min(1)
        hidden_states = hidden_states + torch.tanh(self.mix(history))
        return hidden_states + self.mlp(self.norm(hidden_states))


class TinyBackbone(nn.Module):
    def __init__(self, vocab: int, width: int, layers: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, width)
        self.layers = nn.ModuleList([TinyDecoderBlock(width) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        for block in self.layers:
            hidden = block(hidden, attention_mask=attention_mask, use_cache=False)
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class TinyCausalLM(nn.Module):
    def __init__(self, vocab=64, width=32, layers=3):
        super().__init__()
        self.model = TinyBackbone(vocab, width, layers)
        self.lm_head = nn.Linear(width, vocab, bias=False)
        self.config = SimpleNamespace(model_type="dust_tiny", vocab_size=vocab,
                                      hidden_size=width, _commit_hash=None, use_cache=False,
                                      tie_word_embeddings=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def _backbone(model):
    base = getattr(model, "model", None)
    if base is None:
        raise ValueError("This runtime requires a decoder model exposing .model and an output head")
    return getattr(base, "language_model", base)


@dataclass
class SuffixReplay:
    start: int
    hidden: torch.Tensor
    kwargs: list[dict[str, Any]]


@dataclass
class LocalRuntime:
    model: nn.Module
    adapters: dict[str, LowRankLinear]
    config: TrainConfig
    device: torch.device
    dtype: torch.dtype
    resolved_revision: str | None = None
    full_forwards: int = 0
    suffix_forwards: int = 0
    tokenization_fingerprint: str | None = None
    tokenization_metadata: dict[str, Any] = field(default_factory=dict)
    _verified_targets: set[str] = field(default_factory=set)
    _replay_rejected: set[str] = field(default_factory=set)

    @property
    def backbone(self):
        return _backbone(self.model)

    @property
    def output_head(self):
        return self.model.get_output_embeddings()

    def autocast(self):
        if self.device.type == "cuda" and self.dtype != torch.float32:
            return torch.autocast("cuda", dtype=self.dtype)
        return nullcontext()

    def clear_interventions(self):
        for adapter in self.adapters.values():
            adapter.perturbation = None
            adapter.capture_input = False
            adapter.last_input = None

    def hidden(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.full_forwards += 1
        ids = batch["input_ids"]
        attention = batch["attention_mask"]
        embedding = self.model.get_input_embeddings()
        embedding_device = embedding.weight.device
        kwargs = {"attention_mask": attention, "use_cache": False, "return_dict": True,
                  "position_ids": (attention.long().cumsum(-1) - 1).clamp_min(0)}
        if embedding_device != self.device:
            # Only the frozen embedding lookup runs on CPU. Decoder states stay on GPU.
            embeds = embedding(ids.to(embedding_device)).to(device=self.device, dtype=self.dtype)
            kwargs["inputs_embeds"] = embeds
        else:
            kwargs["input_ids"] = ids
        with self.autocast():
            outputs = self.backbone(**kwargs)
        return outputs.last_hidden_state

    def losses(self, batch: dict[str, torch.Tensor], replay: SuffixReplay | None = None):
        hidden = self.hidden(batch) if replay is None else self.replay_hidden(replay)
        return self.losses_from_hidden(hidden, batch)

    def losses_from_hidden(self, hidden, batch):
        """Return [B,T] contributions to the mean assistant next-token loss.

        Entry t predicts labels[t+1]. Masked context tokens remain in the model
        and receive indirect credit through later losses. The final entry is 0.
        Only supervised positions are projected into the large vocabulary.
        """
        labels = batch["labels"]
        attention = batch["attention_mask"].bool()
        valid = (labels[:, 1:] != -100) & attention[:, 1:] & attention[:, :-1]
        indices = valid.reshape(-1).nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            raise ValueError("Batch has no supervised next-token targets")
        selected = hidden[:, :-1, :].reshape(-1, hidden.shape[-1]).index_select(0, indices)
        targets = labels[:, 1:].reshape(-1).index_select(0, indices)
        chunks = []
        size = self.config.training.loss_chunk_size
        for start in range(0, len(targets), size):
            hs = selected[start:start + size]
            ys = targets[start:start + size]

            def project_loss(h, target=ys):
                with self.autocast():
                    logits = self.output_head(h)
                return F.cross_entropy(logits.float(), target, reduction="none")

            if torch.is_grad_enabled() and hs.requires_grad:
                # Retain hidden-state-sized checkpoints, not all vocabulary logits.
                from torch.utils.checkpoint import checkpoint
                values = checkpoint(project_loss, hs, use_reentrant=False,
                                    preserve_rng_state=False)
            else:
                values = project_loss(hs)
            chunks.append(values)
        losses = torch.cat(chunks).float() / len(targets)
        flat = torch.zeros(valid.numel(), device=hidden.device, dtype=torch.float32)
        flat = flat.index_copy(0, indices, losses)
        return F.pad(flat.view(labels.shape[0], -1), (0, 1))

    def _replay_supported(self) -> bool:
        return (getattr(self.model.config, "model_type", "") in
                {"qwen3_5", "qwen3_5_text", "dust_tiny"}
                and hasattr(self.backbone, "layers") and hasattr(self.backbone, "norm"))

    def _reject_replay(self, target, reason):
        self._replay_rejected.add(target)
        self._verified_targets.discard(target)
        message = f"Suffix replay unavailable for {target}: {reason}"
        if self.config.model.replay_mode == "suffix":
            raise RuntimeError(message + "; refusing to train in strict suffix mode")
        warnings.warn(message + "; using full-forward probes", RuntimeWarning)

    @torch.no_grad()
    def capture(self, target: str, batch) -> tuple[torch.Tensor, torch.Tensor, SuffixReplay | None]:
        """Clean forward, adapter input capture, and optionally verified suffix cache."""
        self.clear_interventions()
        adapter = self.adapters[target]
        start = _layer_index(target)
        want = (self.config.model.replay_mode != "full" and self._replay_supported()
                and start is not None and target not in self._replay_rejected)
        if self.config.model.replay_mode == "suffix" and not want:
            raise ValueError(f"Verified suffix replay is unavailable for {target}")
        adapter.capture_input = True
        captures: dict[int, tuple[torch.Tensor, dict[str, Any]]] = {}
        capture_errors = []
        handles = []
        if want:
            for index in range(start, len(self.backbone.layers)):
                def capture_input(module, args, kwargs, index=index):
                    if any(kwargs.get(key) is not None for key in
                           ("past_key_values", "past_key_value", "cache_params")):
                        capture_errors.append("decoder supplied recurrent/KV state")
                        return
                    if kwargs.get("use_cache", False):
                        capture_errors.append("decoder enabled use_cache")
                        return
                    value = args[0] if args else kwargs.get("hidden_states")
                    if not isinstance(value, torch.Tensor):
                        capture_errors.append("decoder input is not a hidden-state tensor")
                        return
                    saved = dict(kwargs)
                    saved.pop("hidden_states", None)
                    captures[index] = (value.detach() if index == start else None, saved)
                handles.append(self.backbone.layers[index].register_forward_pre_hook(
                    capture_input, with_kwargs=True))
        try:
            hidden = self.hidden(batch)
            clean = self.losses_from_hidden(hidden, batch)
            captured = adapter.last_input
            if captured is None:
                raise RuntimeError(f"Adapter {target} did not execute during the clean forward")
        finally:
            adapter.capture_input = False
            for handle in handles:
                handle.remove()
        replay = None
        if want and (capture_errors or len(captures) != len(self.backbone.layers) - start):
            self._reject_replay(target, "; ".join(capture_errors) or "not every suffix block executed")
        elif want:
            replay = SuffixReplay(start, captures[start][0],
                                  [captures[index][1] for index in range(start, len(self.backbone.layers))])
            if target not in self._verified_targets:
                try:
                    replayed = self.replay_hidden(replay)
                    tol = 1e-5 if hidden.dtype == torch.float32 else 2e-3
                    agrees = torch.allclose(hidden, replayed, atol=tol, rtol=tol)
                    failure = "hidden states disagree with a clean full forward"
                except torch.OutOfMemoryError:
                    raise
                except (TypeError, ValueError, RuntimeError) as error:
                    agrees = False
                    failure = f"canonical decoder replay failed ({type(error).__name__}: {error})"
                if not agrees:
                    self._reject_replay(target, failure)
                    replay = None
                else:
                    self._verified_targets.add(target)
        return clean, captured.detach(), replay

    def replay_hidden(self, replay: SuffixReplay):
        self.suffix_forwards += 1
        # Every call starts at the clean block input; no generated caches are reused.
        hidden = replay.hidden.clone()
        with self.autocast():
            for block, kwargs in zip(self.backbone.layers[replay.start:], replay.kwargs):
                hidden = block(hidden, **kwargs)
                if not isinstance(hidden, torch.Tensor):
                    raise TypeError("Suffix replay supports decoder blocks returning a hidden-state tensor")
            return self.backbone.norm(hidden)

    def metadata(self):
        if self.config.model.replay_mode == "full" or not self._replay_supported():
            execution = "full_forward"
        elif self._replay_rejected:
            execution = "mixed_suffix_and_full_forward" if self._verified_targets else "full_forward"
        else:
            execution = "verified_suffix" if self._verified_targets else "suffix_pending_verification"
        return {
            "model": self.config.model.name,
            "requested_revision": self.config.model.revision,
            "resolved_revision": self.resolved_revision,
            "device": str(self.device), "dtype": str(self.dtype),
            "quantized": bool(getattr(self.model, "is_loaded_in_4bit", False)),
            "targets": list(self.adapters),
            "adapter_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "replay_mode_requested": self.config.model.replay_mode,
            "replay_supported": self._replay_supported(),
            "replay_verified_targets": sorted(self._verified_targets),
            "probe_execution": execution,
            "replay_rejected_targets": sorted(self._replay_rejected),
            "full_forwards": self.full_forwards, "suffix_forwards": self.suffix_forwards,
            "tokenization_fingerprint": self.tokenization_fingerprint,
            "tokenization": self.tokenization_metadata,
        }


def load_runtime(config: TrainConfig) -> LocalRuntime:
    config.validate()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    device = torch.device(config.model.device)
    dtype = getattr(torch, config.model.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use the tiny CPU smoke config or install the CUDA training dependencies.")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("Use model.dtype='float32' on CPU")
    if config.model.name == "tiny":
        model = TinyCausalLM(config.model.tiny_vocab_size, config.model.tiny_hidden_size,
                             config.model.tiny_num_layers).to(device)
    else:
        try:
            from transformers import (AutoConfig, AutoModelForCausalLM,
                                      AutoModelForImageTextToText, BitsAndBytesConfig)
        except ImportError as error:
            raise RuntimeError("Install the training extra: pip install -e '.[train]'") from error
        hf_config = AutoConfig.from_pretrained(config.model.name, revision=config.model.revision,
                                               local_files_only=config.model.local_files_only,
                                               trust_remote_code=False)
        model_class = (AutoModelForImageTextToText if hf_config.model_type == "qwen3_5"
                       else AutoModelForCausalLM)
        kwargs: dict[str, Any] = {
            "revision": config.model.revision, "local_files_only": config.model.local_files_only,
            "trust_remote_code": False, "dtype": dtype,
            "attn_implementation": config.model.attn_implementation,
            "device_map": {"": str(device)}, "low_cpu_mem_usage": True,
        }
        existing_quant = getattr(hf_config, "quantization_config", None)
        if config.model.load_in_4bit:
            if device.type != "cuda":
                raise ValueError("This project's four-bit bitsandbytes runtime requires CUDA")
            if existing_quant:
                # Preserve the checkpoint's dynamic exclusion list and quantization state.
                quant_dict = existing_quant.to_dict() if hasattr(existing_quant, "to_dict") else dict(existing_quant)
                if quant_dict.get("quant_method") != "bitsandbytes":
                    raise ValueError("Use a bitsandbytes checkpoint or original weights; this runtime does not load GGUF/AWQ")
                quant_dict["bnb_4bit_compute_dtype"] = config.model.dtype
                kwargs["quantization_config"] = BitsAndBytesConfig.from_dict(quant_dict)
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype,
                    llm_int8_skip_modules=["lm_head", "visual", "in_proj_a", "in_proj_b", "in_proj_qkv"],
                )
        elif existing_quant:
            raise ValueError("Checkpoint is prequantized but load_in_4bit=false; choose original weights")
        try:
            model = model_class.from_pretrained(config.model.name, config=hf_config, **kwargs)
        except torch.OutOfMemoryError as error:
            raise RuntimeError("Model loading exhausted VRAM before embedding offload. Close inference processes; this loader initially places the checkpoint on the GPU.") from error
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    if hasattr(_backbone(model), "config"):
        _backbone(model).config.use_cache = False
    if config.model.offload_embedding and device.type == "cuda":
        embedding = model.get_input_embeddings()
        if embedding.weight.data_ptr() == model.get_output_embeddings().weight.data_ptr():
            raise ValueError("Embedding offload is not supported for tied embeddings/output weights")
        embedding.to("cpu")
    if config.model.offload_vision and hasattr(getattr(model, "model", None), "visual"):
        model.model.visual.to("cpu")
    targets = select_adapter_targets(model, config)
    adapters = inject_lora(model, targets, rank=config.adapter.rank, alpha=config.adapter.alpha)
    model.eval()
    # Wrappers start in FP32; only their small factors are trainable.
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return LocalRuntime(model, adapters, config, device, dtype,
                        resolved_revision=getattr(model.config, "_commit_hash", None))


def environment_report(config: TrainConfig) -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "bitsandbytes", "peft", "accelerate", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            info = torch.cuda.get_device_properties(index)
            gpus.append({"index": index, "name": info.name, "vram_bytes": info.total_memory,
                         "compute_capability": f"{info.major}.{info.minor}"})
    return {"packages": packages, "cuda_available": torch.cuda.is_available(), "gpus": gpus,
            "model": config.model.name, "downloads_allowed": not config.model.local_files_only,
            "training_data": config.data.train, "validation_data": config.data.validation,
            "probe_replay": "Supported Qwen3.5/3.8 decoder suffixes are checked against clean full forwards; unsupported models use full forwards.",
            "quantized_loading_note": "The entire quantized checkpoint initially loads on GPU; embedding/vision CPU offload follows loading."}
