"""Local training, evaluation, and resumable adapter checkpoints."""
from __future__ import annotations

from array import array
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import time
from typing import Any

import torch

from .adapters import (adapter_parameters, adapter_state_dict, assign_lora_gradients,
                       load_adapter_state_dict)
from .config import TrainConfig, resume_fingerprint
from .noise import NoiseConfig, estimate_activation_error
from .runtime import LocalRuntime, environment_report, load_runtime


class TokenDataset:
    """An indexed JSONL dataset. Token arrays stay on disk until a batch is read."""
    def __init__(self, path: str | Path, *, max_length: int, vocab_size: int | None = None):
        self.path = Path(path)
        self.offsets = array("Q")
        self.groups: set[str] = set()
        self.tokens = self.supervised_tokens = 0
        self.max_length = max_length
        self.vocab_size = vocab_size
        digest = hashlib.sha256()
        identifiers: set[str] = set()
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                digest.update(line)
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    self._validate(row)
                except (ValueError, TypeError, KeyError) as error:
                    raise ValueError(f"Invalid tokenized record at byte {offset} in {self.path}: {error}") from error
                identifier = row.get("id")
                if identifier:
                    if identifier in identifiers:
                        raise ValueError(f"Duplicate example id {identifier!r} in {self.path}")
                    identifiers.add(identifier)
                if row.get("group_id"):
                    self.groups.add(str(row["group_id"]))
                self.offsets.append(offset)
                self.tokens += len(row["input_ids"])
                self.supervised_tokens += sum(value != -100 for value in row["labels"][1:])
        self.fingerprint = digest.hexdigest()

    def _validate(self, row):
        if not isinstance(row, dict):
            raise ValueError("Expected an object with input_ids and labels")
        for field in ("id", "group_id"):
            if field in row and (not isinstance(row[field], str) or not row[field]):
                raise ValueError(f"{field} must be a nonempty string")
        ids, labels = row["input_ids"], row["labels"]
        if not isinstance(ids, list) or not isinstance(labels, list) or len(ids) != len(labels):
            raise ValueError("input_ids and labels must be equal-length lists")
        if not 2 <= len(ids) <= self.max_length:
            raise ValueError(f"Length {len(ids)} is outside [2,{self.max_length}]; retokenize, never silently truncate")
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in ids):
            raise ValueError("Token IDs must be nonnegative integers")
        if self.vocab_size is not None and any(x >= self.vocab_size for x in ids):
            raise ValueError("Token ID exceeds model vocabulary; check tokenizer/checkpoint identity")
        if any(isinstance(y, bool) or not isinstance(y, int) or y not in (-100, x)
               for x, y in zip(ids, labels)):
            raise ValueError("Use unshifted HF labels: each entry equals its input token or -100")
        if not any(y != -100 for y in labels[1:]):
            raise ValueError("No supervised next-token targets")
        if "attention_mask" in row and row["attention_mask"] != [1] * len(ids):
            raise ValueError("Store unpadded examples; the trainer creates right padding per batch")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline())

    def metadata(self):
        return {"sha256": self.fingerprint, "examples": len(self), "tokens": self.tokens,
                "supervised_tokens": self.supervised_tokens, "groups": len(self.groups)}


def collate(rows, device, pad_id=0):
    width = max(len(row["input_ids"]) for row in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        length = len(row["input_ids"])
        ids[index, :length] = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
        labels[index, :length] = torch.tensor(row["labels"], dtype=torch.long, device=device)
        attention[index, :length] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": attention}


def _seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _vocab_size(runtime):
    head = runtime.output_head
    return getattr(head, "out_features", None)


def _record_tokenization(runtime, datasets):
    """Check tokenizer/model identity and keep manifest changes resume-visible."""
    metadata = {}
    for split, dataset in datasets.items():
        if dataset is None:
            continue
        path = dataset.path.parent / "manifest.json"
        if not path.exists():
            continue
        content = path.read_bytes()
        manifest = json.loads(content)
        if manifest.get("model_name") != runtime.config.model.name:
            raise ValueError("Tokenization manifest model_name differs from the training checkpoint; retokenize with the selected model")
        tokenizer_revision = manifest.get("resolved_revision")
        if tokenizer_revision and runtime.resolved_revision and tokenizer_revision != runtime.resolved_revision:
            raise ValueError("Tokenizer and model resolved revisions differ; pin the same revision and retokenize")
        if not tokenizer_revision and runtime.config.model.revision != manifest.get("revision"):
            raise ValueError("Tokenization manifest revision differs from the training revision")
        recorded = manifest.get("splits", {}).get(split, {}).get("sha256")
        if recorded and recorded != dataset.fingerprint:
            raise ValueError(f"Tokenized {split} data changed since its manifest was written")
        metadata[split] = {
            "manifest_sha256": hashlib.sha256(content).hexdigest(),
            "model_name": manifest.get("model_name"),
            "revision": manifest.get("revision"),
            "resolved_revision": tokenizer_revision,
            "chat_template_sha256": manifest.get("chat_template_sha256"),
            "template_kwargs": manifest.get("template_kwargs"),
            "context_policy": manifest.get("context_policy"),
        }
    runtime.tokenization_metadata = metadata
    runtime.tokenization_fingerprint = (hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
                                        if metadata else None)


def _datasets(config, runtime):
    train_data = TokenDataset(config.data.train, max_length=config.data.max_length,
                              vocab_size=_vocab_size(runtime))
    if not len(train_data):
        raise ValueError("Training dataset is empty")
    validation = None
    if config.data.validation:
        path = Path(config.data.validation)
        if not path.exists():
            raise FileNotFoundError(f"Configured validation dataset does not exist: {path}")
        validation = TokenDataset(path, max_length=config.data.max_length,
                                   vocab_size=_vocab_size(runtime))
        overlap = train_data.groups & validation.groups
        if overlap:
            raise ValueError(f"Train/validation leakage: {len(overlap)} shared group IDs")
    _record_tokenization(runtime, {"train": train_data, "validation": validation})
    return train_data, validation


def _json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def _write_adapter(directory, runtime):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().contiguous() for name, value in
             adapter_state_dict(runtime.model).items()}
    try:
        from safetensors.torch import save_file
        save_file(state, str(directory / "adapter_model.safetensors"))
    except ImportError:
        torch.save(state, directory / "adapter_model.bin")
    config = runtime.config
    _json(directory / "adapter_config.json", {
        "base_model_name_or_path": config.model.name,
        "revision": runtime.resolved_revision or config.model.revision,
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "inference_mode": True,
        "r": config.adapter.rank, "lora_alpha": config.adapter.alpha,
        "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False,
        "target_modules": list(runtime.adapters), "modules_to_save": None,
        "use_rslora": False, "use_dora": False,
    })
    _json(directory / "runtime.json", runtime.metadata())


def _load_adapter(directory, runtime):
    directory = Path(directory)
    adapter_config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    for key, expected in {
        "base_model_name_or_path": runtime.config.model.name,
        "revision": runtime.resolved_revision or runtime.config.model.revision,
        "r": runtime.config.adapter.rank, "lora_alpha": runtime.config.adapter.alpha,
    }.items():
        if adapter_config.get(key) != expected:
            raise ValueError(f"Adapter checkpoint {key} differs from the configured base or adapter")
    if (directory / "adapter_model.safetensors").exists():
        from safetensors.torch import load_file
        state = load_file(str(directory / "adapter_model.safetensors"))
    else:
        state = torch.load(directory / "adapter_model.bin", map_location="cpu", weights_only=True)
    load_adapter_state_dict(runtime.model, state)


def _rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG/device count differs from the checkpoint")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(output, runtime, optimizer, *, step, epoch, cursor, dataset_fingerprint,
                    validation_fingerprint, elapsed_seconds):
    output = Path(output)
    destination = output / f"checkpoint-{step:06d}"
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists: {destination}")
    temp = Path(tempfile.mkdtemp(prefix=f".checkpoint-{step:06d}-", dir=output))
    _write_adapter(temp, runtime)
    torch.save({"optimizer": optimizer.state_dict(), "rng": _rng_state()}, temp / "training_state.pt")
    _json(temp / "trainer_state.json", {
        "format_version": 1, "step": step, "epoch": epoch, "cursor": cursor,
        "dataset_fingerprint": dataset_fingerprint,
        "validation_fingerprint": validation_fingerprint,
        "config_fingerprint": resume_fingerprint(runtime.config),
        "resolved_revision": runtime.resolved_revision,
        "tokenization_fingerprint": runtime.tokenization_fingerprint,
        "elapsed_seconds": elapsed_seconds,
        "full_forwards": runtime.full_forwards, "suffix_forwards": runtime.suffix_forwards,
    })
    os.replace(temp, destination)
    _json(output / "latest.json", {"checkpoint": destination.name, "step": step})
    return destination


def restore_checkpoint(path, runtime, optimizer, dataset_fingerprint, validation_fingerprint):
    path = Path(path)
    state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    expected = {"dataset_fingerprint": dataset_fingerprint,
                "validation_fingerprint": validation_fingerprint,
                "config_fingerprint": resume_fingerprint(runtime.config),
                "tokenization_fingerprint": runtime.tokenization_fingerprint,
                "resolved_revision": runtime.resolved_revision}
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"Cannot resume: {key} changed")
    _load_adapter(path, runtime)
    training = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(training["optimizer"])
    # Adam's checkpoint tensors move with their corresponding FP32 adapter masters.
    for parameter, values in optimizer.state.items():
        for key, value in values.items():
            if isinstance(value, torch.Tensor) and key != "step":
                values[key] = value.to(parameter.device)
    _restore_rng(training["rng"])
    runtime.full_forwards = state.get("full_forwards", 0)
    runtime.suffix_forwards = state.get("suffix_forwards", 0)
    return state


@torch.no_grad()
def evaluate_runtime(runtime, dataset, *, max_batches=0, batch_size=1):
    runtime.clear_interventions()
    total_nll = 0.0
    targets = examples = 0
    start = time.perf_counter()
    for offset in range(0, len(dataset), batch_size):
        if max_batches and offset // batch_size >= max_batches:
            break
        rows = [dataset[index] for index in range(offset, min(len(dataset), offset + batch_size))]
        batch = collate(rows, runtime.device)
        count = int((batch["labels"][:, 1:] != -100).sum())
        loss = float(runtime.losses(batch).sum())
        if not math.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss")
        total_nll += loss * count
        targets += count
        examples += len(rows)
    return {"loss": total_nll / targets if targets else None,
            "perplexity": math.exp(min(total_nll / targets, 80)) if targets else None,
            "supervised_tokens": targets, "examples": examples,
            "seconds": time.perf_counter() - start}


def _memory(runtime):
    if runtime.device.type != "cuda":
        return {"cuda_peak_allocated_bytes": 0, "cuda_peak_reserved_bytes": 0}
    return {"cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(runtime.device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(runtime.device)}


def _check_output_directory(output, *, resume, overwrite):
    """Reject history collisions; overwrite archives only a recognized run."""
    if not output.exists() or not any(output.iterdir()):
        return False
    if resume:
        state = json.loads((Path(resume) / "trainer_state.json").read_text(encoding="utf-8"))
        restored_step = state["step"]
        later = []
        for path in output.iterdir():
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            if match and int(match.group(1)) > restored_step:
                later.append(int(match.group(1)))
        metrics = output / "metrics.jsonl"
        if metrics.is_file():
            with metrics.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        recorded_step = json.loads(line).get("step", 0)
                        if recorded_step > restored_step:
                            later.append(recorded_step)
        if later:
            raise ValueError(f"Run directory contains progress after checkpoint step {restored_step}; resume this checkpoint into a new output_dir")
        return False
    if not overwrite:
        raise FileExistsError(f"Run directory is not empty: {output}; choose another path or resume")
    recognized = {"config.json", "environment.json", "dataset.json", "runtime.json",
                  "metrics.jsonl", "summary.json", "latest.json", "adapter"}
    unknown = [path.name for path in output.iterdir()
               if path.name not in recognized and not re.fullmatch(r"checkpoint-\d+", path.name)
               and not path.name.startswith(".checkpoint-")]
    if unknown or not (output / "config.json").is_file():
        raise ValueError("Refusing overwrite: output directory is not a recognized training run; choose a new output_dir")
    return True


def train(config: TrainConfig, *, resume: str | None = None,
          runtime: LocalRuntime | None = None) -> dict[str, Any]:
    config.validate()
    _seed(config.training.seed)
    resume = resume or config.training.resume
    output = Path(config.training.output_dir)
    archive_existing = _check_output_directory(output, resume=resume, overwrite=config.training.overwrite)
    output.mkdir(parents=True, exist_ok=True)
    runtime = runtime or load_runtime(config)
    if resume_fingerprint(runtime.config) != resume_fingerprint(config):
        raise ValueError("Supplied runtime configuration differs from training configuration")
    train_data, validation = _datasets(config, runtime)
    previous_run = None
    if archive_existing:
        for dataset in (train_data, validation):
            if dataset is not None and dataset.path.resolve().is_relative_to(output.resolve()):
                raise ValueError("Cannot archive a run that contains the current input dataset; choose a new output_dir")
        previous_run = output.with_name(f"{output.name}.previous-{time.time_ns()}")
        os.replace(output, previous_run)
        output.mkdir()
    validation_hash = validation.fingerprint if validation is not None else None
    parameters = list(adapter_parameters(runtime.model))
    optimizer = torch.optim.AdamW(parameters, lr=config.training.learning_rate,
                                 weight_decay=config.training.weight_decay)
    start_step = epoch = cursor = 0
    previous_elapsed = 0.0
    if resume:
        restored = restore_checkpoint(resume, runtime, optimizer, train_data.fingerprint, validation_hash)
        start_step, epoch, cursor = restored["step"], restored["epoch"], restored["cursor"]
        previous_elapsed = restored["elapsed_seconds"]
    if start_step > config.training.max_steps:
        raise ValueError("Requested max_steps is before the resumed checkpoint")
    _json(output / "config.json", config.to_dict())
    _json(output / "environment.json", environment_report(config))
    _json(output / "dataset.json", {"train": train_data.metadata(),
                                    "validation": validation.metadata() if validation else None})
    _json(output / "runtime.json", runtime.metadata())
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)
    start = time.perf_counter()
    targets = list(runtime.adapters)
    pad_id = getattr(runtime.model.config, "pad_token_id", 0) or 0
    last_checkpoint = str(resume) if resume else None
    steps_completed = start_step
    last_validation = None

    def order_for_epoch(number):
        order = list(range(len(train_data)))
        if config.training.shuffle:
            random.Random(config.training.seed + number).shuffle(order)
        return order

    order = order_for_epoch(epoch)

    def next_batch():
        nonlocal epoch, cursor, order
        rows = []
        for _ in range(config.training.batch_size):
            if cursor >= len(order):
                epoch += 1
                cursor = 0
                order = order_for_epoch(epoch)
            rows.append(train_data[order[cursor]])
            cursor += 1
        return collate(rows, runtime.device, pad_id=pad_id)

    metrics_path = output / "metrics.jsonl"
    with metrics_path.open("a" if resume else "w", encoding="utf-8") as log:
        for step in range(start_step, config.training.max_steps):
            before = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            module_name = targets[step % len(targets)]
            batches = [next_batch() for _ in range(config.training.gradient_accumulation_steps)]
            target_counts = [int((batch["labels"][:, 1:] != -100).sum()) for batch in batches]
            total_targets = sum(target_counts)
            clean_loss = 0.0
            probe_reports = []
            replay_modes = []
            full_before, suffix_before = runtime.full_forwards, runtime.suffix_forwards
            try:
                for micro, (batch, count) in enumerate(zip(batches, target_counts)):
                    weight = count / total_targets
                    if config.training.method == "backprop":
                        runtime.clear_interventions()
                        loss = runtime.losses(batch).sum() * weight
                        clean_loss += float(loss.detach())
                        loss.backward()
                    else:
                        with torch.no_grad():
                            clean, captured, replay = runtime.capture(module_name, batch)
                            clean_loss += float(clean.sum()) * weight
                            wrapper = runtime.adapters[module_name]

                            def evaluate_probe(perturbation):
                                wrapper.perturbation = perturbation
                                return runtime.losses(batch, replay) * weight

                            noise_config = NoiseConfig(
                                pairs=config.noise.pairs, sigma=config.noise.sigma,
                                seed=config.training.seed + step * 1_000_003 + micro * 101,
                                probe=config.noise.probe, credit=config.noise.credit,
                            )
                            shape = (*captured.shape[:-1], wrapper.out_features)
                            gradient, stats = estimate_activation_error(
                                evaluate_probe, shape, noise_config, runtime.device,
                                token_mask=batch["attention_mask"])
                            wrapper.perturbation = None
                            assign_lora_gradients(wrapper, captured, gradient, accumulate=micro > 0)
                            probe_reports.append(asdict(stats) if is_dataclass(stats) else dict(stats))
                            replay_modes.append("suffix" if replay is not None else "full_forward")
                if not math.isfinite(clean_loss):
                    raise FloatingPointError("Non-finite clean training loss")
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, config.training.max_grad_norm,
                                                                error_if_nonfinite=True))
                optimizer.step()
            finally:
                runtime.clear_interventions()
            steps_completed = step + 1
            if runtime.device.type == "cuda":
                torch.cuda.synchronize(runtime.device)
            row = {"step": steps_completed, "epoch": epoch, "cursor": cursor,
                   "method": config.training.method,
                   "target_module": module_name if config.training.method == "noise" else "all_adapters",
                   "clean_loss": clean_loss, "grad_norm": grad_norm,
                   "input_tokens": sum(int(batch["attention_mask"].sum()) for batch in batches),
                   "supervised_tokens": total_targets,
                   "step_seconds": time.perf_counter() - before,
                   "full_forwards": runtime.full_forwards - full_before,
                   "suffix_forwards": runtime.suffix_forwards - suffix_before,
                   "probe_replay": replay_modes, "probes": probe_reports, **_memory(runtime)}
            row["input_tokens_per_second"] = row["input_tokens"] / row["step_seconds"]
            row["supervised_tokens_per_second"] = total_targets / row["step_seconds"]
            if validation is not None and len(validation) and config.training.eval_every and steps_completed % config.training.eval_every == 0:
                last_validation = evaluate_runtime(runtime, validation,
                                                   max_batches=config.training.eval_batches,
                                                   batch_size=config.training.batch_size)
                row["validation"] = last_validation
            log.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps({key: row[key] for key in ("step", "clean_loss", "grad_norm", "step_seconds", "probe_replay")}), flush=True)
            if config.training.checkpoint_every and steps_completed % config.training.checkpoint_every == 0:
                last_checkpoint = str(save_checkpoint(
                    output, runtime, optimizer, step=steps_completed, epoch=epoch, cursor=cursor,
                    dataset_fingerprint=train_data.fingerprint, validation_fingerprint=validation_hash,
                    elapsed_seconds=previous_elapsed + time.perf_counter() - start))
    if steps_completed > start_step and (not config.training.checkpoint_every or steps_completed % config.training.checkpoint_every):
        last_checkpoint = str(save_checkpoint(
            output, runtime, optimizer, step=steps_completed, epoch=epoch, cursor=cursor,
            dataset_fingerprint=train_data.fingerprint, validation_fingerprint=validation_hash,
            elapsed_seconds=previous_elapsed + time.perf_counter() - start))
    if validation is not None and len(validation):
        last_validation = evaluate_runtime(runtime, validation, max_batches=config.training.eval_batches,
                                           batch_size=config.training.batch_size)
    _write_adapter(output / "adapter", runtime)
    summary = {"steps": steps_completed, "elapsed_seconds": previous_elapsed + time.perf_counter() - start,
               "checkpoint": last_checkpoint, "adapter": str(output / "adapter"),
               "previous_run": str(previous_run) if previous_run else None,
               "validation": last_validation, "runtime": runtime.metadata(), **_memory(runtime)}
    _json(output / "summary.json", summary)
    return summary


def evaluate(config: TrainConfig, *, checkpoint: str | None = None,
             runtime: LocalRuntime | None = None):
    """Default: load the frozen base. An explicitly supplied runtime is evaluated as-is."""
    _seed(config.training.seed)
    runtime = runtime or load_runtime(config)
    if checkpoint:
        _load_adapter(checkpoint, runtime)
    if not config.data.validation:
        raise ValueError("Configure data.validation for evaluation")
    dataset = TokenDataset(config.data.validation, max_length=config.data.max_length,
                            vocab_size=_vocab_size(runtime))
    _record_tokenization(runtime, {"validation": dataset})
    if not len(dataset):
        raise ValueError("Validation dataset is empty")
    result = evaluate_runtime(runtime, dataset, max_batches=config.training.eval_batches,
                              batch_size=config.training.batch_size)
    result["checkpoint"] = checkpoint
    result["dataset_sha256"] = dataset.fingerprint
    return result
