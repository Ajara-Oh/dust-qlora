"""Strict, dependency-free configuration for local training."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
import tomllib
from typing import Any


@dataclasses.dataclass
class ModelConfig:
    name: str = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"
    revision: str | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    load_in_4bit: bool = True
    offload_embedding: bool = True
    offload_vision: bool = True
    attn_implementation: str = "sdpa"
    local_files_only: bool = False
    replay_mode: str = "auto"
    tiny_vocab_size: int = 64
    tiny_hidden_size: int = 32
    tiny_num_layers: int = 3


@dataclasses.dataclass
class AdapterConfig:
    rank: int = 8
    alpha: float = 8.0
    last_n_layers: int = 4
    target_modules: list[str] = dataclasses.field(default_factory=lambda: ["mlp.down_proj"])
    explicit_targets: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class NoiseSettings:
    pairs: int = 4
    sigma: float = 0.02
    probe: str = "hadamard"
    credit: str = "causal"


@dataclasses.dataclass
class DataConfig:
    train: str = "data/tokenized/train.jsonl"
    validation: str | None = "data/tokenized/validation.jsonl"
    max_length: int = 2048


@dataclasses.dataclass
class TrainingConfig:
    method: str = "noise"
    output_dir: str = "runs/qwen38-noise"
    max_steps: int = 100
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 42
    checkpoint_every: int = 25
    eval_every: int = 25
    eval_batches: int = 8
    loss_chunk_size: int = 32
    shuffle: bool = True
    resume: str | None = None
    overwrite: bool = False


@dataclasses.dataclass
class TrainConfig:
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    adapter: AdapterConfig = dataclasses.field(default_factory=AdapterConfig)
    noise: NoiseSettings = dataclasses.field(default_factory=NoiseSettings)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> "TrainConfig":
        positive = {
            "adapter.rank": self.adapter.rank,
            "adapter.last_n_layers": self.adapter.last_n_layers,
            "noise.pairs": self.noise.pairs,
            "training.max_steps": self.training.max_steps,
            "training.batch_size": self.training.batch_size,
            "training.gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "training.loss_chunk_size": self.training.loss_chunk_size,
            "data.max_length": self.data.max_length,
        }
        for key, value in positive.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key, value in {
            "noise.sigma": self.noise.sigma,
            "adapter.alpha": self.adapter.alpha,
            "training.learning_rate": self.training.learning_rate,
            "training.max_grad_norm": self.training.max_grad_norm,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be a finite positive number")
        if not math.isfinite(self.training.weight_decay) or self.training.weight_decay < 0:
            raise ValueError("training.weight_decay must be finite and nonnegative")
        for key in ("checkpoint_every", "eval_every", "eval_batches"):
            value = getattr(self.training, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"training.{key} must be a nonnegative integer")
        if self.training.method not in {"noise", "backprop"}:
            raise ValueError("training.method must be noise or backprop")
        if self.noise.probe not in {"hadamard", "rademacher"}:
            raise ValueError("noise.probe must be hadamard or rademacher")
        if self.noise.credit != "causal":
            raise ValueError("Only causal future-token credit is supported for transcript training")
        if self.model.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("model.dtype must be float32, float16, or bfloat16")
        if self.model.dtype == "float16":
            raise ValueError("Use bfloat16 on the RTX 3090 or float32 on CPU; float16 gradient scaling is not implemented")
        if self.model.replay_mode not in {"auto", "full", "suffix"}:
            raise ValueError("model.replay_mode must be auto, full, or suffix")
        if self.data.max_length < 2:
            raise ValueError("data.max_length must be at least 2")
        if not self.adapter.explicit_targets and not self.adapter.target_modules:
            raise ValueError("At least one adapter target must be configured")
        if not isinstance(self.adapter.target_modules, list) or not isinstance(self.adapter.explicit_targets, list):
            raise ValueError("Adapter targets must be arrays of module names")
        for name in self.adapter.target_modules + self.adapter.explicit_targets:
            if not isinstance(name, str) or not name:
                raise ValueError("Adapter targets must be nonempty module names")
        for name in ("load_in_4bit", "offload_embedding", "offload_vision", "local_files_only"):
            if not isinstance(getattr(self.model, name), bool):
                raise ValueError(f"model.{name} must be a boolean")
        for name in ("shuffle", "overwrite"):
            if not isinstance(getattr(self.training, name), bool):
                raise ValueError(f"training.{name} must be a boolean")
        if isinstance(self.training.seed, bool) or not isinstance(self.training.seed, int) or self.training.seed < 0:
            raise ValueError("training.seed must be a nonnegative integer")
        for name in ("name", "device", "attn_implementation"):
            if not isinstance(getattr(self.model, name), str) or not getattr(self.model, name):
                raise ValueError(f"model.{name} must be a nonempty string")
        if self.model.revision is not None and not isinstance(self.model.revision, str):
            raise ValueError("model.revision must be a string")
        for name in ("tiny_vocab_size", "tiny_hidden_size", "tiny_num_layers"):
            value = getattr(self.model, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"model.{name} must be a positive integer")
        return self


def load_config(path: str | Path) -> TrainConfig:
    """Load TOML. Relative dataset/run paths are relative to the working directory."""
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    section_types = {"model": ModelConfig, "adapter": AdapterConfig, "noise": NoiseSettings,
                     "data": DataConfig, "training": TrainingConfig}
    unknown = set(raw) - set(section_types)
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    sections: dict[str, Any] = {}
    for name, cls in section_types.items():
        values = raw.get(name, {})
        if not isinstance(values, dict):
            raise ValueError(f"[{name}] must be a TOML table")
        extra = set(values) - {field.name for field in dataclasses.fields(cls)}
        if extra:
            raise ValueError(f"Unknown keys in [{name}]: {sorted(extra)}")
        sections[name] = cls(**values)
    return TrainConfig(**sections).validate()


def resume_fingerprint(config: TrainConfig) -> str:
    """Identity of learning semantics; duration/output/evaluation cadence may change."""
    raw = config.to_dict()
    for key in ("max_steps", "output_dir", "resume", "overwrite", "checkpoint_every",
                "eval_every", "eval_batches"):
        raw["training"].pop(key, None)
    raw["data"].pop("train", None)
    raw["data"].pop("validation", None)
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
