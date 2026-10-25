"""Small, exportable LoRA modules with explicit activation interventions.

The base module is never dequantized or copied by this module.  In particular,
bitsandbytes Linear4bit can remain the frozen ``base`` implementation.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from numbers import Integral

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_target(base: nn.Module, rank: int, alpha: float) -> None:
    if isinstance(rank, bool) or not isinstance(rank, Integral) or rank < 1:
        raise ValueError("LoRA rank must be a positive integer")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA alpha must be finite and positive")
    if not hasattr(base, "in_features") or not hasattr(base, "out_features"):
        raise TypeError("Only linear modules with in_features/out_features are supported")
    if rank > min(base.in_features, base.out_features):
        raise ValueError("LoRA rank exceeds the selected layer dimensions")
    weight = getattr(base, "weight", None)
    if weight is None:
        raise TypeError("The base linear module must expose its weight device")
    if weight.device.type == "meta":
        raise ValueError("Load the base weights before injecting adapters")


class LowRankLinear(nn.Module):
    """Frozen linear layer plus FP32 LoRA masters and an optional output probe.

    ``capture_input`` captures only a detached input reference, in its original
    dtype.  The training loop must turn it off for perturbation forwards and
    clear ``last_input`` after using it.  ``perturbation`` is added after the
    adapter, so probes retain a signal with the usual zero-B initialization.

    Adapter matmuls follow the activation dtype.  Their trainable master tensors
    stay FP32 for both manual Noise updates and the ordinary-backprop baseline.
    """

    def __init__(
        self,
        base: nn.Module,
        rank: int,
        alpha: float,
        *,
        name: str = "",
    ) -> None:
        super().__init__()
        _validate_target(base, rank, alpha)
        self.base = base
        self.name = name
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        weight = base.weight
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, self.in_features, device=weight.device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, self.rank, device=weight.device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.capture_input = False
        self.last_input: Tensor | None = None
        self.perturbation: Tensor | None = None

    @property
    def weight(self) -> Tensor:
        """Expose the base weight for model code that inspects layer metadata."""
        return self.base.weight

    @property
    def bias(self) -> Tensor | None:
        return getattr(self.base, "bias", None)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.capture_input:
            self.last_input = x.detach()
        output = self.base(x, *args, **kwargs)
        if not isinstance(output, Tensor):
            raise TypeError("The adapted base module must return a Tensor")
        # Casting the small matrices is differentiable, retaining FP32 masters.
        a = self.lora_A.to(dtype=x.dtype)
        b = self.lora_B.to(dtype=x.dtype)
        delta = F.linear(F.linear(x, a), b).to(output.dtype)
        result = output + delta * self.scaling
        if self.perturbation is not None:
            if self.perturbation.shape != result.shape:
                raise ValueError(
                    f"Probe shape {tuple(self.perturbation.shape)} does not match "
                    f"{self.name or 'adapter'} output {tuple(result.shape)}"
                )
            # Preserve the model's compute dtype; sigma must exceed its rounding
            # floor.  Zero-difference probe statistics make this visible.
            result = result + self.perturbation.to(device=result.device, dtype=result.dtype)
        return result

    def clear_capture(self) -> None:
        self.capture_input = False
        self.last_input = None
        self.perturbation = None


def inject_lora(
    model: nn.Module,
    target_names: Sequence[str],
    rank: int,
    alpha: float,
) -> dict[str, LowRankLinear]:
    """Freeze the model and replace *exact* qualified module names.

    Explicit names prevent an accidental all-layer or vision-layer training
    run.  Validate the complete target list before mutating the model.
    """
    names = list(target_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Provide a non-empty list of unique adapter target names")
    modules = dict(model.named_modules())
    for name in names:
        if not name or name not in modules:
            raise ValueError(f"Unknown adapter target: {name!r}")
        if isinstance(modules[name], LowRankLinear):
            raise ValueError(f"Target already has an adapter: {name}")
        if any(other != name and other.startswith(name + ".") for other in names):
            raise ValueError("Nested adapter targets are not supported")
        _validate_target(modules[name], rank, alpha)
    wrappers = {name: LowRankLinear(modules[name], rank, alpha, name=name) for name in names}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, wrapper in wrappers.items():
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        parent._modules[child_name] = wrapper
        wrapper.lora_A.requires_grad_(True)
        wrapper.lora_B.requires_grad_(True)
    return wrappers


def iter_adapters(model: nn.Module) -> Iterator[tuple[str, LowRankLinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LowRankLinear):
            yield name, module


def adapter_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    for _, adapter in iter_adapters(model):
        yield adapter.lora_A
        yield adapter.lora_B


def adapter_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Return CPU FP32 tensors using PEFT's saved-adapter naming convention."""
    state: dict[str, Tensor] = {}
    for name, adapter in iter_adapters(model):
        prefix = "base_model.model." + name
        state[prefix + ".lora_A.weight"] = adapter.lora_A.detach().float().cpu().clone(memory_format=torch.contiguous_format)
        state[prefix + ".lora_B.weight"] = adapter.lora_B.detach().float().cpu().clone(memory_format=torch.contiguous_format)
    if not state:
        raise ValueError("The model has no LowRankLinear adapters")
    return state


@torch.no_grad()
def load_adapter_state_dict(model: nn.Module, state: dict[str, Tensor]) -> None:
    """Strictly restore compatible PEFT-keyed adapter weights; never base weights."""
    expected = {
        "base_model.model." + name + "." + factor + ".weight": getattr(module, factor)
        for name, module in iter_adapters(model)
        for factor in ("lora_A", "lora_B")
    }
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise ValueError(f"Adapter keys differ (missing={missing}, extra={extra})")
    for key, parameter in expected.items():
        if state[key].shape != parameter.shape:
            raise ValueError(f"Adapter shape differs for {key}")
        if not torch.isfinite(state[key]).all():
            raise ValueError(f"Non-finite adapter tensor: {key}")
    for key, parameter in expected.items():
        parameter.copy_(state[key].to(device=parameter.device, dtype=parameter.dtype))


@torch.no_grad()
def assign_lora_gradients(
    adapter: LowRankLinear,
    clean_input: Tensor,
    output_error: Tensor,
    *,
    accumulate: bool = False,
    token_chunk_size: int = 256,
) -> tuple[Tensor, Tensor]:
    """Convert an output error into LoRA gradients with bounded FP32 scratch.

    The supplied output error already includes the loss's batch/token mean.
    There is deliberately no additional division by token count here.
    ``output_error`` can be either an exact derivative or a Noise estimate.
    """
    if token_chunk_size < 1:
        raise ValueError("token_chunk_size must be positive")
    if clean_input.shape[:-1] != output_error.shape[:-1]:
        raise ValueError("Input and output-error batch/token dimensions differ")
    if clean_input.shape[-1] != adapter.in_features:
        raise ValueError("Input feature count differs from the adapter")
    if output_error.shape[-1] != adapter.out_features:
        raise ValueError("Output-error feature count differs from the adapter")
    if clean_input.device != adapter.lora_A.device or output_error.device != clean_input.device:
        raise ValueError("Adapter, clean input and error must be on the same device")
    if not torch.isfinite(output_error).all():
        raise FloatingPointError("Non-finite estimated activation error")
    x = clean_input.detach().reshape(-1, adapter.in_features)
    g = output_error.detach().reshape(-1, adapter.out_features)
    ga = torch.zeros_like(adapter.lora_A, dtype=torch.float32)
    gb = torch.zeros_like(adapter.lora_B, dtype=torch.float32)
    # Explicitly disable surrounding mixed-precision contexts: these are FP32
    # master gradients, not another low-precision model forward.
    with torch.autocast(device_type=x.device.type, enabled=False):
        for start in range(0, x.shape[0], token_chunk_size):
            xc = x[start : start + token_chunk_size].float()
            gc = g[start : start + token_chunk_size].float()
            z = F.linear(xc, adapter.lora_A.float())
            gb.add_(gc.T @ z, alpha=adapter.scaling)
            ga.add_((gc @ adapter.lora_B.float()).T @ xc, alpha=adapter.scaling)
    if not torch.isfinite(ga).all() or not torch.isfinite(gb).all():
        raise FloatingPointError("Non-finite LoRA gradient")
    for parameter, gradient in ((adapter.lora_A, ga), (adapter.lora_B, gb)):
        if accumulate and parameter.grad is not None:
            parameter.grad.add_(gradient)
        else:
            parameter.grad = gradient
    return ga, gb
