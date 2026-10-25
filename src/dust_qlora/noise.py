"""Noise-inspired structured activation probes for forward-only adapter updates.

This portable estimator uses randomized Hadamard cycles and paired finite
differences.  It measures full causal downstream loss, rather than copying
Noise's GPT-specific stage/readout approximations.  See docs/algorithm.md.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor


@dataclass(frozen=True)
class NoiseConfig:
    pairs: int = 8
    sigma: float = 0.02
    seed: int = 42
    probe: str = "hadamard"
    credit: str = "causal"

    def __post_init__(self) -> None:
        if isinstance(self.pairs, bool) or not isinstance(self.pairs, Integral) or self.pairs < 1:
            raise ValueError("pairs must be a positive integer")
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        if self.probe not in {"hadamard", "rademacher"}:
            raise ValueError("probe must be 'hadamard' or 'rademacher'")
        if self.credit not in {"causal", "sequence"}:
            raise ValueError("credit must be 'causal' or 'sequence'")


@dataclass(frozen=True)
class ProbeStats:
    pairs: int
    forward_evaluations: int
    elapsed_seconds: float
    mean_absolute_loss_difference: float
    zero_difference_fraction: float
    error_norm: float
    probe: str
    seed: int
    sigma: float


def _validate_shape(shape: tuple[int, ...]) -> tuple[int, int, int]:
    if len(shape) != 3 or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in shape):
        raise ValueError("Activation shape must be positive [batch, tokens, features]")
    if shape[-1] > 2**30:
        raise ValueError("Activation width exceeds the supported Hadamard index range")
    return shape[0], shape[1], shape[2]


def _canonical_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def hadamard_rows(rows: Tensor, width: int) -> Tensor:
    """Return requested Sylvester rows, truncated to ``width`` columns.

    The conceptual matrix has order next_power_of_two(width).  Bit parity
    computes rows directly, avoiding an O(width**2) matrix allocation.  For a
    non-power-of-two width, full-cycle COLUMN covariance is identity; truncated
    rows themselves are not generally orthogonal.
    """
    if isinstance(width, bool) or not isinstance(width, Integral) or width < 1 or width > 2**30:
        raise ValueError("width must be in [1, 2**30]")
    order = 1 << (width - 1).bit_length()
    if rows.dtype not in {torch.int32, torch.int64}:
        raise TypeError("Hadamard row indices must be integers")
    if torch.any(rows < 0) or torch.any(rows >= order):
        raise ValueError("Hadamard row index outside its padded order")
    columns = torch.arange(width, device=rows.device, dtype=torch.int32)
    parity = torch.bitwise_and(rows.to(torch.int32).unsqueeze(-1), columns)
    for shift in (16, 8, 4, 2, 1):
        parity.bitwise_xor_(parity >> shift)
    return (1 - 2 * (parity & 1)).to(torch.float32)


def iter_activation_probes(
    shape: tuple[int, int, int],
    pairs: int,
    *,
    seed: int,
    device: torch.device | str,
    kind: str = "hadamard",
    token_mask: Tensor | None = None,
) -> Iterator[Tensor]:
    """Yield unit-variance probes without allocating a whole population.

    Following Noise's orthogonal-probe construction, each token receives its
    own row offset, each cycle receives random column signs, and each draw has
    independent token-wise signs.  The last signs suppress cross-token credit
    interference; reusing one scalar sign for an entire sequence would not.

    A full padded Hadamard cycle has exact identity within-token covariance.
    A partial cycle is isotropic in expectation over its random starting row.
    No extra factor of width is needed when averaging score * probe.
    """
    batch, tokens, width = _validate_shape(tuple(shape))
    if isinstance(pairs, bool) or not isinstance(pairs, Integral) or pairs < 1:
        raise ValueError("pairs must be positive")
    if kind not in {"hadamard", "rademacher"}:
        raise ValueError("Unknown probe kind")
    device = _canonical_device(device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    if token_mask is not None:
        if tuple(token_mask.shape) != (batch, tokens):
            raise ValueError("token_mask shape must match [batch, tokens]")
        token_mask = token_mask.to(device=device, dtype=torch.bool)

    def signs(size: tuple[int, ...]) -> Tensor:
        return torch.randint(0, 2, size, device=device, generator=generator, dtype=torch.int32).float().mul_(2).sub_(1)

    order = 1 << (width - 1).bit_length()
    offset = None
    column_signs = None
    for index in range(pairs):
        if kind == "rademacher":
            probe = signs((batch, tokens, width))
        else:
            if index % order == 0:
                offset = torch.randint(0, order, (batch, tokens), device=device, generator=generator, dtype=torch.int32)
                column_signs = signs((width,))
            rows = (offset + index % order) % order
            probe = hadamard_rows(rows, width)
            probe.mul_(column_signs)
            probe.mul_(signs((batch, tokens, 1)))
        if token_mask is not None:
            probe.mul_(token_mask.unsqueeze(-1))
        yield probe


def causal_credits(loss_differences: Tensor) -> Tensor:
    """Sum *future* supervised losses within each sequence, never across batch.

    Loss position t must mean the prediction emitted from activation position
    t (normally the target token at t+1).  Masked observations receive credit
    from later assistant predictions even though their own direct loss is zero.
    """
    if loss_differences.ndim != 2:
        raise ValueError("Per-token losses must have shape [batch, tokens]")
    return loss_differences.flip(-1).cumsum(-1).flip(-1)


@torch.no_grad()
def estimate_activation_error(
    evaluate: Callable[[Tensor | None], Tensor],
    shape: tuple[int, int, int],
    config: NoiseConfig,
    device: torch.device | str,
    *,
    token_mask: Tensor | None = None,
) -> tuple[Tensor, ProbeStats]:
    """Estimate dL/dy using serial paired activation interventions.

    ``evaluate(delta)`` runs the same example/model state with ``delta`` added
    at ONE adapted layer.  It returns FP32 per-token losses [B,T], already
    masked and divided by the objective's supervised-token count.  It must
    restore all transient model/recurrent state before each call.  It must NOT
    mutate weights, and the caller must put the model in eval mode (no dropout).

    Torch RNG state is restored between + and - evaluations and after the
    operation.  Probes use a private generator, so they do not disturb dataset
    shuffling.  Inference-mode caches from a previous intervention must not leak
    into the next callback.  ``token_mask`` is the attention/padding mask, NOT
    the assistant loss mask: masked context tokens must still be perturbed.
    """
    batch, tokens, width = _validate_shape(tuple(shape))
    device = _canonical_device(device)
    error = torch.zeros((batch, tokens, width), device=device, dtype=torch.float32)
    abs_difference_sum = 0.0
    zero_differences = 0
    observations = 0
    started = time.perf_counter()
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        for probe in iter_activation_probes(
            shape, config.pairs, seed=config.seed, device=device, kind=config.probe, token_mask=token_mask
        ):
            cpu_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            positive = evaluate(probe * config.sigma).detach().float()
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, device)
            negative = evaluate(probe * -config.sigma).detach().float()
            if positive.shape != (batch, tokens) or negative.shape != (batch, tokens):
                raise ValueError("evaluate must return losses aligned to [batch, tokens]")
            if positive.device != device or negative.device != device:
                raise ValueError("evaluate must return losses on the activation device")
            if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():
                raise FloatingPointError("Non-finite perturbed loss")
            difference = positive - negative
            abs_difference_sum += float(difference.abs().sum().item())
            zero_differences += int((difference == 0).sum().item())
            observations += difference.numel()
            if config.credit == "causal":
                credit = causal_credits(difference)
            else:
                credit = difference.sum(-1, keepdim=True).expand_as(difference)
            # +/-1 coordinate variance is one: the normalization is K, not K*d.
            error.add_(probe * credit.unsqueeze(-1), alpha=1.0 / (2.0 * config.sigma * config.pairs))
    if not torch.isfinite(error).all():
        raise FloatingPointError("Non-finite accumulated activation error")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stats = ProbeStats(
        pairs=config.pairs,
        forward_evaluations=2 * config.pairs,
        elapsed_seconds=time.perf_counter() - started,
        mean_absolute_loss_difference=abs_difference_sum / observations,
        zero_difference_fraction=zero_differences / observations,
        error_norm=float(error.norm().item()),
        probe=config.probe,
        seed=config.seed,
        sigma=config.sigma,
    )
    return error, stats
