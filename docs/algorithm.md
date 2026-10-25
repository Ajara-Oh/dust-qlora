# Noise-inspired forward-only LoRA training

This implementation trains ordinary LoRA factors on a frozen, optionally
bitsandbytes four-bit base model. Its learning signal comes from paired
interventions on one adapted layer's output, using randomized Hadamard probes.
It does not backpropagate through the transformer in `method = "noise"`.

The structured probes are inspired by
[Noise's primary speedrun implementation, pinned at `3343973`](https://github.com/qlabs-eng/noise/blob/3343973b6f1af119effcdea2622f1fe672c51b1e/train/speedrun/run.py).
In particular, that implementation uses token-specific row offsets, random
column signs, and independent token signs to construct orthogonal probe cycles.
The bit-parity row generator here is an original implementation of the standard
Sylvester Hadamard construction. It also supports output dimensions that are not
powers of two.

This is a portable adaptation of that idea for existing Hugging Face models and
exportable LoRA adapters. It does not reproduce Noise's custom GPT architecture,
identity-path initialization, specialized stage/readout training rules, or
published speed results. The [DUST research page](https://qlabs.sh/research/dust)
provides the original motivation for exploring forward-only local training;
this repository's specific estimator and its limitations are defined below.

## 1. What is trainable

For a selected linear module, let an input token be a column vector
`x ∈ R^(d_in)`. The wrapped layer computes

\[
y = W_0x + sBAx,\qquad
A\in\mathbb{R}^{r\times d_{in}},\quad
B\in\mathbb{R}^{d_{out}\times r},\quad s=\alpha/r.
\]

The base module `W0`, including its bias if present, remains frozen. The wrapper
calls the original module, including `bitsandbytes.nn.Linear4bit` when used; it
does not materialize a trainable dense copy of the base weight. `A` and `B` are
FP32 master parameters, while their forward matrix multiplies follow the model's
activation dtype. `A` receives the usual random initialization and `B` starts at
zero, so a newly injected adapter leaves model outputs unchanged.

The default configuration selects MLP `down_proj` modules in the last few
decoder layers. Noise training visits one selected module per optimizer step in
a deterministic round-robin order. Other adapters retain their current values.
The backpropagation baseline updates all selected adapters on each step. For a
fair comparison, report elapsed time, data exposure, and validation quality;
equal step counts do not imply equal work or equal update counts per adapter.

## 2. A correctly aligned, masked causal objective

Training records use unshifted Hugging Face labels: `labels[t]` is either the
input token at position `t`, or `-100` when that target should be ignored.
User messages, system instructions, and tool output remain context. Only the
chosen assistant completion is supervised by the transcript tokenization path.

The runtime returns a loss tensor with the same `[batch, sequence]` dimensions
as the selected layer's activations. **Entry `t` is the loss for predicting
`labels[t + 1]` from activation position `t`.** The final entry is zero. Invalid,
masked, and padded targets contribute zero. Every nonzero entry is already
divided by the number of supervised next-token targets in the objective.
Gradient accumulation additionally weights each microbatch by its share of the
total supervised targets, so variable answer lengths do not change the intended
token mean.

If `ell[b,t]` denotes these already normalized contributions, the objective is

\[
L = \sum_{b,t}\ell_{b,t}.
\]

An activation at token `t` can affect its own prediction and later predictions.
A zero direct loss at that position does **not** mean its activation gradient is
zero. Prompt and tool context can influence the later assistant answer. The
estimator therefore assigns future supervised credit to context positions and
does not use the assistant loss mask as a perturbation mask. Its optional
`token_mask` is exclusively an attention/padding mask.

## 3. Structured paired activation interventions

The selected layer first runs cleanly to capture its input `x`. For probe `k`,
the model is evaluated with an intervention added **after the complete adapted
linear output**:

\[
y^+ = y + \sigma u_k,\qquad y^- = y - \sigma u_k.
\]

The probe `u_k` has shape `[batch, sequence, d_out]`. Each coordinate is `+1`
or `-1`, with unit coordinate variance. Probes are generated and evaluated one
at a time; the population is never expanded into the model's batch dimension.
There are `2 * pairs` perturbed evaluations per microbatch, in addition to the
clean capture and any initial replay verification.

Output-space intervention matters at initialization: with `B = 0`, perturbing
only `Ax` before multiplying by `B` would not change the model's output. Adding
the intervention after `sBAx` retains a measurable output gradient and allows
`B` to learn immediately. Once `B` becomes nonzero, `A` can also learn.

### Hadamard construction and normalization

Let `P` be the smallest power of two at least `d_out`. A Sylvester Hadamard
matrix `H` of order `P` has entries

\[
H_{i,j}=(-1)^{\operatorname{parity}(i\mathbin{\&}j)}.
\]

For each cycle, each token gets a random starting row, and columns get random
signs. Successive probes traverse successive rows cyclically. Each probe also
gets independent scalar signs at every batch/token position. These token signs
are necessary to remove cross-token perturbation terms in expectation; using a
single scalar sign for the whole sequence would not provide that property.

Rows are computed directly by bit parity. There is no dense `P × P` allocation.
For a power-of-two output width, a complete per-token cycle is orthogonal. For a
non-power-of-two width, the matrix is truncated to its first `d_out` columns.
Its rows are then generally **not** orthogonal, but its complete-cycle column
covariance remains exactly identity:

\[
\frac{1}{P}H_{[:, :d]}^\top H_{[:, :d]}=I_d.
\]

For example, an output width of 5,120 uses an implicit order of 8,192 without
requiring the underlying model width to change. A short partial cycle is
isotropic in expectation over its randomized starting row and signs. Exact
full-cycle covariance is a mathematical property, not a recommendation to run
thousands of probes for every training step.

The estimator averages by `pairs`. **There is no extra division or
multiplication by `d_out`**, because the coordinate variance is already one.
Using unit-length probes would require a different normalization.
`probe = "rademacher"` provides an independent-sign ablation; `"hadamard"` is
the default structured probe.

### Future loss credit

For each paired evaluation, form

\[
\Delta\ell_{b,t}=\ell^+_{b,t}-\ell^-_{b,t},\qquad
C_{b,t}=\sum_{q\ge t}\Delta\ell_{b,q}.
\]

`C` is a reverse cumulative sum along the token dimension. It never sums across
batch examples. The estimated output error is

\[
\widehat g_{b,t}
=\frac{1}{K}\sum_{k=1}^K
u_{k,b,t}\frac{C_{k,b,t}}{2\sigma}.
\]

This uses all causal future credit, with no discount or truncation. In a
first-order expansion, the probe covariance preserves the desired gradient and
independent token signs cancel contributions from other perturbed tokens in
expectation. Excluding losses before `t` removes terms that cannot depend on
that activation in a causal model. It does not discard valid future credit.

The estimator is noisy at finite `K`. Paired finite differences also introduce
finite-`sigma` error for nonlinear objectives. In smooth, exact arithmetic,
central differences have second-order error in `sigma`; this does **not** make a
finite-probe, low-precision estimate an exact or unbiased model gradient. The
`credit = "sequence"` option on the low-level estimator is an ablation that
includes the entire sequence loss; the transcript training configuration
requires causal credit.

Causal credit is deliberately conservative about downstream dependencies. For
example, after a final MLP followed only by tokenwise normalization and an output
head, future-token losses do not depend on an earlier token's MLP output. The
generic causal sum still includes those losses, adding variance that cancels
only in expectation. A few probe pairs across thousands of features and context
positions can therefore be very noisy. This implementation does not infer a
more selective credit graph or claim the variance reductions of Noise's custom
local objectives.

## 4. Turning an output error into standard LoRA gradients

Once the output error is estimated, the local LoRA derivatives are ordinary
matrix algebra:

\[
\widehat{\nabla_A L}
=s\sum_{b,t}(B^\top\widehat g_{b,t})x_{b,t}^\top,\qquad
\widehat{\nabla_B L}
=s\sum_{b,t}\widehat g_{b,t}(Ax_{b,t})^\top.
\]

The implementation accumulates these contractions in FP32, in token chunks,
and assigns only the adapter parameters' `.grad` fields. It does not divide by
token count again: normalization already happened in the loss callback.
The trainer clips the adapter gradients and calls AdamW. At `B = 0`, the `A`
gradient is correctly zero, while the `B` gradient can be nonzero.

These are the exact local derivatives of the real-valued LoRA equation when
supplied an exact output gradient. In mixed precision, the forward rounds
activations and factors, while local master-gradient contractions use FP32;
the resulting arithmetic need not be bit-identical to reduced-precision
autograd. CPU FP32 tests compare the contractions directly against autograd.

Saved factors use PEFT-compatible keys such as
`base_model.model.model.language_model.layers.61.mlp.down_proj.lora_A.weight`.
The actual qualified target names come from the loaded model. Exported adapters
must be loaded with the matching base architecture, revision, scaling, and
target modules. The base model itself is not included in adapter checkpoints.

## 5. Execution, precision, and memory

The model runs in evaluation mode, disabling dropout. Each positive/negative
pair uses identical Torch RNG state; the estimator restores the caller's Torch
RNG state afterward. Probe generation uses a private seeded generator. The
callback must otherwise keep weights, inputs, masks, positions, and recurrent
state identical. Python/NumPy randomness inside a custom callback is outside
this Torch RNG guarantee.

The runtime disables generation caches. For supported decoder architectures,
it can capture the clean input to the selected decoder block, then replay that
block and the remaining decoder suffix for each intervention. The initial
clean suffix result is compared with a full forward for the selected target.
Automatic mode falls back to full forwards when the supported replay fails its
equivalence check. Explicit full-forward mode is available for diagnosis.
Every replay starts from the same clean block input and fresh recurrent/KV
state. This is architecture-aware reuse of an unaffected prefix, rather than
Noise's custom stage/readout training algorithm.

The Noise path retains no transformer autograd graph. It still requires the
frozen base model, one current forward's working memory, the captured adapter
input, an FP32 output-error accumulator, and temporary probes. The captured
input is retained in its original compute dtype; local gradient multiplication
converts token chunks to FP32. Vocabulary cross-entropy is computed in chunks
at supervised positions so the whole `[batch, sequence, vocabulary]` tensor is
not retained. AdamW state exists only for the small adapter parameters.

These design choices reduce specific memory costs. They do not guarantee that
every 27B checkpoint, context length, loader, or kernel fits into 24 GB. Loading
the checkpoint may have a different peak from training it. For timing, the
default small `pairs` value is a starting point to measure, not a validated
quality/speed optimum. Full-forward fallback can be expensive; even suffix
replay requires two suffix evaluations per probe pair.

`sigma` is an absolute perturbation per output coordinate. A token's probe norm
is approximately `sigma * sqrt(d_out)`, so width matters to the size of the
intervention. If `sigma` is too small, BF16 rounding or loss cancellation can
erase the measured difference; if too large, the nonlinear finite-difference
error grows. Losses, credits, and accumulators use FP32, but the intervention
still enters the model's compute dtype. Logs include the mean absolute paired
loss difference and the fraction of zero differences. The latter includes
deliberately masked losses and the final token, so a large fraction alone is
not proof that the perturbation failed. An entirely zero measured difference
deserves investigation before continuing a long run.

## 6. What has been verified, and what still needs measurement

The core CPU numerical tests cover:

- Power-of-two Hadamard orthogonality, truncated-width column covariance,
  deterministic seeds, and padding masks.
- Exact recovery of a linear output gradient with a complete probe cycle,
  including the correct width-independent normalization.
- Future credit through masked prompt positions and isolation between batch
  sequences.
- Alignment with an autograd gradient on a small causal masked objective and a
  decrease in that objective after an estimated update.
- A complete forward-only update from zero `B`, and agreement of local LoRA
  contractions with exact FP32 autograd.
- Paired Torch randomness, absence of dense base gradients, preservation of a
  frozen weight-only quantized test layer, FP32 adapter masters, and strict
  adapter serialization.
- Integration with the runtime's actual masked causal loss, equal updates from
  variable-length microbatch accumulation versus full-batch autograd when an
  exact output-error oracle is supplied, and skipped AdamW momentum/weight decay
  for inactive round-robin adapters.

The quantized test layer is a CPU test double; it does not validate CUDA
bitsandbytes kernels. Tiny-model and runtime replay checks do not establish
27B convergence, a 3090 peak-memory figure, throughput, or domain quality.

Before a months-long transcript run, compare the frozen base, Noise adapters,
and the backpropagation baseline on a session-separated validation split and
representative held-out harness tasks. Record wall-clock time, peak allocated
and reserved VRAM, tokens seen, per-adapter updates, validation loss, and task
success. Check several seeds and probe/sigma settings on a small subset first.
That experiment determines whether forward-only memory savings are useful for
this checkpoint and domain; the estimator alone does not establish an advantage
over conventional QLoRA.
