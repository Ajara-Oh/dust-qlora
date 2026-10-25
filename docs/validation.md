# Implementation validation

Validated on **2026-10-25** in a CPU environment. No user transcript archive,
RTX 3090, or full 27B model was available to this implementation session.

Final result: **93 tests passed, plus 2 unittest subtests** with the optional
cached-Qwen-tokenizer checks enabled. Ruff and Git whitespace checks passed.
The default offline suite skips the two real-tokenizer checks when their local
assets are not supplied.

## Environment

| Component | Tested version |
| --- | --- |
| Python | 3.12.14 |
| PyTorch | 2.14.1+cpu |
| Transformers | 5.20.2 |
| PEFT | 0.21.2 |
| Accelerate | 1.15.0 |
| Safetensors | 0.8.0 |
| Pytest | 9.1.1 |
| Ruff | 0.17.2 |

These are the versions used for the checks below, not a claim that every
combination allowed by the dependency ranges has been tested. CUDA
bitsandbytes was not installed or exercised in this environment.

## Checks performed

- **Estimator numerics:** complete Hadamard-cycle linear gradient recovery,
  covariance at non-power-of-two widths, causal future credit, paired RNG,
  cross-example isolation, zero-`B` learning, and LoRA contractions against
  FP32 autograd. A small actual-runtime gradient alignment test uses 1,024
  pairs; it does not establish four-pair quality at a 27B hidden width.
- **Training integration:** real optimizer updates, frozen base parameters,
  unequal-length microbatch weighting, inactive-adapter AdamW isolation,
  checkpoint identity checks, and bitwise CPU uninterrupted/resumed equivalence
  for both Noise and backpropagation.
- **Native architecture:** instantiate tiny real Qwen3.5 text models with both
  DeltaNet and full attention, compare clean and perturbed suffix replay to full
  forwards, save/load a tiny multimodal checkpoint through the actual model
  loader, and reload the exported adapter with PEFT to compare model loss.
  These tests need no external weights.
- **Transcripts and masks:** synthetic Pi/Claude trees, split assistant blocks,
  parallel tools, missing results, interruptions, errors, compaction, context
  edits, missing images, reviews, exact deduplication, family splits, loss masks,
  complete assistant targets, protected current-task observations, and private
  output permissions.
- **Command line:** install the package, run an offline four-step CPU smoke
  experiment with validation/checkpoints/adapter export, run metadata preflight,
  and run a real first-batch probe on the tiny model without an optimizer step.
- **Automation:** parse the workflow and require `workflow_dispatch` to be its
  only trigger. No push, pull request, scheduled, or deployment workflow exists.

## Actual Qwen tokenizer check

Only public tokenizer assets were downloaded from
[`unsloth/Qwen3.8-27B-unsloth-bnb-4bit`](https://huggingface.co/unsloth/Qwen3.8-27B-unsloth-bnb-4bit/tree/8aa5f05d26b7205477066e1449e0af13f762a299),
at commit `8aa5f05d26b7205477066e1449e0af13f762a299`. No 27B weights were
downloaded. Native template SHA-256:

```
12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce
```

The synthetic Pi and Claude fixtures produced two canonical conversations and
three assistant-decision examples: **148 input tokens / 54 supervised tokens**.
The actual Qwen tokenizer preserved assistant terminators, rendered reasoning
and tool calls, excluded user/tool text from target masks, and restored parallel
tool results to their call order. Non-thinking text-only examples also passed.
The native tokenizer snapshot revision is resolved and saved in the tokenization
manifest before assets are loaded, so a moving Hub branch cannot mix snapshots.

Optional real-tokenizer tests run offline when a local tokenizer is supplied:

```bash
DUST_QWEN_TOKENIZER=/absolute/path/to/cached/tokenizer \
HF_HUB_OFFLINE=1 python -m pytest -q tests/test_qwen_tokenizer.py
```

These two tests skip by default rather than downloading assets in CI. The rest
of the suite is offline and uses synthetic data and random tiny model weights.

## Transcript window-search check

On a synthetic 100-turn conversation with a 400-token limit, the actual Qwen
tokenizer generated 100 decisions in about **3.34 seconds**, with 838 encoding
attempts and no oversized example. This measures a small local CPU case, not
throughput for an arbitrary transcript archive.

The window search starts from the required current task and grows a suffix of
complete prior tasks, then refines it. It avoids repeatedly encoding a full
session while deleting one message at a time. It still checks every accepted
rendering exactly; long active tasks that cannot fit are counted and skipped.

## Still to measure on the target machine

1. Loading and maximum-length training peaks for the actual checkpoint and its
   quantization exclusions, including CUDA bitsandbytes kernels. Embedding and
   vision offload currently happen **after** the full checkpoint loads on GPU.
2. Whether the native DeltaNet kernel configuration provides usable speed on
   the 3090, and whether suffix replay verifies for the installed runtime.
3. Paired-loss signal at BF16 precision for the chosen absolute `sigma`, probe
   variance at a practical number of pairs, and task/context distributions.
4. Held-out domain task success, regression rates, memorization, and a matched
   comparison against ordinary backpropagation QLoRA.
5. Coverage and conversion reports for the user's actual Pi/Claude versions and
   archives. A synthetic schema regression suite is not an inspection of those
   private logs.

Start with a short pilot using `preflight --probe` and the provided configuration.
Use its measured memory, timings, and validation results before committing to a
long training run. No 3090 fit, runtime estimate, 27B convergence result, or
advantage over ordinary QLoRA is claimed by the CPU checks.
