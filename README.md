# dust-qlora

Local **Noise-inspired, forward-only LoRA training** on a frozen four-bit model,
with converters for **Pi and Claude Code transcripts**. The initial configuration
targets Qwen3.8-27B on an RTX 3090 with 24 GB VRAM and 64 GB system RAM.

The repository includes an executable training loop, native Qwen decoder suffix
replay, numerical tests, resumable checkpoints, PEFT-format adapter exports, and
a conventional backpropagation baseline. It is an experimental adaptation of
[DUST](https://qlabs.sh/research/dust) and
[Noise](https://github.com/qlabs-eng/noise), not a reproduction of the original
GPT speedrun or a demonstrated speed advantage for 27B QLoRA.

**CI is manual only.** The sole GitHub Actions workflow uses `workflow_dispatch`.
Pushes and pull requests do not trigger it; there is no deployment workflow.
Run **Actions → Manual CPU checks → Run workflow** when you want checks.

## What the trainer does

- Loads a Hugging Face bitsandbytes four-bit checkpoint, or quantizes compatible
  original weights to NF4. Existing GGUF/AWQ inference files are not training
  inputs for this runtime.
- Freezes the base model and places rank-8 LoRA adapters on `mlp.down_proj` in the
  last four decoder layers by default. Both LoRA factors can learn; zero-initialized
  `B` receives a signal because perturbations enter the adapted layer's **output**.
- Applies randomized Hadamard probes in positive/negative pairs, estimates
  causal future-token credit, and computes the small LoRA gradients locally.
  Noise mode does not call transformer backpropagation.
- Visits one adapter per optimizer step. The default four probe pairs require
  eight perturbed suffix evaluations plus a clean capture, not one free forward.
- Uses the actual Qwen3.5/3.8 decoder blocks, including hybrid attention, with
  fresh recurrent state. It verifies clean suffix replay against a full forward
  and falls back to full forwards in automatic mode if replay is unsupported.
- Computes vocabulary losses in chunks at supervised positions, uses FP32
  adapter parameters/gradient accumulation, and records timing, probe signals,
  forward counts, validation loss, and peak CUDA allocation/reservation locally.

See [the algorithm and its limits](docs/algorithm.md) for the estimator,
normalization, causal masking, and numerical checks.

## 1. Install locally

Use Linux/WSL2, Python 3.11 or newer, and a recent NVIDIA driver. First install a
CUDA-enabled PyTorch build using the command from the
[official PyTorch installer](https://pytorch.org/get-started/locally/), then:

```bash
git clone https://github.com/Ajara-Oh/dust-qlora.git
cd dust-qlora
python3 -m venv .venv
source .venv/bin/activate
# Install CUDA-enabled PyTorch into this virtual environment, then:
python -m pip install -e '.[train,test]'
python -m pytest -q
dust-qlora smoke --output-dir runs/cpu-smoke
```

The smoke command is fully offline and uses a tiny randomly initialized CPU
model with synthetic token rows. It exercises the real estimator, optimizer,
evaluation, checkpointing, and adapter export. It is not a GPU fit or domain
quality benchmark. For CPU-only development, install CPU PyTorch, then use
`python -m pip install -e '.[tokenize,test]' peft accelerate safetensors` instead.

Conversion alone needs no ML packages: `python -m pip install -e .` is enough
for `dust-qlora prepare`. Tokenization needs `.[tokenize]`.

No training data is uploaded. Conversion uses no network. Tokenization/model
loading may download public Hugging Face assets unless local-only mode is set.
After downloading the chosen checkpoint, use a local model path or the cached
model with `local_files_only = true`, pass `--local-files-only` to tokenization,
and optionally set `HF_HUB_OFFLINE=1`. Telemetry defaults are disabled; metrics
are written to local JSONL without an external experiment tracker.

## 2. Convert the transcript archive

Point the converter at the directories that actually hold your logs. Common
locations are `~/.pi/agent/sessions` and `~/.claude/projects`; repeat `--input`
for archives or additional locations:

```bash
dust-qlora prepare \
  --input ~/.pi/agent/sessions \
  --input ~/.claude/projects \
  --output-dir data/prepared \
  --validation-fraction 0.1
```

This writes:

| File | Purpose |
| --- | --- |
| `train.jsonl` | Canonical conversations with explicit per-message target eligibility |
| `validation.jsonl` | Held-out session families, split before training windows are made |
| `report.json` | Source counts, masking/skipping/redaction/dedup counts, and split sizes |
| `review_template.jsonl` | Stable target IDs for optional accept/reject/correction review |

The converter reconstructs parent-linked sessions, coalesces Claude assistant
blocks, recovers parallel tool-call/result siblings, and handles recorded Pi
compaction/context edits. It preserves actual tool arguments and observations.
Only eligible assistant messages are targets. Users, system prompts, and tool
results remain context. Failed/incomplete responses and missing-image-dependent
targets are masked by default; error context remains available for learning
recovery. Reasoning is retained separately if it was actually recorded.

Repeated prefixes and linked forks/resumes/subagents are grouped before the
train/validation split. Deduplication is exact, not semantic. A very small
archive may have no validation family; inspect the report before training.
`--branch-policy latest` is the default. Use `--branch-policy all` to include all
eligible branches. Source files are never modified.

**All eligible observed answers are not all correct answers.** For a curated
training set, copy `review_template.jsonl` to a separate local file, mark the
targets you accept (or supply corrections), then run:

```bash
dust-qlora prepare \
  --input ~/.pi/agent/sessions \
  --input ~/.claude/projects \
  --selection reviewed \
  --review-file data/reviews.jsonl \
  --output-dir data/prepared-reviewed
```

The [data format guide](docs/data-format.md) documents review records, source
formats, filtering, privacy handling, and parser limits. The converter redacts
common literal secrets by default, but this is not complete de-identification.
Raw/converted logs, checkpoints, and model weights are gitignored. Only explicitly
synthetic test fixtures belong in Git.

## 3. Tokenize for the exact model and harness

Use the same model name/path and revision in tokenization and training. Pin a
Hub commit with `--revision COMMIT_SHA` and `model.revision = "COMMIT_SHA"` for
reproducible runs. The trainer checks available tokenizer/model identity and
dataset fingerprints; it does not translate tokens between models.

```bash
dust-qlora tokenize \
  --input-dir data/prepared \
  --output-dir data/tokenized \
  --model unsloth/Qwen3.8-27B-unsloth-bnb-4bit \
  --revision 8aa5f05d26b7205477066e1449e0af13f762a299 \
  --max-length 2048 \
  --context-policy recent
```

If your archive lacks the actual deployment system prompt or tool schemas,
provide them using `--system-file path/to/system.txt` and
`--tools-file path/to/tools.json`. A tool file is an array of actual function
schemas, in OpenAI `{"type":"function","function":...}` form or plain
`{"name":...,"description":...,"parameters":...}` form. The pipeline does not
invent missing schemas or convert Pi tools into Claude Code tools: train and
evaluate with the vocabulary of the harness you will deploy.

The tokenizer renders the model's native chat template. Each example supervises
one complete assistant decision, including tool-call syntax and recorded
reasoning, and masks all preceding context. The completed chat must extend the
exact native generation prompt; incompatible templates fail instead of guessing
a loss mask. The tokenizer, template settings, hashes, token counts, and skipped
target counts are saved beside the token rows in `manifest.json`.

The default `recent` policy keeps the system prompt and the entire latest user
task, including all subsequent tool/assistant context, then drops older context
exchanges as necessary. Tool calls and their results are removed together.
Parallel results are put back in call order for Qwen's unlabeled response
template. Missing or malformed tool exchanges are counted and skipped. Every
context reduction is counted; current-task observations and assistant targets
are never truncated. Use `--context-policy complete` to skip targets that require
any context reduction. For long coding sessions, examine the reduction/skip rate;
a 2K window cannot faithfully teach every task that depended on a long context.

Thinking is enabled by default, with `--reasoning-effort medium` and preserved
recorded reasoning. No hidden reasoning is synthesized. `--thinking off` is for
data that matches a non-thinking generation prompt; incompatible recorded
reasoning is rejected by the exact-prefix check.

## 4. Preflight the 3090

Stop the inference server so training can use the GPU. Edit
[`configs/qwen38-27b-3090.toml`](configs/qwen38-27b-3090.toml) if your checkpoint,
dataset paths, or device differ. Relative paths are relative to your working
directory, so run these commands from the repository root.

```bash
# Metadata only: no model downloads or weight loading.
dust-qlora preflight --config configs/qwen38-27b-3090.toml

# Explicitly load the model and probe the first batch without an optimizer step.
dust-qlora preflight --config configs/qwen38-27b-3090.toml --probe
```

The second command reports actual loading/probe behavior, chosen replay mode,
paired-loss signal, timing and GPU memory. It uses the first batch, so check a
representative maximum-length batch before assuming the entire dataset fits.

The default enables CPU embedding and vision-tower offload after loading, BF16
compute, small vocabulary-loss chunks, batch size one, and late-layer adapters.
**The current loader initially loads the full quantized checkpoint on the GPU**;
offload is applied afterward. A load-time OOM cannot be solved by lowering the
sequence length. Close other GPU processes or use a smaller compatible
checkpoint. Forward-time OOMs can often be investigated with shorter tokenized
windows, smaller loss chunks, or fewer/lower-rank adapters. Fewer probe pairs
reduce compute, but probes are already streamed and do not multiply batch VRAM.

This repository has no measured 3090/27B VRAM or throughput claim. A successful
CPU smoke test does not verify bitsandbytes CUDA kernels, the full checkpoint,
or long-context memory. The backpropagation baseline can also have a different
peak and should be measured separately.

## 5. Run a short pilot, evaluate, and resume

```bash
# Frozen-base validation before training.
dust-qlora evaluate --config configs/qwen38-27b-3090.toml

# Initial 100-step experiment (override --max-steps for a shorter pilot).
dust-qlora train --config configs/qwen38-27b-3090.toml

# Evaluate the exported adapter; 0 means every validation batch.
dust-qlora evaluate \
  --config configs/qwen38-27b-3090.toml \
  --checkpoint runs/qwen38-noise/adapter \
  --max-batches 0

# Continue the same run to a TOTAL of 500 optimizer steps.
dust-qlora train \
  --config configs/qwen38-27b-3090.toml \
  --resume runs/qwen38-noise/checkpoint-000100 \
  --max-steps 500
```

Training saves local metrics, resolved configuration/environment, data hashes,
runtime metadata, atomic checkpoint directories, and `adapter/` containing
`adapter_model.safetensors` plus `adapter_config.json`. Checkpoints include
optimizer state, RNG state, data cursor, and compatibility fingerprints. Resuming
with changed training semantics or changed data is rejected. The ordinary
adapter export does not include the frozen base weights or optimizer state.
To branch from an older checkpoint after a run has advanced, choose a new
`--output-dir`; the trainer rejects conflicting histories. The optional
`training.overwrite = true` archives a recognized old run to a sibling directory
instead of deleting its checkpoints. Keep local base-model directories immutable
between runs: local weight files are not individually hashed.

Load the export with PEFT against the matching base model and revision. The
training loader uses `AutoModelForImageTextToText` for the current Qwen3.8
multimodal architecture, although this pipeline trains only text transcripts.
Your serving runtime must support that model's LoRA module names and format;
GGUF adapter conversion is outside this repository.

Run the conventional baseline on the same data and placement:

```bash
dust-qlora train --config configs/qwen38-27b-backprop.toml
```

Noise updates one adapter per step; backprop updates every adapter per step.
Compare wall time, input/supervised tokens, per-adapter exposure, held-out loss,
and actual harness task success. More months of transcripts or more probes do
not automatically improve domain quality. Tune on a small, reviewed subset and
keep held-out task/session families separate before a long local run.

## Development and verification

```bash
python -m ruff check src tests
python -m pytest -q
dust-qlora smoke --output-dir runs/another-cpu-smoke
```

Tests cover the estimator against known gradients, zero-`B` initialization,
frozen base weights, prompt-to-future credit, tokenization masks, transcript
reconstruction, split leakage, checkpoint resume, and offline training. Runtime
tests instantiate small real Qwen configurations without downloading weights.
All fixtures are synthetic. See [validation notes](docs/validation.md) for the
tested environment and remaining hardware checks.

MIT licensed. [NOTICE](NOTICE) records upstream inspiration and attribution.
