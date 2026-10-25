"""Local conversion, tokenization, preflight, training and evaluation commands."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def _print(value):
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _parser():
    parser = argparse.ArgumentParser(
        prog="dust-qlora", description="Local Noise-inspired QLoRA and harness transcript preparation"
    )
    parser.add_argument("--debug", action="store_true", help="Show exception tracebacks")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Convert local Pi/Claude Code JSONL logs")
    prepare.add_argument("--input", action="append", required=True, dest="inputs",
                         help="Local file or directory; repeat to combine sources")
    prepare.add_argument("--output-dir", default="data/prepared")
    prepare.add_argument("--source", choices=("auto", "pi", "claude"), default="auto")
    prepare.add_argument("--validation-fraction", type=float, default=0.1)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--branch-policy", choices=("latest", "all"), default="latest")
    prepare.add_argument("--selection", choices=("all", "reviewed"), default="all")
    prepare.add_argument("--review-file")
    prepare.add_argument("--include-sidechains", action="store_true")
    prepare.add_argument("--no-redact-secrets", action="store_true")
    prepare.add_argument("--allow-missing-images", action="store_true")
    prepare.add_argument("--keep-failed-tool-call-targets", action="store_true")
    prepare.add_argument("--max-line-mib", type=int, default=16)
    prepare.add_argument("--max-file-mib", type=int, default=256)
    prepare.add_argument("--min-assistant-chars", type=int, default=1)
    prepare.add_argument("--max-sessions", type=int)
    prepare.add_argument("--overwrite", action="store_true")

    tokenize = commands.add_parser("tokenize", help="Apply the model's native template and assistant masks")
    tokenize.add_argument("--input-dir", default="data/prepared")
    tokenize.add_argument("--output-dir", default="data/tokenized")
    tokenize.add_argument("--model", default="unsloth/Qwen3.8-27B-unsloth-bnb-4bit")
    tokenize.add_argument("--revision")
    tokenize.add_argument("--max-length", type=int, default=2048)
    tokenize.add_argument("--context-policy", choices=("recent", "complete"), default="recent")
    tokenize.add_argument("--tools-file", help="JSON array of actual deployment tool schemas")
    tokenize.add_argument("--system-file", help="Text file containing the actual deployment system prompt")
    tokenize.add_argument("--thinking", choices=("on", "off"), default="on")
    tokenize.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"), default="medium")
    tokenize.add_argument("--local-files-only", action="store_true")
    tokenize.add_argument("--overwrite", action="store_true")

    preflight = commands.add_parser("preflight", help="Inspect environment; optionally load and probe the model")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--load-model", action="store_true", help="Explicitly load configured model weights")
    preflight.add_argument("--probe", action="store_true",
                           help="Run one real training-batch probe without an optimizer update; implies --load-model")

    training = commands.add_parser("train", help="Train local adapters")
    training.add_argument("--config", required=True)
    training.add_argument("--resume", help="Checkpoint directory, e.g. runs/name/checkpoint-000025")
    training.add_argument("--max-steps", type=int, help="Total optimizer steps, including resumed steps")
    training.add_argument("--output-dir")

    evaluation = commands.add_parser("evaluate", help="Score held-out targets with the base or a local adapter")
    evaluation.add_argument("--config", required=True)
    evaluation.add_argument("--checkpoint", help="Local checkpoint or exported adapter directory")
    evaluation.add_argument("--max-batches", type=int, help="0 evaluates all validation batches")

    smoke = commands.add_parser("smoke", help="Offline CPU training on a tiny synthetic causal model")
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--steps", type=int, default=4)
    smoke.add_argument("--method", choices=("noise", "backprop"), default="noise")
    return parser


def _preflight(args, config):
    from .runtime import environment_report, load_runtime
    result = environment_report(config)
    if not (args.load_model or args.probe):
        return result
    import time
    import torch
    runtime = load_runtime(config)
    result["runtime"] = runtime.metadata()
    if args.probe:
        from dataclasses import asdict
        from .noise import NoiseConfig, estimate_activation_error
        from .adapters import assign_lora_gradients
        from .training import TokenDataset, collate, _record_tokenization
        dataset = TokenDataset(config.data.train, max_length=config.data.max_length,
                               vocab_size=runtime.output_head.out_features)
        if not len(dataset):
            raise ValueError("No tokenized training examples; run prepare and tokenize first")
        _record_tokenization(runtime, {"train": dataset})
        # Probe the earliest adapted block: this has the longest decoder suffix.
        rows = [dataset[index % len(dataset)] for index in range(config.training.batch_size)]
        batch = collate(rows, runtime.device, pad_id=getattr(runtime.model.config, "pad_token_id", 0) or 0)
        if runtime.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime.device)
        start = time.perf_counter()
        module_name = next(iter(runtime.adapters))
        try:
            if config.training.method == "backprop":
                losses = runtime.losses(batch)
                losses.sum().backward()
                result["probe"] = {"clean_loss": float(losses.sum().detach()), "method": "backprop"}
            else:
                with torch.no_grad():
                    clean, captured, replay = runtime.capture(module_name, batch)
                    adapter = runtime.adapters[module_name]

                    def callback(perturbation):
                        adapter.perturbation = perturbation
                        return runtime.losses(batch, replay)

                    gradient, stats = estimate_activation_error(
                        callback, (*captured.shape[:-1], adapter.out_features),
                        NoiseConfig(pairs=config.noise.pairs, sigma=config.noise.sigma,
                                    probe=config.noise.probe, credit=config.noise.credit,
                                    seed=config.training.seed), runtime.device,
                        token_mask=batch["attention_mask"]
                    )
                    assign_lora_gradients(adapter, captured, gradient)
                result["probe"] = {"clean_loss": float(clean.sum()), "method": "noise",
                                   "module": module_name, "replay": "suffix" if replay else "full_forward",
                                   "estimator": asdict(stats)}
            if runtime.device.type == "cuda":
                torch.cuda.synchronize(runtime.device)
                result["probe"]["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(runtime.device)
                result["probe"]["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(runtime.device)
            result["probe"]["seconds"] = time.perf_counter() - start
            result["probe"]["input_tokens"] = int(batch["attention_mask"].sum())
            result["probe"]["note"] = "Measured on the first batch, not a maximum-length VRAM guarantee. No optimizer step."
            result["runtime"] = runtime.metadata()
        finally:
            runtime.clear_interventions()
            runtime.model.zero_grad(set_to_none=True)
    return result


def _smoke(output_dir, steps, method):
    from .config import TrainConfig
    from .training import train
    import torch
    if steps < 1:
        raise ValueError("Smoke steps must be positive")
    output = Path(output_dir).expanduser()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Smoke output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for split, count in (("train", 8), ("validation", 2)):
        with (output / f"{split}.jsonl").open("w") as handle:
            for index in range(count):
                ids = [1] + [(index + token) % 28 + 2 for token in range(15)]
                handle.write(json.dumps({"id": f"{split}-{index}", "group_id": split,
                                         "input_ids": ids, "labels": [-100] * 8 + ids[8:]}) + "\n")
    config = TrainConfig()
    config.model.name = "tiny"
    config.model.device = "cpu"
    config.model.dtype = "float32"
    config.model.load_in_4bit = False
    config.model.local_files_only = True
    config.model.tiny_hidden_size = 16
    config.model.tiny_vocab_size = 32
    config.adapter.rank = 4
    config.adapter.alpha = 4
    config.adapter.last_n_layers = 2
    config.noise.pairs = 2
    config.data.train = str(output / "train.jsonl")
    config.data.validation = str(output / "validation.jsonl")
    config.data.max_length = 32
    config.training.method = method
    config.training.output_dir = str(output / "run")
    config.training.max_steps = steps
    config.training.checkpoint_every = 2
    config.training.eval_every = 2
    config.training.eval_batches = 0
    config.training.learning_rate = 1e-3
    return train(config)


def main(argv=None):
    args = _parser().parse_args(argv)
    for key, value in (("HF_HUB_DISABLE_TELEMETRY", "1"), ("DO_NOT_TRACK", "1"), ("WANDB_DISABLED", "true")):
        os.environ.setdefault(key, value)
    try:
        if args.command == "prepare":
            from .data import prepare_transcripts
            result = prepare_transcripts(
                args.inputs, args.output_dir, validation_fraction=args.validation_fraction,
                seed=args.seed, source=args.source, branch_policy=args.branch_policy,
                selection=args.selection, review_file=args.review_file,
                redact_secrets=not args.no_redact_secrets, allow_missing_images=args.allow_missing_images,
                mask_failed_tool_calls=not args.keep_failed_tool_call_targets,
                include_sidechains=args.include_sidechains,
                max_line_bytes=args.max_line_mib * 1024**2, max_file_bytes=args.max_file_mib * 1024**2,
                min_assistant_chars=args.min_assistant_chars, max_sessions=args.max_sessions,
                overwrite=args.overwrite,
            )
        elif args.command == "tokenize":
            from .tokenization import tokenize_prepared
            result = tokenize_prepared(
                args.input_dir, args.output_dir, model_name=args.model, revision=args.revision,
                max_length=args.max_length, context_policy=args.context_policy,
                local_files_only=args.local_files_only, tools_file=args.tools_file,
                system_file=args.system_file, enable_thinking=args.thinking == "on",
                reasoning_effort=args.reasoning_effort, overwrite=args.overwrite,
            )
        elif args.command == "smoke":
            result = _smoke(args.output_dir, args.steps, args.method)
        else:
            from .config import load_config
            config = load_config(args.config)
            if args.command == "preflight":
                result = _preflight(args, config)
            elif args.command == "train":
                from .training import train
                if args.max_steps is not None:
                    config.training.max_steps = args.max_steps
                if args.output_dir:
                    config.training.output_dir = args.output_dir
                result = train(config.validate(), resume=args.resume)
            else:
                from .training import evaluate
                if args.max_batches is not None:
                    config.training.eval_batches = args.max_batches
                result = evaluate(config.validate(), checkpoint=args.checkpoint)
        _print(result)
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        if args.debug:
            raise
        print(f"dust-qlora: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
