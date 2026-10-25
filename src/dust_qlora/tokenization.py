"""Render native chat templates and supervise complete assistant decisions only.

No assumptions about model-specific role-token spellings are made. Every target
must extend the exact generation prompt; token offsets prevent prefix leakage.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any


class TemplateMismatch(ValueError):
    """The template rewrites history or cannot align a completion safely."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _message(message: dict[str, Any]) -> dict[str, Any]:
    result = {key: message[key] for key in
              ("role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content")
              if key in message}
    if result.get("role") not in {"system", "user", "assistant", "tool"}:
        raise ValueError("Unsupported canonical message role")
    result.setdefault("content", "")
    # Qwen's native template expects argument objects, not JSON-encoded strings.
    if result.get("tool_calls"):
        calls = []
        for original in result["tool_calls"]:
            call = dict(original)
            function = dict(call.get("function", call))
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be an object")
            function["arguments"] = arguments
            calls.append({"id": call.get("id", ""), "type": "function", "function": function})
        result["tool_calls"] = calls
    return result


def _tools(tools: Any) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise ValueError("Recorded tools must be an array of actual tool schemas")
    result = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("Each tool schema must be an object")
        if "function" in tool:
            result.append(tool)
        elif "name" in tool:
            result.append({"type": "function", "function": tool})
        else:
            raise ValueError("A tool schema has no function name")
    return result


def _groups(history: list[dict[str, Any]]) -> list[list[int]]:
    """A prior user task and its complete exchanges form one context unit."""
    groups: list[list[int]] = []
    for index, message in enumerate(history):
        if message["role"] not in {"system", "user"} and groups and history[groups[-1][0]]["role"] != "system":
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def _ordered_tool_results(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restore call order for templates that omit response IDs (including Qwen).

    Parallel tools may finish in any order. Missing or noncontiguous responses
    cannot safely be rendered as an unlabeled tool exchange for a later target.
    """
    result = []
    index = 0
    while index < len(history):
        message = history[index]
        if message["role"] == "tool":
            raise ValueError("Orphan tool response")
        result.append(message)
        index += 1
        calls = message.get("tool_calls", [])
        if not calls:
            continue
        ids = [call.get("id") for call in calls]
        if any(not identifier for identifier in ids) or len(set(ids)) != len(ids):
            raise ValueError("Tool calls need distinct nonempty IDs")
        observations = {}
        while index < len(history) and history[index]["role"] == "tool":
            observation = history[index]
            identifier = observation.get("tool_call_id")
            if identifier not in ids or identifier in observations:
                raise ValueError("Duplicate or unmatched tool response")
            observations[identifier] = observation
            index += 1
        if set(observations) != set(ids):
            raise ValueError("A tool call is missing its result before the next assistant decision")
        result.extend(observations[identifier] for identifier in ids)
    return result


def _render(tokenizer: Any, messages: list[dict[str, Any]], tools: Any,
            template_kwargs: dict[str, Any], generation: bool) -> str:
    kwargs = dict(template_kwargs)
    if tools:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=generation, **kwargs
    )


def _encode_completion(tokenizer: Any, history: list[dict[str, Any]],
                       target: dict[str, Any], tools: Any,
                       template_kwargs: dict[str, Any]) -> tuple[list[int], list[int]]:
    prompt = _render(tokenizer, history, tools, template_kwargs, True)
    text = _render(tokenizer, history + [target], tools, template_kwargs, False)
    if not isinstance(prompt, str) or not isinstance(text, str) or not text.startswith(prompt):
        raise TemplateMismatch(
            "The completed chat does not extend its generation prompt. Check the model's "
            "chat template and thinking settings; no unsafe fallback mask was generated."
        )
    boundary = len(prompt)
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(encoded["input_ids"])
        offsets = encoded["offset_mapping"]
        if len(offsets) != len(ids):
            raise TemplateMismatch("Tokenizer returned inconsistent character offsets")
        # A token crossing the boundary stays masked: no prompt text is a target.
        labels = [token if start >= boundary and end > start else -100
                  for token, (start, end) in zip(ids, offsets, strict=True)]
    except (NotImplementedError, TypeError):
        ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        prefix = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if ids[:len(prefix)] != prefix:
            raise TemplateMismatch("A fast tokenizer with offsets is required for this boundary") from None
        labels = [-100] * len(prefix) + ids[len(prefix):]
    if ids:
        labels[0] = -100  # No preceding position can predict the first input token.
    return ids, labels


def decision_examples(row: dict[str, Any], tokenizer: Any, *, max_length: int = 2048,
                      context_policy: str = "recent", tools_override: Any = None,
                      system_override: str | None = None,
                      template_kwargs: dict[str, Any] | None = None,
                      stats: Counter | None = None):
    """Yield one supervised assistant decision per example, with causal context.

    recent: preserve system + the entire latest user task and its subsequent
    tool/assistant context, then discard older exchanges until the example fits.
    complete: skip any target requiring context reduction. Target outputs are
    never cut, and no future observation is ever included in an input.
    """
    if max_length < 2 or context_policy not in {"recent", "complete"}:
        raise ValueError("Invalid maximum length or context policy")
    stats = stats if stats is not None else Counter()
    kwargs = template_kwargs or {"enable_thinking": True, "preserve_thinking": True,
                                  "reasoning_effort": "medium"}
    raw_messages = row.get("messages", [])
    if not isinstance(raw_messages, list):
        raise ValueError("Canonical row messages must be a list")
    messages = [_message(item) for item in raw_messages]
    tools = _tools(tools_override if tools_override is not None else row.get("tools"))
    if not tools and any(message.get("tool_calls") for message in messages):
        stats["rows_without_recorded_tool_schemas"] += 1
    if system_override is not None:
        # This is an explicitly supplied deployment prompt, never a generated one.
        if messages and messages[0]["role"] == "system":
            messages[0] = {"role": "system", "content": system_override}
        else:
            messages.insert(0, {"role": "system", "content": system_override})
            raw_messages = [{"trainable": False}] + raw_messages
    for index, target in enumerate(messages):
        if target["role"] != "assistant" or not raw_messages[index].get("trainable", False):
            continue
        stats["candidate_targets"] += 1
        if not kwargs.get("enable_thinking", True) and target.get("reasoning_content"):
            raise TemplateMismatch(
                "Recorded reasoning cannot extend a non-thinking generation prompt. "
                "Use --thinking on, or explicitly prepare a corpus without reasoning."
            )
        try:
            history = _ordered_tool_results(messages[:index])
        except ValueError:
            stats["skipped_invalid_tool_context"] += 1
            continue
        if not any(message["role"] == "user" for message in history):
            stats["skipped_no_user_context"] += 1
            continue
        groups = _groups(history)
        latest_user = max(i for i, message in enumerate(history) if message["role"] == "user")
        protected = {i for i, group in enumerate(groups)
                     if any(j >= latest_user or history[j]["role"] == "system" for j in group)}
        def encode(selected):
            selected_history = [history[j] for group in selected for j in groups[group]]
            stats["encoding_attempts"] += 1
            return _encode_completion(tokenizer, selected_history, target, tools, kwargs)

        dropped = 0
        if context_policy == "complete":
            ids, labels = encode(list(range(len(groups))))
        else:
            # Start with required context, then grow a suffix of prior tasks.
            # This avoids repeatedly tokenizing an entire months-long prefix.
            # Binary refinement need not find the absolute largest fit for an
            # exotic non-monotone template; every accepted candidate is still
            # rendered and length-checked exactly, and preserves dependencies.
            removable = [group for group in range(len(groups)) if group not in protected]

            def selection(keep):
                return sorted(protected | set(removable[-keep:] if keep else []))

            ids, labels = encode(selection(0))
            kept = 0
            if len(ids) <= max_length and removable:
                candidate = 1
                too_many = None
                while True:
                    attempt_ids, attempt_labels = encode(selection(candidate))
                    if len(attempt_ids) > max_length:
                        too_many = candidate
                        break
                    ids, labels, kept = attempt_ids, attempt_labels, candidate
                    if candidate == len(removable):
                        break
                    candidate = min(candidate * 2, len(removable))
                if too_many is not None:
                    low, high = kept + 1, too_many - 1
                    while low <= high:
                        candidate = (low + high) // 2
                        attempt_ids, attempt_labels = encode(selection(candidate))
                        if len(attempt_ids) <= max_length:
                            ids, labels, kept = attempt_ids, attempt_labels, candidate
                            low = candidate + 1
                        else:
                            high = candidate - 1
            dropped = sum(len(groups[group]) for group in removable[:len(removable) - kept])
        if len(ids) > max_length:
            stats["skipped_over_length"] += 1
            continue
        supervised = sum(label != -100 for label in labels)
        if not supervised:
            stats["skipped_empty_target_mask"] += 1
            continue
        stats["examples"] += 1
        stats["input_tokens"] += len(ids)
        stats["supervised_tokens"] += supervised
        stats["max_example_tokens"] = max(stats["max_example_tokens"], len(ids))
        if dropped:
            stats["examples_with_reduced_context"] += 1
            stats["dropped_context_messages"] += dropped
        target_id = raw_messages[index].get("source_id", str(index))
        example_id = hashlib.sha256(f"{row.get('id', '')}:{target_id}".encode()).hexdigest()[:24]
        yield {"id": example_id, "group_id": row.get("group_id", row.get("session_id")),
               "session_id": row.get("session_id"), "source": row.get("source"),
               "target_id": target_id, "input_ids": ids, "labels": labels,
               "metadata": {"dropped_context_messages": dropped,
                            "context_policy": context_policy, "supervised_tokens": supervised}}


def tokenize_prepared(input_dir: str | Path, output_dir: str | Path, *,
                      model_name: str = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit",
                      revision: str | None = None, max_length: int = 2048,
                      context_policy: str = "recent", local_files_only: bool = False,
                      tools_file: str | Path | None = None,
                      system_file: str | Path | None = None,
                      enable_thinking: bool = True, reasoning_effort: str = "medium",
                      overwrite: bool = False, tokenizer: Any = None) -> dict[str, Any]:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if output_dir.resolve() == input_dir.resolve() or output_dir.resolve() in input_dir.resolve().parents:
        raise ValueError("Tokenized output must be separate from its canonical input")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; choose another or --overwrite")
    if not (input_dir / "train.jsonl").is_file():
        raise FileNotFoundError(f"Missing canonical training data: {input_dir / 'train.jsonl'}")
    resolved_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash") if tokenizer else None
    if tokenizer is None:
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("DO_NOT_TRACK", "1")
        try:
            from transformers import AutoTokenizer
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise RuntimeError("Install dust-qlora[tokenize] before tokenizing") from None
        if not Path(model_name).expanduser().is_dir():
            # Resolve once, then load all tokenizer assets from that exact snapshot.
            config_path = Path(hf_hub_download(model_name, "tokenizer_config.json", revision=revision,
                                               local_files_only=local_files_only))
            if re.fullmatch(r"[0-9a-f]{40}", config_path.parent.name):
                resolved_revision = config_path.parent.name
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=resolved_revision or revision, use_fast=True,
                                                 local_files_only=local_files_only, trust_remote_code=False)
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        raise ValueError("The selected tokenizer has no native chat template")
    tools_override = json.loads(Path(tools_file).read_text()) if tools_file else None
    system_override = Path(system_file).read_text() if system_file else None
    kwargs = {"enable_thinking": enable_thinking, "preserve_thinking": True,
              "reasoning_effort": reasoning_effort}
    manifest: dict[str, Any] = {
        "format_version": 1, "model_name": model_name, "revision": revision,
        "resolved_revision": resolved_revision or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "chat_template_sha256": hashlib.sha256(str(template).encode()).hexdigest(),
        "template_kwargs": kwargs, "max_length": max_length, "context_policy": context_policy,
        "label_convention": "same_token_index; runtime shifts labels by one",
        "tools_override_sha256": file_sha256(Path(tools_file)) if tools_file else None,
        "system_override_sha256": file_sha256(Path(system_file)) if system_file else None,
        "splits": {},
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dust-tokenize-", dir=output_dir.parent) as temp:
        work = Path(temp) / "result"
        work.mkdir(mode=0o700)
        split_groups: dict[str, set[str]] = {}
        for split in ("train", "validation"):
            source = input_dir / f"{split}.jsonl"
            destination = work / f"{split}.jsonl"
            stats: Counter = Counter()
            split_groups[split] = set()
            seen: set[str] = set()
            with destination.open("w", encoding="utf-8") as out:
                if source.is_file():
                    with source.open(encoding="utf-8") as handle:
                        for line_number, line in enumerate(handle, 1):
                            if not line.strip():
                                continue
                            row = json.loads(line)
                            if not isinstance(row, dict):
                                raise ValueError(f"{source.name}:{line_number}: expected an object")
                            stats["canonical_rows"] += 1
                            group = row.get("group_id") or row.get("session_id")
                            if not group:
                                raise ValueError("Canonical rows need group_id/session_id for split auditing")
                            split_groups[split].add(str(group))
                            try:
                                for example in decision_examples(
                                    row, tokenizer, max_length=max_length, context_policy=context_policy,
                                    tools_override=tools_override, system_override=system_override,
                                    template_kwargs=kwargs, stats=stats,
                                ):
                                    if example["id"] in seen:
                                        stats["duplicate_examples"] += 1
                                        continue
                                    seen.add(example["id"])
                                    out.write(json.dumps(example, ensure_ascii=False) + "\n")
                            except (TemplateMismatch, ValueError, KeyError) as exc:
                                # Fail rather than silently train malformed or incorrectly masked data.
                                raise ValueError(f"{source.name}:{line_number}: {exc}") from exc
            manifest["splits"][split] = {**dict(stats), "groups": len(split_groups[split]),
                                         "input_sha256": file_sha256(source) if source.is_file() else None,
                                         "sha256": file_sha256(destination)}
        if split_groups["train"] & split_groups["validation"]:
            raise ValueError("Session/group overlap between training and validation; re-run prepare")
        if not manifest["splits"]["train"].get("examples"):
            raise ValueError("No training targets fit. Inspect prepare report, increase max-length, or adjust context policy")
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(work / "tokenizer")
        (work / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for root, directories, filenames in os.walk(work):
            os.chmod(root, 0o700)
            for filename in filenames:
                os.chmod(Path(root) / filename, 0o600)
        if output_dir.exists():
            if not any(output_dir.iterdir()):
                output_dir.rmdir()
            else:
                # Only after the complete new dataset has passed validation.
                old = Path(temp) / "previous"
                output_dir.rename(old)
                try:
                    work.rename(output_dir)
                except BaseException:
                    old.rename(output_dir)
                    raise
                shutil.rmtree(old)
                return manifest
        work.rename(output_dir)
    return manifest
