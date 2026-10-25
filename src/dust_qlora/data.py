"""Local, dependency-free conversion of Pi and Claude Code session transcripts.

The output is model-neutral chat data. Tokenization and loss masking belong to
the training layer; ``trainable`` is an explicit per-message decision here.
No network requests, telemetry, or transcript uploads are performed.
"""

from __future__ import annotations

import copy
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from collections import Counter
from typing import Any, Iterable, Iterator


SCHEMA_VERSION = 1
_OUTPUT_NAMES = ("train.jsonl", "validation.jsonl", "report.json", "review_template.jsonl")
_SECRET_KEYS = re.compile(r"^(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd|secret[_-]?key)$", re.I)
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z ]*PRIVATE KEY)-----[\s\S]*?-----END (?:[A-Z ]*PRIVATE KEY)-----"),
    re.compile(r"\b(?:sk-(?:ant-|proj-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})\b"),
    re.compile(r"(?i)(?<=Bearer )[A-Za-z0-9._~+/-]{12,}=*"),
)
_ASSIGNMENT = re.compile(r"(?i)(\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password|passwd)\s*[:=]\s*['\"]?)([^\s'\";,}{]{8,})")
_PLACEHOLDER = re.compile(r"^(?:\[.*\]|<.*>|\$.*|\*+|redacted|changeme|example|your[_-].*)$", re.I)
_PI_METADATA = {"session", "thinking_level_change", "model_change", "usage", "custom", "label", "session_info"}
_CLAUDE_METADATA = {"progress", "file-history-snapshot", "queue-operation", "summary", "last-prompt", "pr-link", "agent-name", "agent-color"}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256((_json(value) if not isinstance(value, str) else value).encode("utf-8")).hexdigest()


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite JSON number")
    return number


def _redact(value: Any, counts: Counter) -> Any:
    if isinstance(value, str):
        for pattern in _SECRET_PATTERNS:
            value, n = pattern.subn("[REDACTED_SECRET]", value)
            counts["redactions"] += n

        def replace(match: re.Match) -> str:
            if _PLACEHOLDER.fullmatch(match[2]):
                return match[0]
            counts["redactions"] += 1
            return match[1] + "[REDACTED_SECRET]"

        return _ASSIGNMENT.sub(replace, value)
    if isinstance(value, list):
        return [_redact(item, counts) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if _SECRET_KEYS.fullmatch(key) and isinstance(item, str) and item and not _PLACEHOLDER.fullmatch(item):
                result[key] = "[REDACTED_SECRET]"
                counts["redactions"] += 1
            else:
                result[key] = _redact(item, counts)
        return result
    return value


class _Groups:
    """Union session families before assigning train/validation partitions."""

    def __init__(self) -> None:
        self.parents: dict[str, str] = {}

    def find(self, key: str) -> str:
        self.parents.setdefault(key, key)
        root = key
        while self.parents[root] != root:
            root = self.parents[root]
        while self.parents[key] != key:
            parent = self.parents[key]
            self.parents[key] = root
            key = parent
        return root

    def union(self, a: str, b: str) -> None:
        a, b = self.find(a), self.find(b)
        self.parents[max(a, b)] = min(a, b)


def _discover(inputs: Iterable[str | Path] | str | Path, output_dir: Path) -> list[Path]:
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    found: set[Path] = set()
    for input_path in inputs:
        expanded = os.path.expanduser(str(input_path))
        matches = glob.glob(expanded, recursive=True) if glob.has_magic(expanded) else [expanded]
        if not matches:
            raise FileNotFoundError(f"No transcript files match input: {input_path}")
        for match in matches:
            path = Path(match)
            if not path.exists():
                raise FileNotFoundError(f"Transcript input does not exist: {input_path}")
            candidates = path.rglob("*.jsonl") if path.is_dir() else [path]
            for candidate in candidates:
                resolved = candidate.resolve()
                if resolved.is_file() and resolved != output_dir and output_dir not in resolved.parents:
                    found.add(resolved)
    if not found:
        raise ValueError("No JSONL transcript files found outside the output directory")
    return sorted(found)


def _read_records(path: Path, counts: Counter, max_line_bytes: int, max_file_bytes: int) -> list[dict]:
    if path.stat().st_size > max_file_bytes:
        counts["oversized_files"] += 1
        return []
    records = []
    damaged = False
    total = 0
    with path.open("rb") as handle:
        line_number = 0
        while True:
            raw = handle.readline(max_line_bytes + 1)
            if not raw:
                break
            line_number += 1
            total += len(raw)
            counts["lines_read"] += 1
            if total > max_file_bytes:
                counts["oversized_files"] += 1
                return []
            if len(raw) > max_line_bytes:
                counts["oversized_lines"] += 1
                damaged = True
                while raw and not raw.endswith(b"\n"):
                    raw = handle.readline(max_line_bytes + 1)
                    total += len(raw)
                    if total > max_file_bytes:
                        counts["oversized_files"] += 1
                        return []
                continue
            try:
                text = raw.decode("utf-8-sig" if line_number == 1 else "utf-8").strip()
                if not text:
                    continue
                value = json.loads(text, parse_float=_finite_float, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Non-finite JSON number")))
                if not isinstance(value, dict):
                    raise ValueError("JSONL records must be objects")
            except (UnicodeError, ValueError, RecursionError):
                counts["malformed_lines"] += 1
                damaged = True
                continue
            value["_line"] = line_number
            value["_damaged_before"] = damaged
            records.append(value)
    return records


def _source(records: list[dict], requested: str) -> str | None:
    if requested != "auto":
        return requested
    if any(e.get("type") == "session" for e in records):
        return "pi"
    if any(e.get("type") in {"user", "assistant"} and isinstance(e.get("message"), dict) for e in records):
        return "claude"
    return None


def _tool_ids(entry: dict, kind: str) -> set[str]:
    message = entry.get("message")
    if not isinstance(message, dict):
        return set()
    content = message.get("content", [])
    if not isinstance(content, list):
        return set()
    key = "id" if kind == "tool_use" else "tool_use_id"
    return {b[key] for b in content if isinstance(b, dict) and b.get("type") == kind and isinstance(b.get(key), str)}


def _claude_response(entry: dict) -> str | None:
    if entry.get("type") != "assistant":
        return None
    message = entry.get("message")
    if not isinstance(message, dict):
        return None
    value = message.get("id")
    return value if isinstance(value, str) and value else None


def _writer(entry: dict) -> tuple:
    return bool(entry.get("isSidechain")), entry.get("agentId"), entry.get("teamName")


def _recover_parallel(chain: list[dict], nodes: dict[str, dict], counts: Counter) -> list[dict]:
    """Recover the sibling content-block records and parallel tool replies."""
    anchors: dict[str, str] = {}
    writers: dict[str, tuple] = {}
    for entry in chain:
        response = _claude_response(entry)
        if response:
            anchors[response] = entry["_id"]
            writers[response] = _writer(entry)
    if not anchors:
        return chain
    siblings: dict[str, list[dict]] = {}
    owners: dict[str, str | None] = {}
    for entry in nodes.values():
        response = _claude_response(entry)
        if response:
            siblings.setdefault(response, []).append(entry)
            for call in _tool_ids(entry, "tool_use"):
                if call not in owners:
                    owners[call] = response
                elif owners[call] != response:
                    owners[call] = None
    seen = {e["_id"] for e in chain}
    answered = {call for e in chain for call in _tool_ids(e, "tool_result")}
    inserts: dict[str, list[dict]] = {}
    for response, anchor in anchors.items():
        group = [e for e in siblings[response] if _writer(e) == writers[response]]
        member_ids = {e["_id"] for e in group}
        extra = [e for e in group if e["_id"] not in seen]
        results = []
        for e in nodes.values():
            if e["_id"] in seen or _writer(e) != writers[response]:
                continue
            calls = _tool_ids(e, "tool_result")
            owns_call = any(owners.get(call) == response for call in calls)
            if owns_call and (e.get("_parent") in member_ids or any(call not in answered for call in calls)):
                results.append(e)
                answered.update(calls)
        recovered = sorted(extra, key=lambda e: e["_line"]) + sorted(results, key=lambda e: e["_line"])
        inserts[anchor] = recovered
        seen.update(e["_id"] for e in recovered)
        counts["parallel_records_recovered"] += len(recovered)
    return [e for original in chain for e in [original, *inserts.get(original["_id"], [])]]


def _chains(records: list[dict], source: str, branch_policy: str, include_sidechains: bool, counts: Counter) -> Iterator[list[dict]]:
    nodes: dict[str, dict] = {}
    previous = None
    header = next((e for e in records if e.get("type") == "session"), {})
    legacy = source == "pi" and header.get("version", 1) < 2
    for entry in records:
        if source == "pi" and entry.get("type") == "session":
            continue
        key = entry.get("id" if source == "pi" else "uuid")
        if not isinstance(key, str) or not key:
            if source == "pi" and legacy:
                key = f"legacy-line-{entry['_line']}"
            else:
                counts["metadata_records_ignored"] += 1
                continue
        entry["_id"] = key
        entry["_parent"] = previous if legacy else entry.get("parentId" if source == "pi" else "parentUuid")
        if entry["_parent"] is not None and not isinstance(entry["_parent"], str):
            counts["invalid_parent_ids"] += 1
            # Keep an unresolvable sentinel so the branch is rejected, rather
            # than silently treating the malformed record as a new root.
            entry["_parent"] = "__invalid_parent__" + str(entry["_line"])
        previous = key
        if key in nodes:
            counts["duplicate_record_ids"] += 1
        nodes[key] = entry
    if legacy:
        for entry in nodes.values():
            if entry.get("type") == "compaction" and isinstance(entry.get("firstKeptEntryIndex"), int):
                index = entry["firstKeptEntryIndex"]
                if 0 <= index < len(records):
                    entry["firstKeptEntryId"] = records[index].get("_id")
    children = {e.get("_parent") for e in nodes.values()}
    leaves = [e for key, e in nodes.items() if key not in children]
    if source == "claude":
        def usable(e: dict) -> bool:
            return e.get("type") in {"user", "assistant"} and not e.get("isMeta") and (include_sidechains or not any(_writer(e)))

        selected = []
        for leaf in leaves:
            visited = set()
            while leaf and not usable(leaf) and leaf["_id"] not in visited:
                visited.add(leaf["_id"])
                leaf = nodes.get(leaf.get("_parent"))
            if leaf and usable(leaf):
                selected.append(leaf)
        leaves = selected
    leaves = sorted({e["_id"]: e for e in leaves}.values(), key=lambda e: e["_line"], reverse=True)
    if not leaves and nodes:
        counts["cyclic_sessions"] += 1
        return
    if branch_policy == "latest":
        chosen = leaves[:1]
        # Compaction may start a disconnected root. Its original turns remain
        # useful examples, each with the context available before compaction.
        if source == "claude":
            boundaries = [e["_line"] for e in nodes.values() if e.get("subtype") == "compact_boundary"]
            for boundary in boundaries:
                old = [e for e in nodes.values() if e["_line"] < boundary and e.get("type") in {"user", "assistant"} and (include_sidechains or not any(_writer(e))) and not e.get("isMeta")]
                if old:
                    chosen.append(max(old, key=lambda e: e["_line"]))
        counts["branches_not_selected"] += max(0, len(leaves) - len({e["_id"] for e in chosen}))
        leaves = sorted({e["_id"]: e for e in chosen}.values(), key=lambda e: e["_line"])
    for leaf in leaves:
        chain = []
        visited = set()
        current = leaf
        broken = False
        while current is not None:
            if current["_id"] in visited:
                counts["cycles"] += 1
                broken = True
                break
            visited.add(current["_id"])
            chain.append(current)
            parent = current.get("_parent")
            if parent is not None and parent not in nodes:
                counts["missing_parents"] += 1
                broken = True
                break
            current = nodes.get(parent)
        if broken:
            counts["broken_branches_skipped"] += 1
            continue
        chain.reverse()
        if source == "claude":
            chain = _recover_parallel(chain, nodes, counts)
        yield chain


def _blocks(content: Any, counts: Counter) -> tuple[str, str, list[dict], list[dict], bool]:
    """Return text, reasoning, calls, result blocks, and incomplete-content flag."""
    if isinstance(content, str):
        return content, "", [], [], False
    if content is None:
        return "", "", [], [], False
    if not isinstance(content, list):
        counts["unknown_content"] += 1
        return "[UNSUPPORTED_CONTENT]", "", [], [], True
    text, thinking, calls, results = [], [], [], []
    incomplete = False
    for block in content:
        if not isinstance(block, dict):
            incomplete = True
            counts["unknown_content"] += 1
            continue
        kind = block.get("type")
        if kind == "text":
            text.append(str(block.get("text", "")))
        elif kind == "thinking":
            value = block.get("thinking", block.get("text", ""))
            if isinstance(value, str):
                thinking.append(value)
        elif kind == "redacted_thinking":
            counts["opaque_reasoning_blocks_omitted"] += 1
        elif kind in {"tool_use", "toolCall"}:
            arguments = block.get("input", block.get("arguments", {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = None
            if not isinstance(arguments, dict) or not isinstance(block.get("id"), str) or not isinstance(block.get("name"), str) or block.get("argumentsError"):
                counts["invalid_tool_calls"] += 1
                incomplete = True
                continue
            calls.append({"id": block["id"], "type": "function", "function": {"name": block["name"], "arguments": arguments}})
        elif kind == "tool_result":
            results.append(block)
        elif kind in {"image", "image_url", "document", "audio", "video"}:
            counts["nontext_blocks"] += 1
            incomplete = True
            text.append("[NON_TEXT_CONTENT_OMITTED]")
        else:
            counts["unknown_content"] += 1
            incomplete = True
            text.append("[UNSUPPORTED_CONTENT]")
    return "".join(text), "".join(thinking), calls, results, incomplete


def _merge_blocks(entries: list[dict], counts: Counter) -> dict:
    """Merge Claude block records, including repeated cumulative snapshots."""
    merged = copy.deepcopy(entries[0])
    blocks: list = []
    fingerprints: set[str] = set()
    call_positions: dict[str, int] = {}
    for entry in entries:
        message = entry.get("message", {})
        content = message.get("content", [])
        content = [{"type": "text", "text": content}] if isinstance(content, str) else content
        if not isinstance(content, list):
            content = [content]
        for block in content:
            fingerprint = _hash(block)
            if fingerprint in fingerprints:
                counts["duplicate_content_blocks"] += 1
                continue
            fingerprints.add(fingerprint)
            if isinstance(block, dict) and block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                call = block["id"]
                if call in call_positions:
                    blocks[call_positions[call]] = block
                else:
                    call_positions[call] = len(blocks)
                    blocks.append(block)
            else:
                blocks.append(block)
        for key in ("stop_reason", "stopReason"):
            if message.get(key) is not None:
                merged["message"][key] = message[key]
        merged["isApiErrorMessage"] = bool(merged.get("isApiErrorMessage") or entry.get("isApiErrorMessage"))
        merged["_damaged_before"] = bool(merged.get("_damaged_before") or entry.get("_damaged_before"))
    merged["message"]["content"] = blocks
    merged["_response_ids"] = [e["_id"] for e in entries]
    return merged


def _coalesce(chain: list[dict], counts: Counter) -> list[dict]:
    responses: dict[str, list[dict]] = {}
    for entry in chain:
        response = _claude_response(entry)
        if response:
            responses.setdefault(response, []).append(entry)
    output, emitted = [], set()
    for entry in chain:
        response = _claude_response(entry)
        if response:
            if response in emitted:
                continue
            emitted.add(response)
            output.append(_merge_blocks(responses[response], counts))
        else:
            output.append(entry)
    return output


def _message(role: str, content: str, source_id: str, **extra: Any) -> dict:
    return {"role": role, "content": content, "trainable": False, "source_id": source_id, **extra}


def _normal_messages(entry: dict, source: str, session_id: str, counts: Counter) -> list[dict]:
    kind = entry.get("type")
    raw = entry.get("message", {})
    raw = raw if isinstance(raw, dict) else {}
    role = raw.get("role", kind if source == "claude" else None)
    raw_id = _claude_response(entry) or entry["_id"]
    sid = f"{source}:{session_id}:{raw_id}"
    if source == "pi" and kind == "custom_message":
        raw, role = entry, "user"
    if source == "pi" and kind == "branch_summary":
        return [_message("user", "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n" + str(entry.get("summary", "")) + "</summary>", sid)]
    if role in {"custom", "hookMessage"}:
        role = "user"
    if role == "bashExecution":
        if raw.get("excludeFromContext"):
            return []
        text = f"Ran `{raw.get('command', '')}`\n"
        text += f"```\n{raw['output']}\n```" if raw.get("output") else "(no output)"
        if raw.get("cancelled"):
            text += "\n\n(command cancelled)"
        elif raw.get("exitCode") not in (None, 0):
            text += f"\n\nCommand exited with code {raw['exitCode']}"
        if raw.get("truncated") and raw.get("fullOutputPath"):
            text += f"\n\n[Output truncated. Full output: {raw['fullOutputPath']}]"
            counts["truncated_tool_outputs"] += 1
        return [_message("user", text, sid)]
    if role not in {"user", "assistant", "toolResult", "system"}:
        if kind not in _PI_METADATA | _CLAUDE_METADATA | {"system", "attachment"}:
            counts["unknown_record_types"] += 1
        if kind == "attachment":
            counts["unrepresented_attachments"] += 1
            return [_message("user", "[ATTACHMENT_CONTEXT_UNAVAILABLE]", sid, _incomplete=True)]
        if kind == "message":
            return [_message("user", "[UNSUPPORTED_MESSAGE_CONTEXT]", sid, _incomplete=True)]
        return []
    # Claude system records are harness diagnostics, not the model's system prompt.
    if source == "claude" and role == "system":
        return []
    text, thinking, calls, results, incomplete = _blocks(raw.get("content"), counts)
    messages = []
    if role == "toolResult":
        call_id = raw.get("toolCallId")
        if not isinstance(call_id, str):
            counts["invalid_tool_results"] += 1
            return [_message("user", "[TOOL_RESULT_WITHOUT_CALL_ID]", sid, _incomplete=True)]
        messages.append(_message("tool", text, sid, tool_call_id=call_id, name=raw.get("toolName", ""), is_error=bool(raw.get("isError")), _incomplete=incomplete))
    else:
        for index, result in enumerate(results):
            result_text, _, _, _, result_incomplete = _blocks(result.get("content"), counts)
            call_id = result.get("tool_use_id")
            if not isinstance(call_id, str):
                counts["invalid_tool_results"] += 1
                incomplete = True
                continue
            messages.append(_message("tool", result_text, sid + f":result:{index}", tool_call_id=call_id, is_error=bool(result.get("is_error")), _incomplete=result_incomplete))
        if text or calls or thinking or role == "assistant":
            item = _message(role, text, sid, _incomplete=incomplete)
            if calls:
                item["tool_calls"] = calls
            if thinking:
                item["reasoning_content"] = thinking
            if role == "assistant":
                stopped = raw.get("stopReason", raw.get("stop_reason"))
                item["_eligible"] = stopped not in {"error", "aborted", "length", "max_tokens", "refusal"} and not bool(raw.get("errorMessage") or entry.get("isApiErrorMessage"))
                item["_damaged"] = bool(entry.get("_damaged_before"))
            messages.append(item)
    for item in messages:
        item["_entry_id"] = entry["_id"]
    return messages


def _apply_system(raw: dict, state: dict, counts: Counter, reset: bool = False) -> None:
    if reset:
        state.clear()
    state.setdefault("content", [])
    state.setdefault("sections", {})
    state.setdefault("tools", {})
    text, _, _, _, incomplete = _blocks(raw.get("content", ""), counts)
    if text:
        state["content"].append(text)
    state["incomplete"] = bool(state.get("incomplete") or incomplete)
    sections = raw.get("sections", {})
    if isinstance(sections, dict):
        for name, value in sections.items():
            if value is None:
                state["sections"].pop(name, None)
            elif isinstance(value, str):
                state["sections"][name] = value
    removed_tools = raw.get("toolsRemoved", [])
    if not isinstance(removed_tools, list):
        counts["unrepresented_tool_definitions"] += 1
        removed_tools = []
    for removed in removed_tools:
        name = removed.get("name") if isinstance(removed, dict) else removed
        if isinstance(name, str):
            state["tools"].pop(name, None)
    added_tools = raw.get("toolsAdded", [])
    if not isinstance(added_tools, list):
        counts["unrepresented_tool_definitions"] += 1
        added_tools = []
    for tool in added_tools:
        if isinstance(tool, dict) and isinstance(tool.get("name"), str) and isinstance(tool.get("parameters", tool.get("input_schema")), dict):
            function = {"name": tool["name"], "parameters": tool.get("parameters", tool.get("input_schema"))}
            if isinstance(tool.get("description"), str):
                function["description"] = tool["description"]
            state["tools"][tool["name"]] = {"type": "function", "function": function}
        else:
            counts["unrepresented_tool_definitions"] += 1


def _system_message(state: dict, session_id: str) -> list[dict]:
    text = "\n\n".join([*state.get("content", []), *state.get("sections", {}).values()])
    return [_message("system", text, f"pi:{session_id}:system", _incomplete=bool(state.get("incomplete")))] if text else []


def _read_reviews(path: str | Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    decisions = {}
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            decision = json.loads(line)
            if not isinstance(decision, dict) or not isinstance(decision.get("source_id"), str) or (decision.get("accept") is not None and type(decision.get("accept")) is not bool):
                raise ValueError(f"Invalid review decision on line {number}")
            if decision.get("replacement") is not None:
                replacement = decision["replacement"]
                if not isinstance(replacement, dict) or not set(replacement) <= {"content", "reasoning_content"} or not all(isinstance(v, str) for v in replacement.values()):
                    raise ValueError(f"Review replacement on line {number} must contain only text content/reasoning_content")
            if decision["source_id"] in decisions:
                raise ValueError(f"Duplicate review source_id on line {number}")
            decisions[decision["source_id"]] = decision
    return decisions


def _finalize(messages: list[dict], options: dict, reviews: dict, counts: Counter) -> list[dict]:
    result = copy.deepcopy(messages)
    calls: dict[str, dict] = {}
    pending_calls: set[str] = set()
    blocked = False
    has_prompt = False
    for item in result:
        role = item["role"]
        if item.get("_incomplete"):
            if options["allow_missing_images"]:
                counts["incomplete_content_overrides"] += 1
            else:
                blocked = True
        if role == "tool":
            owner = calls.get(item["tool_call_id"])
            if owner is None:
                counts["orphan_tool_results"] += 1
                blocked = True
            elif item.get("is_error") and options["mask_failed_tool_calls"]:
                if owner.get("trainable"):
                    counts["failed_tool_call_targets_masked"] += 1
                owner["trainable"] = False
            pending_calls.discard(item["tool_call_id"])
            has_prompt = True
        if role == "user":
            has_prompt = True
        if role == "assistant":
            decision = reviews.get(item["source_id"], {})
            eligible = item.get("_eligible", False) and not item.get("_already_trained", False)
            eligible = eligible and not item.get("_damaged", False) and has_prompt and not blocked
            eligible = eligible and (bool(item.get("tool_calls")) or len(item.get("content", "")) >= options["min_assistant_chars"])
            if eligible and pending_calls:
                counts["missing_tool_result_targets_masked"] += 1
                eligible = False
            if options["selection"] == "reviewed":
                eligible = eligible and decision.get("accept") is True
            elif decision.get("accept") is False:
                eligible = False
            if decision.get("accept") is True and decision.get("replacement"):
                item.update(decision["replacement"])
                item["corrected"] = True
            item["trainable"] = bool(eligible)
            if not eligible:
                counts["assistant_occurrences_masked"] += 1
            for call in item.get("tool_calls", []):
                calls[call["id"]] = item
                pending_calls.add(call["id"])
    for item in result:
        for key in list(item):
            if key.startswith("_"):
                del item[key]
    return result


def _trajectories(chain: list[dict], source: str, session_id: str, options: dict, reviews: dict, counts: Counter) -> Iterator[dict]:
    if source == "claude":
        chain = _coalesce(chain, counts)
    context: list[dict] = []
    state: dict = {}
    segment = 0

    def flush() -> dict | None:
        nonlocal segment
        messages = _finalize(context, options, reviews, counts)
        for item in context:
            if item["role"] == "assistant":
                item["_already_trained"] = True
        if not any(m["trainable"] for m in messages):
            return None
        row = {"schema_version": SCHEMA_VERSION, "source": source, "session_id": session_id, "messages": messages, "metadata": {"leaf_id": chain[-1]["_id"], "segment": segment}}
        if state.get("tools"):
            row["tools"] = list(state["tools"].values())
        segment += 1
        return row

    for entry in chain:
        kind = entry.get("type")
        raw = entry.get("message", {})
        if source == "pi" and kind == "message" and isinstance(raw, dict) and raw.get("role") == "system":
            row = flush()
            if row:
                yield row
            _apply_system(raw, state, counts)
            context = _system_message(state, session_id) + [m for m in context if m["role"] != "system"]
            continue
        if source == "pi" and kind == "context_edit":
            row = flush()
            if row:
                yield row
            target = entry.get("targetId")
            replacement = entry.get("replacement")
            matched = [m for m in context if m.get("_entry_id") == target]
            if not matched:
                counts["context_edits_outside_context"] += 1
            elif replacement is None:
                context = [m for m in context if m.get("_entry_id") != target]
            elif isinstance(replacement, dict) and "content" in replacement:
                original = next((e for e in chain if e["_id"] == target), None)
                if original:
                    revised = copy.deepcopy(original)
                    if isinstance(revised.get("message"), dict):
                        revised["message"]["content"] = replacement["content"]
                    else:
                        revised["content"] = replacement["content"]
                    normalized = _normal_messages(revised, source, session_id, counts)
                    for m in normalized:
                        m["_already_trained"] = True
                    start = next(i for i, m in enumerate(context) if m.get("_entry_id") == target)
                    context = [m for m in context[:start] if m.get("_entry_id") != target] + normalized + [m for m in context[start:] if m.get("_entry_id") != target]
            else:
                counts["invalid_context_edits"] += 1
                context.append(_message("user", "[INVALID_CONTEXT_EDIT]", f"pi:{session_id}:{entry['_id']}", _incomplete=True))
            continue
        if source == "pi" and kind == "compaction":
            row = flush()
            if row:
                yield row
            first_kept = entry.get("firstKeptEntryId")
            kept = []
            found = first_kept == entry["_id"]
            retain = False
            for m in context:
                if m.get("_entry_id") == first_kept:
                    retain = found = True
                if retain and m["role"] != "system":
                    kept.append(m)
            if not found:
                counts["unresolved_compaction_ranges"] += 1
            if isinstance(entry.get("systemMessage"), dict):
                _apply_system(entry["systemMessage"], state, counts, reset=True)
            summary = _message("user", "The conversation history before this point was compacted into the following summary:\n\n<summary>\n" + str(entry.get("summary", "")) + "\n</summary>", f"pi:{session_id}:{entry['_id']}", _entry_id=entry["_id"], _incomplete=not found)
            context = _system_message(state, session_id) + [summary] + kept
            counts["compactions"] += 1
            continue
        if source == "claude" and (entry.get("subtype") == "compact_boundary" or entry.get("isCompactSummary")):
            row = flush()
            if row:
                yield row
            context = []
            if entry.get("subtype") == "compact_boundary":
                counts["compactions"] += 1
                continue
        context.extend(_normal_messages(entry, source, session_id, counts))
    row = flush()
    if row:
        yield row


def _semantic(message: dict) -> dict:
    return {key: value for key, value in message.items() if key not in {"source_id", "trainable", "corrected"}}


def prepare_transcripts(
    inputs: Iterable[str | Path] | str | Path,
    output_dir: str | Path,
    *,
    validation_fraction: float = 0.1,
    seed: int = 42,
    source: str = "auto",
    branch_policy: str = "latest",
    selection: str = "all",
    review_file: str | Path | None = None,
    redact_secrets: bool = True,
    allow_missing_images: bool = False,
    mask_failed_tool_calls: bool = True,
    include_sidechains: bool = False,
    max_line_bytes: int = 16 * 1024 * 1024,
    max_file_bytes: int = 256 * 1024 * 1024,
    min_assistant_chars: int = 1,
    max_sessions: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Convert local JSONL sessions to train/validation chat JSONL and a report.

    Splitting is deterministic at the session-family level, before token
    chunking. Resumed copies and exact repeated context/targets join the same
    family. Input files are never modified. Existing outputs require overwrite.
    ``selection='all'`` means all eligible observed targets, not verified wins.
    """
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if source not in {"auto", "pi", "claude"} or branch_policy not in {"latest", "all"} or selection not in {"all", "reviewed"}:
        raise ValueError("Invalid source, branch_policy, or selection")
    if max_line_bytes < 1 or max_file_bytes < max_line_bytes or min_assistant_chars < 0 or (max_sessions is not None and max_sessions < 1):
        raise ValueError("Invalid parser size/count limit")
    output = Path(output_dir).expanduser().resolve()
    files = _discover(inputs, output)
    if not overwrite and any((output / name).exists() for name in _OUTPUT_NAMES):
        raise FileExistsError("Prepared output already exists; choose another output directory or overwrite=True")
    reviews = _read_reviews(review_file)
    options = {"allow_missing_images": allow_missing_images, "mask_failed_tool_calls": mask_failed_tool_calls, "selection": selection, "min_assistant_chars": min_assistant_chars}
    counts: Counter = Counter()
    groups = _Groups()
    file_groups: dict[str, str] = {}
    parent_links = []
    file_reports = []
    output.mkdir(parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output) as scratch:
        temporary = Path(scratch)
        connection = sqlite3.connect(temporary / "spool.sqlite3")
        connection.executescript("CREATE TABLE examples (seq INTEGER PRIMARY KEY, family TEXT, payload TEXT); CREATE TABLE identities (identity TEXT PRIMARY KEY, family TEXT, trained INTEGER); CREATE TABLE reviews (source_id TEXT PRIMARY KEY, session_id TEXT, eligible INTEGER);")
        try:
            for path in files:
                if max_sessions is not None and counts["files_processed"] >= max_sessions:
                    counts["files_omitted_by_limit"] += 1
                    continue
                before = counts.copy()
                records = _read_records(path, counts, max_line_bytes, max_file_bytes)
                counts["files_processed"] += 1
                detected = _source(records, source)
                if detected is None:
                    counts["unrecognized_files"] += 1
                    file_reports.append({"file_id": _hash(str(path))[:16], "source": None, "counts": dict(counts - before)})
                    continue
                header = next((e for e in records if e.get("type") == "session"), {})
                sid = header.get("id") if detected == "pi" else next((e.get("sessionId") for e in records if isinstance(e.get("sessionId"), str)), None)
                sid = sid if isinstance(sid, str) and sid else _hash(str(path))[:32]
                family = f"{detected}:{sid}"
                groups.find(family)
                file_groups[str(path)] = family
                if detected == "claude":
                    # Stable provider response IDs link resumed/exported copies
                    # even when a local correction changes their target text.
                    for entry in records:
                        response = _claude_response(entry)
                        if response:
                            identity = "claude-response:" + response
                            previous = connection.execute("SELECT family FROM identities WHERE identity=?", (identity,)).fetchone()
                            if previous:
                                groups.union(family, previous[0])
                            else:
                                connection.execute("INSERT INTO identities VALUES (?,?,0)", (identity, family))
                if detected == "pi" and isinstance(header.get("parentSession"), str):
                    parent = Path(header["parentSession"]).expanduser()
                    parent = parent if parent.is_absolute() else path.parent / parent
                    parent_links.append((family, str(parent.resolve())))
                # Subagent IDs vary; the enclosing session directory is a second
                # stable family link when sessionId is absent or agent-local.
                if detected == "claude" and "subagents" in path.parts:
                    index = path.parts.index("subagents")
                    if index:
                        groups.union(family, f"claude:{path.parts[index - 1]}")
                for chain in _chains(records, detected, branch_policy, include_sidechains or "subagents" in path.parts, counts):
                    counts["branches_processed"] += 1
                    for row in _trajectories(chain, detected, sid, options, reviews, counts):
                        if redact_secrets:
                            row = _redact(row, counts)
                        prefix = _hash(row.get("tools", []))
                        for message in row["messages"]:
                            prefix = _hash([prefix, _semantic(message)])
                            if message["role"] != "assistant":
                                continue
                            trainable = bool(message["trainable"])
                            identity = "context:" + prefix
                            match = connection.execute("SELECT family, trained FROM identities WHERE identity=?", (identity,)).fetchone()
                            if match:
                                groups.union(family, match[0])
                                if trainable and match[1]:
                                    message["trainable"] = False
                                    counts["duplicate_targets_masked"] += 1
                                elif trainable:
                                    connection.execute("UPDATE identities SET trained=1 WHERE identity=?", (identity,))
                            else:
                                connection.execute("INSERT INTO identities VALUES (?,?,?)", (identity, family, int(trainable)))
                            connection.execute("INSERT INTO reviews VALUES (?,?,?) ON CONFLICT(source_id) DO UPDATE SET eligible=MAX(eligible,excluded.eligible)", (message["source_id"], sid, int(trainable)))
                        if not any(m["trainable"] for m in row["messages"]):
                            counts["examples_without_new_targets"] += 1
                            continue
                        row["id"] = _hash([detected, sid, row["metadata"], row["messages"]])[:32]
                        row["metadata"]["source_file_id"] = _hash(str(path))[:16]
                        connection.execute("INSERT INTO examples(family,payload) VALUES (?,?)", (family, _json(row)))
                        counts["examples_prepared"] += 1
                file_reports.append({"file_id": _hash(str(path))[:16], "source": detected, "session_id": sid, "counts": dict(counts - before)})
                connection.commit()
            for child, parent in parent_links:
                if parent in file_groups:
                    groups.union(child, file_groups[parent])
                else:
                    # Missing parent files still join siblings forked from it.
                    groups.union(child, "parent-file:" + _hash(parent))
            split_counts: Counter = Counter()
            split_groups: dict[str, set[str]] = {"train": set(), "validation": set()}
            with (temporary / "train.jsonl").open("w", encoding="utf-8") as train, (temporary / "validation.jsonl").open("w", encoding="utf-8") as validation:
                for family, payload in connection.execute("SELECT family,payload FROM examples ORDER BY seq"):
                    root = groups.find(family)
                    group_id = _hash(root)[:32]
                    fraction = int(_hash([seed, root])[:16], 16) / 2**64
                    partition = "validation" if fraction < validation_fraction else "train"
                    row = json.loads(payload)
                    row["group_id"] = group_id
                    row["split"] = partition
                    (validation if partition == "validation" else train).write(_json(row) + "\n")
                    split_counts[partition + "_examples"] += 1
                    split_counts[partition + "_assistant_targets"] += sum(m["trainable"] for m in row["messages"])
                    split_groups[partition].add(group_id)
            with (temporary / "review_template.jsonl").open("w", encoding="utf-8") as handle:
                for source_id, sid, eligible in connection.execute("SELECT source_id,session_id,eligible FROM reviews ORDER BY source_id"):
                    handle.write(_json({"source_id": source_id, "session_id": sid, "eligible": bool(eligible), "accept": None}) + "\n")
            warnings = []
            if validation_fraction and not split_counts["validation_examples"]:
                warnings.append("No validation family was selected. Add sessions or adjust seed/fraction; never split one session's turns between partitions.")
            if not split_counts["train_examples"]:
                warnings.append("No training examples remain. Inspect the counts and selection/review filters before training.")
            if not redact_secrets:
                warnings.append("Literal-secret redaction was disabled.")
            warnings.append("Automated redaction is best effort. Review locally before using or sharing generated files.")
            report = {"schema_version": SCHEMA_VERSION, "counts": dict(counts), "splits": {key: {"examples": split_counts[key + "_examples"], "assistant_targets": split_counts[key + "_assistant_targets"], "groups": len(split_groups[key])} for key in ("train", "validation")}, "settings": {"validation_fraction": validation_fraction, "seed": seed, "source": source, "branch_policy": branch_policy, "selection": selection, "redact_secrets": redact_secrets, "allow_missing_images": allow_missing_images, "mask_failed_tool_calls": mask_failed_tool_calls, "include_sidechains": include_sidechains, "max_line_bytes": max_line_bytes, "max_file_bytes": max_file_bytes}, "files": file_reports, "warnings": warnings, "outputs": {name: str(output / name) for name in _OUTPUT_NAMES}}
            (temporary / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            for name in _OUTPUT_NAMES:
                os.chmod(temporary / name, 0o600)
                os.replace(temporary / name, output / name)
            return report
        finally:
            connection.close()
