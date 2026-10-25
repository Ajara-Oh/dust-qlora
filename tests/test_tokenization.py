"""Regression checks for the data-to-loss boundary, not model-specific role IDs."""
from collections import Counter
import json

import pytest

from dust_qlora.tokenization import TemplateMismatch, decision_examples, tokenize_prepared


class CharacterTokenizer:
    chat_template = "synthetic-native-chat-template-v1"
    init_kwargs = {"_commit_hash": "synthetic"}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        rendered = ""
        for message in messages:
            body = message.get("reasoning_content", "") + message["content"]
            if message.get("tool_calls"):
                body += json.dumps(message["tool_calls"], sort_keys=True)
            rendered += f"<{message['role']}>" + body + "</>"
        return rendered + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def save_pretrained(self, path):
        path.mkdir()
        (path / "tokenizer_config.json").write_text("{}")


def message(role, content, trainable=False, **kwargs):
    return {"role": role, "content": content, "trainable": trainable, **kwargs}


def row(messages, group="one"):
    return {"id": group, "session_id": group, "group_id": group, "source": "synthetic", "messages": messages}


def decode(ids):
    return "".join(chr(token) for token in ids if token != -100)


def test_only_selected_assistant_decision_is_supervised_and_no_future_leaks():
    data = row([message("user", "private prompt"),
                message("assistant", "ignored bad answer"),
                message("user", "correct it"),
                message("assistant", "good answer", True),
                message("tool", "FUTURE OBSERVATION", tool_call_id="absent")])
    example, = decision_examples(data, CharacterTokenizer())
    assert decode(example["labels"]) == "good answer</>"
    assert "private prompt" in decode(example["input_ids"])
    assert "FUTURE" not in decode(example["input_ids"])
    assert all(label == -100 or label == token
               for token, label in zip(example["input_ids"], example["labels"], strict=True))


def test_reasoning_and_tool_call_arguments_are_targets_but_observations_are_not():
    call = {"id": "call-1", "function": {"name": "read", "arguments": '{"path":"x.py"}'}}
    data = row([message("user", "read the file"),
                message("assistant", "", True, tool_calls=[call], reasoning_content="look first"),
                message("tool", "file contents", tool_call_id="call-1"),
                message("assistant", "I found the issue", True)])
    first, second = decision_examples(data, CharacterTokenizer())
    assert "look first" in decode(first["labels"])
    assert '"arguments": {"path": "x.py"}' in decode(first["labels"])
    assert "file contents" not in decode(first["input_ids"])
    assert decode(second["labels"]) == "I found the issue</>"
    assert "file contents" in decode(second["input_ids"])


def test_recent_context_preserves_system_task_and_whole_tool_exchanges():
    call = {"id": "c", "function": {"name": "read", "arguments": {"path": "a"}}}
    data = row([message("system", "RULE"), message("user", "old task"),
                message("assistant", "", tool_calls=[call]),
                message("tool", "large observation " * 20, tool_call_id="c"),
                message("user", "NEW TASK"), message("assistant", "ANSWER", True)])
    stats = Counter()
    example, = decision_examples(data, CharacterTokenizer(), max_length=90, stats=stats)
    text = decode(example["input_ids"])
    assert "RULE" in text and "NEW TASK" in text and "ANSWER" in text
    assert "<tool>" not in text and "<assistant>{" not in text
    assert example["metadata"]["dropped_context_messages"] == 3
    assert stats["examples_with_reduced_context"] == 1


def test_complete_context_skips_instead_of_truncating_and_oversized_target_is_never_cut():
    data = row([message("user", "old" * 40), message("assistant", "context"),
                message("user", "task"), message("assistant", "answer", True)])
    stats = Counter()
    assert not list(decision_examples(data, CharacterTokenizer(), max_length=70,
                                      context_policy="complete", stats=stats))
    assert stats["skipped_over_length"] == 1
    data["messages"][-1]["content"] = "huge answer" * 100
    assert not list(decision_examples(data, CharacterTokenizer(), max_length=70))


def test_rewritten_template_is_rejected_instead_of_guessing_a_mask():
    class BadTemplate(CharacterTokenizer):
        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            return "PROMPT" if add_generation_prompt else "DIFFERENT COMPLETION"
    data = row([message("user", "task"), message("assistant", "answer", True)])
    with pytest.raises(TemplateMismatch, match="does not extend"):
        list(decision_examples(data, BadTemplate()))


def test_boundary_straddling_token_remains_masked():
    class MergingTokenizer(CharacterTokenizer):
        def __call__(self, text, *, return_offsets_mapping=False, **kwargs):
            boundary = text.index("answer")
            encoded = super().__call__(text, return_offsets_mapping=True, **kwargs)
            encoded["offset_mapping"][boundary] = (boundary - 1, boundary + 1)
            return encoded
    data = row([message("user", "task"), message("assistant", "answer", True)])
    example, = decision_examples(data, MergingTokenizer())
    assert decode(example["labels"]) == "nswer</>"


def test_invalid_tool_context_is_counted_and_never_used_as_a_target():
    data = row([message("user", "task"), message("tool", "orphan", tool_call_id="none"),
                message("assistant", "answer", True)])
    stats = Counter()
    assert not list(decision_examples(data, CharacterTokenizer(), stats=stats))
    assert stats["skipped_invalid_tool_context"] == 1


def test_current_task_tool_observation_cannot_be_removed_to_make_answer_fit():
    call = {"id": "c", "function": {"name": "read", "arguments": {"path": "a"}}}
    data = row([message("user", "Read the value"), message("assistant", "", tool_calls=[call]),
                message("tool", "large output " * 100 + "the value is 42", tool_call_id="c"),
                message("assistant", "The value is 42.", True)])
    stats = Counter()
    assert not list(decision_examples(data, CharacterTokenizer(), max_length=250, stats=stats))
    assert stats["skipped_over_length"] == 1


def test_parallel_results_are_rendered_in_call_order_and_missing_results_skip():
    calls = [{"id": str(i), "function": {"name": "read", "arguments": {"path": str(i)}}} for i in (1, 2)]
    data = row([message("user", "Read two files"), message("assistant", "", tool_calls=calls),
                message("tool", "SECOND_RESULT", tool_call_id="2"),
                message("tool", "FIRST_RESULT", tool_call_id="1"),
                message("assistant", "Answer", True)])
    example, = decision_examples(data, CharacterTokenizer())
    text = decode(example["input_ids"])
    assert text.index("FIRST_RESULT") < text.index("SECOND_RESULT")
    del data["messages"][3]
    stats = Counter()
    assert not list(decision_examples(data, CharacterTokenizer(), stats=stats))
    assert stats["skipped_invalid_tool_context"] == 1


def test_dataset_manifest_hashes_masks_and_saves_tokenizer(tmp_path):
    source = tmp_path / "prepared"
    source.mkdir()
    for split in ("train", "validation"):
        data = row([message("user", "task"), message("assistant", "answer", True)], group=split)
        (source / f"{split}.jsonl").write_text(json.dumps(data) + "\n")
    dest = tmp_path / "tokenized"
    manifest = tokenize_prepared(source, dest, tokenizer=CharacterTokenizer(), model_name="test")
    assert manifest["splits"]["train"]["examples"] == 1
    assert len(manifest["splits"]["train"]["sha256"]) == 64
    assert (dest / "tokenizer/tokenizer_config.json").is_file()
    stored = json.loads((dest / "train.jsonl").read_text())
    assert decode(stored["labels"]) == "answer</>"
    with pytest.raises(FileExistsError):
        tokenize_prepared(source, dest, tokenizer=CharacterTokenizer())


def test_split_leakage_rejection_is_atomic_even_with_overwrite(tmp_path):
    source = tmp_path / "prepared"
    source.mkdir()
    data = row([message("user", "task"), message("assistant", "answer", True)])
    for split in ("train", "validation"):
        (source / f"{split}.jsonl").write_text(json.dumps(data) + "\n")
    dest = tmp_path / "tokenized"
    dest.mkdir()
    (dest / "existing.txt").write_text("keep this")
    with pytest.raises(ValueError, match="overlap"):
        tokenize_prepared(source, dest, tokenizer=CharacterTokenizer(), overwrite=True)
    assert (dest / "existing.txt").read_text() == "keep this"
    assert not (dest / "train.jsonl").exists()


def test_system_override_does_not_shift_target_mask_or_identity():
    data = row([message("user", "task"), message("assistant", "answer", True, source_id="target")])
    example, = decision_examples(data, CharacterTokenizer(), system_override="ACTUAL SYSTEM")
    assert example["target_id"] == "target"
    assert decode(example["labels"]) == "answer</>"
    assert decode(example["input_ids"]).startswith("<system>ACTUAL SYSTEM</>")


def test_long_archive_window_search_avoids_reencoding_every_prefix():
    messages = []
    for index in range(100):
        messages += [message("user", f"task {index}"),
                     message("assistant", "a small complete answer", True, source_id=str(index))]
    stats = Counter()
    examples = list(decision_examples(row(messages), CharacterTokenizer(), max_length=250, stats=stats))
    assert len(examples) == 100
    assert stats["encoding_attempts"] < 700
    assert all(len(example["input_ids"]) <= 250 for example in examples)


def test_non_thinking_with_recorded_reasoning_has_an_actionable_error():
    data = row([message("user", "task"),
                message("assistant", "answer", True, reasoning_content="recorded reasoning")])
    with pytest.raises(TemplateMismatch, match="Use --thinking on"):
        list(decision_examples(data, CharacterTokenizer(), template_kwargs={"enable_thinking": False}))
