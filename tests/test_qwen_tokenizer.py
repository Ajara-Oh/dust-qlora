"""Optional real-tokenizer checks. Never download assets during the test suite.

Set DUST_QWEN_TOKENIZER to a local Qwen3.8 tokenizer directory to enable these.
"""
import os

import pytest

from dust_qlora.tokenization import decision_examples


TOKENIZER_PATH = os.environ.get("DUST_QWEN_TOKENIZER")
pytestmark = pytest.mark.skipif(not TOKENIZER_PATH, reason="Set DUST_QWEN_TOKENIZER to cached tokenizer assets")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True, trust_remote_code=False)


def supervised_text(tokenizer, example):
    return tokenizer.decode([token for token in example["labels"] if token != -100])


def test_actual_qwen_reasoning_tool_call_and_completion_masks(tokenizer):
    calls = [{"id": f"c{i}", "type": "function",
              "function": {"name": "read", "arguments": {"path": f"file-{i}.py"}}} for i in (1, 2)]
    row = {"id": "qwen-test", "group_id": "qwen-test", "messages": [
        {"role": "system", "content": "ACTUAL_SYSTEM_INSTRUCTION"},
        {"role": "user", "content": "Read both files."},
        {"role": "assistant", "content": "", "reasoning_content": "Inspect both files first.",
         "tool_calls": calls, "trainable": True},
        {"role": "tool", "name": "read", "tool_call_id": "c2", "content": "SECOND_OBSERVATION"},
        {"role": "tool", "name": "read", "tool_call_id": "c1", "content": "FIRST_OBSERVATION"},
        {"role": "assistant", "content": "FINAL_ANSWER", "trainable": True}],
        "tools": [{"type": "function", "function": {
            "name": "read", "description": "Read a file", "parameters": {
                "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]}
    first, second = decision_examples(row, tokenizer)
    first_target = supervised_text(tokenizer, first)
    second_target = supervised_text(tokenizer, second)
    assert "Inspect both files first." in first_target and "<tool_call>" in first_target
    assert "<function=read>" in first_target and "file-1.py" in first_target
    assert "OBSERVATION" not in tokenizer.decode(first["input_ids"])
    assert "FINAL_ANSWER" in second_target
    for target in (first_target, second_target):
        assert target.rstrip().endswith("<|im_end|>")
        assert "ACTUAL_SYSTEM_INSTRUCTION" not in target
        assert "OBSERVATION" not in target
    rendered_second = tokenizer.decode(second["input_ids"])
    assert rendered_second.index("FIRST_OBSERVATION") < rendered_second.index("SECOND_OBSERVATION")


def test_actual_qwen_non_thinking_without_recorded_reasoning(tokenizer):
    row = {"id": "plain", "group_id": "plain", "messages": [
        {"role": "user", "content": "Say hello."},
        {"role": "assistant", "content": "Hello.", "trainable": True}]}
    example, = decision_examples(row, tokenizer, template_kwargs={"enable_thinking": False,
                                                                 "preserve_thinking": True})
    assert supervised_text(tokenizer, example).rstrip() == "Hello.<|im_end|>"
