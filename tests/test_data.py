"""Synthetic-only regression tests for transcript provenance and loss selection."""

import json
from pathlib import Path
import tempfile
import unittest

from dust_qlora.data import prepare_transcripts


def pi_header(session="pi-session", **extra):
    return {"type": "session", "version": 3, "id": session, **extra}


def pi_message(identifier, parent, role, content, **extra):
    return {"type": "message", "id": identifier, "parentId": parent,
            "message": {"role": role, "content": content, **extra}}


def claude_message(identifier, parent, role, content, session="claude-session", response=None, **extra):
    message = {"role": role, "content": content}
    if response:
        message["id"] = response
    return {"type": role, "uuid": identifier, "parentUuid": parent,
            "sessionId": session, "message": message, **extra}


class TranscriptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.out = self.root / "prepared"

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, entries):
        path = self.raw / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(e) if not isinstance(e, str) else e for e in entries) + "\n", encoding="utf-8")
        return path

    def prepare(self, **kwargs):
        kwargs.setdefault("validation_fraction", 0)
        self.report = prepare_transcripts(self.raw, self.out, **kwargs)
        rows = []
        for partition in ("train", "validation"):
            rows.extend(json.loads(line) for line in (self.out / (partition + ".jsonl")).read_text().splitlines())
        return rows

    def targets(self, rows):
        return [m for row in rows for m in row["messages"] if m["trainable"]]

    def test_simple_pi_preserves_user_and_assistant_mask(self):
        self.write("simple.jsonl", [pi_header(), pi_message("u", None, "user", "Please fix this."), pi_message("a", "u", "assistant", [{"type": "text", "text": "Use the corrected expression."}], stopReason="stop")])
        rows = self.prepare()
        self.assertEqual(len(rows), 1)
        self.assertEqual([m["trainable"] for m in rows[0]["messages"]], [False, True])
        self.assertEqual(rows[0]["messages"][1]["source_id"], "pi:pi-session:a")
        self.assertNotIn("tools", rows[0])

    def test_latest_branch_does_not_concatenate_alternatives(self):
        self.write("branch.jsonl", [pi_header(), pi_message("u", None, "user", "Choose."), pi_message("old", "u", "assistant", "First attempt."), pi_message("new", "u", "assistant", "Revised attempt.")])
        targets = self.targets(self.prepare())
        self.assertEqual([m["content"] for m in targets], ["Revised attempt."])
        self.assertEqual(self.report["counts"]["branches_not_selected"], 1)

    def test_all_branches_share_family_and_prefix_is_not_retrained(self):
        self.write("branch.jsonl", [pi_header(), pi_message("u", None, "user", "Begin."), pi_message("a", "u", "assistant", "Shared response."), pi_message("u2", "a", "user", "Continue."), pi_message("old", "u2", "assistant", "Alternative one."), pi_message("new", "u2", "assistant", "Alternative two.")])
        rows = self.prepare(branch_policy="all")
        self.assertEqual(len({r["group_id"] for r in rows}), 1)
        self.assertEqual(len(self.targets(rows)), 3)
        self.assertEqual(sum(m["content"] == "Shared response." for m in self.targets(rows)), 1)

    def test_pi_failed_tool_call_masked_but_recovery_is_target(self):
        self.write("error.jsonl", [pi_header(), pi_message("u", None, "user", "Read file."), pi_message("a", "u", "assistant", [{"type": "toolCall", "id": "c", "name": "read", "arguments": {"path": "missing"}}]), pi_message("t", "a", "toolResult", [{"type": "text", "text": "Missing file"}], toolCallId="c", toolName="read", isError=True), pi_message("b", "t", "assistant", "I will use the correct path.")])
        rows = self.prepare()
        self.assertEqual([m["trainable"] for m in rows[0]["messages"]], [False, False, False, True])
        tool = rows[0]["messages"][2]
        self.assertTrue(tool["is_error"])
        self.assertEqual(tool["tool_call_id"], "c")

    def test_tool_failure_can_be_included_explicitly(self):
        self.write("error.jsonl", [pi_header(), pi_message("u", None, "user", "Try."), pi_message("a", "u", "assistant", [{"type": "toolCall", "id": "c", "name": "read", "arguments": {}}]), pi_message("t", "a", "toolResult", "Error", toolCallId="c", isError=True)])
        self.assertEqual(len(self.targets(self.prepare(mask_failed_tool_calls=False))), 1)

    def test_aborted_and_incomplete_assistants_are_context_only(self):
        self.write("abort.jsonl", [pi_header(), pi_message("u", None, "user", "Try."), pi_message("a", "u", "assistant", "Partial", stopReason="aborted"), pi_message("b", "a", "assistant", "Complete", stopReason="stop")])
        self.assertEqual([m["content"] for m in self.targets(self.prepare())], ["Complete"])

    def test_reasoning_is_separate_and_signature_not_exported(self):
        self.write("think.jsonl", [pi_header(), pi_message("u", None, "user", "Compute."), pi_message("a", "u", "assistant", [{"type": "thinking", "thinking": "Check the arithmetic", "signature": "opaque"}, {"type": "text", "text": "The result is 4."}])])
        target = self.targets(self.prepare())[0]
        self.assertEqual(target["reasoning_content"], "Check the arithmetic")
        self.assertEqual(target["content"], "The result is 4.")
        self.assertNotIn("opaque", json.dumps(target))

    def test_images_block_only_dependent_targets(self):
        self.write("images.jsonl", [pi_header(), pi_message("u", None, "user", "Hello."), pi_message("a", "u", "assistant", "Hello back."), pi_message("img", "a", "user", [{"type": "image", "data": "synthetic-base64"}, {"type": "text", "text": "What is visible?"}]), pi_message("b", "img", "assistant", "A chart.")])
        rows = self.prepare()
        self.assertEqual([m["content"] for m in self.targets(rows)], ["Hello back."])
        self.assertEqual(self.report["counts"]["nontext_blocks"], 1)
        self.assertNotIn("synthetic-base64", json.dumps(rows))

    def test_image_override_is_explicit_and_reported(self):
        self.write("image.jsonl", [pi_header(), pi_message("u", None, "user", [{"type": "image", "data": "synthetic"}]), pi_message("a", "u", "assistant", "A chart.")])
        rows = self.prepare(allow_missing_images=True)
        self.assertEqual(len(self.targets(rows)), 1)
        self.assertGreater(self.report["counts"]["incomplete_content_overrides"], 0)

    def test_corrupt_line_masks_later_context_without_losing_earlier_target(self):
        self.write("broken.jsonl", [pi_header(), pi_message("u", None, "user", "First."), pi_message("a", "u", "assistant", "First reply."), "{broken", pi_message("u2", "a", "user", "Second."), pi_message("b", "u2", "assistant", "Second reply.")])
        self.assertEqual([m["content"] for m in self.targets(self.prepare())], ["First reply."])
        self.assertEqual(self.report["counts"]["malformed_lines"], 1)

    def test_missing_ancestry_is_not_fabricated(self):
        self.write("missing.jsonl", [pi_header(), pi_message("u", "not-present", "user", "Missing past."), pi_message("a", "u", "assistant", "Do not train this.")])
        self.assertEqual(self.prepare(), [])
        self.assertEqual(self.report["counts"]["missing_parents"], 1)

    def test_wrong_parent_type_is_reported_without_printing_content(self):
        self.write("bad-parent.jsonl", [pi_header(), pi_message("u", ["private-transcript-marker"], "user", "Private prompt"), pi_message("a", "u", "assistant", "Private answer")])
        self.assertEqual(self.prepare(), [])
        self.assertEqual(self.report["counts"]["invalid_parent_ids"], 1)
        self.assertNotIn("private-transcript-marker", json.dumps(self.report))

    def test_tree_cycle_is_reported(self):
        self.write("cycle.jsonl", [pi_header(), pi_message("u", "a", "user", "Cycle"), pi_message("a", "u", "assistant", "Cycle")])
        self.assertEqual(self.prepare(), [])
        self.assertEqual(self.report["counts"]["cyclic_sessions"], 1)

    def test_legacy_linear_pi(self):
        self.write("legacy.jsonl", [{"type": "session", "id": "old"}, {"type": "message", "message": {"role": "user", "content": "Old prompt"}}, {"type": "message", "message": {"role": "assistant", "content": "Old reply"}}])
        self.assertEqual([m["content"] for m in self.targets(self.prepare())], ["Old reply"])

    def test_compaction_never_places_future_summary_in_earlier_target(self):
        self.write("compact.jsonl", [pi_header(), pi_message("u", None, "user", "Original prompt."), pi_message("a", "u", "assistant", "Original answer."), {"type": "compaction", "id": "compact", "parentId": "a", "firstKeptEntryId": "compact", "summary": "Later summary."}, pi_message("u2", "compact", "user", "Continue."), pi_message("b", "u2", "assistant", "Post-compaction reply.")])
        rows = self.prepare()
        self.assertEqual(len(rows), 2)
        self.assertNotIn("Later summary", json.dumps(rows[0]))
        self.assertIn("Later summary", rows[1]["messages"][0]["content"])
        self.assertNotIn("Original prompt", json.dumps(rows[1]))

    def test_compaction_retained_assistant_is_context_only(self):
        self.write("keep.jsonl", [pi_header(), pi_message("u", None, "user", "First"), pi_message("a", "u", "assistant", "Retained"), {"type": "compaction", "id": "c", "parentId": "a", "firstKeptEntryId": "a", "summary": "Summary"}, pi_message("u2", "c", "user", "Next"), pi_message("b", "u2", "assistant", "New")])
        rows = self.prepare()
        self.assertEqual([m["content"] for m in self.targets(rows)], ["Retained", "New"])
        self.assertFalse(next(m for m in rows[1]["messages"] if m["content"] == "Retained")["trainable"])

    def test_context_edit_only_affects_later_targets(self):
        self.write("edit.jsonl", [pi_header(), pi_message("u", None, "user", "Original"), pi_message("a", "u", "assistant", "Answer before edit"), {"type": "context_edit", "id": "edit", "parentId": "a", "targetId": "u", "replacement": {"content": "Revised"}}, pi_message("u2", "edit", "user", "Continue"), pi_message("b", "u2", "assistant", "Answer after edit")])
        rows = self.prepare()
        self.assertEqual(rows[0]["messages"][0]["content"], "Original")
        self.assertEqual(rows[1]["messages"][0]["content"], "Revised")
        self.assertEqual(len(self.targets(rows)), 2)

    def test_recorded_system_and_tools_replayed_without_invention(self):
        self.write("system.jsonl", [pi_header(), pi_message("s", None, "system", "Base instructions", sections={"policy": "Old policy"}, toolsAdded=[{"name": "read", "description": "Read a path", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}]), pi_message("u", "s", "user", "Begin"), pi_message("a", "u", "assistant", "Before"), pi_message("s2", "a", "system", "", sections={"policy": "New policy"}, toolsRemoved=[{"name": "read"}]), pi_message("u2", "s2", "user", "Continue"), pi_message("b", "u2", "assistant", "After")])
        rows = self.prepare()
        self.assertIn("Old policy", rows[0]["messages"][0]["content"])
        self.assertIn("New policy", rows[1]["messages"][0]["content"])
        self.assertEqual(rows[0]["tools"][0]["function"]["name"], "read")
        self.assertNotIn("tools", rows[1])

    def test_claude_parallel_call_siblings_and_results_recovered(self):
        self.write("parallel.jsonl", [claude_message("u", None, "user", "Read both files."), claude_message("a1", "u", "assistant", [{"type": "tool_use", "id": "c1", "name": "Read", "input": {"file_path": "a"}}], response="msg-shared"), claude_message("a2", "a1", "assistant", [{"type": "tool_use", "id": "c2", "name": "Read", "input": {"file_path": "b"}}], response="msg-shared"), claude_message("r2", "a2", "user", [{"type": "tool_result", "tool_use_id": "c2", "content": "Second output"}]), claude_message("r1", "a1", "user", [{"type": "tool_result", "tool_use_id": "c1", "content": "First output"}]), claude_message("final", "r1", "assistant", [{"type": "text", "text": "Both read."}], response="msg-final")])
        rows = self.prepare()
        messages = rows[0]["messages"]
        assistant = next(m for m in messages if m.get("tool_calls"))
        self.assertEqual({c["id"] for c in assistant["tool_calls"]}, {"c1", "c2"})
        self.assertEqual({m["tool_call_id"] for m in messages if m["role"] == "tool"}, {"c1", "c2"})
        self.assertEqual(len(self.targets(rows)), 2)
        self.assertGreater(self.report["counts"]["parallel_records_recovered"], 0)

    def test_answer_after_incomplete_parallel_results_is_masked(self):
        calls = [{"type": "toolCall", "id": identifier, "name": "read", "arguments": {"path": identifier}} for identifier in ("c1", "c2")]
        self.write("missing-result.jsonl", [pi_header(), pi_message("u", None, "user", "Read both files."), pi_message("a", "u", "assistant", calls), pi_message("r1", "a", "toolResult", "First output", toolCallId="c1"), pi_message("b", "r1", "assistant", "Both files were read.")])
        rows = self.prepare()
        self.assertEqual([m["source_id"] for m in self.targets(rows)], ["pi:pi-session:a"])
        self.assertEqual(self.report["counts"]["missing_tool_result_targets_masked"], 1)

    def test_late_result_clears_pending_calls_without_fabricating_cancellation(self):
        self.write("interruption.jsonl", [pi_header(), pi_message("u", None, "user", "Read file."), pi_message("a", "u", "assistant", [{"type": "toolCall", "id": "c", "name": "read", "arguments": {"path": "file"}}]), pi_message("u2", "a", "user", "Are you done?"), pi_message("b", "u2", "assistant", "Premature answer."), pi_message("r", "b", "toolResult", "Actual result.", toolCallId="c"), pi_message("c", "r", "assistant", "Answer after result.")])
        rows = self.prepare()
        self.assertEqual([m["source_id"] for m in self.targets(rows)], ["pi:pi-session:a", "pi:pi-session:c"])
        self.assertEqual(self.report["counts"]["missing_tool_result_targets_masked"], 1)

    def test_terminal_tool_call_remains_an_eligible_decision(self):
        self.write("terminal-call.jsonl", [pi_header(), pi_message("u", None, "user", "Read file."), pi_message("a", "u", "assistant", [{"type": "toolCall", "id": "c", "name": "read", "arguments": {"path": "file"}}])])
        rows = self.prepare()
        self.assertEqual([m["source_id"] for m in self.targets(rows)], ["pi:pi-session:a"])
        self.assertEqual(self.report["counts"].get("missing_tool_result_targets_masked", 0), 0)

    def test_duplicate_claude_blocks_do_not_duplicate_response(self):
        self.write("duplicate.jsonl", [claude_message("u", None, "user", "Hello"), claude_message("a1", "u", "assistant", [{"type": "text", "text": "Hello back"}], response="msg-1"), claude_message("a2", "a1", "assistant", [{"type": "text", "text": "Hello back"}], response="msg-1")])
        self.assertEqual([m["content"] for m in self.targets(self.prepare())], ["Hello back"])

    def test_claude_compaction_retains_both_training_eras(self):
        self.write("compact.jsonl", [claude_message("u", None, "user", "Earlier"), claude_message("a", "u", "assistant", "Earlier answer", response="msg-1"), {"type": "system", "subtype": "compact_boundary", "uuid": "boundary", "parentUuid": None, "logicalParentUuid": "a", "sessionId": "claude-session"}, claude_message("summary", "boundary", "user", "Summary of earlier work", isCompactSummary=True), claude_message("b", "summary", "assistant", "Continued answer", response="msg-2")])
        rows = self.prepare()
        self.assertEqual([m["content"] for m in self.targets(rows)], ["Earlier answer", "Continued answer"])
        self.assertNotIn("Summary of earlier work", json.dumps(rows[0]))
        self.assertNotIn("Earlier answer", json.dumps(rows[1]))

    def test_orphan_tool_results_mask_dependent_targets(self):
        self.write("orphan.jsonl", [claude_message("u", None, "user", [{"type": "tool_result", "tool_use_id": "missing", "content": "Cannot recover original call"}]), claude_message("a", "u", "assistant", "Unsafe context", response="msg-1")])
        self.assertEqual(self.prepare(), [])
        self.assertEqual(self.report["counts"]["orphan_tool_results"], 1)

    def test_literal_secrets_redacted_in_text_and_tool_arguments(self):
        secret = "sk-ant-" + "x" * 30
        self.write("secrets.jsonl", [pi_header(), pi_message("u", None, "user", "Token " + secret), pi_message("a", "u", "assistant", [{"type": "toolCall", "id": "c", "name": "configure", "arguments": {"api_key": "synthetic-private-value"}}])])
        rows = self.prepare()
        data = json.dumps(rows)
        self.assertNotIn(secret, data)
        self.assertNotIn("synthetic-private-value", data)
        self.assertIn("[REDACTED_SECRET]", data)
        self.assertGreaterEqual(self.report["counts"]["redactions"], 2)

    def test_review_selects_only_accepted_targets_and_allows_text_correction(self):
        self.write("review.jsonl", [pi_header(), pi_message("u", None, "user", "First"), pi_message("a", "u", "assistant", "Incorrect text"), pi_message("u2", "a", "user", "Second"), pi_message("b", "u2", "assistant", "Unreviewed")])
        reviews = self.root / "reviews.jsonl"
        reviews.write_text(json.dumps({"source_id": "pi:pi-session:a", "accept": True, "replacement": {"content": "Corrected text"}}) + "\n")
        rows = self.prepare(selection="reviewed", review_file=reviews)
        targets = self.targets(rows)
        self.assertEqual([m["content"] for m in targets], ["Corrected text"])
        self.assertTrue(targets[0]["corrected"])

    def test_review_template_contains_ids_but_no_transcript_text(self):
        self.write("review.jsonl", [pi_header(), pi_message("u", None, "user", "Private source text"), pi_message("a", "u", "assistant", "Private target text")])
        self.prepare()
        template = (self.out / "review_template.jsonl").read_text()
        self.assertIn("pi:pi-session:a", template)
        self.assertNotIn("Private", template)
        self.assertIsNone(json.loads(template)["accept"])

    def test_exact_shared_prefix_unions_families_before_split(self):
        self.write("a.jsonl", [pi_header("family-a"), pi_message("u", None, "user", "Shared prompt"), pi_message("a", "u", "assistant", "Shared reply")])
        self.write("b.jsonl", [pi_header("family-b"), pi_message("u", None, "user", "Shared prompt"), pi_message("a", "u", "assistant", "Shared reply"), pi_message("u2", "a", "user", "New prompt"), pi_message("b", "u2", "assistant", "New reply")])
        rows = self.prepare(validation_fraction=0.5)
        self.assertEqual(len({r["group_id"] for r in rows}), 1)
        self.assertEqual(len({r["split"] for r in rows}), 1)
        self.assertEqual(len(self.targets(rows)), 2)

    def test_claude_response_identity_groups_corrected_resumed_copies(self):
        self.write("a.jsonl", [claude_message("u", None, "user", "First context", session="first"), claude_message("a", "u", "assistant", "Original response", session="first", response="shared-provider-id")])
        self.write("b.jsonl", [claude_message("u", None, "user", "Changed context", session="second"), claude_message("a", "u", "assistant", "Copied response", session="second", response="shared-provider-id")])
        rows = self.prepare(validation_fraction=0.5)
        self.assertEqual(len({r["group_id"] for r in rows}), 1)

    def test_nonfinite_json_number_is_a_malformed_line(self):
        self.write("number.jsonl", [pi_header(), '{"type":"message","id":"u","parentId":null,"message":{"role":"user","content":"Private marker","number":1e999}}'])
        self.assertEqual(self.prepare(), [])
        self.assertEqual(self.report["counts"]["malformed_lines"], 1)
        self.assertNotIn("Private marker", json.dumps(self.report))

    def test_pi_fork_parent_groups_even_without_shared_text(self):
        parent = self.write("parent.jsonl", [pi_header("parent"), pi_message("u", None, "user", "Parent question"), pi_message("a", "u", "assistant", "Parent answer")])
        self.write("child.jsonl", [pi_header("child", parentSession=str(parent)), pi_message("u2", None, "user", "Child question"), pi_message("b", "u2", "assistant", "Child answer")])
        rows = self.prepare(validation_fraction=0.5)
        self.assertEqual(len({r["group_id"] for r in rows}), 1)

    def test_split_is_deterministic_and_groups_disjoint(self):
        for i in range(40):
            self.write(f"{i:02}.jsonl", [pi_header(str(i)), pi_message("u", None, "user", f"Question {i}"), pi_message("a", "u", "assistant", f"Answer {i}")])
        first = self.prepare(validation_fraction=0.3)
        before = {r["session_id"]: r["split"] for r in first}
        second = self.prepare(validation_fraction=0.3, overwrite=True)
        self.assertEqual(before, {r["session_id"]: r["split"] for r in second})
        self.assertGreater(self.report["splits"]["validation"]["examples"], 0)
        train = {r["group_id"] for r in first if r["split"] == "train"}
        validation = {r["group_id"] for r in first if r["split"] == "validation"}
        self.assertFalse(train & validation)

    def test_bounded_line_skips_oversized_content_and_reports(self):
        self.write("large.jsonl", [pi_header(), pi_message("u", None, "user", "Question"), pi_message("a", "u", "assistant", "Good answer"), "x" * 600, pi_message("u2", "a", "user", "Later"), pi_message("b", "u2", "assistant", "Later answer")])
        self.assertEqual([m["content"] for m in self.targets(self.prepare(max_line_bytes=500, max_file_bytes=4000))], ["Good answer"])
        self.assertEqual(self.report["counts"]["oversized_lines"], 1)

    def test_oversized_file_is_bounded_and_reported(self):
        self.write("large.jsonl", ["x" * 1000])
        self.assertEqual(self.prepare(max_line_bytes=100, max_file_bytes=500), [])
        self.assertEqual(self.report["counts"]["oversized_files"], 1)

    def test_output_overwrite_is_explicit_and_inputs_unchanged(self):
        path = self.write("source.jsonl", [pi_header(), pi_message("u", None, "user", "Prompt"), pi_message("a", "u", "assistant", "Answer")])
        original = path.read_bytes()
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual((self.out / "train.jsonl").stat().st_mode & 0o777, 0o600)

    def test_synthetic_checked_in_fixtures(self):
        fixtures = Path(__file__).parent / "fixtures" / "transcripts"
        report = prepare_transcripts(fixtures, self.out, validation_fraction=0)
        self.assertEqual(report["counts"]["files_processed"], 2)
        self.assertGreaterEqual(report["splits"]["train"]["assistant_targets"], 3)


if __name__ == "__main__":
    unittest.main()
