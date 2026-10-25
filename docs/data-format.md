# Local transcript preparation

`dust_qlora.data.prepare_transcripts` converts local Pi and Claude Code JSONL
session files into model-neutral conversations for supervised training. It uses
the Python standard library. It does not contact an API, run a teacher model,
read referenced output files, or upload transcripts. The checked-in examples are
synthetic.

The converter preserves recorded assistant actions and final answers. A
transcript is an observation, not a success label: the default `selection="all"`
means all eligible observed assistant messages. It does not mean all tasks were
completed correctly. Use local review decisions to curate a first training set.

## Prepare a local corpus

The typical input directories are `~/.pi/agent/sessions/` and
`~/.claude/projects/`. Claude's configuration directory may be different when
`CLAUDE_CONFIG_DIR` is set; pass that actual directory explicitly. Inputs can be
files, recursive directory roots, or glob patterns. Nothing is read until you
pass an input path, and source files are never changed.

```python
from pathlib import Path
from dust_qlora.data import prepare_transcripts

report = prepare_transcripts(
    [
        Path.home() / ".pi/agent/sessions",
        Path.home() / ".claude/projects",
    ],
    "data/prepared",
    validation_fraction=0.1,
    seed=42,
    branch_policy="latest",
    redact_secrets=True,
)
print(report["splits"])
print(report["counts"])
```

Only pass directories that exist. The function raises on a missing input so an
incorrect path cannot silently produce an apparently complete dataset.
Directories are scanned for `*.jsonl`; files can have any extension when named
explicitly. The output directory is excluded from discovery. Keep other exports
and review files outside the raw input tree.

The output directory contains:

| File | Contents |
| --- | --- |
| `train.jsonl` | Training conversations with explicit assistant target masks. |
| `validation.jsonl` | Entire held-out session families, with the same schema. |
| `report.json` | Counts, source file fingerprints, settings, split sizes, and warnings. |
| `review_template.jsonl` | Stable assistant IDs and empty review decisions; no conversation excerpts. |

Files are created with mode `0600`, and the output directory uses `0700` on
systems that implement POSIX permissions. Each completed output file is renamed
into place. Existing outputs require `overwrite=True`. The report contains
session IDs and hashed source paths, so it is still local dataset metadata.

## Canonical schema

Each JSONL line is a conversation segment, before tokenization or context-window
chunking:

```json
{
  "schema_version": 1,
  "id": "stable-example-fingerprint",
  "group_id": "stable-family-fingerprint",
  "session_id": "recorded-session-id",
  "source": "pi",
  "split": "train",
  "messages": [
    {
      "role": "user",
      "content": "Inspect the configuration.",
      "source_id": "pi:recorded-session-id:user-entry-id",
      "trainable": false
    },
    {
      "role": "assistant",
      "content": "",
      "source_id": "pi:recorded-session-id:assistant-entry-id",
      "trainable": true,
      "tool_calls": [
        {
          "id": "call-1",
          "type": "function",
          "function": {
            "name": "read",
            "arguments": {"path": "example.json"}
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": "{\"enabled\":true}",
      "source_id": "pi:recorded-session-id:tool-entry-id",
      "tool_call_id": "call-1",
      "name": "read",
      "is_error": false,
      "trainable": false
    },
    {
      "role": "assistant",
      "content": "The feature is enabled.",
      "source_id": "pi:recorded-session-id:final-entry-id",
      "trainable": true
    }
  ],
  "metadata": {
    "leaf_id": "final-entry-id",
    "segment": 0,
    "source_file_id": "hashed-source-path"
  }
}
```

`function.arguments` is a JSON object, not a pre-serialized string. Tool schemas
appear in an optional top-level `tools` list only when a supported source record
actually includes them. Schemas use the `{"type":"function","function":...}`
shape. A tool's `name` is retained on tool results when available. The converter
does not guess argument schemas from calls or reconstruct unrecorded Claude
system instructions.

Recorded thinking text is preserved separately as `reasoning_content` on the
assistant message. Opaque thought signatures and `redacted_thinking` payloads
are omitted. The tokenizer must use the chosen model's chat template and make
its treatment of reasoning explicit. The converter does not insert `<think>`
tags or provider-specific role delimiters.

Only assistant messages can have `trainable: true`. User prompts, tool results,
system instructions, summaries, and historical assistant context have no
supervised loss. The training layer must respect these masks; applying loss to
the entire rendered conversation would defeat the selection performed here.

## Trees, streaming records, and context changes

**Pi:** version 2/3 session entries form a tree through `id` and `parentId`.
The session header is not a message node. `branch_policy="latest"` selects the
most recently appended leaf. `"all"` selects all leaves while suppressing
duplicate shared-prefix targets. Version 1 linear logs receive deterministic
local IDs and are traversed in file order.

**Claude Code:** `uuid` and `parentUuid` identify transcript ancestry. Main
session selection avoids metadata and sidechain leaves by default. Claude can
write separate assistant blocks with the same API `message.id`, and parallel
tool results can form sibling branches. The importer recovers sibling blocks
and matching tool results for the same writer, then coalesces those assistant
blocks into one response. Duplicate UUIDs use the last recorded value; repeated
identical content blocks and tool-call IDs do not become repeated targets.

Pi compaction records, system/tool updates, and `context_edit` operations end a
training segment. Earlier targets retain the context that preceded the change.
Later segments contain the new context, with copied assistant history masked.
Compaction summaries are user context, never assistant targets. Pi retained
message ranges and recorded system checkpoints are honored. Custom messages,
branch summaries, and ordinary `!` shell execution context are normalized into
user context; `!!` executions marked `excludeFromContext` are omitted.

Pi's named system sections and tool additions/removals are replayed into a
current leading system message and current tool schema list. Content is
normalized for a different model's template; this is not a byte-identical
reconstruction of a provider request. A provider's original system instructions
and tool definitions may be absent in older Pi or Claude transcripts.

Claude `compact_boundary` and `isCompactSummary` records start a new context.
The importer does not follow `logicalParentUuid` across a compaction boundary.
It also recovers the last pre-boundary training trajectory so useful earlier
turns are retained without leaking the later summary into them.

Subagent files under `<session-id>/subagents/` are imported as separate
trajectories and grouped with their enclosing session. `include_sidechains=True`
also allows sidechain leaf selection in a main session file. To include all
branches of that file, combine it with `branch_policy="all"`.

## Selection, local review, and corrections

The default rules mask these assistant targets:

- Explicit aborts, API errors, length-limited responses, and refusal stop codes.
- An assistant message containing a tool call with a recorded failed result,
  when `mask_failed_tool_calls=True`. A response with several calls is masked
  as a whole if any one fails. Its actions and error results remain available as
  context for a later recovery response.
- Targets after missing image/document/audio/video context or an unsupported
  content block. Earlier text-only targets in that segment can still be used.
- Targets without a preceding user/tool prompt, after a malformed/oversized
  source line, or after an orphan tool result whose call is unavailable.
- Assistant continuations before all preceding tool calls have recorded
  results. A user interruption does not imply cancellation. If the remaining
  results arrive later, the missing-result mask can clear for subsequent turns;
  nonstandard interleaving may still be rejected by the target chat template.
- Targets rejected in a review file or not accepted in reviewed-only mode.
- Repeated targets with an exact same normalized conversation prefix.

Branches with missing parents or cycles are skipped rather than being joined
to unrelated messages. An assistant tool call at the end of a complete branch
can remain a target even when its future result was never recorded. No success
is inferred from that absence, from a successful individual tool call, or from
the assistant saying it is done.

Start with a normal preparation run, then copy `review_template.jsonl` to a
separate location and edit it while inspecting the corresponding local
conversations. Stable `source_id` values connect review decisions to their
messages. Pi uses the raw entry ID; Claude uses the API response ID when
present, with UUID fallback. A decision looks like:

```json
{"source_id":"pi:recorded-session-id:assistant-entry-id","accept":true}
```

Set `accept` to `false` to reject a target. `null` means unreviewed. For a text
correction:

```json
{"source_id":"pi:recorded-session-id:final-entry-id","accept":true,"replacement":{"content":"The corrected answer."}}
```

Only `content` and `reasoning_content` text can be replaced this way. Tool call
and result corrections require editing and validating the canonical dataset as
a coordinated change. Replacing an earlier answer changes subsequent training
context, so review any dependent later targets too. An accepted correction
does not override structural, missing-content, or failed-tool safeguards.

Run preparation again into a fresh output directory:

```python
prepare_transcripts(
    ["/actual/pi/sessions", "/actual/claude/projects"],
    "data/reviewed",
    selection="reviewed",
    review_file="data/reviews/decisions.jsonl",
    seed=42,
)
```

In `selection="reviewed"`, only explicitly accepted targets are eligible.
`selection="all"` still applies explicit rejections and accepted text
corrections. There is no automatic LLM judge. The template enumerates assistant
messages retained in emitted eligible segments; wholly unusable segments are
represented in the report instead.

## Held-out data and deduplication

Split assignment happens at the session-family level, before token chunking.
The same session, Pi fork ancestry, Claude subagents, repeated Claude provider
response IDs, and exact shared conversation/target prefixes join the same
family. A SQLite index tracks prefix
fingerprints and suppresses duplicate targets. Its temporary database is deleted
when preparation finishes. Only the current file's tree is indexed in memory;
the corpus itself is spooled on disk.

The partition is a deterministic hash of the seed and family identity. A 10%
validation fraction is an expected fraction of families, not an exact fraction
of rows or tokens. Small datasets may produce an empty validation or training
partition; the report warns about this. Add independent sessions or adjust the
seed/fraction. Do not split messages from one session to fill a missing
partition.

Deduplication is exact, not semantic. It will not find every paraphrased task,
renamed file, remapped tool ID, or copied repository snippet. A session-family
split does not guarantee that projects are independent. Reserve a separate
project/task benchmark if that is the deployment question you need to measure.

## Redaction, missing content, and resource limits

`redact_secrets=True` replaces common literal API tokens, authorization bearer
values, PEM private keys, and common secret-valued fields/assignments with
`[REDACTED_SECRET]` in exported strings and arguments. It is a best-effort
filter, not a proof that the dataset contains no credentials or private data.
It can also replace illustrative literals in code. Raw inputs remain unchanged.
Review locally, and keep raw/prepared data out of version control.

This text pipeline does not decode or train image inputs. It inserts a visible
placeholder and masks dependent targets by default. `allow_missing_images=True`
explicitly permits training with incomplete non-text/unsupported content
placeholders; the override is counted. Use that only after reviewing what was
lost. File paths mentioned by tools are not opened automatically. Truncated
tool output remains truncated, as it was in the available transcript.

The default maximum physical JSONL line is 16 MiB; the default maximum source
file is 256 MiB. Lines are read with a bounded buffer and oversized files are
reported and skipped. The parsed tree for one source file can occupy several
times its serialized size, so this is a bounded per-file approach rather than
constant-memory parsing of arbitrary-size trees. Increase `max_line_bytes` and
`max_file_bytes` deliberately for a known export. `max_sessions` limits the
number of source files processed for a pilot run.

Inspect `report.json` before tokenization. Relevant counters include
`malformed_lines`, `oversized_lines`, `oversized_files`, `missing_parents`,
`cycles`, `nontext_blocks`, `unknown_content`, `unresolved_compaction_ranges`,
`orphan_tool_results`, `failed_tool_call_targets_masked`,
`missing_tool_result_targets_masked`, `parallel_records_recovered`,
`duplicate_targets_masked`, and `redactions`.
Counters with zero occurrences may be absent. File reports contain counts and
IDs, not raw message excerpts. The importer handles the source shapes covered
by its synthetic regressions; validate a small sample of your actual client
versions before converting months of history.

## Source formats and tests

The implementation was checked against these primary sources:

- [Pi session format](https://pi.dev/docs/latest/session-format) and
  [message types](https://pi.dev/docs/latest/message-types).
- [Pi session manager](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/src/core/session-manager.ts):
  version migrations, parent-linked trees, compaction projections, and context edits.
- [Pi message conversion](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/src/core/messages.ts):
  summaries, custom messages, and shell execution context.
- [Pi message/tool type definitions](https://github.com/earendil-works/pi/blob/main/packages/ai/src/types.ts).
- [Anthropic's Claude Agent SDK session reader](https://github.com/anthropics/claude-agent-sdk-python/blob/main/src/claude_agent_sdk/_internal/sessions.py):
  local transcript paths, parent traversal, parallel assistant-block/tool-result
  recovery, compaction behavior, and subagent paths.
- [Claude Code hook documentation](https://code.claude.com/docs/en/hooks):
  the local `transcript_path` field.

Run the converter tests without loading a model or requiring a GPU:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_data.py -v
```

Tests use generated temporary logs and the two explicitly synthetic fixtures in
`tests/fixtures/transcripts/`. No personal conversation is checked in.
