# Lean Semantic MuSR Blackboard Dashboard Retention — Handoff

**Date:** 2026-09-03

## Outcome

A new first-class `dashboard_semantic` artifact profile writes a lean semantic
sidecar directly while the episode runs. It is not a full-profile run followed
by deletion. The profile keeps existing compact scientific Parquet, round, and
microscopic trajectory artifacts unchanged for aggregation.

Configuration:

```yaml
storage:
  artifact_profile: dashboard_semantic
  checkpoint_mode: episode
  options:
    expected_public_message_characters: 240
logging:
  options:
    prompt_examples: {count: 0, scope: cell}
```

It currently applies to `relational_imitation_round_feedback` in board mode.
Prompt examples above zero are rejected.

## Files and schemas

Added:

- `src/mas_cc/storage/dashboard_semantic.py`
- `src/mas_cc/planning/semantic_storage.py`
- `tests/mas_cc/test_dashboard_semantic_retention.py`

Changed configuration, recorder/runtime, resume, dashboard, study discovery,
preflight, identity, tests, and reference documentation.

Semantic episode layout:

```text
round_records/<episode-id>/
  dashboard_semantic.jsonl
  dashboard_semantic_complete.json
  round_trajectory.jsonl
  micro_slot_trajectory.jsonl
```

`dashboard_semantic.jsonl` schema version 1 has typed records:

- `header`
- `initialization`
- `round_start`
- `validation`
- `update`
- `round_end`
- `completion`

The completion seal records the whole-stream SHA-256, row count, identity, final
cursor, and completion time. Strict validation requires one header,
initialization, and completion; one final completed status; identical identity
on every row; unique cursors; contiguous global updates; and the configured
expected update count.

## Retained semantics

- stable run/cell/episode identity and seed;
- resolved-config and prompt-definition hashes;
- task identity, truth, controller settings, population, rounds, q, and rho;
- initial votes and exact active/historical fact IDs per agent;
- focal vote deltas and population count/share summaries;
- sampled public message identities;
- validated parsed public vote/message action;
- public message creation, replies, expiry fields, and active round snapshots;
- evidence acquisitions, refreshes, reactivations, and dawn deactivations;
- controller sensing, target, action, directives, and exposure fields;
- round/update cursors and timestamps;
- compact validation attempts, issue-field codes, and repair markers.

Microscopic semantic rows do not repeat full population vote vectors. The
reader reconstructs these from initialization plus focal vote deltas. Round
snapshots retain only currently active public messages.

## Excluded artifacts and content

The semantic profile does not write:

- compiled/system/user prompts;
- raw provider responses or malformed raw values;
- private model reasoning;
- request/response envelopes or provider details;
- `audit_traces.jsonl`;
- `api_call_status.jsonl`;
- `usage_cost.jsonl`;
- `budget_events.jsonl`;
- `prompt_block_traces.jsonl`;
- `experiment.log`;
- prompt Markdown or prompt-candidate archives;
- dashboard caches or duplicate semantic tables.

Semantic event logging uses a strict cursor/status allowlist. Compact failure
outcomes retain the exception class but omit arbitrary exception text.

## Dashboard capabilities

`BlackboardRunReader` auto-detects a full episode or a semantic stream and
produces the same timeline/snapshot API.

Available under `dashboard_semantic`:

- round and microscopic-update navigation;
- population votes;
- public board and message lifecycle;
- active/historical evidence;
- focal public decision;
- controller state;
- compact validation attempts;
- live follow.

The prompt, raw-response, and provider-detail panels explicitly say:

```text
Not retained by dashboard_semantic profile
```

The study reader recognizes semantic episodes as inspectable, follows their
microscopic stream for live activity/progress, caches the selected reader, and
never mutates source artifacts.

## Resume behavior

A semantic episode is reusable only when its normal scientific checkpoint and
semantic stream/seal both validate. The scientific checkpoint is published
before the semantic completion seal. An unsealed prior semantic attempt is
replaced before a rerun, so histories are never concatenated. A sealed semantic
stream cannot be appended.

Retention fields are excluded from scientific protocol fingerprinting, so
changing retention does not redefine the scientific protocol. Reuse still
requires the requested observability artifact to exist and validate.

## Scientific equivalence and tests

A deterministic mock-provider test runs the same seed under `full`,
`dashboard_semantic`, and `results_only`. The retained round scientific records
match after excluding storage-only provenance labels. Full and semantic readers
reconstruct matching final vote counts and truth share. Existing estimators and
canonical scientific writers are unchanged.

Focused validation command:

```text
.venv/bin/python -m pytest \
  tests/mas_cc/test_dashboard_semantic_retention.py \
  tests/mas_cc/test_blackboard_dashboard.py \
  tests/mas_cc/test_blackboard_study_dashboard.py \
  tests/mas_cc/test_results_only_resume.py \
  tests/mas_cc/test_config.py \
  tests/mas_cc/test_relational_blackboard.py \
  tests/mas_cc/test_relational_musr_blackboard.py -q
```

Result: **97 passed**.

JavaScript syntax, patch whitespace, and editor diagnostics also pass.

## Measured retention

The deterministic checked-in N=24, 2-round blackboard smoke produced:

| Profile | Episode bytes | Episode files |
|---|---:|---:|
| `full` | 2,078,589 | 13 |
| `dashboard_semantic` | 281,971 | 4 |
| `results_only` | 109,669 | 2 |

These are measured fixture values, not a claimed universal reduction. The
semantic profile was substantially smaller than full while remaining larger
than results-only, as intended.

For the planned 270-episode study, multiplying this measured semantic episode
size gives approximately 76.1 MB for the per-episode round-record directories,
excluding shared cell/run manifests and tables. Preflight separately emits a
conservative configuration-derived estimate and supports
`preflight.storage_ceiling_bytes` denial.

## Current-run limitation

Existing `results_only` episodes cannot gain board/evidence history
retroactively. Do not switch the active study in place. Use this profile for a
new, separately identified run or study arm. No production job was submitted,
no scheduler state was changed, and no existing scientific artifact was
modified during implementation.
