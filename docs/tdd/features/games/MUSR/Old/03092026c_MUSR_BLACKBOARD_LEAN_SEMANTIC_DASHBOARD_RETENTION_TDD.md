# TDD: Lean Semantic Retention for the Live MuSR Blackboard Dashboard

**Date:** 2026-09-03  
**Status:** implementation plan  
**Scope:** retention and dashboard reconstruction; no scientific/game changes

## 1. Objective

Add a lean, dashboard-capable artifact profile for blackboard experiments.
It must preserve the semantic evolution of every episode sufficiently to use
the detailed live dashboard without retaining the heavy full audit/archive
tree.

The profile must support live and completed inspection of:

- population votes over time;
- public blackboard contents and message lifecycle;
- focal-agent decisions and public actions;
- active and historical evidence state;
- evidence acquisition, reactivation, refresh, and deactivation;
- controller sensing, action, target, and posted directives;
- round and microscopic-update timing/progress;
- validation attempts and compact repair outcomes.

It must not retain full prompts, raw provider responses, request bodies,
provider logs, duplicated state histories, or analysis caches.

## 2. New artifact profile

Introduce a first-class profile named:

```yaml
storage:
  artifact_profile: dashboard_semantic
  checkpoint_mode: episode
```

Supported profiles then have distinct contracts:

| Profile | Purpose |
|---|---|
| `full` | detailed debugging/audit, including prompts and raw responses |
| `dashboard_semantic` | lean live semantic inspection plus scientific analysis |
| `results_only` | smallest scientific/reaggregation handoff without detailed episode replay |

`dashboard_semantic` is not an alias for `full`. Its allowlist must be explicit
and tested. Do not implement it as “write everything and delete some files
later” during normal production runs.

## 3. Scientific invariants

This work must not change:

- prompts or provider calls;
- model/provider settings;
- random seeds or episode identities;
- blackboard sampling, lifetime, or exclusion rules;
- controller, persistence, sensing, or update semantics;
- validation or repair behavior;
- canonical scientific observations;
- estimators, aggregation, or analysis mathematics;
- episode-boundary resume behavior.

Retention must be a passive observer. Runs with `full`, `dashboard_semantic`,
and `results_only` must produce scientifically identical events for identical
seeds and provider responses.

## 4. Semantic data contract

Write one compact append-only semantic stream per episode, for example:

```text
dashboard_semantic.jsonl
```

The exact name may follow an existing artifact convention, but there must be
one authoritative stream rather than several duplicated dashboard tables.

### 4.1 Episode identity/header

Retain:

- schema version and game protocol version;
- study, config, run, cell, and episode identities;
- repetition index and seed;
- task/world identity and semantic hash;
- population size, rounds, q, persistence, and relevant controller settings;
- truth option and controller target;
- resolved-config and prompt-definition hashes;
- expected update counts.

Do not duplicate the entire resolved configuration inside every event.

### 4.2 Initialization record

Retain the initial semantic state:

- vote by agent and population vote counts;
- active and historical fact IDs by agent;
- initial evidence acquisition/provenance identifiers;
- empty/initial public board state;
- initialization completion timestamp or monotonic elapsed time.

### 4.3 Microscopic update record

For every update retain only the semantic transition needed for replay:

- round, within-round, and global update indices;
- focal agent;
- focal vote before and after;
- population vote counts/shares before and after;
- sampled public message IDs and author IDs;
- validated parsed public action/message;
- message ID, type, text, shared fact ID, `reply_to`, author, creation cursor,
  and expiry cursor for newly posted messages;
- messages expired at this cursor;
- focal active/historical fact IDs before and after, or an equivalent exact
  delta plus a recoverable snapshot boundary;
- fact acquisition, refresh, reactivation, and deactivation events with source;
- controller exposure/directive attribution where applicable;
- compact validation result: attempt count, valid/invalid, issue codes, and
  whether repair succeeded;
- elapsed timing necessary to display episode/update progress.

Do not retain:

- compiled prompt text;
- system/user prompt messages;
- raw provider response text;
- request/response HTTP envelopes;
- token-by-token output;
- private hidden reasoning;
- repeated copies of the complete board or every agent state on each update.

If the model contract contains a short, explicit user-facing rationale that is
scientifically part of the parsed decision, decide separately whether it is
part of the semantic schema. Do not preserve arbitrary raw response text under
the name “rationale.” The default lean profile should omit it unless a concrete
scientific/dashboard requirement is documented.

### 4.4 Round boundary record

Retain:

- round vote state;
- active and historical evidence summaries and exact agent fact-ID sets;
- active board messages after expiry/application;
- persistence deactivations/reactivations;
- controller sensing/action summary;
- round-level semantic counts;
- round completion time.

Round snapshots provide bounded random access. Microscopic records between two
round snapshots should be deltas, avoiding repeated population-wide state.

### 4.5 Completion record

Retain:

- completed, failed, or aborted status;
- final cursor and counts;
- semantic-stream row count and SHA-256;
- schema/config/episode identities;
- compact validation-failure summary;
- completion timestamp.

Publish completion atomically and validate the stream before marking the
episode reusable.

## 5. Dashboard capabilities

Teach `BlackboardRunReader` to detect explicit source capabilities and use a
dedicated adapter:

```text
full artifacts
    -> FullBlackboardAdapter

dashboard_semantic.jsonl
    -> SemanticBlackboardAdapter

both adapters
    -> existing canonical dashboard snapshot
    -> existing episode UI
```

The semantic adapter must restore the old dashboard’s important scientific
views:

- synchronized round/update navigation;
- population vote bars and trajectories;
- live/expired public board with replies;
- active and historical evidence matrix;
- focal-agent vote, visible sampled messages, parsed public action, and evidence
  transition;
- controller inspector;
- validation/retry counts;
- live auto-follow.

Features deliberately absent from lean retention must show a precise
capability message:

- compiled prompt: `Not retained by dashboard_semantic profile`;
- raw response: `Not retained by dashboard_semantic profile`;
- provider request details: `Not retained by dashboard_semantic profile`.

Their absence must not disable the rest of the episode inspector.

## 6. Storage and performance constraints

The profile is intended for hundreds of episodes and must be measurably lean.

- One semantic stream plus minimal manifest/seal files per episode.
- No per-update Markdown files.
- No persistent API-call, usage, budget-event, or experiment log streams in
  the scientific result tree.
- No prompt or response archive.
- No duplicate CSV/Parquet/JSON mirrors of the semantic stream.
- No persistent analysis caches or resampling draws.
- No complete population/board snapshot repeated at every microscopic update.
- Write JSONL incrementally so the live dashboard can safely read complete
  lines while an episode runs.
- Cache replayed snapshots in dashboard memory by file signature; do not write
  another cache directory.
- Use compact keys/records only where schema clarity remains adequate; do not
  use opaque binary encoding in the first version.

Preflight should estimate semantic-retention storage from `N`, rounds,
repetitions, cells, q, and expected message length. It should warn or deny when
a declared study storage ceiling is exceeded.

Acceptance target: demonstrate that a representative 30-round, N=24 episode is
substantially smaller than `full` retention. Record actual before/after bytes
and file counts. Do not claim a reduction percentage before measuring it.

## 7. Privacy and semantic boundaries

The public-board panel may contain only messages actually posted publicly.
Private active/historical fact IDs may appear only in the agent/evidence
inspectors, as they do in the scientific game state.

Do not retain or expose:

- hidden chain of thought;
- credentials or authorization headers;
- provider request metadata containing secrets;
- raw malformed responses;
- correction prompts;
- private prompt content merely for debugging convenience.

Compact validation issue codes and correction-attempt counts are sufficient
for normal dashboard health inspection.

## 8. Resume and lifecycle

Keep episode-boundary resume unchanged.

- A completed semantic stream and valid episode seal are durable.
- An incomplete episode may restart under the existing policy.
- Live partial JSONL tails are ignored until newline-complete.
- A completed stream with a malformed/truncated record is invalid and must not
  be silently repaired.
- Restart must not append a second logical history to a completed episode.
- Failed attempts and later successful attempts must remain distinguishable in
  compact status/provenance without retaining raw provider payloads.

The dashboard may inspect an active stream, but it must never become part of
the writer or resume protocol.

## 9. Aggregation and packaging

Canonical scientific aggregation continues to consume the established compact
scientific tables. The dashboard semantic stream is an observability sidecar,
not a replacement estimator input.

Standard scientific analysis ZIPs should not automatically include all
episode dashboard streams. Provide a separate, clearly named dashboard bundle
only when requested, or document that live semantic streams remain under the
study result root. Do not regress the compact standardized analysis-package
contract.

If a portable dashboard export is requested, include only:

- the selected semantic streams;
- minimal study/cell/episode manifests;
- dashboard assets;
- explicit capability metadata.

Never include SLURM logs, raw prompts/responses, provider logs, caches, or
analysis intermediates.

## 10. Implementation sequence

1. Inventory the exact fields currently consumed by every episode-dashboard
   panel.
2. Classify each field as semantic-required, debugging-only, derivable, or
   unavailable under the lean profile.
3. Define and version the semantic event schema.
4. Add the typed `dashboard_semantic` retention policy and configuration
   validation.
5. Add a direct semantic observer/writer in the game/orchestrator path.
6. Ensure the writer receives already-produced game events and never changes
   game decisions.
7. Add atomic episode validation and completion metadata.
8. Add `SemanticBlackboardAdapter` to the existing dashboard reader.
9. Restore all supported old episode views from the canonical snapshot model.
10. Add capability messages for deliberately omitted full-audit fields.
11. Integrate study-wide episode selection with semantic-profile episodes.
12. Measure size, file count, refresh latency, and replay memory.
13. Document launch and retention guidance for Potsdam blackboard studies.

## 11. Required tests

Add tests for:

1. parsing and validation of `artifact_profile: dashboard_semantic`;
2. rejection of unknown profile values;
3. identical game outcomes across all retention profiles using a deterministic
   provider;
4. identical canonical scientific tables and estimators;
5. semantic initialization record completeness;
6. vote reconstruction at every cursor;
7. board creation, replies, expiry, and visibility reconstruction;
8. active/historical evidence reconstruction;
9. acquisition, refresh, reactivation, and deactivation provenance;
10. controller sensing/action/directive reconstruction;
11. q=1 and q>1 board sampling semantics;
12. compact validation-attempt summaries;
13. live partial trailing JSONL handling;
14. malformed completed stream rejection;
15. episode completion seal/hash validation;
16. safe episode-boundary resume;
17. absence of compiled prompts and raw provider responses;
18. absence of provider/request, token, budget, and verbose experiment logs;
19. absence of duplicate dashboard tables and persistent caches;
20. full old-dashboard navigation for supported semantic panels;
21. explicit unavailable messages for omitted audit panels;
22. study-wide navigation into a running semantic episode;
23. refresh without rewriting or mutating source artifacts;
24. bounded file count per episode;
25. measured size comparison among `full`, `dashboard_semantic`, and
   `results_only` fixtures.

## 12. Acceptance criteria

The feature is complete when:

1. A running `dashboard_semantic` episode opens from the study dashboard.
2. Votes, board contents, message lifecycle, agent evidence, controller state,
   and semantic transitions can be followed at every retained cursor.
3. The old episode navigation experience is preserved for supported panels.
4. Raw prompts, raw responses, request logs, verbose logs, and caches are not
   written.
5. Unsupported audit-only panels clearly explain their retention limitation.
6. Scientific outputs are equivalent across retention profiles.
7. Episode resume and completion validation remain correct.
8. Storage/file-count measurements demonstrate a substantial reduction from
   `full` retention.
9. A 270-episode study remains practical to refresh and inspect live.
10. No game, controller, prompt, provider, or estimator behavior changes.

## 13. Migration and current-run limitation

This profile cannot reconstruct semantic details that a prior `results_only`
episode never retained. Do not fabricate missing board or evidence history.

Do not switch retention profiles in the middle of an existing scientific cell:
that would create heterogeneous observability and complicate provenance.
Apply `dashboard_semantic` to new studies or explicitly new, separately
identified arms/runs.

Existing `results_only` studies remain scientifically analyzable and may use
the study overview and compact vote trajectories, but their unavailable
episode details must remain labeled honestly.

## 14. Required handoff

After implementation, report:

- files and schemas added or changed;
- exact retained and excluded artifacts;
- supported dashboard panels under each profile;
- deterministic scientific-equivalence results;
- resume and live-tail test results;
- measured bytes/files per representative episode and projected study size;
- proof that a running study episode can be opened from the study dashboard;
- confirmation that no scientific/runtime semantics changed.
