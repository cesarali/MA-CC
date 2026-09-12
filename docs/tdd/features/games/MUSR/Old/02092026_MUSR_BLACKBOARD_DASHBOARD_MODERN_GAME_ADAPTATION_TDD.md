# TDD: Adapt the MuSR Blackboard Dashboard to the Modern Game Standards

**Date:** 2026-09-02  
**Status:** implementation plan  
**Scope:** compatibility and refinement only

## 1. Objective

Adapt the existing interactive MuSR blackboard dashboard to the repository's
current game-specific standards and runtime artifacts.

The existing dashboard must remain the basis of the work. Preserve its user
interface, scientific meaning, navigation model, plots, inspectors, filtering,
live-follow behavior, and static export as closely as possible. Do not redesign
the dashboard merely because the game architecture has evolved.

The original dashboard plan and implemented interface are documented in:

```text
docs/tdd/features/games/02092026_MUSR_INTERACTIVE_BLACKBOARD_DASHBOARD_PLAN.md
```

The primary implementation and focused tests are:

```text
src/mas_cc/blackboard_dashboard/
tests/mas_cc/test_blackboard_dashboard.py
```

## 2. Required first assessment

Before changing code:

1. Locate the dashboard CLI, server, artifact readers, normalization layer,
   snapshot schema, frontend assets, and static exporter.
2. Inspect the current game contracts and the game-specific artifact standards.
3. Run the existing dashboard tests.
4. Run a read-only smoke against a recent result produced by the modern game.
5. Determine whether the dashboard already works fully, partially, or not at
   all.
6. Record each concrete incompatibility before implementing a fix.

Classify incompatibilities as:

- renamed, moved, or versioned fields;
- changed result-directory layout;
- new game-specific metadata or event types;
- changed cursor, round, or microscopic-update representation;
- changed prompt, response, retry, controller, blackboard, or evidence records;
- genuinely missing retained data;
- presentation-only defects;
- malformed or unsupported source artifacts.

Do not infer missing scientific values and do not translate absent data into
zeros.

## 3. Compatibility architecture

Keep one canonical internal dashboard snapshot model. Modern and older game
artifacts should enter that model through explicit, side-effect-free readers
and adapters.

Preferred flow:

```text
source run artifacts
    -> schema/version detection
    -> game-specific artifact adapter
    -> canonical dashboard snapshot
    -> shared server/API/static-export path
    -> existing frontend
```

Game-specific translation must not be scattered through frontend components or
HTTP handlers. Add or refine an adapter boundary where the source schema
requires it.

Adapters must:

- retain stable scientific IDs and exact indices;
- distinguish initialization from social rounds;
- preserve before/after update semantics;
- preserve active versus historical evidence;
- preserve message liveness, expiry, reply links, and exact evidence transfer;
- preserve prompt, response, validation, repair, and retry provenance;
- preserve controller disabled, no-action, and missing-record distinctions;
- expose unsupported fields explicitly;
- avoid rewriting source artifacts.

If the modern result already satisfies the canonical dashboard contract, do
not add an unnecessary adapter.

## 4. Scientific and behavioral invariants

This work must not change:

- game behavior or controller behavior;
- provider requests or prompts;
- episode seeds or scientific identities;
- blackboard, evidence, persistence, or sensing semantics;
- estimator definitions or aggregation behavior;
- retained experimental observations;
- completed result folders.

The dashboard remains a read-only observability layer. It may normalize and
present retained data, but it must not create replacement scientific values or
silently reinterpret the game.

The phrase **private model rationale** must remain distinct from hidden chain of
thought. Private rationale may be shown only when it is explicitly present in
the retained model response and appropriate for the selected inspector.

## 5. Existing behavior to preserve

Preserve, where supported by the source run:

- synchronized episode, phase, round, and microscopic-update navigation;
- initialization as a distinct phase;
- explicit before/after state semantics;
- live auto-follow and manual navigation;
- overview and run-health summaries;
- vote distribution and truth-share display;
- public blackboard filtering, liveness, expiry, and reply navigation;
- active and historical evidence matrices;
- acquisition, refresh, reactivation, deactivation, and forgetting events;
- agent prompt, response, parsed action, public message, and retry inspection;
- controller sensing, action, target, controlled positions, and directives;
- provider-attempt and budget information;
- portable static export;
- localhost-only server defaults and existing security controls.

Do not remove a feature solely because one modern game result does not retain
the corresponding field. Show an explicit unavailable/unsupported state.

## 6. Schema and version handling

The dashboard must detect supported source formats using explicit schema,
manifest, game, and artifact metadata where available. Avoid weak detection
based only on the presence of one filename.

For every supported source format, document:

- identifying metadata;
- required files and fields;
- optional capabilities;
- normalization rules;
- unsupported features;
- errors that should stop loading.

Malformed completed records must remain errors. Partial trailing JSONL records
during live execution remain a normal `waiting_for_writer` condition.

Unknown schema versions must fail clearly rather than displaying plausible but
incorrect data.

## 7. Backward compatibility

Where reasonably possible, retain support for the completed pilot used to
develop the original dashboard. Backward compatibility must not force the
modern game to emit obsolete duplicate artifacts.

Use two acceptance fixtures:

1. a recent completed result produced by the modern game implementation;
2. the older completed dashboard pilot, when available.

The modern result is authoritative for current game integration. The older
fixture is a regression check for established dashboard behavior.

## 8. Implementation sequence

### Phase A: assess

1. Run existing focused tests.
2. Load both fixtures without modifying them.
3. compare source artifacts with the canonical snapshot contract;
4. write down the smallest required compatibility changes.

### Phase B: normalize

1. Add or update schema detection.
2. Add narrowly scoped game-specific adapters.
3. Keep reusable readers pure and side-effect-free.
4. Preserve canonical cursor and snapshot semantics.
5. Add explicit capability/unavailable reporting.

### Phase C: integrate

1. Reuse the same canonical snapshots for API and static export.
2. Update frontend assumptions only where the canonical contract genuinely
   changed.
3. Preserve existing paths and CLI commands where practical.
4. Do not introduce another dashboard implementation.

### Phase D: verify

1. Run focused unit and API tests.
2. Smoke the modern completed result.
3. Export and inspect a portable dashboard.
4. Regression-test the older pilot.
5. Hash source scientific artifacts before and after inspection to prove they
   were not changed.

## 9. Tests

Add or update tests for:

1. modern schema detection;
2. older schema detection where retained;
3. normalization into the canonical snapshot;
4. cursor ordering and initialization semantics;
5. round and microscopic-update reconstruction;
6. blackboard creation, expiry, replies, and visibility;
7. active and historical evidence reconstruction;
8. modern evidence event names and provenance;
9. prompt, response, validation-repair, and retry joins;
10. controller no-action, disabled, and unavailable distinctions;
11. missing optional capabilities;
12. malformed and unknown schemas;
13. partial live JSONL writes;
14. API behavior for modern and older fixtures;
15. static export equivalence with server snapshots;
16. read-only operation;
17. escaping and path-traversal protections;
18. existing dashboard regression behavior.

Tests should compare scientific values and stable IDs, not only assert that a
page renders.

## 10. Acceptance criteria

The adaptation is complete when:

1. The dashboard loads a current modern-game result without modifying it.
2. All available rounds and microscopic updates are navigable.
3. Blackboard, votes, evidence, agents, controller records, and provider status
   match retained artifacts at selected cursors.
4. Missing capabilities are labeled explicitly.
5. Live server and static export use the same normalized data.
6. The existing dashboard interaction model and visual organization remain
   substantially unchanged.
7. The older pilot still works where backward compatibility is practical.
8. Unknown or malformed source schemas fail clearly.
9. No game, controller, estimator, or experimental code changes are required
   solely for dashboard compatibility.
10. Focused tests and both fixture smokes pass.
11. Source scientific artifact hashes are unchanged after inspection/export.

## 11. Non-goals

- redesigning the dashboard;
- building a general analytics platform;
- modifying or rerunning experiments;
- changing the modern game standards to match the old dashboard;
- duplicating modern artifacts in a legacy layout;
- changing scientific estimators or observables;
- introducing a database or frontend framework without demonstrated need;
- adding editing or experiment-control capabilities.

## 12. Required handoff

At completion report:

- files modified;
- whether the existing dashboard already worked and to what extent;
- exact incompatibilities found;
- adapter and schema decisions;
- preserved and unavailable capabilities;
- fixtures used;
- tests and smoke checks run;
- source-artifact hash verification;
- remaining limitations.

The desired result is the existing dashboard, refined to consume the modern
game outputs correctly and cleanly while preserving its established behavior
and scientific meaning.
