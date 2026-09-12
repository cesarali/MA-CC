# TDD: Study-Wide Live MuSR Blackboard Dashboard

**Date:** 2026-09-03  
**Status:** implementation plan  
**Scope:** extend the existing read-only episode dashboard to inspect a complete standardized study

## 1. Objective

Extend the existing MuSR blackboard dashboard so one server can open a
standardized study root and show all scientific cells and episodes live.

The study view must answer, at a glance:

- how many cells and episodes are pending, running, complete, or failed;
- which cells and episodes are currently running;
- the resolved scientific parameters of every cell;
- progress within each running episode;
- population-vote time series for every episode;
- which cell/episode should be opened in the existing detailed dashboard.

This is an observability extension, not a new execution or analysis system.
The existing episode dashboard, canonical snapshot model, and scientific
artifacts remain authoritative.

## 2. Existing implementation to reuse

Preserve and extend:

```text
src/mas_cc/blackboard_dashboard/data.py
src/mas_cc/blackboard_dashboard/server.py
src/mas_cc/blackboard_dashboard/assets/
tests/mas_cc/test_blackboard_dashboard.py
```

The current `BlackboardRunReader` remains responsible for detailed inspection
of one episode. Add a study-level reader/catalog above it rather than copying
its reconstruction logic.

The standardized study layout already provides the required hierarchy:

```text
study root
  execution_manifest.csv
  execution_plan.json
  submission.json
  study_manifest.json
  runs/<config>/shards/<cell>/...
    cells/<cell>/overrides.json
    round_records/<episode>/round_trajectory.jsonl
    data/episodes/<episode>/...
```

`execution_manifest.csv` is the primary cell index. Scientific identity and
parameters come from the resolved config and `overrides.json`, never from the
SLURM array index alone.

## 3. User interface

### 3.1 Study overview

Add a study landing page containing:

- study name and submission/job identity;
- expected and discovered cell counts;
- expected, running, completed, failed, and pending episode counts;
- active SLURM array tasks when scheduler information is available;
- last refresh time and live/finished state;
- filters for experiment block, controller condition, cell status, and
  scientific parameter values;
- a sortable cell table or compact cell-card grid.

Each cell row/card must show:

- stable config name and scientific cell ID;
- resolved coordinates relevant to that cell, including controller condition,
  `rho`, `b`, and other configured sweep values;
- repetitions expected and episodes discovered;
- episode counts by lifecycle state;
- current round/update for running episodes;
- elapsed time where it can be established without inference;
- a compact vote-trajectory preview;
- an action to open the cell and its episodes.

Unknown or unavailable values must be displayed as unavailable, not as zero.

### 3.2 Cell detail

Selecting a cell opens a cell page with:

- its complete resolved parameter table;
- all episode IDs and deterministic seeds;
- status and progress for each episode;
- one overlaid or small-multiple vote trajectory per episode;
- optional cell mean trajectory clearly labeled as descriptive and based only
  on currently available records;
- truth, false-controller-target, and other-option shares where retained;
- controls to hide/show individual repetitions;
- a link from each episode to the existing detailed episode dashboard.

The display must make partial data explicit. A live mean over 4/10 episodes
must say `4/10 available`; it must not appear to be a complete cell estimate.

### 3.3 Episode detail

Reuse the current episode dashboard without removing features. The navigation
path becomes:

```text
study -> cell -> episode -> existing round/microscopic-update inspector
```

Add breadcrumbs and previous/next episode navigation. All existing blackboard,
agent, evidence, controller, prompt, retry, and cursor views remain intact.

## 4. Status semantics

Use two deliberately separate status sources.

### 4.1 Durable scientific status

Determine episode/cell completion from manifests, completion seals, retained
records, and established repository resume semantics. These artifacts are the
source of truth for completed scientific work.

Classify episodes as:

- `completed`;
- `running` when records are advancing or an explicit live marker applies;
- `failed` or `aborted` only when recorded as such;
- `pending` when expected but not yet started;
- `unknown` when artifacts are inconsistent or insufficient.

Do not call an episode complete merely because its SLURM task exited, and do
not call it failed merely because a process is absent.

### 4.2 Optional scheduler status

When running on Potsdam and `submission.json` contains a job ID, an optional
read-only scheduler adapter may enrich the display with `squeue`/`sacct` state:

- array task running/pending/held;
- assigned node;
- elapsed scheduler time;
- terminal exit state.

Scheduler data are operational annotations only. The dashboard must continue
to work after SLURM accounting expires, outside Potsdam, and for copied study
folders. Scheduler commands must be bounded, cached briefly, and never invoked
once per episode.

The dashboard must never submit, cancel, hold, release, or retry jobs.

## 5. Data architecture

Add one study-level, read-only layer:

```text
study manifests + execution manifest + run/cell artifacts
    -> BlackboardStudyReader / study catalog
    -> compact study and cell summaries
    -> lazy BlackboardRunReader for selected episode
    -> existing canonical episode snapshots
```

Recommended internal objects:

- `StudyDescriptor`: study identity, expected totals, configs, submission;
- `CellDescriptor`: qualified cell identity, resolved coordinates, path;
- `EpisodeDescriptor`: episode identity, seed, status, progress, path;
- `VoteSeries`: round/update coordinates and vote-share series;
- `SchedulerSnapshot`: optional job/array operational state.

Stable qualified identity should include config/run identity plus cell and
episode IDs. Cell names such as `cell-0000` are not globally unique across a
multi-config study.

## 6. Performance and live refresh

This extension must remain lightweight even for hundreds of episodes.

- Load study/cell metadata eagerly because it is small.
- Read only compact manifests, seals, and round trajectories for the overview.
- Do not load audit traces, full prompts, raw responses, microscopic
  trajectories, or checkpoints until an episode is selected.
- Cache parsed files by path, modification time, and size, following the
  current reader pattern.
- On refresh, re-read only changed files.
- Tolerate a partial trailing JSONL line while its writer is active.
- Reject malformed records in a completed/sealed episode.
- Expose one batched study-summary endpoint rather than issuing one request per
  episode from the browser.
- Use polling initially; WebSockets, a database, and a background indexing
  service are not required.

For a 270-episode study, a normal refresh should read changed compact records,
not rebuild 270 detailed episode snapshots.

## 7. API changes

Preserve existing episode endpoints and add approximately:

```text
GET /api/study
GET /api/study/cells
GET /api/study/cell/<qualified-cell-id>
GET /api/study/cell/<qualified-cell-id>/votes
GET /api/study/episode/<qualified-episode-id>/status
```

The exact routing may differ, but identifiers must be validated and resolved
through the catalog. User input must never be converted directly into an
arbitrary filesystem path.

Allow the existing CLI to accept either an episode/run directory or a study
root. Prefer automatic, explicit schema/layout detection, with an optional
`--study-dir` spelling if needed for clarity. Existing commands must remain
valid.

## 8. Vote time-series contract

Build vote trajectories only from retained scientific trajectory records.
For every point retain:

- qualified study/config/cell/episode identity;
- repetition and seed when available;
- phase and round index;
- truth share;
- controller-target share when applicable;
- counts/shares for all vote options;
- number of agents represented;
- record completeness/live status.

Initialization must remain distinct from social rounds. The frontend must not
silently interpolate missing rounds, convert missing observations to zero, or
mix episodes with different option meanings.

The overview preview may downsample for rendering, but the API and selected
cell view must preserve the actual recorded points.

## 9. Implementation sequence

### Phase A: fixtures and discovery

1. Use a copied/minimal study fixture with multiple configs, cells, and
   episodes, including pending, running, completed, and failed examples.
2. Document the exact current study/run/cell/episode paths and seals.
3. Add study-root detection and manifest validation.
4. Resolve qualified cell identities and scientific overrides.

### Phase B: compact live model

1. Implement `BlackboardStudyReader` and immutable descriptors.
2. Join expected episodes to discovered artifacts without inventing records.
3. Implement compact vote-series extraction from round trajectories.
4. Add cached optional scheduler enrichment.
5. Add incremental refresh based on file signatures.

### Phase C: API and interface

1. Add batched study/cell endpoints.
2. Add the study overview and filtering.
3. Add the cell parameter/status table and vote plots.
4. Add cell detail with episode trajectories.
5. Route selected episodes into the existing detailed inspector.
6. Preserve live auto-follow and localhost-only serving.

### Phase D: verification

1. Run focused unit, API, frontend, and security tests.
2. Smoke against a live standardized blackboard study read-only.
3. Smoke against a completed copied study without SLURM access.
4. Verify that source artifact hashes do not change.
5. Confirm bounded refresh time and memory on a 270-episode fixture.

## 10. Required tests

Add tests for:

1. study-root detection and rejection of unrelated/unknown layouts;
2. `execution_manifest.csv` mapping across multiple configs;
3. globally qualified cell and episode identities;
4. resolved parameter/override presentation;
5. expected versus discovered cells and episodes;
6. pending, running, completed, failed, aborted, and unknown status handling;
7. sealed scientific status versus scheduler status;
8. held and running array-task annotations;
9. operation when `squeue`/`sacct` are unavailable;
10. live partial JSONL tails;
11. malformed sealed JSONL rejection;
12. vote counts and truth/controller-target shares over time;
13. initialization versus round indexing;
14. partial-cell descriptive trajectory labeling;
15. cache invalidation when a trajectory grows;
16. no detailed audit/prompt loading on overview refresh;
17. batched API behavior and path traversal protection;
18. navigation from study to cell to existing episode inspector;
19. current single-episode dashboard regression tests;
20. strictly read-only operation and unchanged source hashes.

## 11. Acceptance criteria

The work is complete when:

1. One dashboard process can open a standardized multi-config study root.
2. It reports expected and observed totals for every cell and episode.
3. It identifies currently running cells/episodes and optional SLURM task/node
   state without treating scheduler state as scientific completion.
4. Every cell shows its resolved working parameters.
5. Every episode exposes its recorded population-vote time series.
6. A user can navigate into the existing full episode inspector.
7. Live refresh reads only changed compact artifacts and remains responsive on
   a 270-episode study.
8. Partial studies and partial trajectories are labeled honestly.
9. The dashboard performs no provider calls and modifies no experiment files.
10. Existing single-episode behavior and tests remain valid.

## 12. Non-goals

- changing game, controller, persistence, prompt, or validation behavior;
- changing study submission, retries, aggregation, or estimators;
- controlling SLURM jobs from the dashboard;
- recomputing phase diagrams or inferential statistics live;
- storing a second scientific dataset or introducing a database;
- loading every prompt/audit artifact merely to render the study overview;
- replacing the existing detailed episode dashboard;
- creating a study-specific dashboard implementation.

## 13. Handoff requirements

After implementation, report:

- files changed;
- supported study and episode layouts;
- status and parameter provenance rules;
- API additions;
- refresh-time and memory measurements on a representative large study;
- tests and live/completed-study smoke results;
- confirmation that scientific artifacts and scheduler state were not changed.
