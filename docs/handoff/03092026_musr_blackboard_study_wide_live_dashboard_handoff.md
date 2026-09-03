# MuSR Blackboard Study-Wide Live Dashboard — Implementation Handoff

**Date:** 2026-09-03

## Outcome

`mas-cc blackboard dashboard` now accepts either an existing full episode/run directory or a standardized study result root. A study process provides a study overview, cell details, compact vote trajectories, and links into the existing detailed episode inspector when full artifacts were retained.

The corrective pass preserves the detailed inspector as the inner view rather
than replacing it. Study-selected episodes use the same `BlackboardRunReader`
as direct episode loading.

Use either:

```text
mas-cc blackboard dashboard --study-dir <study-root>
mas-cc blackboard dashboard --run-dir <study-root>
```

The server remains localhost-only and read-only.

## Files changed

- `src/mas_cc/blackboard_dashboard/study_data.py`
- `src/mas_cc/blackboard_dashboard/server.py`
- `src/mas_cc/blackboard_dashboard/__init__.py`
- `src/mas_cc/blackboard_dashboard/assets/index.html`
- `src/mas_cc/blackboard_dashboard/assets/app.js`
- `src/mas_cc/blackboard_dashboard/assets/style.css`
- `src/mas_cc/cli/main.py`
- `tests/mas_cc/test_blackboard_study_dashboard.py`

## Supported layouts

A study root must contain `study_manifest.json` and `submission_manifest.csv`. It may also contain `execution_manifest.csv`, `execution_plan.json`, and `submission.json`.

Nested cell-array shard wrappers are resolved through the existing study discovery layer. Discovered cells support:

- full episodes under `data/episodes/<episode-id>`;
- compact trajectories under `round_records/<episode-id>`;
- compact completion artifacts under `.resume/<episode-id>` and sealed `scientific_events.parquet`.

Copied studies can be opened when their original submitted YAML paths are unavailable, provided retained resolved configs and execution metadata are present.

## Scientific status rules

- `completed`: an explicit completed full manifest, a validated compact
	per-episode artifact, or a valid cell seal covering the episode.
- `failed`: an explicit persisted failed outcome.
- `aborted`: an explicit aborted or `skipped_aborted` outcome.
- `pending`: expected from the resolved repetition count but no retained activity exists.
- `running`: a compact trajectory advances between refreshes, or the cell's scheduler task is running.
- `unknown`: records exist but no durable terminal state or observed live signal exists, or seal validation fails.

Scientific outcomes and live activity are separate. Activity is `advancing`,
`started_unchanged`, or `not_started` and comes only from episode-local
records. SLURM (Simple Linux Utility for Resource Management, the cluster
scheduler) data are cell annotations only. `squeue` or `sacct` is queried once
per job with a three-second bound and a short cache. Scheduler state never
marks an episode active, complete, or failed.

## Corrected path resolution

The original study implementation retained only the scientific cell path and
then passed that path as a run root. This hid the authoritative run manifest
and could report a retained episode as unavailable.

`ResolvedDashboardCellPaths` now preserves the execution shard root,
discovered run root, scientific cell root, full episode root, round-record
root, resume root, summary, seal, and scientific table. Status, trajectories,
and detail inspection all use this one resolution. Ambiguous duplicate run
matches fail explicitly. The detailed reader receives the discovered run root.

## Parameter provenance

Resolved cell configuration is flattened into the parameter table. Grid overrides replace matching values. Convenience labels include controller condition, `rho` (epistemic persistence), `b` (intervention budget), task, population, sensor size, controller target, and ground truth.

The study page now has an exact `rho` dropdown populated from available cell
values. The cell page shows a concise scientific summary and keeps the complete
configuration in a native, collapsible `details` section.

## APIs

- `GET /api/study`
- `GET /api/study/cells`
- `GET /api/study/cell/<qualified-cell-id>`
- `GET /api/study/cell/<qualified-cell-id>/votes`
- `GET /api/study/episode/<qualified-episode-id>/status`
- `GET /api/study/episode/<qualified-episode-id>/timeline`
- `GET /api/study/episode/<qualified-episode-id>/snapshot`

Identifiers are looked up in the catalog. They are never joined directly to filesystem paths.

## Refresh and retention behavior

Metadata are discovered once. Compact JSON Lines (JSONL, one JSON object per line) files are cached by path, modification time, and size. Cell seal validation is cached by seal/table signatures. Overview refreshes do not load prompts, audit traces, raw responses, checkpoints, or microscopic trajectories.

A partial final JSONL line is tolerated while an episode is live. It is rejected once the episode is classified complete.

Initialization is represented separately from social round zero. Missing values remain unavailable. Descriptive cell means state the available/expected episode count and do not interpolate missing rounds.

Detailed inspection is available only when full episode artifacts exist. Results-only episodes show an explicit explanation because prompts and microscopic updates were not retained.

The cell page separates its compact episode table from a dedicated
Trajectories tab. The table no longer embeds one chart per row. URL state
records the cell, episode, cell tab, episode tab, cursor, agent, follow mode,
`rho` filter, trajectory selections, and parameter disclosure. Polling updates
the selected model without calling the navigation path again.

## Verification

Focused validation:

```text
.venv/bin/python -m pytest tests/mas_cc/test_blackboard_dashboard.py tests/mas_cc/test_blackboard_study_dashboard.py -q
```

Result: 16 passed.

`node --check src/mas_cc/blackboard_dashboard/assets/app.js` passed. `git diff --check` passed. Editor diagnostics report no errors in changed Python files.

The wider study/resume/CLI selection completed with 37 passing tests and seven unrelated failures caused by study configuration files already absent from this checkout. No dashboard-related failure occurred.

## Remaining verification

A real 270-episode study root was not available in this local checkout, so a
truthful before/after live count cannot be reported here. The nested fixture
proves that study-selected and direct readers return identical timelines and
snapshots for the same full episode. Refresh-time and peak-memory measurements,
live Potsdam smoke testing, and a completed copied-study smoke remain to be
recorded on representative data. No SLURM command or scientific artifact was
changed during implementation or tests.
