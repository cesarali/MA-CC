# MuSR Blackboard Study-Wide Live Dashboard — Implementation Handoff

**Date:** 2026-09-03

## Outcome

`mas-cc blackboard dashboard` now accepts either an existing full episode/run directory or a standardized study result root. A study process provides a study overview, cell details, compact vote trajectories, and links into the existing detailed episode inspector when full artifacts were retained.

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

- `completed`: an explicit completed full manifest or a valid cell seal covering the episode.
- `failed`: an explicit persisted failed outcome.
- `aborted`: an explicit aborted or `skipped_aborted` outcome.
- `pending`: expected from the resolved repetition count but no retained activity exists.
- `running`: a compact trajectory advances between refreshes, or the cell's scheduler task is running.
- `unknown`: records exist but no durable terminal state or observed live signal exists, or seal validation fails.

SLURM (Simple Linux Utility for Resource Management, the cluster scheduler) data are annotations only. `squeue` or `sacct` is queried once per job with a three-second bound and a short cache. Scheduler state never marks scientific work complete or failed.

## Parameter provenance

Resolved cell configuration is flattened into the parameter table. Grid overrides replace matching values. Convenience labels include controller condition, `rho` (epistemic persistence), `b` (intervention budget), task, population, sensor size, controller target, and ground truth.

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

## Verification

Focused validation:

```text
.venv/bin/python -m pytest tests/mas_cc/test_blackboard_dashboard.py tests/mas_cc/test_blackboard_study_dashboard.py -q
```

Result: 12 passed.

`node --check src/mas_cc/blackboard_dashboard/assets/app.js` passed. `git diff --check` passed. Editor diagnostics report no errors in changed Python files.

The wider study/resume/CLI selection completed with 37 passing tests and seven unrelated failures caused by study configuration files already absent from this checkout. No dashboard-related failure occurred.

## Remaining verification

A real 270-episode study root was not available in this local checkout. Refresh-time and peak-memory measurements, live Potsdam smoke testing, and a completed copied-study smoke therefore remain to be recorded on representative data. No SLURM command or scientific artifact was changed during implementation or tests.
