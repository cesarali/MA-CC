# Results-only Per-cell and Final Information Analysis

**Date:** 2026-08-12  
**Status:** implementation plan  
**Scope:** HiddenBench imitation grids using `storage.artifact_profile: results_only`,
cell-completion orchestration, configured CMI/controller diagnostics, local retention,
resume, and master-only Comet publication.

## Goal

Make every completed results-only grid cell independently analysis-ready immediately,
without waiting for the remaining cells, while retaining the existing final analysis over
the complete grid.

The required lifecycle is:

1. finish and durably seal one cell's episode shards;
2. run the configured HiddenBench imitation analysis for that cell;
3. retain a compact human- and machine-readable cell report;
4. publish that cell's report through the existing master Comet writer when enabled;
5. continue running other cells without making a derived-analysis failure fail or corrupt
   completed episode data;
6. after the whole grid finishes, run a separate grid-wide analysis containing every
   completed cell and all cross-cell comparisons.

This feature changes when configured analysis runs and where its outputs live. It must not
change any estimator, conditioning variable, bootstrap/null procedure, episode seed,
provider call, or scientific event retained by `results_only`.

---

## 1. Current behavior and gap

The orchestrator already has a cell-completion boundary:

- `_CellCompletion.record(cell_id)` identifies the last episode in a cell;
- `_ResultsOnlyFinalizer.seal(cell_id)` merges validated episode shards into the cell's
  sealed `scientific_events.parquet`;
- `GridAggregator.aggregate(cell_id)` computes and publishes trajectory aggregates;
- the master process remains the only Comet writer.

Configured HiddenBench imitation information analysis does **not** use that boundary.
`run_configured_analysis(base, grid_dir)` is called only after all grid tasks and cell
summaries finish. Consequently, a completed first cell has correct compact scientific data
but no local CMI/entropy report until the entire grid ends.

This conflicts with the operational requirement that a long run killed or delayed after
one completed cell still exposes the complete scientific analysis for that cell.

---

## 2. Required output layout

Per-cell and final reports must have separate, stable paths:

```text
<grid-run>/
├── cells/
│   └── cell-0000/
│       ├── scientific_events.parquet
│       ├── cell_complete.json
│       └── hidden_bench_imitation_analysis/
│           ├── information_estimates.md
│           ├── information_estimates.csv
│           ├── support_diagnostics.csv
│           ├── analysis_summary.json
│           ├── plots/
│           └── analysis_complete.json
└── hidden_bench_imitation_analysis/
    ├── information_estimates.md
    ├── information_estimates.csv
    ├── support_diagnostics.csv
    ├── analysis_summary.json
    ├── plots/
    └── analysis_complete.json
```

The cell directory contains estimates for exactly that cell. The root directory contains
the final multi-cell report. Neither pass may overwrite or ambiguously reuse the other.

For `results_only`, retain the same compact final-analysis policy used today: the Markdown
report, information table, support diagnostics, plots, and summary remain; bulky derived
intermediates are removed only after the retained outputs are valid. If a configured
diagnostic is requested but currently removed by results-only cleanup, either retain its
compact table or incorporate all its values into the Markdown/summary before deletion.
The configured CMI and controller-entropy values must never exist only transiently.

---

## 3. Analysis semantics

### 3.1 Per-cell analysis

Use the completed cell directory as the analysis input. The existing event reader already
supports a cell-local sealed `scientific_events.parquet`.

Use the exact resolved cell config, not merely the grid base config, for:

- configured statistic and diagnostic names;
- bootstrap resamples;
- null permutations;
- confidence level;
- analysis seed;
- artifact profile;
- resolved-config identity recorded in the analysis summary;
- Comet tags/parameters describing task and control mechanism.

Every requested statistic must be available, including:

- sensing MI and population/target/focal actuation CMI;
- order-parameter sensing and actuation CMI;
- controller action entropy, both unconditional and conditioned on population or each
  order parameter;
- actuation information fractions;
- signed actuation diagnostics;
- action-overlap/support diagnostics;
- confidence intervals and configured permutation-null results.

`n_episodes` and `n_events` in the report must reflect only the cell being analyzed.

### 3.2 Final grid analysis

After all cells finish, retain the current root-level analysis over the entire grid. This
pass must recompute from sealed cell tables rather than concatenate already summarized
cell estimates. Bootstrap samples, null permutations, support diagnostics, and future
cross-cell statistics require event-level inputs.

The final report must include one row/block per scientific cell and preserve the grid
coordinates needed to compare tasks and controller mechanisms.

### 3.3 Partial or failed cells

Run automatic per-cell analysis only when the cell reaches its existing completion
boundary and its compact table is sealed successfully. Do not label an incomplete cell
report as final.

If some episodes failed but the orchestrator considers the cell finished, analyze the
successfully persisted episodes and record all inclusion counts and failures prominently in
`analysis_summary.json` and the Markdown preamble. If no episode completed, emit a small
failed-analysis marker with the reason and do not create numerical estimate tables.

---

## 4. Orchestrator design

Introduce one narrow cell-analysis coordinator rather than embedding analysis policy in
`_run_episode_task`.

Suggested interface:

```python
class CellAnalysisCoordinator:
    async def analyze_completed_cell(
        self,
        *,
        cell_id: str,
        cell_config: RunConfig,
        cell_dir: Path,
        cell_result: GridCellResult | None,
    ) -> CellAnalysisResult: ...
```

Cell-completion order must be:

```text
last episode finishes
  -> seal compact scientific table
  -> validate cell completion seal/table
  -> run trajectory aggregation
  -> run configured per-cell information analysis off the event loop
  -> validate retained analysis outputs
  -> atomically write analysis_complete.json
  -> add analysis artifacts to the cell completion hashes
  -> notify the master Comet sink
  -> release/continue normal grid work
```

Analysis is CPU/file work and must execute via `asyncio.to_thread` (or an equivalent bounded
analysis executor), never on the event loop that schedules provider requests. Only one
analysis task should run at a time by default to prevent bootstrap/permutation work from
competing with active episodes for all CPUs. Provider calls must continue while analysis
runs unless explicit resource measurements prove this harmful.

The episode semaphore must be released before cell analysis begins. Cell analysis must not
hold an episode-parallelism slot.

Derived-analysis exceptions must be caught and recorded. They may mark the analysis failed,
but must not change completed episode outcomes or delete the sealed scientific table. The
final grid pass must retry any cell whose earlier derived analysis failed.

---

## 5. Idempotency and resume

Add an atomic `analysis_complete.json` seal containing at least:

- schema version;
- analysis scope (`cell` or `grid`);
- cell ID for cell scope;
- resolved-config hash;
- scientific table hash(es);
- requested statistics and diagnostics;
- bootstrap/null/confidence/seed settings;
- analysis implementation/version identifier;
- start and finish timestamps;
- retained artifact paths, byte sizes, and SHA-256 hashes;
- status and failure summary.

On resume:

1. validate the scientific completion seal;
2. validate `analysis_complete.json` against config, scientific hashes, and output hashes;
3. skip analysis only when all identities and artifacts match;
4. otherwise recompute analysis without rerunning provider episodes.

A crash during analysis may leave temporary files, but no valid completion seal. The next
run replaces the derived directory transactionally and retries.

When a previously analyzed cell is discovered during `_prime_resumed_outcomes`, ensure its
analysis is present/valid and publish it to the new master Comet experiment if necessary.
Do not assume that prior Comet publication survived merely because local analysis did.

---

## 6. Comet behavior

Master-only logging remains mandatory.

At cell completion:

- upload the cell `information_estimates.md`, compact CSV tables, and plots;
- log scalar estimates, intervals, support flags, entropy ceilings, and overlap measures
  with names prefixed by `cell-000N_` when `cell_reporting: master`;
- with per-cell experiments enabled, use the cell experiment and unprefixed statistic names;
- attach task/control coordinates and `analysis_scope=cell`;
- never create episode-level Comet writers.

At grid completion:

- upload the distinct root-level report with `analysis_scope=grid`;
- do not overwrite same-named cell assets accidentally;
- log cross-cell/comparison outputs only from the final grid pass.

Comet failures remain non-fatal. The local analysis and completion seal are authoritative.

---

## 7. Configuration

The first implementation should make per-cell analysis the default whenever all of the
following are true:

```text
storage.artifact_profile == results_only
analysis.enabled == true
grid has more than one cell
```

Do not require users to duplicate the estimator list in a second section.

If an explicit policy knob is needed for resource control, use one analysis scheduling
field rather than another estimator selection:

```yaml
analysis:
  enabled: true
  scheduling:
    per_cell: true
    final_grid: true
    max_concurrent: 1
```

Defaults for `results_only` should be `per_cell: true`, `final_grid: true`, and
`max_concurrent: 1`. Preserve current behavior for `full` unless explicitly enabled, to
avoid an unexpected increase in derived artifacts for legacy runs.

The implementation must decide whether this requires a schema-version bump. Unknown
scheduling fields must not be silently ignored.

---

## 8. Retention and hashing

Today `_record_cell_hashes` may run immediately after aggregation. Move or repeat that seal
update after per-cell analysis succeeds so the cell completion metadata covers the retained
analysis artifacts.

Required rules:

- never delete `scientific_events.parquet` after analysis;
- never remove per-cell reports during final grid cleanup;
- never let the root final pass write beneath `cells/cell-XXXX/`;
- exclude temporary analysis files from hashes;
- ensure a result-only transfer includes each completed cell's analysis without `.resume`;
- allow the root manifest to hash both cell reports and the final report;
- document whether `controller_diagnostics.csv` is retained. Since the requested entropy,
  overlap, signed-actuation, and information-fraction values are scientific results, the
  preferred decision is to retain a compact `controller_diagnostics.csv` in results-only.

---

## 9. Testing strategy

### 9.1 Unit tests

1. A cell-local compact table produces the same information and diagnostic values as the
   equivalent rich event input.
2. The per-cell runner uses the resolved cell config and configured estimator subset.
3. `analysis_complete.json` rejects changed scientific data, config, estimator settings,
   seed, or corrupted outputs.
4. Results-only cleanup retains all requested numerical outputs.
5. Empty and partially failed cells produce explicit, non-misleading summaries.

### 9.2 Orchestrator integration tests

1. In a two-cell mock grid, cell 0 analysis exists before cell 1 finishes.
2. Provider episode work continues while cell 0 analysis executes off the event loop.
3. A per-cell analysis exception leaves episodes/cell table intact and the grid continues.
4. The final root analysis runs after all cells and contains both cells.
5. Cell and root reports use distinct paths and neither overwrites the other.
6. Resume skips a valid cell analysis and recomputes a missing/corrupt one without provider
   calls.
7. `full` and `results_only` produce numerically identical configured estimates.
8. Master-only Comet receives per-cell analysis once and final-grid analysis once.

### 9.3 Operational acceptance test

Run a provider-free two-cell HiddenBench imitation fixture with several episodes per cell:

- pause execution after cell 0 completes;
- verify its Markdown, CSV, support diagnostics, plots, and seal are readable;
- verify the root final report does not yet claim completion;
- resume and complete cell 1;
- verify the final report contains both cells and matches an offline recomputation;
- delete only derived analysis and resume again;
- verify analysis is restored without any new provider request.

---

## 10. Implementation sequence

1. Add analysis scheduling config/model/schema and resolved-config coverage if a policy knob
   is accepted.
2. Extract a reusable configured-analysis argument builder that accepts the resolved cell
   config and explicit destination/scope.
3. Implement atomic analysis completion seals and validation.
4. Add the bounded cell-analysis coordinator.
5. Wire it after results-only cell sealing/aggregation and before final cell hash sealing.
6. Teach resume priming to validate/recompute cell analysis.
7. Separate and label cell versus final-grid Comet publication.
8. Retain the compact controller diagnostic table required to recover every configured
   entropy/overlap/signed statistic.
9. Add unit, integration, resume, failure-isolation, and Comet-parity tests.
10. Update the architecture overview, config reference, results-only TDD plan, and runbook.

---

## 11. Definition of done

The feature is complete when a user can inspect a finished first cell during an ongoing
results-only grid and obtain every configured CMI, entropy, information-fraction, signed
actuation, overlap, confidence, null, and support value from durable local artifacts; when
the same artifacts can be republished after a Comet outage; and when the final root report
still recomputes the complete multi-cell analysis without rerunning or modifying episodes.
