# Study aggregation and analysis-package contract

This document defines how a completed MA-CC study is validated, aggregated, and packaged. It is the
operational companion to [`metrics.md`](metrics.md), which explains the scientific meaning of the
relational blackboard metrics and estimators.

The aggregation stage is offline. It reads retained observations and makes no large language model
(LLM) or provider calls.

## 1. Concepts

- A **study** is a collection of related experiment configurations.
- A **cell** is one fixed combination of scientific settings.
- An **episode** is one independent simulation inside a cell.
- A **round** is one population-level update.
- A **micro-slot** is one individual agent-update opportunity inside a round.
- **Canonical tables** are the standard normalized tables retained so analysis can be reproduced.
- A **seal** is a machine-readable statement that expected outputs passed integrity checks.

The data path is:

```text
submitted scientific configs
  -> cells and episodes
  -> retained observations and completion seals
  -> validation and canonical tables
  -> configured estimates and derived quantities
  -> plots, reports, provenance, and ZIP archive
```

## 2. Standard commands

Aggregate a complete study:

```text
mas-cc study aggregate --study-dir <study-result-root>
```

Produce an explicitly provisional package from incomplete data:

```text
mas-cc study aggregate --study-dir <study-result-root> --allow-incomplete
```

Use `--allow-incomplete` only when partial analysis is intentional. It does not make an incomplete
study complete.

Convert a historical CSV analysis package to compact Parquet without rerunning estimators:

```text
mas-cc study compact-analysis --study-dir <study-result-root>
```

**Parquet** is a compressed table format for analysis. New analysis packages use Parquet; historical
CSV packages remain readable.

Compacting one old run is a different operation:

```text
mas-cc experiment compact --run-dir <run-result-root> --profile results_only
```

`experiment compact` changes one run's retained storage. `study aggregate` validates and analyzes a
whole study.

### Potsdam

On Potsdam, run every command through the dedicated `MA-CC` Conda environment:

```text
/home/ojedamarin/.local/share/miniforge3/bin/conda run \
  --live-stream \
  -n MA-CC \
  mas-cc study aggregate \
  --study-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/<study-name>
```

Conda is the environment manager that selects the required Python interpreter and packages.
`--live-stream` keeps progress visible and does not change scientific calculations.

## 3. Required study identity

A submitted study root must contain:

```text
study_manifest.json
submission_manifest.csv
```

`study_manifest.json` identifies the study and records its analysis recipe. `submission_manifest.csv`
records expected configs, hashes, cells, episodes, and output locations.

Run artifacts are discovered at the output paths recorded in the submission manifest. A literal
`<study-root>/runs/` directory is not mandatory. Source run trees may also be absent when the study
is reaggregated from a previously retained complete set of canonical tables.

The submitted experiment YAML files remain the authority for scientific design. A study folder's
`analysis.yaml` is the **analysis recipe**: the requested estimates, resampling settings, derived
quantities, views, and plots.

## 4. Retained simulation artifacts

The exact run tree depends on the game and storage profile. Compact relational blackboard runs
normally retain:

```text
<run-root>/
  manifest.json
  resolved_base_config.yaml
  scientific_events.parquet
  grid_summary.csv
  cells/<cell-id>/
    overrides.json
    resolved_config.yaml
    scientific_events.parquet
    cell_complete.json
    cell_summary.json
    aggregate.json
    round_records/<episode-id>/
      round_trajectory.jsonl
      micro_slot_trajectory.jsonl
```

`round_trajectory.jsonl` contains one JSON record per population round. It is the main source for
population state, controller observation and action, action probability, evidence state,
communication, information estimates, and response estimates.

`micro_slot_trajectory.jsonl` contains individual within-round updates. It supports microscopic
currents, effective affinity, kinetic compliance, message exposure, and communication audits.

`cell_complete.json` records expected episode identities, row counts, schemas, and file hashes.
Aggregation validates this seal rather than assuming that an existing directory is complete.

Compact scientific storage is not a full provider audit. It can omit private prompts, model
responses, reasoning traces, request logs, and verbose runtime events. See [`metrics.md`](metrics.md)
for the exact `results_only`, `dashboard_semantic`, and `full` distinctions.

## 5. Aggregation stages

`mas-cc study aggregate` performs these stages:

1. **Discovery:** locate submitted runs and scientific cells from manifests and resolved configs.
2. **Canonical selection:** retain valid completed scientific records and remove failed-attempt or
   superseded retry rows.
3. **Canonicalization:** build normalized cells, episodes, rounds, and micro-slot tables while
   preserving source identity.
4. **Validation:** check expected counts, identities, schemas, seals, hashes, and required records.
5. **Estimation:** run each requested estimate independently inside its physical scientific cell or
   requested state group.
6. **Derivation:** calculate requested quantities from primary estimates and canonical observations.
7. **Views:** build optional state-local, occupancy, blackboard, communication, causal, endpoint, or
   theory-comparison tables.
8. **Rendering:** create configured plots and human-readable reports.
9. **Provenance:** preserve the configs, manifests, hashes, and exact analysis contract.
10. **Packaging:** create the final analysis ZIP compressed archive.

SLURM (the cluster job scheduler) task numbers and shard boundaries are execution details. They are
never scientific coordinates. Observations from cells with different scientific settings must not
be mixed into one cell estimate.

## 6. Validation behavior

Validation is written before scientific estimation:

```text
analysis/validation.json
analysis/validation.md
```

It checks:

- expected and found configurations and cells;
- expected, completed, failed, and aborted episodes;
- cell completion seals;
- duplicate run, cell, episode, round, and micro-slot identities;
- source-config and retained-artifact hashes;
- scientific schema versions;
- round and micro-slot availability;
- paired initialization when required by the design.

### Strict default

Without `--allow-incomplete`, invalid study data causes aggregation to:

1. write validation files;
2. stop before producing new final estimator tables or a ZIP;
3. return exit code 2.

### Explicit incomplete mode

With `--allow-incomplete`, aggregation:

1. preserves `valid: false` and `complete: false`;
2. analyzes the available canonical records;
3. marks every output as provisional;
4. writes tables, plots, reports, manifest, and ZIP;
5. returns exit code 1.

This option bypasses completeness failure only. It does not bypass a malformed recipe, unknown
estimator, impossible theory setting, or unsupported endpoint classifier.

## 7. Canonical tables

The observation layer is written under `analysis/tables/`:

| Table | One row represents |
| --- | --- |
| `cells.parquet` | One qualified scientific cell, including coordinates and completion state. |
| `episodes.parquet` | One episode, including identity, seed, status, counts, and execution summary. |
| `rounds.parquet` | One retained population round plus all available game-specific fields. |
| `micro_slots.parquet` | One retained individual update plus all available microscopic fields. |

A **qualified cell ID** includes its source-config identity. This prevents two local directories both
named `cell-0000` from being mistaken for the same scientific cell.

These four tables are sufficient for supported reaggregation when source run trees are unavailable.
The reader prefers Parquet and accepts legacy CSV.

## 8. Analysis tables

These estimator-layer tables are always part of the table contract, although they can be empty when
nothing applicable was requested:

| Table | Contents |
| --- | --- |
| `primary_estimates.parquet` | Primary estimates, intervals, null summaries, units, support, sample counts, and analysis identity. |
| `information_estimates.parquet` | Information and response rows from the established round-feedback estimator. |
| `support_diagnostics.parquet` | Action overlap, state support, sparsity, and sample-size checks. |
| `derived_observables.parquet` | Quantities constructed from primary estimates and retained observations. |

Optional recipe-driven tables can include state-local estimates, occupancy maps, blackboard
summaries, causal response, communication efficiency, initialization diagnostics, endpoint
classifications, and theory comparisons. Their scientific meanings are documented in
[`metrics.md`](metrics.md).

### Uncertainty and null summaries

Most estimators use whole episodes as the bootstrap unit. A **bootstrap** repeatedly samples the
observed units with replacement to measure sampling uncertainty. Causal-response analysis instead
resamples complete shared-initialization blocks so matched episodes stay together.

A **null model** deliberately breaks the tested relationship while preserving relevant structure.
The estimator stores compact null summaries, not every generated draw.

Typical estimate fields include:

```text
estimate, ci_low, ci_high, confidence
null_type, null_mean, null_std, p_value
bootstrap_resamples, null_permutations
n_observations, n_episodes, units, support_status
analysis_hash
```

Individual bootstrap and null draws are temporary and are not retained.

## 9. Plots, reports, provenance, and manifest

A completed analysis normally contains:

```text
analysis/
  validation.json
  validation.md
  analysis_manifest.json
  analysis_recipe.yaml
  tables/
  plots/
  reports/
  provenance/
  <study-id>_analysis.zip
```

Plots are views of source tables, not independent evidence. Unsupported estimator rows are masked,
and missing scientific cells are not interpolated.

`analysis_manifest.json` is the machine-readable package index. It records scientific and analysis
hashes, requested statistics, resampling settings, theory provenance, generated tables and plots,
and the retention contract.

`provenance/` preserves the submitted design and source identity. It can contain study and
submission manifests, submission metadata, source config copies, recovery metadata, and lineage
records.

The ZIP includes validation, manifest, recipe, tables, plots, reports, and provenance. It excludes
source run trees, provider logs, caches, temporary files, and individual resampling draws.

## 10. Reaggregation

When all four canonical observation tables and validation metadata remain available, rerunning:

```text
mas-cc study aggregate --study-dir <study-result-root>
```

can recalculate estimates, derived quantities, plots, reports, and the ZIP without source run trees
or provider calls. Bootstrap and null calculations run again because no persistent estimator cache
is part of the current contract.

Use `study compact-analysis` only for format migration without estimator recomputation.

## 11. What aggregation does not do

Study aggregation does not:

- launch episodes or call a provider;
- change game dynamics, seeds, or scientific coordinates;
- repair missing source measurements;
- invent support for an unavailable estimate;
- mix heterogeneous cells into one conditional mutual information estimate;
- preserve complete private prompt or response history;
- retain individual bootstrap or null draws;
- turn an incomplete study into a complete one.

## 12. Review order and acceptance checklist

Review a package in this order:

1. `validation.md` for completeness and errors.
2. `analysis_manifest.json` for formats, settings, hashes, and package contents.
3. `analysis_recipe.yaml` for intended estimates and plots.
4. `tables/cells.parquet` and `tables/episodes.parquet` for scientific coordinates and completion.
5. `tables/support_diagnostics.parquet` before interpreting estimates.
6. `tables/primary_estimates.parquet` and `tables/derived_observables.parquet` for results.
7. `tables/rounds.parquet` and `tables/micro_slots.parquet` when independently reconstructing a result.
8. `reports/methods.md` for exact semantics and units.
9. `plots/` only after checking their source rows.
10. `provenance/` to recover the submitted design.

Before accepting a final handoff, verify:

```text
[ ] validation says valid and complete
[ ] found configs, cells, and completed episodes equal expected counts
[ ] failed and aborted episode counts are zero
[ ] identities are not duplicated
[ ] config and retained-artifact hashes pass
[ ] canonical Parquet tables are readable
[ ] estimator rows include units and support status
[ ] bootstrap and null counts match analysis.yaml
[ ] unsupported estimates remain explicitly unsupported
[ ] requested plots have source-table rows
[ ] analysis_manifest.json lists emitted tables and plots
[ ] provenance contains submitted configs and manifests
[ ] the final ZIP opens and contains the documented handoff
```

## 13. Implementation map

| Source file | Responsibility |
| --- | --- |
| `src/mas_cc/cli/main.py` | Command parsing, dispatch, and exit codes. |
| `src/mas_cc/studies/discovery.py` | Run and cell discovery. |
| `src/mas_cc/studies/canonical.py` | Canonical cells, episodes, rounds, and micro-slots. |
| `src/mas_cc/studies/validation.py` | Completeness and integrity checks. |
| `src/mas_cc/studies/table_io.py` | Parquet and legacy CSV table handling. |
| `src/mas_cc/studies/aggregation.py` | Estimation orchestration, views, plots, reports, provenance, and packaging. |
| `src/mas_cc/studies/episode_endpoints.py` | Optional endpoint classifications. |
| `src/mas_cc/analysis/causal_response.py` | Causal response and communication outputs. |
| `src/mas_cc/analysis/single_affinity.py` | Single-affinity definitions and derived quantities. |

For scientific definitions, units, metric names, estimator meanings, and blackboard-specific
interpretation rules, use [`metrics.md`](metrics.md).
