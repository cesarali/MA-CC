# Results-only experiment data, metrics, and analysis

This document explains how a configured experiment is organized on disk when
`storage.artifact_profile: results_only` is used. The concrete example is
[`configs/runs/relational_reasoning/population_study_04`](../../configs/runs/relational_reasoning/population_study_04/README.md),
but the run/cell/episode hierarchy and compact scientific table are shared by
all experiment grids that use this profile.

The most important distinction is:

```text
game state -> runtime metrics -> cell aggregation
          \-> scientific records -> configured estimators -> analysis tables/reports
```

Runtime **metrics**, cell **aggregation**, and post-run **analysis estimators**
are separate layers. They have separate config sections, consume different
views of the data, and produce different files.

## 1. From a YAML config to a run directory

Study 04 consists of three launchable YAML files:

```text
configs/runs/relational_reasoning/population_study_04/
  relational_population_study04_qc06.yaml
  relational_population_study04_qc12.yaml
  relational_population_study04_qc18.yaml
```

Each file fixes one sensing sample size, `q_c`, and defines a six-cell grid:

```yaml
grid:
  control.options.intervention_budget: [6, 12, 18]
  game.options.task_id: [task_0001, task_0002]
```

The order of the axes and values determines the stable `cell-NNNN` numbers.
Every cell gets the base config plus its two overrides, and every cell runs
`execution.repetitions: 30` episodes. A run therefore has this identity
hierarchy:

```text
one YAML/config process
  -> one run directory
    -> six grid cells
      -> thirty episode identities per cell
        -> round and micro-slot observations
```

By default the qc12 config lands under:

```text
results/
  relational_imitation_round_feedback/
    relational-study04-qc12-resource-grid/
      relational-study04-qc12-resource-grid-20260821/
```

The final component is the run ID, formed from `experiment.name` and
`execution.seed`. Passing `experiment run --output-dir ...` replaces the
`storage.output_dir` root, but the game/experiment/run-ID components are still
created beneath it.

Do not manually infer a cell's scientific condition from its number. Read
`cells/cell-NNNN/overrides.json`, which is the authoritative map from a cell ID
to the swept values.

## 2. The completed results-only tree

A successful Study 04 row has approximately this shape. Optional files are
marked explicitly.

```text
<run-dir>/
  manifest.json
  resolved_base_config.yaml
  budget_state.json
  grid_summary.csv
  grid_progress.png
  comet_summary.json
  timing_study.md
  scientific_events.parquet

  cells/
    cell-0000/
      overrides.json
      resolved_config.yaml
      cell_summary.json
      aggregate.json
      cell_complete.json
      prompt_examples.md                 # when prompt sample count > 0
      scientific_events.parquet
      metrics/
        plots/
          <configured-cell-metric>.png
      round_records/
        cell-0000-0000/
          round_trajectory.jsonl
          micro_slot_trajectory.jsonl
        ...
      reports/                           # only with per_cell_reports: true
        <compact per-cell reports>

  relational_imitation_round_feedback_analysis/
    analysis_summary.json
    round_information_estimates.csv
    round_information_estimates.md
    round_information_nulls.csv
    round_support_diagnostics.csv
    controller_action_summary.csv
    episode_epistemic_regime.csv
    round_epistemic_trajectory.csv
    theory_comparison.csv
    theory_state_curves.csv
    currents/
      episode_currents.csv
      cell_current_summary.csv
      current_analysis.md                # or one task/cell subfolder per report
```

The Study 04 configs set `analysis.options.per_cell_reports: false`, so their
`cells/*/reports/` directories are normally absent. Per-cell estimates still
exist as rows in the run-level analysis CSVs; the option controls only whether
an additional compact human-readable report is rendered as each cell closes.

### Run-level files

| File | Purpose |
| --- | --- |
| `resolved_base_config.yaml` | Fully resolved base config used by the run. This is preferable to reconstructing settings from the source YAML after it may have changed. |
| `manifest.json` | Run identity and, after successful results-only finalization, hashes and sizes of retained artifacts. |
| `budget_state.json` | Durable request/token/cost accounting used across resume. |
| `grid_summary.csv` | One summary row per grid cell, including completion/failure counts and overrides. |
| `grid_progress.png` | Local grid progress/status visualization. |
| `comet_summary.json` | Status/reference for optional master-level Comet reporting. Comet is a monitoring view, not the authoritative store. |
| `timing_study.md` | Compact run/cell timing summary. The separate per-episode and per-request timing CSVs belong to the `timing_study` profile, not `results_only`. |
| `scientific_events.parquet` | Run-wide merge of every sealed cell's compact transition table. Convenient for cross-cell loading. |

### Cell-level files

| File | Purpose |
| --- | --- |
| `overrides.json` | The exact grid-axis values that define this cell. |
| `resolved_config.yaml` | Complete config after applying the cell overrides. |
| `scientific_events.parquet` | All completed episodes in the cell, ordered by episode and interaction. |
| `cell_complete.json` | Completion seal: episode IDs, row counts, schema version, Parquet hash, and hashes of retained cell artifacts. |
| `cell_summary.json` | Counts of completed, resumed, failed, and aborted episodes; compact failure details if present. |
| `aggregate.json` | Generic across-episode metric curves, bands, scalars, counts, aggregation policy, and excluded-episode list. |
| `metrics/plots/*.png` | Plots made from the configured `aggregation.cell_metrics`. |
| `prompt_examples.md` | A bounded deterministic sample for prompt inspection; not a complete prompt/response history. |

## 3. What `results_only` retains

`results_only` changes local retention, not the game dynamics, provider calls,
or requested scientific computation. For the same resolved config, `full` and
`results_only` are intended to make equivalent provider calls and yield the
same metrics and estimates.

For each completed episode, the writer first publishes an atomic Parquet shard
under:

```text
cells/<cell-id>/.resume/<episode-id>/scientific_events.parquet
```

That shard is the episode checkpoint. On resume, its config, prompt-definition,
pricing, game, cell, episode, seed, and scientific-schema identities are
validated before the episode is skipped. Resume is at **episode granularity**:
an episode interrupted in flight starts again from round zero with the same
seed.

When every expected episode in a cell is complete, the shards are validated
and merged into:

```text
cells/<cell-id>/scientific_events.parquet
cells/<cell-id>/cell_complete.json
```

Only after the cell is sealed is its `.resume/` directory removed. The
run-level `scientific_events.parquet` is then assembled from sealed cell
tables. If a cell is incomplete, successful episode shards remain under
`.resume/` and the cell is not falsely marked complete.

The compact Parquet has a versioned schema. Conceptually, one row is one
interaction transition and contains:

- identity/provenance: run, cell, episode, seed, interaction index, config and
  prompt hashes, pricing hash, game/dynamics/control/task identities;
- scientific state: answer alphabet and target, population count vectors
  before/after, sensed state, controller action, truth/order/control variables,
  and focal state before/after;
- small controller/behavior diagnostics;
- terminal status, timestamps, interaction count, and token/request totals;
- JSON-encoded runtime metric values needed to reconstruct an `EpisodeFrame`
  for aggregation.

The relational game additionally retains two deliberately scientific JSONL
channels per episode:

- `round_trajectory.jsonl` is one record per population round. It is the
  authoritative input for the Study 04 sensing, actuation, epistemic-memory,
  support, response, and matched-theory analysis.
- `micro_slot_trajectory.jsonl` records within-round focal update slots. It is
  used for microscopic/current consistency checks and analyses that require a
  finer clock than one population round.

These round records carry Study 04's scientific coordinates directly, such as
`sensor_sample_size`, `sensing_fraction`, `intervention_budget`,
`actuation_fraction`, `controller_beta`, and `controller_threshold`.

### What is intentionally not retained

Unlike `full`, `results_only` does not keep the verbose per-episode recorder
tree after successful cell sealing. In particular, do not expect complete
copies of:

- `events.jsonl`, `trajectory.jsonl`, and `experiment.log`;
- per-request status, usage, budget, prompt-block, and audit JSONL files;
- per-episode `metrics/streaming.csv` and `metrics/final.csv`;
- every prompt, message, response, or reasoning trace;
- legacy checkpoint manifests and rich analysis intermediates.

Use `artifact_profile: full` for prompt/provider debugging or forensic work
that needs those artifacts. A bounded `prompt_examples.md` under
`results_only` is an inspection sample, not an audit log.

## 4. Metrics, aggregation, and estimators

### Runtime metrics: `metrics:`

Runtime metrics observe game state while an episode is running. For the
relational round-feedback game, the registered metrics live in
`src/mas_cc/games/relational_reasoning/imitation_round_feedback/metrics.py`.
They include vote observables such as `m_truth`, `m_ctrl`, `m_order`, vote
entropy, action shares, and epistemic observables such as supporting-fact
coverage and full-proof-agent share.

Study 04 uses:

```yaml
metrics:
  enabled: true
  comet_export: []
```

The values are computed locally. Under `results_only`, their per-round values
are embedded in `scientific_events.parquet` rather than retained as standalone
per-episode CSVs. `comet_export: []` means none of these episode metric names is
selected for direct Comet export.

### Cell aggregation: `aggregation:`

Aggregation operates after episodes have written their metrics. It combines
many episode series within one fixed grid cell, applying the recorded
forward-fill, percentile, rolling-window, and relabeling policy.

Study 04 requests:

```yaml
aggregation:
  forward_fill: absorbing
  percentiles: [10, 50, 90]
  rolling_window: 1
  cell_metrics: [m_ctrl, m_truth, m_order, dominant_action_share]
  sweep_metrics: []
```

The numerical result is `cells/<cell-id>/aggregate.json`; plots go under that
cell's `metrics/plots/`. Because the compact Parquet retains the metric payload,
these aggregates can be recomputed without model calls:

```bash
python -m mas_cc.cli.main experiment aggregate --run-dir <run-dir>
```

Passing `--config <config.yaml>` intentionally substitutes only that config's
`aggregation:` section, for example to change percentile bands or the fill
policy. It does not rerun episodes.

### Post-run estimators: `analysis:`

Analysis reads the scientific round records across episodes. It does not
observe live game state and does not make provider calls. In Study 04 the
explicit `analysis.estimators` list is the measurement plan. Preflight checks
every name against the supported round-analysis vocabulary before the run is
allowed to spend.

The configured names fall into these families:

| Family | Examples | Question |
| --- | --- | --- |
| Sensing | `round_sensing_mi`, sensor MAE/MSE | How informative is the controller's sample about the population? |
| Controller entropy | action entropy, entropy conditioned on population | Is there enough action variation for actuation information to be estimable? |
| Actuation information | population/target/truth/order CMI | How much information does controller action put into the next state, conditional on the current state? |
| Support/sparsity | dual-action fractions, state count, singleton fraction | Does the data support the requested conditional estimate? |
| Signed response | target/truth/order signed actuation | In which direction did the controller push? CMI alone is unsigned. |
| Memory-aware conditioning | memory, epistemic, phi, susceptible, kappa CMI/response | Does the result persist after conditioning on richer epistemic state? |

MI and CMI use the direct-counting implementation in
`src/mas_cc/analysis/estimators.py`. Each estimate exposes unsmoothed,
Jeffreys-smoothed, and Miller-Madow variants; the round-feedback pipeline's
reported main variant is currently `unsmoothed`. Confidence intervals are
episode-level bootstrap intervals. Actuation statistics use a
policy-conditional randomization null; sensing MI uses a sensor-permutation
null. Study 04 asks for 1,000 bootstrap resamples, 1,000 null permutations,
95% confidence, and four bins per axis for the joint epistemic diagnostic.

Always read information estimates beside their support diagnostics. A finite
CMI value does not by itself show that enough conditioning states saw both
actions. In particular, a controller with no action entropy cannot identify an
action-effect information channel.

## 5. Reading the analysis output

For Study 04, configured analysis runs automatically after the run-level
scientific table is assembled. Its main files are:

| File | How to use it |
| --- | --- |
| `round_information_estimates.csv` | Primary machine-readable table. Rows are keyed by `cell_id` and statistic, plus a `pooled` slice. Includes estimator variants, bootstrap interval, null summary, units, and statistic-matched support fields. |
| `round_information_estimates.md` | Human-readable version of the estimates, with the matched classical theory section appended. |
| `round_information_nulls.csv` | Individual permutation/randomization-null draws. Use for null-distribution inspection, not just the stored mean. |
| `round_support_diagnostics.csv` | Cell-wide support overview: conditioning coverage, dual-action support, and sparsity. Read before interpreting CMI. |
| `controller_action_summary.csv` | Advocate/no-op counts and frequencies, mean policy probability, and controlled-position summaries. |
| `episode_epistemic_regime.csv` | One row per episode with epistemic-regime and controller-action summaries. |
| `round_epistemic_trajectory.csv` | Tidy round-level analysis table with controller action, population changes, epistemic state/bins, and resource coordinates. |
| `theory_comparison.csv` | One matched finite-`N` q-voter comparison per cell, including empirical residuals and ratios. |
| `theory_state_curves.csv` | State-resolved empirical/theory curves underlying the comparison. |
| `currents/episode_currents.csv` | Per-episode integrated-current quantities and consistency checks. |
| `currents/cell_current_summary.csv` | Bootstrap summaries of empirical and theory currents by task/cell. |
| `analysis_summary.json` | Compact audit/index: data counts, estimator settings, memory-support warnings, theory applicability, current-analysis results, and optional Comet status. |

Each Study 04 row contains cells with three distinct actuation budgets. Its
pooled empirical estimates are valid descriptive calculations, but one pooled
classical reference is not: `analysis_summary.json` and
`theory_comparison.csv` mark the pooled theory row not applicable because it
spans multiple controller parameter tuples. Use the **per-cell** theory rows.

The three qc06/qc12/qc18 YAMLs produce three separate run directories. There
is no automatic parent directory that merges all three processes. For the full
3 x 3 sensing/actuation surface, concatenate the per-cell rows from the three
runs and retain the source run ID and cell ID as provenance. The coordinates
are available in the round trajectory and are re-emitted as `theory_qc`,
`theory_sensing_fraction`, `theory_b`, and `theory_c` in
`theory_comparison.csv`.

## 6. Running, resuming, and re-analyzing

Run these commands from the repository root. Preflight validates the config,
estimator names, budget, and call estimates without making model calls:

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment preflight \
  --config configs/runs/relational_reasoning/population_study_04/relational_population_study04_qc12.yaml \
  --output-dir results/inspection/relational_study04_qc12_preflight
```

Launch the row:

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment run \
  --config configs/runs/relational_reasoning/population_study_04/relational_population_study04_qc12.yaml
```

The experiment runner invokes configured analysis automatically. To rerun the
relational analysis explicitly, offline and without model calls:

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main analysis relational-round-feedback \
  --run-dir results/relational_imitation_round_feedback/relational-study04-qc12-resource-grid/relational-study04-qc12-resource-grid-20260821
```

The explicit CLI uses its own argument defaults unless they are supplied. In
particular, its default seed is `1`, whereas configured Study 04 analysis uses
the execution seed. The exact explicit rerun is:

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main analysis relational-round-feedback \
  --run-dir results/relational_imitation_round_feedback/relational-study04-qc12-resource-grid/relational-study04-qc12-resource-grid-20260821 \
  --bootstrap-resamples 1000 --null-permutations 1000 \
  --confidence 0.95 --seed 20260821 --epistemic-bins 4
```

Normal `experiment run` behavior resumes by episode. Point the same unchanged
design at the same output root to continue it. If the scientific config,
prompt definition, pricing identity, or schema has changed, validation rejects
the old checkpoint rather than mixing incompatible data. Use a new experiment
name/output root for a changed design. `storage.wipe_and_recompute: true` is a
destructive clean recomputation and should be used deliberately.

## 7. Source map

These are the principal implementation locations when the format or analysis
needs to be audited or extended:

| Concern | Source |
| --- | --- |
| Config data models and retention policy | `src/mas_cc/config/models.py` |
| Config parsing and results-only validation/defaults | `src/mas_cc/config/loader.py` |
| Run/cell/episode orchestration, resume, sealing, and final analysis | `src/mas_cc/experiments/orchestrator.py` |
| Configured estimator validation and dispatch | `src/mas_cc/experiments/configured_analysis.py` |
| Runtime recorder and round-record writers | `src/mas_cc/observability/recorder.py` |
| Compact Parquet schema, validation, merging, and read adapters | `src/mas_cc/storage/scientific.py` |
| Legacy full-run compaction | `src/mas_cc/storage/compaction.py` |
| Generic cell aggregation | `src/mas_cc/experiments/aggregation.py`, `src/mas_cc/metrics/aggregate.py`, `src/mas_cc/metrics/cell.py` |
| Relational runtime metrics | `src/mas_cc/games/relational_reasoning/imitation_round_feedback/metrics.py` |
| Shared direct-counting MI/CMI estimators | `src/mas_cc/analysis/estimators.py` |
| Shared round-feedback estimator vocabulary and resampling | `src/mas_cc/games/hidden_bench/imitation_round_feedback/analysis.py` |
| Relational record adapter, report tables, and theory comparison | `src/mas_cc/games/relational_reasoning/imitation_round_feedback/analysis.py` |
| Relational current analysis | `src/mas_cc/games/relational_reasoning/imitation_round_feedback/current.py` |

For execution parallelism and SLURM layouts, see
[`slurm_jobs_cells_parallelism_and_aggregation.md`](slurm_jobs_cells_parallelism_and_aggregation.md).
For every config field, see
[`config_reference.md`](../howto/launch/config_reference.md).
