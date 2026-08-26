# Standardized Study Submission and Aggregation — Implementation Handoff

Implementation snapshot: 2026-08-22.

## 2026-08-26 incremental-analysis correction

Sealed cell workers now prepare deterministic atomic per-cell resampling
fragments. Post-hoc aggregation reuses them, processes missing cell groups in
parallel according to `SLURM_CPUS_PER_TASK`, persists every completed group,
and reports progress in `analysis/progress.json`. Aggregation preserves each
physical cell coordinate and does not create a heterogeneous study-wide pool.

## 2026-08-22 resource-planning correction

The original config-array-only implementation underused Study 06 and inherited
unsafe cluster defaults (one CPU, 4 GB, and a 24-hour limit for an estimated
35-hour task). The study workflow now supports `execution.mode: auto`, which
expands grids into deterministic original-cell shards, writes
`execution_manifest.csv` and `execution_plan.json`, derives an RPM-bounded
array throttle, and passes explicit CPU, memory, and time requests to the
generic `run_study_cell_array.job` launcher. Each cell shard retains its
original grid index, cell ID, overrides, and episode seeds. Canonical study
aggregation continues to reconstruct cells and pool raw observations; scheduler
IDs never enter scientific coordinates.

### Potsdam output-root correction

Production study data belongs under
`/work/ojedamarin/Projects/LanguageGames/MA-CC/results`, not the home source
checkout. Study policies can declare `execution.results_root` and
`execution.require_results_under`; submission rejects an override outside the
required root. Both generic launchers have a `/work/.../logs` direct-submit
fallback, while the normal submitter overrides this with absolute
`<study-result-root>/logs/slurm-%A_%a.{out,err}` paths.

This handoff covers the implementation of
`docs/tdd/features/orchestrator/22082026_TDD_standardized_study_submission_and_aggregation.md`.
The code is the source of truth; this document explains the operational path,
important invariants, extension points, verification performed, and remaining
limitations for the next maintainer.

## 1. User-facing outcome

The normal study workflow is now:

```bash
mas-cc study submit --config-dir <CONFIG_FOLDER>
mas-cc study aggregate --study-dir <RESULTS_FOLDER>
```

Submission performs all experiment preflights before calling SLURM. Aggregation
discovers the resulting normal run trees, validates them, creates canonical
study tables, invokes or reuses the established scientific estimators,
computes requested derived observables, renders configured plots and reports,
and writes one ZIP package.

Useful options are:

```bash
mas-cc study submit \
  --config-dir <CONFIG_FOLDER> \
  --results-dir <RESULTS_FOLDER> \
  --throttle <MAX_SIMULTANEOUS_ARRAY_TASKS>

mas-cc study aggregate \
  --study-dir <RESULTS_FOLDER> \
  --allow-incomplete
```

`--allow-incomplete` is intentionally explicit. It permits exploratory output
after validation failures, but `validation.json`, `analysis_manifest.json`,
the CLI exit status, and the reports continue to mark the package incomplete.

## 2. Main implementation map

| File | Responsibility |
|---|---|
| `src/mas_cc/cli/main.py` | Adds the `study submit` and `study aggregate` CLI surfaces and error/status reporting |
| `src/mas_cc/studies/manifest.py` | Reads `study.yaml`, discovers configs, validates stable ordering, duplicates, paths, execution mode, and the analysis recipe |
| `src/mas_cc/studies/submission.py` | Resolves configs, computes deterministic identities/counts/output roots, preflights every config, writes manifests, submits one `sbatch` command |
| `src/mas_cc/studies/array_worker.py` | Maps `SLURM_ARRAY_TASK_ID` to exactly one manifest row and calls the existing experiment CLI in the same Python process |
| `src/mas_cc/studies/discovery.py` | Locates ordinary and sharded run/cell trees using manifests and resolved configs rather than directory ordering |
| `src/mas_cc/studies/canonical.py` | Builds study-wide cells, episodes, rounds, and micro-slot tables with qualified provenance |
| `src/mas_cc/studies/validation.py` | Validates expected/found counts, identities, schemas, seals, row consistency, source hashes, and retained artifact hashes |
| `src/mas_cc/studies/aggregation.py` | Orchestrates validation, table writing, estimator execution/reuse, caching, derived observables, plotting, reports, provenance, and ZIP packaging |
| `src/mas_cc/analysis/effective_affinity.py` | Reusable form of the established Study-05 effective-affinity and kinetic-compliance analysis |
| `scripts/Potsdam/SLURM/run_config_array.job` | Generic config-array SLURM launcher; no study-specific job file is required |
| `tests/mas_cc/test_studies.py` | Focused manifest, submission, array mapping, aggregation, cache, incomplete-study, and affinity contracts |

The package export is `src/mas_cc/studies/__init__.py`:

```python
from mas_cc.studies import aggregate_study, discover_study, submit_study
```

## 3. Configuration-folder contract

A folder may contain:

```text
study_folder/
    study.yaml
    config_a.yaml
    config_b.yaml
    analysis.yaml
```

Supported `study.yaml` shape:

```yaml
study:
  name: example_study

configs:
  - config_a.yaml
  - config_b.yaml

execution:
  mode: config_array
  throttle: 4

analysis:
  recipe: analysis.yaml
```

Important behavior:

- Listed config order is preserved exactly.
- Without `study.yaml`, `*.yaml` and `*.yml` experiment configs are sorted by
  filename.
- `study.yaml` and `analysis.yaml` are never treated as experiment configs.
- Duplicate configs, missing configs, malformed YAML, paths escaping the
  config folder, and unsupported execution modes fail before submission.
- `study.yaml` never duplicates game/controller/scientific parameters. Normal
  experiment YAMLs remain authoritative.
- An unlisted `analysis.yaml` is used automatically when present.

## 4. Submission behavior and identities

`submit_study()` performs these steps:

1. Discover and validate the study folder.
2. Resolve every config through `load_run_config_or_grid()`.
3. Compute each config's expected cell and episode counts.
4. Derive stable output roots under `<study-dir>/runs/<config-stem>`.
5. Run `run_experiment_preflight()` for every config in a temporary directory.
6. Abort without publishing a new study root when any launch status is not
   `permitted`.
7. Copy successful preflight artifacts into `<study-dir>/preflight/`.
8. Write `study_manifest.json` and deterministic `submission_manifest.csv`.
9. Call `sbatch` exactly once with one array index per config.
10. Parse the returned SLURM job ID and write `submission.json`.

The CSV columns are:

```text
array_index
config_path
config_hash
resolved_config_hash
output_dir
expected_cell_count
expected_episode_count
execution_seed
git_commit
```

`config_hash` is the SHA-256 of the submitted YAML bytes.
`resolved_config_hash` binds the resolved scientific configuration and grid
mapping. For grids, expected episodes are the sum of repetitions over resolved
cells, so a swept `execution.repetitions` value remains correctly represented.

Rerunning submission with unchanged inputs produces the same config-to-output
mapping. The existing experiment runner therefore sees the same output tree
and applies its established episode resume/checkpoint validation. The study
layer introduces no checkpoint format.

## 5. SLURM execution path

The submitted command is equivalent to:

```bash
sbatch --array=0-N%THROTTLE \
  scripts/Potsdam/SLURM/run_config_array.job \
  <study-dir>/submission_manifest.csv
```

The job executes:

```bash
python -m mas_cc.studies.array_worker <submission_manifest>
```

The worker reads `SLURM_ARRAY_TASK_ID`, validates it against the contiguous
zero-based array mapping, and dispatches the equivalent of:

```bash
mas-cc experiment run \
  --config <selected-config> \
  --output-dir <selected-output-root> \
  --no-progress
```

The worker calls the existing CLI in-process. One array task therefore remains
one Python worker executing one ordinary MA-CC run. Grid cells and repetitions
continue to use the current orchestrator and `execution.parallelism`.

The generic job intentionally contains no study-specific resources. Cluster
partitions that require modules, Conda activation, GPUs, memory, or special
wall time should supply those through the surrounding scheduler environment
or a genuinely topology-specific generic job variant—not a scientific
study-number job.

## 6. Result discovery and canonical identity

Aggregation starts from `study_manifest.json` and
`submission_manifest.csv`. Each expected output root is searched for ordinary
MA-CC run manifests with a sibling `resolved_config.yaml` or
`resolved_base_config.yaml`.

This supports:

- normal single-config runs;
- normal complete-grid runs;
- config-array studies;
- disjoint cell-sharded trees beneath one expected config output root;
- explicitly permitted partial studies.

Scientific identity never comes from a SLURM job ID, array index alone,
directory ordering, or a convenient directory label. It is recovered from:

- the submission entry;
- the run manifest;
- the resolved base/cell config;
- `overrides.json`;
- compact scientific identities where available.

Execution-local cell labels are qualified study-wide:

```text
config-0000/cell-0000
config-0001/cell-0000
```

The original label remains in `source_cell_id`. This prevents ordinary
`cell-0000` and episode-label reuse across configs from becoming false
scientific duplicates.

Grid coordinates are taken from `overrides.json`. Dotted coordinate paths are
retained, and an unambiguous leaf-name alias is also emitted for declarative
plotting. Common non-grid coordinates such as task ID, population size,
sensor sample size, and intervention budget are recovered from resolved
configs when present.

## 7. Validation contract

Aggregation always writes these before scientific analysis:

```text
analysis/validation.json
analysis/validation.md
```

Validation currently checks and reports:

- expected versus found config indices;
- expected versus found cells;
- sealed and scientifically incomplete cells;
- expected, completed, failed, and aborted episodes;
- duplicate run, cell, and episode identities;
- scientific schema versions;
- source-config hash changes after submission;
- compact cell seals and row-count consistency via
  `validate_cell_artifact()`;
- compact table hashes and retained artifact hashes recorded by cells/runs;
- missing scientific, round, and micro-slot records;
- canonical round and micro-slot row counts.

Compact episode `resolved_config_hash` values are not compared literally with
the cell-resolved config hash. The runner deliberately derives an
episode-specific seed and hashes that episode-resolved config, so these hashes
represent different levels of the scientific hierarchy. Compact artifact
validation checks their internal consistency instead.

Strict aggregation raises after writing validation when errors exist.
`--allow-incomplete` continues while preserving `valid: false` and
`complete: false`.

## 8. Canonical tables

Successful or explicitly partial aggregation writes:

```text
analysis/tables/
    cells.parquet
    episodes.parquet
    rounds.parquet
    micro_slots.parquet
    primary_estimates.parquet
    information_estimates.parquet
    support_diagnostics.parquet
    derived_observables.parquet
```

Parquet is authoritative; redundant table CSV mirrors are not written.

### Cells

One row per qualified scientific cell, including study/run provenance,
submitted and resolved config hashes, swept coordinates, expected/completed/
failed episode counts, and seal status.

### Episodes

One row per episode. Compact runs are grouped from
`scientific_events.parquet`; full-profile runs use retained episode manifests.
Seeds, status, interaction counts, timestamps, termination reason, usage, and
schema provenance are retained when available.

### Rounds

Rich `round_trajectory.jsonl` records are preferred. When a supported
round-feedback run retains only compact scientific transitions, those rows are
adapted through `compact_row_to_imitation_event()` and the existing round
adapter. Heterogeneous nested values are JSON-serialized only at the Parquet
write boundary; estimator inputs retain their native structures in memory.

### Micro-slots

`micro_slot_trajectory.jsonl` records are merged with qualified cell/run
provenance and normalized round/slot indices. These remain the input to
effective-affinity and kinetic-compliance estimation.

## 9. Scientific estimator reuse

The study layer does not contain a second MI/CMI implementation.

When an analysis recipe requests round information statistics, aggregation
loads trajectory observations through the established relational or
HiddenBench adapters, qualifies local identities, merges observations, and
calls:

```python
mas_cc.games.hidden_bench.imitation_round_feedback.analysis.round_information_analysis
```

This preserves:

- direct counting;
- the existing primary estimator variant;
- unsmoothed, Jeffreys, and Miller–Madow values;
- whole-episode bootstrap;
- policy-conditional action randomization;
- sensing permutation nulls;
- memory-conditioned variants;
- existing support diagnostics.

Per-cell groups are estimated from observations. Heterogeneous scientific cells
are not combined into a study-wide estimator group.

When no new estimator recipe is supplied, compatible existing
`round_information_estimates.csv`, `round_information_nulls.csv`, and
`round_support_diagnostics.csv` files are ingested where available.

Recipe aliases added for the plan's vocabulary are:

```text
round_target_actuation_cmi_memory
    -> round_memory_target_actuation_cmi

round_target_actuation_cmi_memory_phi
    -> round_phi_target_actuation_cmi

target_signed_actuation
    -> round_target_signed_actuation
```

The `memory_phi` compatibility name is not phi-only in the estimator state.
`round_phi_target_actuation_cmi` uses the shared augmented-conditioning path,
whose conditioning key is `(target_count_before, conditioning_phi_bin)`.
In other words, it is target history plus phi. A focused test calls the actual
`ROUND_CONDITIONING_STATE` entry and asserts both parts of that tuple.

Episode and cell current requests call the existing relational current engine.

## 10. Effective affinity and kinetic compliance

`src/mas_cc/analysis/effective_affinity.py` refactors the scientific core of
the archived Study-05 `estimate_effective_affinity.py` script into a reusable
module. It retains the established definitions over controlled microscopic
slots in ADVOCATE rounds:

```text
p_plus  = P(non-target -> target | controlled)
p_minus = P(target -> non-target | controlled)

effective_affinity = log(p_plus / p_minus)
kinetic_compliance = p_plus + p_minus
```

Point estimates remain raw transition frequencies. Sparse bootstrap draws use
Jeffreys-stabilized probabilities, and resampling is by whole episode. Pooled
bootstrap draws preserve cell allocation through stratification.

These metrics are emitted only when requested and when eligible micro-slot
records exist. Their exposure/transition counts and probabilities are added to
`support_diagnostics.parquet`.

## 11. Long-format outputs and support

`primary_estimates.parquet` is the union of information estimates and requested
auxiliary primary estimators. Its stable leading columns include:

```text
study_id, source_run_id, cell_id
metric, estimator_version, estimator_variant
grouping_json, conditioning_json
estimate, ci_low, ci_high, confidence
null_type, null_mean, null_std, p_value
n_observations, n_episodes
units, support_status, analysis_hash
```

Null procedures still execute, but only `null_type`, `null_mean`, `null_std`,
`p_value`, and `null_permutations` are retained. Bootstrap output likewise
retains confidence limits and `bootstrap_resamples`, not individual draws.

Information support rows preserve the current estimator diagnostics and add
standardized fields such as action-0/action-1 counts, action entropy,
dual-action support, occupied conditioning states, singleton fraction, and a
sparse-state fraction. Plotting masks `support_status == unsupported`.

## 12. Reaggregation and transient computation

The information `analysis_hash` includes:

- submitted config hashes;
- hashes of scientific, round, micro-slot, and seal artifacts;
- canonical cell and episode identities/counts;
- estimator name/version;
- requested statistics;
- bootstrap/null/confidence/seed settings.

No persistent estimator cache is retained. An unchanged second aggregation
recomputes from `cells.parquet`, `episodes.parquet`, `rounds.parquet`, and
`micro_slots.parquet`. In-memory or invocation-local temporary computation is
allowed, but successful aggregation removes it. Analysis hashes and calculation
settings remain recorded in `analysis_manifest.json`. If the original run trees
are no longer present, `study aggregate` uses these retained canonical tables
and preserves the prior scientific input identity.

## 13. Derived observables and plots

The first derived observable is `eta_ir`. It is constructed downstream from
the existing target CMI, the matching signed response, and empirical action
frequency; no information quantity is re-estimated in the derived layer.
The response dependency is specifically `round_target_signed_actuation`, not
the marginal `round_target_signed_response_share`. CMI and response rows are
joined one-to-one on `study_id`, `source_run_id`, `cell_id`, `grouping_json`,
and `conditioning_json`. A different state or grouping therefore produces no
derived row, and duplicate dependencies at one resolution fail explicitly.
The derived output carries the matched grouping/conditioning and records the
join keys in `dependencies_json`.

Plots accept either named built-ins:

```yaml
plots:
  - target_cmi_x_b
  - eta_ir_x_b
  - memory_conditioning
  - h_eff_phi_b
  - gamma_eff_phi_b
```

or declarative definitions:

```yaml
plots:
  target_cmi_x_b:
    source: primary_estimates
    metric: round_target_actuation_cmi
    x: intervention_budget
    y: target_fraction_bin
    facet: sensor_sample_size
    kind: heatmap
```

When requested coordinates exist, the plotter creates the configured heatmap.
For a named metric without both requested axes, it emits a clearly labeled
per-cell bar view rather than silently omitting available estimates. Requested
metrics with no data produce no misleading empty scientific plot.

## 14. Reports, provenance, and ZIP package

The final analysis tree includes:

```text
analysis/
    validation.json
    validation.md
    analysis_manifest.json
    analysis_recipe.yaml              # when configured
    tables/
    plots/
    reports/
        summary.md
        methods.md
    provenance/
        study_manifest.json
        submission_manifest.csv
        submission.json               # when available
        config-0000-<source-name>.yaml # when source remains available
    <study-name>_analysis.zip
```

ZIP entries are sorted, use fixed timestamps and permissions, and exclude the
ZIP itself and Windows `:Zone.Identifier` sidecars. The archive contains the
machine-readable tables, reports, plots, validation, estimator identity,
recipe, and submission/config provenance needed for handoff.

The standardized study analysis package contains canonical scientific data,
compact estimator summaries, support diagnostics, plots, reports, validation,
and provenance. Bootstrap/permutation draws and analysis caches are transient
computational intermediates and are not retained. Source run trees, SLURM logs,
checkpoint/resume files, and provider/request logs are not packaged.

## 15. Tests and verification performed

Focused tests cover:

- implicit and explicit manifest discovery;
- stable order and duplicate rejection;
- deterministic submission entries and counts;
- array-index resolution and out-of-range failure;
- preflight of every config before one `sbatch` call;
- real compact mock-provider run normalization;
- all nine canonical/analysis Parquet products;
- cache reuse on unchanged aggregation;
- strict incomplete-study failure and explicit partial continuation;
- effective-affinity and kinetic-compliance definitions.
- `memory_phi` as `(target history, phi bin)` conditioning;
- exact-resolution dependency joins for `eta_ir`, including rejection of a
  marginal-response mismatch;
- a one-episode mock execution of the actual Study 06 main base config under
  `results_only`, asserting the required round and micro-slot fields on files
  written by the recorder.

The final affected-path regression command was:

```bash
conda run -n MA-CC --no-capture-output python -m pytest -q \
  tests/mas_cc/test_studies.py \
  tests/mas_cc/test_cli_and_inspection.py \
  tests/mas_cc/test_import_safety.py \
  tests/mas_cc/test_results_only_resume.py \
  tests/mas_cc/test_configured_analysis.py \
  tests/mas_cc/test_relational_round_feedback_analysis.py \
  tests/mas_cc/test_relational_current_analysis.py
```

The latest gate-focused regression command was:

```bash
/home/cesarali/miniconda3/envs/MA-CC/bin/python -m pytest -q \
  tests/mas_cc/test_studies.py \
  tests/mas_cc/test_relational_round_feedback_analysis.py \
  tests/mas_cc/test_results_only_resume.py \
  tests/mas_cc/test_relational_current_analysis.py \
  tests/mas_cc/test_relational_matched_theory.py
```

Result: **109 passed**.

### 2026-08-22 four-gate follow-up

Status at handoff:

| Gate | Status | Evidence |
|---|---|---|
| `memory_phi` means history+ | Pass | Alias resolves to the phi statistic; actual shared conditioning returns `(target_before, phi_bin)` |
| `eta_ir` dependency resolution | Pass after narrow fix | Exact five-key one-to-one join; state-matched actuation only; focused mismatch test |
| Study 06 round/micro retention | Pass | Actual Study 06 base, downscaled to one mock episode with `results_only`; both JSONL schemas and all requested fields asserted |
| Tiny real Potsdam SLURM array | Blocked before submission | Login host is reachable, but every configured local/Windows identity is rejected with `Permission denied (publickey)` |

No SLURM job was submitted and no remote files were changed. Attempts used
`ojedamarin@login1.hpc.uni-potsdam.de`, which is the account/host recorded by
the repository's existing Potsdam jobs and shell history. Both local SSH keys,
the Windows SSH key, WSL OpenSSH, and Windows OpenSSH were tried without an
accepted identity. The local environment has no SSH agent available.

Once key-based access is restored, the remaining gate is deliberately small:
generate a standalone resolved copy of
`configs/runs/old/toy_game_smoke_test.yaml` in a temporary one-config study
folder on the shared filesystem, run `mas-cc study submit` with throttle 1,
confirm that `sbatch` reports an array job (`0-0%1`), wait for `sacct` to report
`COMPLETED`, and verify the toy run seal plus `submission.json`. Do not submit
the Study 06 folder for this gate: that would launch the real study rather than
a tiny scheduler smoke.

A repository-wide `pytest -q` run was also attempted. It is not currently
green for reasons outside this change: numerous legacy tests reference absent
files such as `configs/runs/hidden_bench_grid.yaml`,
`configs/runs/hidden_bench_vanilla.yaml`, and
`configs/runs/hidden_bench/hidden_bench_imitation_classical.yaml`; one older
game-registry assertion also expects a registry predating the currently
registered games. The changed study, CLI, import, resume, configured-analysis,
round-information, and current-analysis paths pass.

## 16. Known limitations and recommended next work

1. **Scheduler environment is site-dependent.**
   `run_config_array.job` assumes `python` resolves to an environment containing
   the installed repository. Add a generic Potsdam environment/bootstrap block
   once the cluster's desired module/Conda policy is fixed.

2. **No `study status` or selective failed-index retry yet.**
   Resubmit the same study mapping; existing episode resume makes this safe.
   A later `sacct`-based convenience layer should not change scientific IDs.

3. **Plotting is intentionally small.**
   It supports declarative heatmaps and per-cell fallback bars. More plot kinds
   should remain downstream views over canonical tables and support status.

4. **`eta_ir` is the first derived registry case.**
   If more derived observables are added, extract dependency registration into
   a dedicated registry rather than growing conditionals in aggregation.

5. **Mixed rich/compact studies are supported, but compact relational rows
   cannot recover fields that were never retained.**
   The generic existing round adapter can recover the established transition
   channels from compact scientific data. Memory/epistemic estimators still
   require those fields to have been retained in rich round records or an
   extended compact schema.

6. **Source paths in submission manifests are absolute.**
   This is appropriate for a shared cluster filesystem and deterministic
   resubmission. The final ZIP copies available source YAMLs into provenance,
   so downstream analysis does not depend on those paths remaining mounted.

7. **Do not average execution-shard information estimates.**
   Any future shard ingestion must continue to merge observations and call the
   established estimator on the complete scientific grouping.

## 17. Safe extension recipes

### Add a derived observable

1. Read dependencies from `primary_estimates` or canonical tables.
2. Add the observable in the derived layer.
3. Include its specification/version in the derived hash.
4. Add it to `analysis.yaml`.
5. Rerun `mas-cc study aggregate`; information caches should remain reusable.

### Add an existing primary estimator

1. Import and call its authoritative implementation.
2. Adapt its result into `PRIMARY_COLUMNS`.
3. Preserve detailed draws/support in their dedicated tables.
4. Include scientific input and estimator settings in its hash.
5. Add an equivalence test against the original analysis entry point.

### Add another scheduler topology

1. Keep scientific config and cell identities unchanged.
2. Add a generic launcher only when allocation topology genuinely differs.
3. Treat scheduler task IDs as execution provenance only.
4. Verify canonical tables and pooled estimator results are invariant to the
   new execution partition.

## 18. Critical invariants to preserve

- Experiment configs and resolved cell overrides are scientific truth.
- SLURM task numbers and directory names are not scientific coordinates.
- Submission preflights every config before the one scheduler mutation.
- Study resubmission reuses existing output roots and checkpoint semantics.
- Canonical tables retain source provenance and qualified identities.
- MI/CMI/bootstrap/null/support come from the established estimator engine.
- Pooled estimates operate on pooled observations, never averaged shard CMI.
- Conditional information estimates travel with support diagnostics.
- Derived observables do not trigger a replacement information estimator.
- Strict validation precedes final analysis unless the user explicitly chooses
  visibly incomplete exploratory output.
- The ZIP is a downstream reproducible handoff artifact, not scientific truth.
