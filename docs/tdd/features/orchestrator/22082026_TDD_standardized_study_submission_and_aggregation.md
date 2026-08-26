# TDD: Standardized Study Submission and Aggregation for MA-CC

**Date:** 2026-08-22  
**Purpose:** implementation specification for a study-level workflow that standardizes experiment submission, SLURM parallelization, cross-run aggregation, existing information-theoretic analysis, derived observables, plotting, and packaging.

---

## 1. Objective

The repository already contains the main scientific and execution machinery:

- YAML experiment configs with `grid:` expansion.
- Stable run / cell / episode identities.
- Episode-level resume and validation.
- `results_only` compact scientific storage.
- Per-cell and run-level `scientific_events.parquet`.
- Relational `round_trajectory.jsonl` and `micro_slot_trajectory.jsonl`.
- Generic cell aggregation.
- Existing configured MI / CMI estimators.
- Episode-level bootstrap confidence intervals.
- Action-randomization and sensor-permutation nulls.
- Support diagnostics.
- Current analysis and other study-specific estimators.
- Existing complete-grid and SLURM-array execution patterns.

The problem is organizational, not that these components are absent.

The target user-facing workflow is deliberately minimal:

```bash
mas_cc study submit --config-dir <CONFIG_FOLDER>
mas_cc study aggregate --study-dir <RESULTS_FOLDER>
```

These should be the normal two commands for a study.

Everything else—manifest creation, preflight, SLURM array mapping, run discovery, merging, validation, existing CMI analysis, derived metrics, plots, reports, and ZIP packaging—should happen internally.

---

## 2. Fundamental architecture

Keep these concepts separate:

```text
scientific design
    ↓
execution topology
    ↓
canonical scientific dataset
    ↓
scientific estimators
    ↓
derived observables
    ↓
plots / reports
```

A SLURM task is an execution shard, not a scientific unit.

The scientific hierarchy remains:

```text
study
  → experiment config / run
    → grid cell
      → episode
        → round
          → micro-slot
```

The authoritative identity of a scientific cell is the resolved config plus its `overrides.json`, never its directory name or SLURM task number.

---

## 3. Study directory

A study is a folder containing several normal experiment YAML configs.

Example:

```text
configs/runs/relational_reasoning/population_study_06/
    study.yaml
    relational_population_study06_qc06.yaml
    relational_population_study06_qc12.yaml
    relational_population_study06_qc18.yaml
    analysis.yaml
```

The individual experiment YAMLs remain ordinary existing MA-CC configs and remain scientifically authoritative.

`study.yaml` is only a lightweight orchestration manifest.

Suggested form:

```yaml
study:
  name: relational_population_study06

configs:
  - relational_population_study06_qc06.yaml
  - relational_population_study06_qc12.yaml
  - relational_population_study06_qc18.yaml

execution:
  mode: config_array

analysis:
  recipe: analysis.yaml
```

Do not duplicate controller/game parameters into `study.yaml`.

---

## 4. `mas_cc study submit`

Example:

```bash
mas_cc study submit \
  --config-dir configs/runs/relational_reasoning/population_study_06
```

Optional explicit result root:

```bash
mas_cc study submit \
  --config-dir configs/runs/relational_reasoning/population_study_06 \
  --results-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/relational_population_study_06
```

On Potsdam, production results must resolve beneath
`/work/ojedamarin/Projects/LanguageGames/MA-CC/results`. The study execution
policy records that root and may reject CLI destinations outside it. Submission
must create `<study-root>/logs` and pass absolute `sbatch --output` and
`--error` patterns there. Relative SLURM log paths in the source repository are
forbidden.

### Required behavior

The command must:

1. Read `study.yaml` if present.
2. Discover or validate the listed experiment configs.
3. Run the existing experiment preflight for every config.
4. Abort before SLURM submission if any config fails preflight.
5. Create one study-level result root.
6. Create a deterministic submission manifest with:
   - array index,
   - config path,
   - config hash,
   - expected output root,
   - expected cell count,
   - expected episode count,
   - execution seed,
   - git commit when available.
7. Submit one SLURM config array.
8. Record the returned job ID and submission metadata.
9. Exit.

The command itself must not become the experiment worker. It prepares the mapping and submits once; SLURM performs the parallel execution.

---

## 5. SLURM execution model

### 5.1 Compatibility mode: one array task per YAML config

For:

```text
qc06.yaml
qc12.yaml
qc18.yaml
```

create:

```csv
array_index,config_path,output_dir
0,.../qc06.yaml,.../runs/qc06
1,.../qc12.yaml,.../runs/qc12
2,.../qc18.yaml,.../runs/qc18
```

Then submit one array:

```bash
sbatch --array=0-2%<THROTTLE> \
  scripts/Potsdam/SLURM/run_config_array.job \
  <submission_manifest>
```

Each array task reads:

```bash
SLURM_ARRAY_TASK_ID
```

and maps that index to exactly one config/output pair.

Conceptually:

```bash
INDEX="$SLURM_ARRAY_TASK_ID"
CONFIG=<manifest row INDEX config_path>
OUTPUT=<manifest row INDEX output_dir>

python -m mas_cc.cli.main experiment run \
  --config "$CONFIG" \
  --output-dir "$OUTPUT" \
  --no-progress
```

Thus:

```text
one array task
  → one Python process
    → one existing MA-CC experiment run
      → that config's ordinary grid cells
        → episodes under existing execution.parallelism
```

This mode is a compatibility fallback, not the preferred topology when YAML
boundaries underuse available cluster or provider capacity.

### 5.2 Preferred mode: planned scientific-cell shards

The scientific unit is the cell and the computational unit is the episode.
Submission must expand original grid cells, preserve their indices and episode
seed derivation, and create cell or cell-bundle execution shards independently
of YAML-file boundaries. Each process runs one episode per configured async
episode slot. Aggregation reconstructs complete cells before cell-level
estimation and pools canonical observations before pooled estimation.

The planner consumes cell/episode demand, request concurrency, live provider
limits, planning latency, node limits, memory, CPUs, and time limits. It writes
`execution_plan.json` and `execution_manifest.csv`, passes explicit resources
to one generic SLURM launcher, and refuses a throttle above the declared RPM
bound. Execution partitioning must not affect identities, observables, or phase
diagrams.

### 5.3 Preserve existing in-process parallelism

If one YAML has:

```yaml
execution:
  repetitions: 30
  parallelism: 8

grid:
  control.options.intervention_budget: [6, 12, 18]
  game.options.task_id: [task_0001, task_0002]
```

the config still creates six scientific cells inside one normal MA-CC process.

The study layer does not rewrite the experiment orchestrator.

### 5.4 Episode sharding and cell reconstruction

Retain a generic cell-array launcher only for configurations that genuinely cannot fit in one allocation.

If used, preserve:
- original `cell-NNNN`,
- original overrides,
- original seed derivation,
- original scientific identity.

Cell sharding must remain an execution detail. Episode-level distribution may
be added when workers write disjoint episode artifacts and cell sealing occurs
only after every expected episode is reconstructed. Until that contract is
implemented, use one original cell per shard and run its episodes concurrently.

---

## 6. Generic SLURM infrastructure

Target stable infrastructure:

```text
scripts/Potsdam/SLURM/
    run_grid.job
    run_config_array.job
    run_cell_array.job
```

`run_config_array.job` should be the normal study launcher.

A new scientific study must not require:

```text
study06.job
study07.job
beta_test.job
thermo_test.job
```

A new `.job` should only be needed if scheduler topology or resources are genuinely different.

---

## 7. Study result layout

Every config folder maps to one study result root.

Example:

```text
results/relational_population_study06/
    study_manifest.json
    submission_manifest.csv
    submission.json

    runs/
        qc06/
            <existing normal run tree>
        qc12/
            <existing normal run tree>
        qc18/
            <existing normal run tree>

    analysis/
        ...
```

Convenience directory labels are not scientific truth.

Always recover scientific coordinates from existing resolved configs and overrides.

---

## 8. Resume behavior

Do not rewrite checkpointing or resume.

The existing runner already validates and resumes completed episodes.

The study layer should simply resubmit unchanged configs to the same output roots.

Initial implementation may safely resubmit the whole config array because completed episodes will be skipped by the existing runner.

A later convenience feature may inspect `sacct` and resubmit only failed/incomplete array indices, but this is not required for the first implementation.

---

## 9. `mas_cc study aggregate`

Example:

```bash
mas_cc study aggregate \
  --study-dir results/relational_population_study06
```

This should be the only normal post-run command.

Internally it performs:

```text
discover
→ validate
→ normalize
→ reuse/run estimators
→ compute derived observables
→ render plots/reports
→ package
```

The user must not need to separately call shard roll-up, CMI scripts, current scripts, plot scripts, and ZIP scripts in the standard path.

---

## 10. Discovery

Prefer `study_manifest.json` as the source of expected runs.

Support:
1. normal complete-grid runs;
2. config-array studies;
3. cell-sharded studies;
4. explicitly allowed partial studies.

For every run collect:
- source run ID,
- source path,
- resolved config,
- config hash,
- cells,
- completion status,
- scientific schema version.

Do not rely on SLURM job IDs or directory ordering for scientific identity.

---

## 11. Validation before analysis

Write:

```text
analysis/validation.json
analysis/validation.md
```

Validate at least:

```text
expected configs / found configs
expected cells / found cells
sealed cells / incomplete cells
expected episodes / completed / failed / aborted
duplicate run identities
duplicate cell identities
duplicate episode identities
schema versions
config hashes
missing scientific_events.parquet
missing round records
missing micro-slot records
row-count consistency
cell_complete.json consistency
artifact hashes when available
```

Example:

```text
Expected configs:        3
Found configs:           3

Expected cells:         18
Sealed cells:           18
Incomplete cells:        0

Expected episodes:     540
Completed episodes:    540
Failed episodes:         0

Duplicate episode IDs:   0
Scientific schemas:      1
Config mismatches:       0
```

Default final aggregation should be strict.

Allow exploratory partial aggregation only through an explicit flag:

```bash
--allow-incomplete
```

Incomplete outputs must remain visibly marked as incomplete.

---

## 12. Canonical study tables

Produce:

```text
analysis/tables/
    cells.parquet
    episodes.parquet
    rounds.parquet
    micro_slots.parquet

    primary_estimates.parquet
    information_estimates.parquet
    information_nulls.parquet
    support_diagnostics.parquet
    derived_observables.parquet
```

CSV mirrors may be emitted for small tables; Parquet should be authoritative for large tables.

### 12.1 `cells.parquet`

One row per scientific cell.

Include:

```text
study_id
source_run_id
source_run_path
cell_id
config_hash
resolved_config_hash
task_id
all swept scientific coordinates
expected_episodes
completed_episodes
failed_episodes
sealed
```

Do not hard-code only current Study 04 coordinates.

### 12.2 `episodes.parquet`

One row per episode.

Include provenance, seed, status, counts, requests/tokens/cost when available, plus useful episode summaries.

### 12.3 `rounds.parquet`

One row per episode × population round.

This becomes the canonical study-wide input for:
- sensing MI,
- action CMI,
- signed response,
- memory-conditioned estimators,
- epistemic diagnostics,
- state-local curves,
- future round-scale observables.

Reuse the existing relational round-record adapter.

### 12.4 `micro_slots.parquet`

One row per micro update slot.

This becomes the canonical input for:
- `p_+`,
- `p_-`,
- effective affinity,
- kinetic compliance,
- microscopic current checks,
- future micro-scale estimators.

Reuse the existing micro-slot adapter/schema.

---

## 13. Existing MI / CMI machinery is authoritative

This is a hard requirement.

The repository already contains:
- direct-counting MI / CMI;
- unsmoothed estimates;
- Jeffreys smoothing;
- Miller–Madow;
- episode bootstrap;
- policy/action randomization nulls;
- sensing permutation nulls;
- support diagnostics;
- memory-conditioned variants;
- estimator preflight/validation.

Do not implement a second CMI engine in the study layer.

### Reuse policy

#### Case A — existing output is valid and complete

If a run already contains compatible:

```text
round_information_estimates.csv
round_information_nulls.csv
round_support_diagnostics.csv
```

ingest/reuse them.

#### Case B — a merged or new analysis is required

Call the existing estimator engine on the canonical merged scientific data.

Examples:
- pooled cross-run descriptive CMI;
- new memory conditioning;
- changed bins;
- changed bootstrap/null settings.

Never estimate pooled CMI by averaging arbitrary shard CMIs.

In general:

```text
mean(CMI from execution shards) != CMI from pooled scientific observations
```

A per-cell CMI may only be reused if it was computed from the complete scientific cell.

---

## 14. Primary estimators versus derived observables

### Primary estimators

Operate directly on trajectory data.

Examples:

```text
sensing_mi
target_cmi
population_cmi
truth_cmi
order_cmi
target_cmi_memory
target_cmi_memory_phi
target_cmi_kappa
signed_target_response
effective_affinity
kinetic_compliance
episode_current
cell_current
```

Where an implementation already exists, wrap or call it; do not copy it.

### Derived observables

Consume primary estimates or canonical tables.

Examples:

```text
eta_IR
theory residuals
empirical/theory ratios
control yield
future thermodynamic efficiencies
future phase-diagram observables
```

Conceptually:

```text
rounds
  ↓
existing CMI estimator
  ↓
T_pi + CI + null + support
  ↓
signed response
  ↓
chi
  ↓
eta_IR
```

A new efficiency must not trigger a new CMI implementation.

---

## 15. Long-format estimator schema

Suggested `primary_estimates.parquet`:

```text
study_id
source_run_id
cell_id

metric
estimator_version
estimator_variant

grouping_json
conditioning_json

estimate
ci_low
ci_high
confidence

null_type
null_mean
null_std
p_value

n_observations
n_episodes

units
support_status
analysis_hash
```

Detailed null draws live in `information_nulls.parquet`.

Detailed support statistics live in `support_diagnostics.parquet`.

No scientifically important quantity should exist only in Markdown.

Reports are views of machine-readable estimator results.

---

## 16. Support diagnostics are first-class

Every conditional information estimate must travel with support diagnostics.

Preserve/compute at least:

```text
n observations
n episodes
action-0 count
action-1 count
action entropy
dual-action support fraction
occupied conditioning states
singleton fraction
sparse-state fraction
```

For richer conditioning, preserve condition-specific coverage.

Plots should be able to mask or flag unsupported regions automatically.

---

## 17. `analysis.yaml`

A study may contain a recipe such as:

```yaml
version: 1

estimators:
  - round_sensing_mi
  - round_target_actuation_cmi
  - round_target_actuation_cmi_memory
  - round_target_actuation_cmi_memory_phi
  - target_signed_actuation
  - effective_affinity
  - kinetic_compliance

resampling:
  bootstrap_resamples: 1000
  null_permutations: 1000
  confidence: 0.95
  seed: 20260821

derived:
  - eta_ir

plots:
  - target_cmi_x_b
  - eta_ir_x_b
  - memory_conditioning
  - h_eff_phi_b
  - gamma_eff_phi_b
```

Reuse the repository's current estimator names wherever possible.

---

## 18. Caching

Repeated aggregation should be cheap.

Define an `analysis_hash` from at least:

```text
scientific input identity
estimator name/version
grouping
conditioning
binning
estimator variant
bootstrap settings
null settings
analysis seed
```

Behavior:

```text
same data + same estimator specification
    → reuse cached result

new derived observable only
    → reuse CMI/nulls/support/response
    → compute only derived observable

changed conditioning/bootstrap/etc.
    → recompute only affected estimator products
```

Do not key caching only by filenames.

---

## 19. Plot recipes

Plots should be downstream declarative views.

Example:

```yaml
plots:

  eta_ir_x_b:
    source: derived_observables
    metric: eta_ir
    x: intervention_budget
    y: target_fraction_bin
    facet: sensor_sample_size
    kind: heatmap

  target_cmi_x_b:
    source: primary_estimates
    metric: round_target_actuation_cmi
    x: intervention_budget
    y: target_fraction_bin
    facet: sensor_sample_size
    kind: heatmap

  h_eff_phi_b:
    source: primary_estimates
    metric: effective_affinity
    x: intervention_budget
    y: phi_bin
    facet: sensor_sample_size
    kind: heatmap
```

Generic plotting code should work from scientific coordinate names rather than Study-04-specific assumptions.

Study-specific plots remain allowed when scientifically necessary.

---

## 20. Final package

A successful aggregate should produce:

```text
analysis/
    validation.json
    validation.md
    analysis_manifest.json

    tables/
        cells.parquet
        episodes.parquet
        rounds.parquet
        micro_slots.parquet
        primary_estimates.parquet
        information_estimates.parquet
        information_nulls.parquet
        support_diagnostics.parquet
        derived_observables.parquet

    plots/
        <configured plots>

    reports/
        summary.md
        methods.md

    <study-name>_analysis.zip
```

The ZIP is the standard handoff artifact.

It should be sufficient for another analyst to:
- understand the study;
- verify completeness;
- trace a plotted point to its source;
- inspect CMI/null/bootstrap/support;
- add another derived observable without reopening arbitrary run folders.

---

## 21. Suggested code locations

Adapt to repository conventions rather than forcing exact paths.

Possible modules:

```text
src/mas_cc/studies/
    models.py
    manifest.py
    submission.py
    discovery.py
    validation.py
    canonical.py
    aggregation.py
    cache.py
    plotting.py
    packaging.py
```

Reuse existing components from:

```text
src/mas_cc/config/models.py
src/mas_cc/config/loader.py
src/mas_cc/experiments/orchestrator.py
src/mas_cc/experiments/aggregation.py
src/mas_cc/experiments/configured_analysis.py
src/mas_cc/storage/scientific.py
src/mas_cc/analysis/estimators.py
src/mas_cc/games/hidden_bench/imitation_round_feedback/analysis.py
src/mas_cc/games/relational_reasoning/imitation_round_feedback/analysis.py
src/mas_cc/games/relational_reasoning/imitation_round_feedback/current.py
```

Do not duplicate scientific implementations simply to fit the new namespace.

---

# 22. Test-driven development plan

Implementation should proceed test-first.

## 22.1 Study manifest discovery

Given:

```text
study/
  study.yaml
  a.yaml
  b.yaml
  c.yaml
```

expect:
- exactly three experiment configs;
- stable order;
- normalized paths;
- no accidental inclusion of `study.yaml` or `analysis.yaml`.

Failures:
- duplicate config;
- missing listed config;
- invalid YAML.

---

## 22.2 Submission manifest

Given three configs, expect:

```text
array_index = 0,1,2
```

and:
- distinct output roots;
- deterministic config hashes;
- expected cell counts;
- expected episode counts.

Regenerating unchanged input must reproduce the same scientific mapping.

---

## 22.3 SLURM array mapping

Mock:

```text
SLURM_ARRAY_TASK_ID=1
```

Verify only manifest row 1 is resolved and the command is equivalent to:

```bash
experiment run --config <row1-config> --output-dir <row1-output>
```

Out-of-range index must fail loudly.

---

## 22.4 Resume

Create a partially completed run using existing checkpoints.

Resubmit unchanged study.

Expect:
- same output roots;
- existing completed episodes preserved;
- only incomplete work rerun by the existing experiment runner.

No new checkpoint format should be introduced.

---

## 22.5 Canonical discovery

Synthetic study with two normal run trees and multiple cells.

Expect:
- correct run count;
- correct cell count;
- correct episode count;
- preserved source provenance.

Rename directories to misleading labels.

Scientific coordinates must remain unchanged because they come from resolved configs/overrides.

---

## 22.6 Shard invariance

Represent identical synthetic scientific data as:

1. one normal complete-grid run;
2. several cell-sharded trees.

After canonical normalization, expect identical scientific:
- cell table,
- episode table,
- round table,
- micro-slot table,

except intentional execution-provenance fields.

---

## 22.7 Critical CMI invariance test

Split identical observations into different arbitrary execution shards.

Compute CMI from the merged canonical observations.

Expect identical result under every shard partition.

Explicitly assert the implementation does **not** substitute:

```text
mean(shard CMI)
```

for pooled CMI.

---

## 22.8 Existing estimator equivalence

Using a Study-04-like fixture:

1. run the existing configured relational analysis;
2. run the new study-level adapter.

Expect the same per-cell:
- sensing MI;
- target CMI;
- signed response;
- bootstrap interval;
- null summary;
- support counts;

to exact equality or documented floating tolerance.

This is a migration gate.

---

## 22.9 Null/permutation equivalence

For same data and seed:

```text
same estimator
same null permutations
same bootstrap resamples
same confidence
```

must reproduce the established output.

---

## 22.10 Cache behavior

First aggregate:
- estimator executes;
- result saved with hash.

Second unchanged aggregate:
- estimator reused.

Add a new derived observable:
- CMI reused;
- nulls reused;
- support reused;
- only new derived observable computed.

Change CMI conditioning:
- affected CMI recomputed;
- unrelated products reused.

---

## 22.11 Validation failures

Fail or clearly mark invalid for:
- duplicate episodes;
- mismatched schema versions;
- inconsistent sealed-cell row counts;
- missing expected cells;
- conflicting config identities;
- corrupted retained hashes when available.

`--allow-incomplete` may continue, but outputs remain explicitly incomplete.

---

## 22.12 End-to-end CLI test

Fixture:

```text
config_folder/
  study.yaml
  config_a.yaml
  config_b.yaml
  analysis.yaml
```

Mock SLURM.

Run:

```bash
mas_cc study submit --config-dir ...
```

Verify:
- preflight for both configs;
- common study result root;
- manifest creation;
- exactly one `sbatch` config-array submission.

Populate fixture result trees.

Run:

```bash
mas_cc study aggregate --study-dir ...
```

Verify:
- validation;
- canonical tables;
- established information estimators;
- derived metrics;
- plots;
- final ZIP.

---

# 23. Migration phases

Do not perform a giant rewrite.

## Phase 1 — submission wrapper

Deliver:
- `study.yaml`;
- deterministic submission manifest;
- generic `run_config_array.job`;
- `mas_cc study submit`;
- common result root.

No scientific estimator changes.

## Phase 2 — canonical cross-run dataset

Deliver:
- discovery;
- validation;
- `cells.parquet`;
- `episodes.parquet`;
- `rounds.parquet`;
- `micro_slots.parquet`;
- packaging.

Reuse current readers/adapters.

## Phase 3 — current analysis integration

Deliver:
- reuse/ingestion of current `round_information_*`;
- invoke existing estimator engine where needed;
- canonical estimate/null/support tables;
- equivalence tests.

## Phase 4 — derived-observable registry

Deliver:
- dependencies;
- caching;
- `derived_observables.parquet`;
- first integration of `eta_ir`.

## Phase 5 — declarative plots

Deliver:
- phase-diagram recipes;
- support masking;
- standard summary report.

## Phase 6 — optional smarter retry

Deliver:
- `study status`;
- `sacct` inspection;
- failed-array-index resubmission.

This is convenience, not required for scientific standardization.

---

# 24. Non-goals

Do not:

1. Rewrite the experiment orchestrator.
2. Rewrite episode resume/checkpointing.
3. Replace `scientific_events.parquet`.
4. Replace round/micro scientific records.
5. Reimplement MI or CMI.
6. Average arbitrary shard-level CMI values.
7. Infer conditions from `cell-NNNN`.
8. Make Comet authoritative.
9. Create one `.job` per study.
10. Require multiple manual analysis commands in the normal workflow.
11. Hard-code current Study 04 parameter names as universal.
12. Delete genuinely study-specific estimators where special scientific designs require them.

---

# 25. Acceptance criteria

The system is successful when this is sufficient:

```bash
mas_cc study submit \
  --config-dir configs/runs/relational_reasoning/population_study_06
```

and later:

```bash
mas_cc study aggregate \
  --study-dir results/relational_population_study06
```

with:
- no new study-specific SLURM job;
- no manual cross-run concatenation;
- no manual shard roll-up in the normal workflow;
- established CMI/MI/bootstrap/null/support machinery preserved;
- one canonical analysis folder;
- one final reproducible ZIP.

Adding a new efficiency or phase diagram should normally require only:

```text
register/add derived observable
edit analysis.yaml
rerun `mas_cc study aggregate`
```

and should not require:
- rerunning LLM episodes;
- writing another `.job`;
- manually locating cells;
- reimplementing CMI;
- manually preparing another ZIP.

---

# 26. Final architectural summary

```text
CONFIG FOLDER
    |
    |  mas_cc study submit
    v
ONE STUDY RESULT ROOT
    |
    |  one SLURM config array
    |  one array task per YAML config
    |  each task uses existing grid / episode machinery
    v
NORMAL EXISTING RUN TREES
    |
    |  mas_cc study aggregate
    v
VALIDATED CANONICAL STUDY DATASET
    |
    +--> existing MI / CMI / bootstrap / null / support code
    |
    +--> current / microscopic estimators
    |
    +--> derived efficiencies and other observables
    |
    +--> standard phase diagrams
    |
    v
ONE REPRODUCIBLE ANALYSIS ZIP
```

The study layer standardizes orchestration and outputs.

It must make the existing scientific machinery easier to use, not replace it.
