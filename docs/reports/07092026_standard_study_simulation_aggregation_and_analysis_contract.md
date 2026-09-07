# Standard MA-CC Study Simulation, Aggregation, and Analysis Contract

**Scope:** study-agnostic description of what a complete MA-CC experiment produces and what the standard study aggregation command calculates  
**Code state checked:** 7 September 2026  
**Primary implementation:** `src/mas_cc/studies/aggregation.py`

## 1. Purpose

This report explains the complete data path for a standard MA-CC study:

```text
scientific YAML configuration
  -> study submission
  -> cells and episodes
  -> retained scientific observations
  -> study validation
  -> canonical study tables
  -> statistical estimators
  -> derived observables
  -> plots, reports, provenance, and ZIP package
```

A **study** is a collection of related experiment configurations. An **experiment configuration** is a YAML file describing one scientific design. A **cell** is one fixed combination of all swept scientific settings. An **episode** is one independent simulation run inside a cell. A **round** is one population-level time step. A **micro-slot** is one individual update inside a round.

The aggregation stage is offline. It reads completed files and does not call a large language model (LLM).

## 2. Important correction to the proposed command description

The current repository implements:

```bash
mas-cc study aggregate --study-dir <study-result-root>
```

and, for an explicitly provisional package:

```bash
mas-cc study aggregate \
  --study-dir <study-result-root> \
  --allow-incomplete
```

The current repository does **not** implement:

```text
mas-cc study compact-analysis
```

There is also an important format correction:

- Current study aggregation writes new analysis tables as **CSV** files.
- CSV means comma-separated values, a portable text table format.
- Existing historical Parquet analysis tables remain readable.
- Run-level compact scientific observations still use `scientific_events.parquet`.
- Parquet means a compressed typed columnar table format.

Therefore, “study aggregation emits compressed Parquet tables” is not the current code contract. If direct Parquet analysis output and a conversion-only `study compact-analysis` command are desired, they still need to be implemented.

A different existing command is:

```bash
mas-cc experiment compact --run-dir <run-result-root>
```

That command converts an older full **run** into validated `results_only` storage. It is not the study aggregation command and does not create the complete study-level analysis package described below.

## 3. Standard Potsdam command

On the Potsdam system, use the dedicated `MA-CC` Conda environment. Conda is the environment manager that selects the required Python interpreter and installed packages.

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run \
  --live-stream \
  -n MA-CC \
  mas-cc study aggregate \
  --study-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/<study-name>
```

Use `--allow-incomplete` only when a provisional analysis of missing or unfinished work is intentional:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run \
  --live-stream \
  -n MA-CC \
  mas-cc study aggregate \
  --study-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/<study-name> \
  --allow-incomplete
```

`--live-stream` keeps long-running progress visible. It does not change scientific calculations.

## 4. Code path

The command-line entry point is registered in `pyproject.toml`:

```text
mas-cc = mas_cc.cli.main:main
```

The parser and dispatch live in:

```text
src/mas_cc/cli/main.py
```

The `study aggregate` branch calls:

```python
aggregate_study(study_dir, allow_incomplete=...)
```

The implementation lives in:

```text
src/mas_cc/studies/aggregation.py
```

Important supporting modules are:

| Source file | Responsibility |
|---|---|
| `src/mas_cc/studies/discovery.py` | Finds submitted runs and scientific cells |
| `src/mas_cc/studies/canonical.py` | Builds canonical cells, episodes, rounds, and micro-slot tables |
| `src/mas_cc/studies/validation.py` | Checks completeness, identities, schemas, seals, and hashes |
| `src/mas_cc/studies/table_io.py` | Reads legacy CSV/Parquet tables and writes verified canonical CSV |
| `src/mas_cc/studies/aggregation.py` | Runs estimators, derives quantities, plots, reports, and packages |
| `src/mas_cc/studies/episode_endpoints.py` | Builds configured episode-level outcome classifications |
| `src/mas_cc/games/hidden_bench/imitation_round_feedback/analysis.py` | Established round-level information and response estimator engine |
| `src/mas_cc/analysis/single_affinity.py` | Revised single-affinity theory definitions and provenance |

## 5. Inputs required at the study root

A submitted study root must contain:

```text
<study-result-root>/
  study_manifest.json
  submission_manifest.csv
  runs/
    ... discovered run and cell artifacts ...
```

The aggregation command refuses a directory without both `study_manifest.json` and `submission_manifest.csv`.

`study_manifest.json` identifies the study and records the analysis recipe path. `submission_manifest.csv` records the expected configurations, their hashes, expected cells, expected episodes, and output locations.

The scientific design remains in the submitted experiment YAML files. Typical configuration content includes:

- game type and population size;
- task and prompt;
- controller mechanism and policy;
- swept grid coordinates;
- number of repetitions;
- random seeds;
- provider and execution limits;
- storage profile;
- metric and analysis requests.

A study folder may contain `analysis.yaml`. This file is the **analysis recipe**, meaning the declarative list of estimators, resampling settings, derived quantities, endpoint analyses, and plots to create after simulation.

## 6. What a simulation produces before study aggregation

The exact run tree depends on the storage profile and game. A `results_only` run normally retains compact scientific artifacts such as:

```text
<run-root>/
  manifest.json
  resolved_base_config.yaml
  scientific_events.parquet
  grid_summary.csv
  cells/
    <cell-id>/
      overrides.json
      resolved_config.yaml
      scientific_events.parquet
      cell_complete.json
      cell_summary.json
      aggregate.json
      round_records/
        <episode-id>/
          round_trajectory.jsonl
          micro_slot_trajectory.jsonl
```

### 6.1 `scientific_events.parquet`

This is the compact scientific transition table. It retains the identity, state, action, outcome, status, and metric fields required for resume, validation, and configured analysis.

It is not a complete provider audit. It does not necessarily retain every prompt, response, reasoning trace, request log, or verbose event.

### 6.2 `round_trajectory.jsonl`

JSONL means one JSON object per line. This file stores one scientific record per population round when the game supports rich round records.

It is normally the richest input for:

- population state before and after a round;
- sensing and controller action;
- controller action probability;
- opinion counts and target/truth shares;
- epistemic or memory state;
- communication and exposure diagnostics;
- state-conditioned information and response analysis.

### 6.3 `micro_slot_trajectory.jsonl`

This stores individual within-round updates when the game exposes a microscopic clock. It supports:

- focal state before and after an update;
- whether the update was controlled;
- message sampling and exposure;
- microscopic currents;
- controlled transition frequencies;
- effective-affinity and kinetic-compliance estimates.

### 6.4 `cell_complete.json`

This is the cell completion seal. A **seal** is a machine-readable statement that the expected episode set and retained files passed integrity checks.

The seal records episode identities, row counts, scientific schema, file hashes, and retained artifact hashes. Aggregation validates the seal instead of assuming that a directory is complete because it exists.

### 6.5 What `results_only` does not preserve

A results-only study is intended for scientific analysis, not complete provider forensics. It may omit:

- all prompts and model responses;
- complete reasoning traces;
- per-request logs;
- full per-episode recorder trees;
- verbose runtime events;
- transient bootstrap and permutation draws;
- analysis caches.

Use a fuller storage profile when later questions require those artifacts.

## 7. Aggregation stages

`aggregate_study()` performs the following stages.

### Stage 1: discover runs and cells

The aggregator reads the submission manifest and locates run trees and cell artifacts. Scheduler identifiers are treated as execution details, not scientific coordinates.

Scientific cell identity comes from submitted configuration identity, resolved configuration, cell overrides, and persisted scientific keys.

### Stage 2: select canonical records

**Canonical** means the authoritative normalized record selected for analysis.

Only episodes marked `completed` or `skipped_resumed` contribute trajectory rows. Failed-attempt prefixes are excluded. If a safe retry wrote the same physical coordinate more than once, the last record is retained.

The command records how many incomplete and superseded retry rows were excluded.

### Stage 3: build canonical tables

The aggregator constructs four study-wide tables:

```text
cells.csv
episodes.csv
rounds.csv
micro_slots.csv
```

These tables join all discovered runs while preserving source provenance.

### Stage 4: validate the study

Validation checks:

- expected and found configurations;
- expected and found cells;
- cell seals;
- expected, completed, failed, and aborted episodes;
- duplicate run, cell, and episode identities;
- source configuration hashes;
- retained artifact hashes;
- scientific schema versions;
- round and micro-slot availability;
- paired initialization, when that contract is present.

Validation is written before estimator execution:

```text
analysis/validation.json
analysis/validation.md
```

### Stage 5: run configured estimators

The aggregator rebuilds estimator inputs from canonical round records and calls the repository’s established estimator engine.

Each physical scientific cell is estimated independently. The code does not create a single conditional mutual information estimate by mixing scientifically different cells.

### Stage 6: calculate derived quantities

Requested quantities are constructed from primary estimates and canonical observations. Examples include susceptibility, information-response efficiency, sensing information, controlled current, effective affinity, and thermodynamic-efficiency diagnostics.

### Stage 7: produce optional views

Depending on `analysis.yaml`, the command may add:

- episode endpoint classifications;
- state-local estimates;
- occupancy maps;
- persistence-aggregated descriptive summaries;
- paired-initialization diagnostics;
- blackboard communication summaries;
- theory-comparison tables.

### Stage 8: render plots

Configured plots are created from canonical or estimator tables. Unsupported estimator rows are masked. A missing table or metric does not produce a misleading empty scientific plot.

### Stage 9: write reports, provenance, and manifest

The command writes human-readable summaries, copies configuration provenance, and records the exact analysis contract in `analysis_manifest.json`.

### Stage 10: package the handoff

The final handoff is compressed into:

```text
analysis/<study-id>_analysis.zip
```

No LLM calls occur in any of these aggregation stages.

## 8. The four canonical study tables

These are the observation layer from which later analysis is reconstructed.

### 8.1 `cells.csv`

One row represents one scientific cell.

Stable content includes:

- study and source-run provenance;
- qualified and source cell IDs;
- submitted and resolved configuration hashes;
- swept coordinates and useful resolved coordinates;
- expected, completed, and failed episode counts;
- seal status.

All grid overrides are retained. Unambiguous leaf-name aliases are added for convenient plotting.

### 8.2 `episodes.csv`

One row represents one realized episode.

Stable content includes:

- study, run, cell, and episode identity;
- repetition index and stable episode key when available;
- episode seed;
- status;
- interaction count;
- provider request and token usage when available;
- timestamps and termination reason;
- scientific schema version.

This is mainly an episode index and execution outcome table. It is not normally the main table of population dynamics.

### 8.3 `rounds.csv`

One row represents one retained population round.

Stable leading columns include provenance and:

```text
cell_id
episode_id
round_index
record_source
```

All available game-specific round fields are retained after those columns. Depending on the game, these can include:

- population state before and after the round;
- controller observation and action;
- action probability;
- target, truth, or order variables;
- opinion counts;
- memory or epistemic state;
- communication mode;
- message, exposure, and adoption counts;
- resource coordinates.

This is the principal input for mutual information, conditional mutual information, entropy, response, state occupancy, and endpoint analyses.

### 8.4 `micro_slots.csv`

One row represents one retained microscopic update.

Stable leading columns include:

```text
cell_id
episode_id
round_index
micro_slot_index
record_source
```

Available game-specific fields can include focal transitions, controlled-slot status, sampled messages, exposure, currents, and evidence changes.

This is the principal input for effective affinity, kinetic compliance, and microscopic current analyses.

## 9. Analysis recipe

A generic `analysis.yaml` can request:

```yaml
theoretical_reference: single_affinity_revised

estimators:
  - round_sensing_mi
  - round_target_actuation_cmi
  - round_target_susceptibility
  - episode_current
  - cell_current
  - effective_affinity
  - kinetic_compliance

resampling:
  bootstrap_resamples: 1000
  null_permutations: 1000
  confidence: 0.95
  seed: 1

derived:
  - round_target_susceptibility
  - eta_ir
  - target_sensing_information_nats
  - controlled_current
  - affinity_weighted_current_nats
  - thermodynamic_control_expenditure_nats
  - eta_th

plots:
  target_information:
    source: primary_estimates
    metric: round_target_actuation_cmi
    x: intervention_budget
    y: threshold
    facet: beta
    kind: heatmap
```

This is an illustration, not a universal required recipe. A requested estimator must be supported by the game’s retained schema. An estimator name in the recipe does not manufacture missing source fields.

Default resampling values, when the recipe omits them, are:

```text
bootstrap resamples: 1000
null permutations: 1000
confidence: 0.95
seed: 1
```

## 10. Primary estimator families

The exact output depends on the recipe and game.

### 10.1 Mutual information

**Mutual information (MI)** is a number measuring how much knowing one variable reduces uncertainty about another.

Examples include sensing MI, which asks how informative a controller observation is about the population state.

### 10.2 Conditional mutual information

**Conditional mutual information (CMI)** measures dependence between two variables after accounting for a third variable.

A typical action-to-next-state quantity is:

```text
I(controller action; next population state | current population state)
```

This measures predictive information flow. It is not, by itself, a causal effect estimate.

### 10.3 Entropy

**Entropy** measures uncertainty or variation. Controller-action entropy is necessary because a controller that always takes the same action supplies no empirical comparison between actions.

### 10.4 Signed response and susceptibility

CMI is unsigned: it can show that actions predict different outcomes without saying which action moves the population toward a target.

Signed response and susceptibility compare action-conditioned changes and retain direction.

### 10.5 Memory-conditioned estimates

A one-dimensional opinion count may omit important internal state. Configured estimators can condition on richer memory, epistemic state, proof coverage, susceptible fraction, or evidence coverage when those fields are retained.

### 10.6 Currents

A **current** is a directed net change accumulated over time. MA-CC distinguishes descriptive terminal episode change from state-matched controlled current used in the revised theory.

### 10.7 Effective affinity and kinetic compliance

**Effective affinity** is a log ratio comparing controlled forward and reverse transition tendencies. **Kinetic compliance** measures the total controlled transition activity in both directions.

These require microscopic transition support. They can be unavailable even when round-level outcomes are present.

## 11. Bootstrap confidence intervals

A **bootstrap confidence interval** measures sampling uncertainty by repeatedly resampling observed units and recalculating the estimate.

The study estimator uses whole episodes as the resampling unit. It does not independently resample rounds from the same episode, because those rounds are dependent.

Compact estimator rows retain:

```text
estimate
ci_low
ci_high
confidence
bootstrap_resamples
n_observations
n_episodes
```

Individual bootstrap draws are not retained in the final package.

For nonlinear derived quantities, the complete quantity is recalculated inside each bootstrap replicate when supported by that derived pipeline. Separate confidence intervals are not combined algebraically.

## 12. Null summaries

A **null distribution** describes estimates expected when the tested relationship is intentionally broken while preserving relevant structure.

The established estimator uses, where appropriate:

- policy-conditional action randomization for action-to-state statistics;
- sensor permutation for sensing statistics.

Compact rows retain:

```text
null_type
null_mean
null_std
p_value
null_permutations
```

Individual null draws are not retained.

A positive raw plug-in MI or CMI estimate should be interpreted beside its null summary, especially in sparse finite samples.

## 13. Support diagnostics

A numerical estimate is not automatically well supported.

`support_diagnostics.csv` can contain:

- number of observations and episodes;
- number of observed actions;
- action counts and action entropy;
- number of occupied conditioning states;
- fraction of states that saw both actions;
- fraction of events lying in dual-action states;
- singleton and sparse-state fractions;
- transition counts for affinity estimates;
- support status.

Typical support labels are:

- `adequate`: usable empirical overlap under the configured checks;
- `limited`: finite but sparse or weakly overlapping;
- `unsupported`: required action or state overlap is absent.

Every scientific interpretation should read the estimate and support row together.

## 14. Core analysis tables

These four estimator-layer tables are always written, although they may be empty when nothing applicable was requested:

### `primary_estimates.csv`

Union of the primary estimator rows. Stable fields include:

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
null_permutations
bootstrap_resamples
n_observations
n_episodes
units
support_status
analysis_hash
```

It may also contain current, affinity, compliance, and state-local estimates.

### `information_estimates.csv`

The information and response rows returned by the established round-feedback estimator.

### `support_diagnostics.csv`

The corresponding action-overlap, state-support, sparsity, and sample-size diagnostics.

### `derived_observables.csv`

Quantities constructed from primary estimates and retained observations. The table records dependencies, units, support, and analysis identity.

## 15. Derived observables

The recipe controls which derived families are requested.

### 15.1 Information-response efficiency

`eta_ir` compares a rigorous lower bound implied by the mean response with the measured action-to-next-state information.

The corrected response coordinate is target fraction. The headline value is an occupancy-weighted ratio of sums, not an unweighted average of state-local efficiencies.

It is an information-response measure, not physical energy efficiency.

### 15.2 Sensing information

The revised single-affinity sensing quantity can use a scalar target-sensing channel in nats. This is not necessarily the same object as full-vector sensing MI in bits.

A **nat** is an information unit using natural logarithms. A **bit** uses logarithms base two.

### 15.3 Controlled current

The revised controlled current is constructed state by state from occupancy, action probability, and susceptibility. It is distinct from final-minus-initial episode target count.

### 15.4 Thermodynamic efficiency

`eta_th` requires supported effective affinity, controlled current, and sensing information under the same theoretical semantics.

The command can emit an unsupported row with an explicit reason. It does not guarantee a number merely because `eta_th` appears in the recipe.

### 15.5 Factorial contrasts

Configured factorial contrasts are descriptive matched differences between levels of a scientific factor. They are labelled `descriptive_only` and are not new MI or CMI estimators.

## 16. Optional tables

Depending on the game and recipe, aggregation may also produce:

```text
episode_endpoints.csv
episode_endpoint_summary.csv
phi_conditioning_comparison.csv
initialization_diagnostics.csv
matched_initialization_audit.csv
state_local_phase_maps.csv
state_occupancy_binned.csv
state_occupancy.csv
rho_aggregated_state_local_maps.csv
rho_aggregated_state_occupancy.csv
rho_aggregated_descriptive_summary.csv
thermodynamic_efficiency_diagnostics.csv
single_affinity_theory_comparison.csv
```

For blackboard studies, `blackboard_population_outputs: true` can add:

```text
blackboard_diagnostics.csv
cell_summary.csv
state_resolved_x_b.csv
sensing_information.csv
transfer_information.csv
susceptibility.csv
efficiencies.csv
rho_b_summary.csv
```

These names describe optional views. They are not guaranteed for every game.

Persistence-aggregated or other coordinate-aggregated state maps are descriptive weighted summaries. They do not replace per-cell estimators unless a separate estimator explicitly pools the canonical observations.

## 17. Episode endpoints

A configured endpoint classifier turns complete round trajectories into one row per episode and one summary row per scientific condition.

Current supported classifier names are:

```text
relational_false_takeover_v1
relational_persistence_exploratory_v1
relational_persistence_refinement_v1
relational_persistence_truth_aligned_v1
```

Endpoint tables can describe final winners, ties, target or truth shares, transient majorities, late-time behavior, and evidence state. Their exact fields depend on the classifier.

An unsupported classifier name causes aggregation to fail rather than silently substituting another definition.

## 18. State-local analysis

A **state-local estimate** is calculated separately in different regions of the current population state.

Supported recipe resolutions are:

```text
x
x_phi
x_kappa
```

Here `x` is target state, while `phi` and `kappa` are additional epistemic summaries when available. Optional `state_local_x_bins` groups the target fraction into a positive integer number of bins.

State-local output should always be paired with:

- state occupancy;
- number of episodes;
- action overlap;
- support status.

An unvisited state is different from a visited but statistically unsupported state. The output preserves that distinction.

## 19. Plot generation

Plots are declared in `analysis.yaml` as built-in names or explicit definitions.

Supported generic plot kinds are:

```text
heatmap
line
```

A plot definition can name:

- source table;
- metric or value column;
- x and y coordinates;
- facet;
- line series;
- filters;
- shared or independent color scales;
- support/status overlay.

The plotter:

- removes `support_status == unsupported` estimator rows;
- can display structural absence, unvisited states, and insufficient support separately;
- does not interpolate missing scientific cells;
- writes PNG files under `analysis/plots/`;
- records plot paths in `analysis_manifest.json`.

If a requested heatmap lacks its required axes but estimates exist, the current generic fallback is a per-cell bar plot. If no selected data exist, no misleading empty plot is written.

## 20. Reports and provenance

A successful or explicitly incomplete aggregation normally creates:

```text
analysis/
  validation.json
  validation.md
  analysis_manifest.json
  analysis_recipe.yaml          # when configured
  tables/
  plots/
  reports/
    summary.md
    methods.md
    state_space_support.md       # when applicable
    false_takeover.md            # when applicable
  provenance/
  <study-id>_analysis.zip
```

### `reports/summary.md`

Contains completion status and high-level counts for runs, cells, episodes, round records, micro-slot records, primary estimates, and derived observables.

### `reports/methods.md`

Records the estimator family, whole-episode resampling, scientific-cell reconstruction, analysis hash, and revised single-affinity semantics when used.

### `provenance/`

Can include:

- study manifest;
- submission manifest;
- submission metadata;
- exact source configuration copies;
- recovery submission metadata;
- study-extension and lineage metadata.

Provenance answers “which design and source artifacts produced this package?”

## 21. `analysis_manifest.json`

The analysis manifest is the machine-readable index of the package.

Current schema version 2 records:

- study ID;
- complete or incomplete status;
- creation time;
- scientific input identity;
- analysis, derived, and auxiliary hashes;
- estimator engine;
- requested statistics;
- resampling settings;
- theoretical reference and theory provenance;
- retention contract;
- generated table names;
- generated plot paths;
- derived semantic/version stamps.

The retention contract currently declares:

```json
{
  "canonical_table_format": "csv",
  "csv_tables": true,
  "parquet_tables": false,
  "compact_estimator_summaries": true,
  "persistent_analysis_cache": false,
  "individual_null_draws": false,
  "individual_bootstrap_draws": false
}
```

The manifest should be consulted before assuming that a historical package follows the current format.

## 22. Final ZIP package

The package is written to:

```text
analysis/<study-id>_analysis.zip
```

It includes, when present:

- `validation.json` and `validation.md`;
- `analysis_manifest.json`;
- `analysis_recipe.yaml`;
- everything under `tables/`;
- everything under `plots/`;
- everything under `reports/`;
- everything under `provenance/`.

It excludes caches, temporary files, pickles, source run trees, provider logs, and individual null draws.

ZIP entries are sorted and receive fixed timestamps and permissions. This makes the archive structure reproducible. The bytes can still differ between aggregations because `analysis_manifest.json` records a new creation timestamp.

## 23. Strict and incomplete behavior

### Strict default

Without `--allow-incomplete`, invalid study data causes this sequence:

1. Write `validation.json` and `validation.md`.
2. Stop aggregation.
3. Do not produce final estimator tables or a new ZIP from that invocation.
4. Return command exit code 2.

### Explicit incomplete mode

With `--allow-incomplete`:

1. Validation remains invalid.
2. The package is marked incomplete.
3. A warning records that analysis continued intentionally.
4. Available canonical records are analyzed.
5. Tables, plots, reports, manifest, and ZIP are produced.
6. The command returns exit code 1.

Exit code 1 is therefore expected for a successfully produced but incomplete package. It does not mean the package was silently treated as complete.

`--allow-incomplete` bypasses study-completeness failure only. It does not bypass malformed recipes, unknown estimator names, impossible theory settings, or unsupported endpoint classifiers.

## 24. Reaggregation from retained canonical tables

If source run trees are unavailable but a previous analysis directory retains all four canonical tables and validation metadata, `study aggregate` can reload:

```text
cells
episodes
rounds
micro_slots
```

The reader prefers CSV and accepts legacy Parquet.

The command then recalculates configured estimators, derived quantities, plots, reports, and the ZIP. It does not call an LLM.

Example:

```bash
mas-cc study aggregate --study-dir <study-result-root>
```

This is reaggregation, not conversion-only packaging. Bootstrap and null calculations can run again. There is no persistent estimator cache in the current contract.

## 25. Run compaction is a different operation

For an older full run, the available compaction command is:

```bash
mas-cc experiment compact \
  --run-dir <run-result-root> \
  --profile results_only
```

Optional flags include:

```text
--archive
--delete-raw
```

This operation validates and creates compact run-level scientific artifacts. It is not a replacement for `study aggregate`.

Use the distinction:

```text
experiment compact
  = change one run's retained storage profile

study aggregate
  = validate and analyze a complete multi-run study
```

## 26. What aggregation does not do

Study aggregation does not:

- launch new episodes;
- call an LLM provider;
- change the game dynamics;
- change seeds or scientific coordinates;
- repair missing source measurements;
- invent support for an unavailable estimator;
- retain individual bootstrap or null draws;
- preserve complete prompt/response histories;
- pool heterogeneous scientific cells into one CMI estimate;
- turn an incomplete study into a complete one.

## 27. How to read a complete analysis package

A practical review order is:

1. `validation.md` for completeness and errors.
2. `analysis_manifest.json` for formats, estimator settings, hashes, and package contents.
3. `analysis_recipe.yaml` for intended estimates and plots.
4. `tables/cells.csv` for scientific coordinates and cell completion.
5. `tables/episodes.csv` for episode identities and outcomes.
6. `tables/support_diagnostics.csv` before interpreting information estimates.
7. `tables/primary_estimates.csv` for point estimates, intervals, and null summaries.
8. `tables/derived_observables.csv` for quantities built from primary estimates.
9. `tables/rounds.csv` and `tables/micro_slots.csv` when a result needs to be independently reconstructed.
10. `reports/methods.md` for exact semantics and units.
11. `plots/` for visual summaries only after checking their source tables.
12. `provenance/` to recover the submitted design.

## 28. Minimal acceptance checklist

Before calling an aggregation deliverable complete, verify:

```text
[ ] validation.json says valid and complete
[ ] found configurations equal expected configurations
[ ] found cells equal expected cells
[ ] completed episodes equal expected episodes
[ ] failed and aborted episodes are zero
[ ] cell, episode, round, and micro-slot identities are not duplicated
[ ] retained artifact and config hashes pass
[ ] one scientific schema version is present
[ ] canonical CSV tables are readable
[ ] estimator rows record units and support status
[ ] bootstrap and null counts match analysis.yaml
[ ] unsupported estimates remain explicitly unsupported
[ ] requested plots have source-table rows
[ ] analysis_manifest.json lists all emitted tables and plots
[ ] provenance contains the submitted configs and manifests
[ ] final ZIP opens and contains the documented handoff
```

## 29. Bottom line

A complete MA-CC study package contains two distinct layers:

1. **Canonical observations:** cells, episodes, rounds, and microscopic updates selected from completed scientific artifacts.
2. **Analysis products:** estimates, uncertainty, null summaries, support diagnostics, derived observables, plots, reports, provenance, and manifest.

The standard command is `mas-cc study aggregate`. It performs validation, normalization, estimator calculation, derivation, plotting, reporting, and ZIP packaging entirely offline.

Current new analysis tables are CSV, not Parquet. Historical Parquet tables remain readable. There is currently no `mas-cc study compact-analysis` command; rerunning `study aggregate` recalculates the analysis from retained canonical tables, while `experiment compact` is the separate command for converting an older run to results-only storage.
