# Creating reports from standardized MA-CC analysis packages

For the short saved-results-only workflow, including frozen archives and the
checkpoint report, see [Report quickstart](report_quickstart.md).
The aggregation steps below describe how analysis packages are produced;
they are not prerequisites to rerun when the user requests only a report.

Agent entry point: [Metrics and aggregation master reference](README.md).

This page explains how we turn the output of `mas-cc study aggregate` into a readable scientific report.

The implemented report command is:

```bash
mas-cc study report --config <report.yaml>
```

It creates:

```text
report.md
report.tex
report.pdf
report_manifest.json
figures/
```

The report command is offline. It does not launch episodes or call a language model.

## 1. The complete workflow

The normal sequence is:

```text
simulation results
    -> study aggregation
    -> canonical Parquet tables and estimates
    -> report.yaml
    -> Markdown, LaTeX, PDF, figures, and report manifest
```

A **Parquet table** is a compressed table that preserves column types. An **aggregation package** is the validated `analysis/` directory, or its ZIP archive, produced after a study run.

### Step 1: aggregate the study

On Potsdam, run:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run \
  --live-stream \
  -n MA-CC \
  mas-cc study aggregate \
  --study-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/<study-name>
```

Use `--allow-incomplete` only for an explicitly provisional analysis:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run \
  --live-stream \
  -n MA-CC \
  mas-cc study aggregate \
  --study-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/<study-name> \
  --allow-incomplete
```

On Potsdam, the default aggregation backend can return after submitting detached analysis jobs.
Wait for the finalizer to publish the package before building the report; use `--backend local`
when synchronous execution is intended.

An incomplete aggregation can still produce a package. It remains marked invalid and incomplete, and the report repeats that warning.

### Step 2: inspect the aggregation package

The study normally contains:

```text
<study-root>/analysis/
  validation.json
  validation.md
  analysis_manifest.json
  analysis_recipe.yaml
  tables/
  plots/
  reports/
  provenance/
  <study-name>_analysis.zip
```

Read these files first:

1. `validation.json`: completion, missing episodes, errors, and warnings.
2. `analysis_manifest.json`: table list, estimator settings, hashes, and package status.
3. `analysis_recipe.yaml`: requested estimators, resampling, derived quantities, and aggregation plots.

The report builder accepts either the study root or its `analysis/` directory through `report.source_analysis`. Extract an analysis ZIP first; a ZIP path is not a supported direct input.

### Step 3: copy the report example

Start from:

```text
configs/reports/state_budget_phase_report.example.yaml
```

Copy it beside the study config or to another convenient location:

```bash
cp configs/reports/state_budget_phase_report.example.yaml \
  configs/runs/<study-folder>/report.yaml
```

Edit at least:

```yaml
report:
  id: my_study_report
  title: My Study Report
  source_analysis: /absolute/path/to/<study-root>/analysis
  output_dir: /absolute/path/to/report-output
```

The output directory must not be inside the source `analysis/` package. This prevents report generation from modifying its own scientific source.

### Step 4: select report sections and metrics

The implementation supports three section kinds:

```text
validation_summary
state_budget_phase_suite
budget_curve_suite
```

`validation_summary` reports study completeness. `state_budget_phase_suite` renders state-by-budget heatmaps from existing aggregation rows.

A typical phase-suite section is:

```yaml
- id: state_budget_phase_diagrams
  title: State-by-budget phase diagrams
  kind: state_budget_phase_suite

  axes:
    state: target_fraction_bin_center
    budget: intervention_budget
    state_domain: [0.0, 1.0]
    minimum_state_bins: 8

  facets:
    mode: auto
    preferred:
      - target_semantics
      - epistemic_persistence
      - social_group_size
      - task_id
      - beta
      - threshold

  views: [resolved, aggregated]
```

The state coordinate is the controller-target fraction `x`. The budget coordinate is normally intervention budget `b`.

### Step 5: run the report builder

On Potsdam:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run \
  --live-stream \
  -n MA-CC \
  mas-cc study report \
  --config /absolute/path/to/report.yaml
```

On a configured local checkout:

```bash
mas-cc study report --config /absolute/path/to/report.yaml
```

Successful output ends with a message similar to:

```text
Study report built (complete): /path/to/report-output/report.pdf
```

or:

```text
Study report built (provisional): /path/to/report-output/report.pdf
```

## 2. What each aggregation file contributes

The report is a view over aggregation outputs. It should not become a second estimator implementation.

| Report content | Preferred source |
| --- | --- |
| Completion and provisional status | `validation.json` |
| Study identity and estimator settings | `analysis_manifest.json` |
| Requested calculations | `analysis_recipe.yaml` |
| Submitted configuration | `provenance/` |
| Scientific coordinates and cell completion | `tables/cells.parquet` |
| Episode identity and status | `tables/episodes.parquet` |
| Direct round measurements | `tables/rounds.parquet` |
| Direct microscopic measurements | `tables/micro_slots.parquet` |
| Primary estimates, intervals, and nulls | `tables/primary_estimates.parquet` |
| Information-estimator subset | `tables/information_estimates.parquet` |
| Estimator reliability | `tables/support_diagnostics.parquet` |
| Derived efficiencies and currents | `tables/derived_observables.parquet` |
| State occupancy | `tables/state_occupancy_binned.parquet` or `state_occupancy.parquet` |
| Persistence-aggregated state maps | `tables/rho_aggregated_state_local_maps.parquet` |
| Persistence-aggregated occupancy | `tables/rho_aggregated_state_occupancy.parquet` |
| Communication diagnostics | `tables/blackboard_diagnostics.parquet`, when requested |
| Thermodynamic availability | `tables/thermodynamic_efficiency_diagnostics.parquet`, when requested |
| Specialized causal susceptibility | Its dedicated aggregation table, when requested |

The report builder can read legacy CSV tables, but compressed Parquet is the current canonical analysis format.

For Task003, the rho report sources now use component-weighted efficiency
ratios from the enabled derived suite. The additional causal and epistemic
summary tables and their interpretation are listed in
[`weighted_study_summaries.md`](weighted_study_summaries.md). Their existence
does not add new section kinds to this report builder.

## 3. Division between aggregation and reporting

The central rule is:

> Aggregation creates scientific estimates. Reporting selects, checks, reshapes, and explains them.

### Aggregation owns

- mutual information (MI);
- conditional mutual information (CMI);
- bootstrap confidence intervals;
- randomization and permutation nulls;
- support diagnostics;
- susceptibility;
- currents;
- effective affinity;
- information-response and thermodynamic efficiencies;
- state-local estimates;
- scientific pooling or weighted aggregation.

**Mutual information** measures how much one variable tells us about another. **Conditional mutual information** measures this after accounting for a third variable. A **bootstrap confidence interval** measures uncertainty by resampling whole episodes and recalculating the estimate.

### Reporting may

- filter existing rows;
- select a metric;
- convert a fraction to a percentage;
- subtract `null_mean` from `estimate` within the same row;
- select minima or maxima;
- count source-backed rows;
- reshape unique rows into a heatmap;
- change the display scale without changing values;
- format labels and units.

### Reporting must not

- recalculate MI or CMI from `rounds.parquet`;
- create a missing bootstrap interval;
- create a new pooled estimate;
- average duplicate plotted coordinates;
- replace one unavailable metric with another;
- turn missing values into zero;
- call an LLM or rerun simulations.

If a scientific quantity is missing, add it to `analysis.yaml` and rerun aggregation. The report should mark it unavailable until then.

## 4. State-by-budget phase diagrams

The phase-suite report was designed around `x` versus `b` diagrams.

### 4.1 Minimum state resolution

Every requested state-local metric must have at least eight existing bins spanning zero to one:

```yaml
axes:
  state_domain: [0.0, 1.0]
  minimum_state_bins: 8
```

The report never invents finer bins. If aggregation produced fewer than eight bins, the panel is gray and unavailable. Increase `state_local_x_bins` in `analysis.yaml` and reaggregate.

The standard aggregation request is:

```yaml
state_local:
  - x
state_local_x_bins: 8
```

### 4.2 Resolved and aggregated views

The report can request:

```yaml
views:
  - mode: resolved
  - mode: aggregated
```

A **resolved view** keeps each varying scientific condition separate. An **aggregated view** reads a dedicated aggregation-produced table. The report does not calculate the aggregate itself.

For example, a persistence-resolved map may facet by `epistemic_persistence`. A persistence-aggregated map reads `rho_aggregated_state_local_maps.parquet`, which aggregation has already produced using its declared weighting rule.

### 4.3 Automatic facets

The builder automatically keeps varying scientific coordinates as separate panels. Preferred coordinates include:

```text
target_semantics
epistemic_persistence
social_group_size
task_id
beta
threshold
sensor_sample_size
receiver_epistemic_disposition
controller_evidence_strategy
controller_actuation_mode
controller_communication_policy
```

This means one report configuration can adapt to studies that vary:

- persistence `rho`;
- social sample size `q`;
- truth, false-target, or no-control semantics;
- task;
- controller settings;
- sensor size.

If additional faceting cannot make each plotted coordinate unique, the panel becomes unavailable. The builder never silently averages duplicate rows.

### 4.4 Required metric examples

The example requests:

```text
T_pi
eta_IF
eta_IR
eta_th
occupancy
susceptibility chi
susceptibility chi with symmetric-log colors
```

A metric entry declares both its resolved and aggregated sources:

```yaml
- id: T_pi
  label: T_pi [bits]
  source: primary_estimates
  metric: round_target_actuation_cmi
  aggregation_source: rho_aggregated_state_local_maps
  aggregation_metric: T_pi
  color_scale: linear
```

`T_pi` is action-to-next-state conditional information. `eta_IF` is the information fraction. `eta_IR` is information-response efficiency. `eta_th` is thermodynamic efficiency and can remain unsupported when effective affinity cannot be identified.

### 4.5 Susceptibility display

The ordinary susceptibility map uses a diverging scale centered at zero:

```yaml
color_scale: diverging
```

The same values can also be displayed with a symmetric logarithmic scale:

```yaml
color_scale:
  kind: symlog
  linthresh: 0.001
```

A **symmetric logarithmic scale** shows positive and negative values while expanding small magnitudes around zero. This changes only the colors, not the estimator.

A genuinely different susceptibility must use a different metric and source table. It must not be substituted automatically.

## 5. Missing and unsupported values

Every missing or unsupported value is gray.

Gray can mean:

- a planned cell was not run;
- the state was never visited;
- the state was visited but estimator support was insufficient;
- a metric is absent;
- a derived quantity is unsupported;
- an aggregated view was not produced.

Gray never means zero.

The report keeps unavailable sections rather than silently removing them. It states the source table or metric that is absent. For example:

```text
This requested view is unavailable in the source aggregation package.
The report did not reconstruct a scientific estimator from raw observations.
```

### 5.1 Support labels

| Support status | Report treatment |
| --- | --- |
| `adequate` | Show normally. |
| `limited` | Show, but interpret cautiously. |
| `unsupported` | Gray or unavailable. |
| missing | Gray or unavailable. |

Read `support_diagnostics.parquet` when a finite estimator needs a more detailed reliability check.

## 6. Incomplete studies

The builder reads `validation.json`.

If the study is incomplete, it still creates Markdown, LaTeX, PDF, figures, and a manifest. It writes:

```text
INCOMPLETE / PROVISIONAL
```

The report includes expected, completed, failed, and aborted episode counts and lists validation errors. It must not describe a partial grid as final.

An incomplete cell can still contain usable observations. Those estimates remain provisional, especially when completed repetitions differ strongly between cells.

## 7. Cell identity and phase-map correctness

All downstream joins must use the canonical scientific `cell_id` stored in the aggregation tables.

Source labels such as:

```text
config-0000/cell-0003
```

are provenance labels. They are not necessarily the canonical scientific ID. Indexed or extended studies can use stable hashed cell IDs.

The correct phase-map join is:

```text
expected canonical cells
  LEFT JOIN occupancy by (cell_id, target_fraction_bin_index)
  LEFT JOIN estimates by (cell_id, target_fraction_bin_index, metric)
```

The status meanings are:

| Status | Meaning |
| --- | --- |
| `structural_cell_not_run` | A planned canonical cell has no retained observations. |
| `state_not_visited` | The cell exists, but this state bin has no observations. |
| `insufficient_estimator_support` | The state was visited, but no finite supported estimate exists. |
| `adequate` or `limited` | A finite estimate exists with that support status. |

If every phase-map row is unexpectedly gray while `primary_estimates.parquet` contains state-local values, compare cell-ID namespaces before interpreting the gray map scientifically.

## 8. Output files

A successful build creates:

```text
<report-output>/
  report.md
  report.tex
  report.pdf
  report_manifest.json
  figures/
```

### `report.md`

Readable report source with validation, figures, unavailable quantities, methods, and limitations.

### `report.tex`

LaTeX source generated from the same resolved data as Markdown.

### `report.pdf`

Compiled report. A LaTeX failure is a failed report build.

### `report_manifest.json`

Records:

- report ID;
- creation time;
- source analysis package;
- source package status;
- source scientific input identity;
- source analysis hash;
- report configuration hash;
- phase-result availability;
- displayed value and panel counts;
- generated figures;
- number of source-backed scientific values validated.

## 9. Source-row validation

Every displayed scientific number must come from a source row.

The builder checks plotted values through an internal source ledger. For each value, it records the report location, source table, source row, source field, and rendered value.

The report fails if it contains no source-backed scientific numbers. The builder also refuses to average duplicate plotted coordinates.

The public report manifest records the source package and the count of validated values. The detailed row ledger is currently transient.

## 10. Example configuration

A compact example is:

```yaml
schema_version: 1

report:
  id: example_report
  title: Example MA-CC Phase Report
  source_analysis: /work/.../results/studies/example/analysis
  output_dir: /work/.../results/reports/example

sections:
  - id: validation
    title: Validation and completeness
    kind: validation_summary

  - id: phase_maps
    title: State-by-budget phase diagrams
    kind: state_budget_phase_suite

    axes:
      state: target_fraction_bin_center
      budget: intervention_budget
      state_domain: [0.0, 1.0]
      minimum_state_bins: 8

    facets:
      mode: auto
      preferred:
        - target_semantics
        - epistemic_persistence
        - social_group_size
        - task_id

    max_panels_per_figure: 6
    views: [resolved, aggregated]

    metrics:
      - id: T_pi
        label: T_pi [bits]
        source: primary_estimates
        metric: round_target_actuation_cmi
        aggregation_source: rho_aggregated_state_local_maps
        aggregation_metric: T_pi
        color_scale: linear

      - id: eta_IR
        label: eta_IR
        source: derived_observables
        metric: eta_ir_state_local
        aggregation_source: rho_aggregated_state_local_maps
        aggregation_metric: eta_IR
        color_scale: linear

      - id: occupancy
        label: Round observations
        source: state_occupancy_binned
        value: n_observations
        aggregation_source: rho_aggregated_state_occupancy
        aggregation_value: n_observations
        color_scale: linear
```

For the complete maintained example, use:

```text
configs/reports/state_budget_phase_report.example.yaml
```

## 11. Troubleshooting

### The report says a metric is unavailable

Check whether the metric exists in the named Parquet table. If not, add it to `analysis.yaml` and rerun aggregation. Do not calculate a substitute in the report.

### The report says fewer than eight bins exist

Set at least:

```yaml
state_local:
  - x
state_local_x_bins: 8
```

Then rerun aggregation. The report will not rebin existing data.

### Aggregated panels are gray but resolved panels work

Inspect:

```text
state_local_phase_maps.parquet
state_occupancy_binned.parquet
rho_aggregated_state_local_maps.parquet
rho_aggregated_state_occupancy.parquet
```

Confirm that phase rows use the same canonical `cell_id` as `cells.parquet` and `primary_estimates.parquet`. Also confirm that the aggregation recipe requested `rho_aggregated_descriptive: true`.

### Every state is gray

Gray means missing or unsupported, not zero. Check:

- state occupancy;
- `support_status`;
- `phase_status`;
- cell-ID alignment;
- whether the metric has state-local rows;
- whether the requested state bin count matches the source.

### Duplicate coordinates are reported

Add another varying scientific coordinate to `facets.preferred`, apply a valid source-row filter, or produce a dedicated aggregate in `analysis.yaml`. Do not average the duplicate rows inside the report.

### PDF compilation fails

Confirm that `latexmk` and a working LaTeX installation are available. The report command keeps PDF compilation as a hard technical requirement.

## 12. Validation checklist

Before delivering a report, check:

```text
[ ] source aggregation package opens
[ ] validation status is copied correctly
[ ] incomplete study is marked provisional
[ ] analysis manifest and recipe are readable
[ ] report.yaml uses schema_version 1
[ ] all requested sources resolve or are marked unavailable
[ ] state-local maps use at least eight existing bins from 0 to 1
[ ] varying rho, q, target semantics, and tasks remain separate facets
[ ] duplicate plot coordinates were not averaged
[ ] unsupported and missing values are gray
[ ] gray is explained as missing, not zero
[ ] every displayed value has a source row
[ ] report.md exists
[ ] report.tex exists
[ ] report.pdf compiles
[ ] PDF was visually inspected
[ ] report_manifest.json records the source package
```

## 13. Implementation map

| Concern | Source |
| --- | --- |
| CLI parser and dispatch | `src/mas_cc/cli/main.py` |
| Report package loader and builder | `src/mas_cc/studies/reporting.py` |
| Scientific table I/O | `src/mas_cc/studies/table_io.py` |
| Study aggregation | `src/mas_cc/studies/aggregation.py` |
| Maintained report example | `configs/reports/state_budget_phase_report.example.yaml` |
| Focused report tests | `tests/mas_cc/test_study_reporting.py` |
| Design protocol | `docs/reports/10092026_standard_report_creation_protocol.md` |

## 14. Current scope

The current report builder is intentionally narrow. It supports validation summaries, budget curves, and adaptive state-by-budget phase suites. It does not yet implement every section listed in the broader report protocol, such as arbitrary outcome tables or free-form narrative templates.

Add new generic section kinds to `src/mas_cc/studies/reporting.py` when several studies need the same presentation. Keep study-specific scientific calculations in aggregation, not in the report builder.

### Budget curves

The `budget_curve_suite` section plots existing whole-cell estimates against
`x: intervention_budget`. It accepts the same metric source and `views` fields
as the phase suite. Use `aggregation_source: study_aggregated_metrics` for
study-level curves. State-bin rows are excluded; varying scientific conditions
become separate curves. Duplicate curve/budget coordinates are rejected rather
than averaged. Optional metric `filters` select exact source column values.
Only adequate or limited finite estimates are shown; unsupported points remain
gaps. Vertical bars show stored `ci_low` and `ci_high` where available.

See the Task003 q12 false-control study's `report.yaml` for T_pi,
occupancy-weighted susceptibility, eta_IF, and eta_IR examples. These curves
appear before the phase diagrams when their section is listed first.

### Epistemic summaries and episode examples

Epistemic budget curves can read `epistemic_aggregated_metrics` using the
`budget_curve_suite` section. The Task003 q12 report requests all ten retained
summary metrics, with their stored confidence intervals.

The `episode_timeseries` section reads a retained round table through `source`.
Configure `filters` as column-to-list mappings, `group_by` as scientific
coordinates, `episodes_per_group` (default 1), and `metrics` with `value`,
`label`, and optional `ylim`. Within each group, examples are selected by
sorted cell/episode ID, independently of outcomes. Duplicate episode-round
coordinates fail validation. `episode_examples.json` records the exact selected
identities. These examples are illustrative, not a representative sample.
For the epistemic table, pre-intervention epistemic quantities and post-round
vote shares must be labeled with their respective measurement boundaries.

### Epistemic phase diagrams

`epistemic_phase_suite` reshapes retained `x_bin` / `phi_star_band` rows into
heatmaps of target vote share against symbolic individual solvability.
Each metric specifies `source`, `value`, `label`, `id`, and optional exact
`filters`. The grid dimensions come from the source analysis recipe's
`blackboard_epistemic_phase_outputs` settings. Panels preserve scientific cell
identity and show budget and persistence. A common symmetric color scale is
used across all panels of each metric; missing or unsupported bins remain gray.
Duplicate cell/bin rows are rejected. The 30x30 Potsdam report includes causal
susceptibility and activation-minus-silence changes in vote share and solvability.

### State-local propensity-weighted causal response

To extend a retained analysis package with x-by-budget response maps, use:

```bash
python scripts/analysis/add_state_local_causal_response.py \
  --source <extracted-analysis> --output <separate-extended-analysis>
```

This calls the existing `estimate_causal_response` on completed, lag-eligible
observations within each initial-target-state bin, for lags 1–3. It preserves
canonical cell IDs and uses the source recipe's bin and bootstrap settings.
Shared initialization blocks are resampled within each selected subset.
The source package is preserved; additional estimates, support tables, and
hashed extension provenance are written to a separate package. No provider
calls are made. Point `report.source_analysis` at that extended package and
use `causal_response_state_local_lag_1` (or `_2`, `_3`) as the phase source,
with metric `propensity_weighted_causal_response`. These are unnormalized
causal-response estimates, distinct from available-mass susceptibility and
`round_target_susceptibility`. No persistence-aggregated phase estimate is
created by this extension.

### Null-model comparisons

`null_summary` displays whole-cell `round_target_actuation_cmi` rows from
`primary_estimates`: budget, persistence, raw estimate, null mean and standard
deviation, stored raw-minus-null estimate, permutation p-value, and draw count.
Duplicate budget/persistence rows are rejected. Values are recorded in the
source ledger. Negative excess information is retained; p-values are explicitly
unadjusted for multiple comparisons. The null standard deviation describes
the randomization distribution, not uncertainty in the observed estimate.
These fields can also be selected as phase-map values when state-local null
rows exist. The current truth-control archive retains null summaries only at
whole-cell resolution, so its report includes the table without null phase maps.

### Target occupancy by persistence

`occupancy_bar_summary` displays a precomputed `source_summary` Parquet table
(path relative to the report YAML), with `epistemic_persistence` and
`mean_target_share`. Generate it with
`scripts/analysis/summarize_target_occupancy.py --source <causal_response_round_inputs.parquet> --output <summary.parquet>`.
The summary weights retained rounds from completed episodes equally and pools
across budgets. It is descriptive occupancy, not a causal intervention effect
or a standardized comparison. The script records source hashes and weighting
in a neighboring JSON file. Labels distinguish false and truth targets.

### No-control reports

Set `report.no_control: true` for population-outcome wording without control
phase-map boilerplate. Use `occupancy_bar_summary` and `episode_timeseries`
sections; omit control estimators and null sections. Bar sections accept a
custom `title` and `description`, and episode sections accept `description`
for measurement-boundary and selection notes.
`scripts/analysis/summarize_no_control_outcomes.py` creates bar inputs from a
validated complete analysis: start/end-round truth share, final episode truth
share, collective/individual solvability, and active fact coverage. The first
and epistemic means weight rounds equally; final truth weights episodes equally.
The no-control Task003 report selects three episodes per persistence by sorted
ID and records those identities in `episode_examples.json`.

Target occupancy summaries support `--mode before` (default), `--mode after`,
and `--mode final`. Before/after average over every retained round from
completed episodes, using `x_t` and `x_after` respectively. Final selects the
last round by `(cell_id, episode_id)` and weights completed episodes equally.
All three pool across budgets; they are descriptive means, not causal effects.
The controlled-study reports display all three definitions explicitly.
