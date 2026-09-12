# Standard MA-CC Report Creation Protocol

**Status:** implemented first version  
**Date:** 10 September 2026  
**Input:** a standardized `mas-cc study aggregate` analysis package  
**Output:** Markdown, LaTeX, and PDF reports

## 1. Purpose

This protocol defines how to turn MA-CC aggregation files into a scientific report.

The intended path is:

```text
completed simulation
  -> mas-cc study aggregate
  -> analysis package with Parquet tables
  -> report.yaml
  -> report builder
  -> Markdown, LaTeX, and PDF
```

A **Parquet table** is a compressed table that preserves column types. A **report builder** is the reusable program that reads the aggregation package and creates the report.

The report must use the aggregation package as its evidence. It must not call a large language model (LLM) or launch new episodes.

The implemented command is:

```bash
mas-cc study report --config <report.yaml>
```

An editable example is available at:

```text
configs/reports/state_budget_phase_report.example.yaml
```

## 2. Agreed decisions

The protocol follows the answers in `10092026_standard_report_workflow_questions.md`.

| Question | Decision |
|---|---|
| Scientific calculations | Both `analysis.yaml` and the report implementation may define calculations. |
| Missing quantities | Mark them unavailable and explain why. |
| Report sources | Use all relevant aggregation Parquet tables, including optional diagnostic tables. |
| Section selection | Use a reusable `report.yaml`. |
| Output formats | Create Markdown, LaTeX, and PDF. |
| Incomplete studies | Build a provisional report with a clear warning. |
| Additional descriptive aggregation | Do not aggregate estimates again during report creation. |
| Missing plot values | Show all missing values as gray cells. |
| Published traceability | Record the source analysis package. |
| Scientific validation failure | Fail if a stated number has no source table row. |

## 3. Division of responsibility

### 3.1 `analysis.yaml`

`analysis.yaml` defines calculations that require scientific estimation across observations.

Examples are:

- mutual information (MI);
- conditional mutual information (CMI);
- bootstrap confidence intervals;
- permutation nulls;
- support classifications;
- susceptibility;
- currents;
- effective affinity;
- information-response efficiency;
- state-local estimates;
- pooled or weighted scientific summaries.

**Mutual information** measures how much one variable tells us about another. **Conditional mutual information** measures this after accounting for another variable. A **bootstrap confidence interval** measures uncertainty by resampling complete episodes.

These calculations belong in aggregation because they combine observations and define scientific estimators.

### 3.2 Report implementation

The report implementation may perform simple deterministic calculations using existing aggregation rows.

Allowed examples are:

- converting a fraction to a percentage;
- calculating `estimate - null_mean` within one estimator row;
- calculating interval width from `ci_low` and `ci_high`;
- selecting the largest or smallest existing estimate;
- counting rows that satisfy a stated condition;
- formatting units and labels;
- reshaping rows into a heatmap without averaging them.

The report implementation must not recreate a missing scientific estimator from `rounds.parquet` or `micro_slots.parquet`.

### 3.3 Resolving the apparent overlap

Both `analysis.yaml` and the report implementation may define calculations, but they have different roles:

```text
analysis.yaml
  = calculations across observations that create scientific estimates

report implementation
  = row-level arithmetic, selection, reshaping, and presentation
```

This boundary avoids two different implementations of the same estimator.

## 4. Source analysis package

The report builder receives either:

```text
<study-root>/analysis/
```

or the corresponding analysis ZIP.

The package should contain:

```text
validation.json
validation.md
analysis_manifest.json
analysis_recipe.yaml

tables/
  cells.parquet
  episodes.parquet
  rounds.parquet
  micro_slots.parquet
  primary_estimates.parquet
  information_estimates.parquet
  support_diagnostics.parquet
  derived_observables.parquet
  ... optional diagnostic tables ...

plots/
reports/
provenance/
```

Not every optional table exists in every study. The table list in `analysis_manifest.json` is the package index.

## 5. Standard file-to-report mapping

The following mapping should be the default.

| Report content | Preferred aggregation source |
|---|---|
| Completion and provisional status | `validation.json` |
| Study identity and analysis settings | `analysis_manifest.json` |
| Requested estimators and resampling | `analysis_recipe.yaml` |
| Submitted scientific configuration | `provenance/` |
| Grid coordinates and cell completion | `cells.parquet` |
| Episode identity, status, and usage | `episodes.parquet` |
| Population trajectories | `rounds.parquet` |
| Microscopic transitions and exposure | `micro_slots.parquet` |
| Primary estimates, intervals, and nulls | `primary_estimates.parquet` |
| Information-estimator subset | `information_estimates.parquet` |
| Estimator reliability | `support_diagnostics.parquet` |
| Efficiencies and other derived quantities | `derived_observables.parquet` |
| Episode outcomes | `episode_endpoint_summary.parquet`, when present |
| State occupancy | `state_occupancy.parquet`, when present |
| Communication diagnostics | `blackboard_diagnostics.parquet`, when present |
| Thermodynamic availability | `thermodynamic_efficiency_diagnostics.parquet`, when present |
| State maps | `state_local_phase_maps.parquet`, when present and valid |

The report builder may use another aggregation table when it is a more direct source for the requested section.

## 6. Source priority

When several tables contain the same quantity, use this priority:

1. A dedicated optional diagnostic or summary table.
2. `derived_observables.parquet` for a derived quantity.
3. `primary_estimates.parquet` for a primary estimate.
4. `information_estimates.parquet` for its information-estimator view.
5. `rounds.parquet` or `micro_slots.parquet` only for direct measurements, trajectories, or verification.

Do not recalculate an estimate from raw observations merely because the raw table is available.

## 7. No additional aggregation rule

The report builder must not combine several estimate rows into a new mean, pooled estimate, weighted estimate, median, or confidence interval.

Examples that are not allowed during report creation are:

```text
mean T_pi across rho
observation-weighted chi across arms
pooled CMI across cells
average eta_IR across tasks
new bootstrap interval across cell estimates
```

If such a quantity is needed, it must be requested in `analysis.yaml` and emitted as an aggregation table row.

### 7.1 Reshaping is allowed

The report builder may reshape existing rows without changing their values.

For example:

```text
rows keyed by (rho, b, estimate)
  -> rho x b heatmap
```

This is presentation, not aggregation.

### 7.2 Duplicate plotted coordinates

If two or more source rows map to the same plotted coordinate, the report builder must not average them.

It must do one of the following:

- add a facet that separates the rows;
- apply a declared filter that selects one row;
- use a dedicated aggregation table where the combination was already calculated;
- mark the plot section unavailable.

## 8. Missing quantity policy

If a requested quantity or table is absent:

1. Do not calculate a replacement from raw observations.
2. Keep the report section.
3. Mark the quantity as unavailable.
4. Explain which source table or metric is missing.
5. Explain what aggregation change would be needed.

Suggested wording:

```text
This quantity is unavailable in the source aggregation package.
The report did not reconstruct it from raw round records.
Add <metric-name> to analysis.yaml and rerun study aggregation.
```

A missing optional quantity does not by itself stop report creation.

## 9. Incomplete study policy

Read completeness from `validation.json`.

If the study is incomplete:

- create all three report formats;
- put `INCOMPLETE / PROVISIONAL` on the title page or at the top;
- state expected, completed, failed, and aborted episodes;
- list the validation errors;
- repeat the provisional warning in the executive summary and limitations;
- avoid wording such as “final” or “complete result.”

Available data may still be described, but the report must not silently treat the partial grid as complete.

## 10. Unsupported and missing plot values

All missing plot values must appear as gray.

This includes:

- a structural scientific cell that was not run;
- an unvisited state;
- a visited state with insufficient estimator support;
- a missing estimate;
- an unavailable derived quantity.

Gray means **no interpretable value is shown**. It must never mean zero.

Every relevant caption should include wording such as:

```text
Gray cells have no displayed estimate. They may be unvisited, unsupported,
structurally absent, or otherwise missing; gray does not mean zero.
```

The chosen simple style does not visually distinguish different missing reasons. The underlying source tables should still preserve those reasons when available.

## 11. Support policy

The report builder reads `support_status` from estimator rows and uses `support_diagnostics.parquet` when a section requires more detail.

Recommended display rules are:

| Support status | Report treatment |
|---|---|
| `adequate` | Display normally. |
| `limited` | Display with a caution in the caption or text. |
| `unsupported` | Display as gray or unavailable. |
| missing | Display as gray or unavailable. |

Including an unsupported row in the input data is not itself a validation failure. Presenting its numerical estimate as supported evidence is not allowed.

## 12. `report.yaml`

A reusable `report.yaml` selects sections, sources, metrics, and plots.

A proposed minimal structure is:

```yaml
schema_version: 1

report:
  id: example_study_report
  title: Example MA-CC Study
  subtitle: Standard Aggregation Report
  source_analysis: /path/to/study/analysis
  output_dir: /path/to/report

sections:
  - id: validation
    title: Validation and completeness
    source: validation.json
    kind: validation_summary

  - id: outcomes
    title: Behavioral outcomes
    source: episode_endpoint_summary
    required: false
    metrics:
      - final_p_truth
      - final_p_target
    plots:
      - kind: heatmap
        x: intervention_budget
        y: epistemic_persistence
        facet: target_semantics

  - id: transfer_information
    title: Transfer information
    source: primary_estimates
    metric: round_target_actuation_cmi
    filters:
      estimator_variant: unsmoothed
    fields:
      - estimate
      - ci_low
      - ci_high
      - null_mean
      - p_value
      - support_status
    plots:
      - kind: heatmap
        x: intervention_budget
        y: epistemic_persistence
        facet: target_semantics

  - id: efficiency
    title: Information-response efficiency
    source: derived_observables
    metric: eta_ir
    fields:
      - estimate
      - ci_low
      - ci_high
      - support_status
```

Table names in `report.yaml` omit `.parquet`. The builder resolves the file using `analysis_manifest.json`.

## 13. Section resolution procedure

For each section, the builder performs these steps:

1. Resolve the named source file in the package.
2. Load the Parquet table.
3. Select the requested metric, if any.
4. Apply declared filters.
5. Check that displayed coordinates identify at most one row.
6. Check units.
7. Check support status.
8. Apply only allowed row-level calculations.
9. Create a section table or plot.
10. Generate prose only from the resulting source-backed rows.

If step 5 finds duplicate coordinates, the section is unavailable until the ambiguity is removed. The builder must not silently average duplicates.

## 14. Report outputs

The standard builder creates:

```text
<report-output>/
  report.md
  report.tex
  report.pdf
  figures/
  report_manifest.json
```

### 14.1 Markdown

`report.md` is the readable text source. It contains the same scientific claims and section order as the final PDF.

### 14.2 LaTeX

`report.tex` is the typesetting source. It is generated from the same resolved section data as Markdown.

### 14.3 PDF

`report.pdf` is compiled from LaTeX. A failed LaTeX compilation is a technical build failure and must not be reported as a successful PDF delivery.

### 14.4 Figures

Every figure is generated from a resolved section table. Figures are not copied from unrelated historical reports.

### 14.5 Report manifest

Published traceability records only the source analysis package, as selected in the questionnaire.

A minimal `report_manifest.json` is:

```json
{
  "schema_version": 1,
  "report_id": "example_study_report",
  "source_analysis_package": "/path/to/study/analysis",
  "source_analysis_zip": "/path/to/study/analysis/example_analysis.zip",
  "source_analysis_status": "complete",
  "created_at": "<timestamp>",
  "outputs": ["report.md", "report.tex", "report.pdf"]
}
```

The public manifest does not need row-level provenance. The build still checks source rows internally before stating numbers.

## 15. Number-to-row validation

This is the selected scientific hard-failure rule:

> Every stated scientific number must have at least one source table row.

A **stated scientific number** includes:

- a number in prose;
- a table value;
- a plotted point or cell;
- a confidence interval;
- a p-value;
- a null estimate;
- a count used in a conclusion.

Allowed exceptions are document structure numbers, such as section numbers, equation labels, or dates.

### 15.1 Internal source ledger

Although the published traceability records only the source package, the builder should keep an internal ledger during validation.

For each stated number, the ledger records:

```text
report location
source table
source row index or key
source field
optional row-level formula
rendered value
```

This ledger can be transient and need not be packaged.

### 15.2 Validation failure

Scientific validation fails when a stated number cannot be linked to a source row.

The builder must stop before declaring the report complete.

## 16. Non-fatal report conditions

Under the selected answers, these conditions do not automatically fail scientific validation:

- a requested optional table is missing;
- a requested optional figure cannot be produced;
- an estimator row is unsupported;
- the study is incomplete;
- a plot contains gray cells;
- a report section is marked unavailable.

They must be disclosed clearly.

## 17. Technical build failures

Technical failures are separate from the selected scientific validation rule.

The builder cannot claim successful delivery when:

- `report.yaml` is invalid;
- the source package cannot be opened;
- `validation.json` or `analysis_manifest.json` cannot be read;
- a Parquet file is corrupt;
- Markdown or LaTeX generation crashes;
- LaTeX does not compile;
- the requested output path cannot be written.

These are execution failures, not scientific-support judgments.

## 18. Standard creation procedure

### Step 1: locate the aggregation package

Provide the analysis directory or ZIP and confirm that it came from `mas-cc study aggregate`.

### Step 2: inspect package metadata

Read:

```text
validation.json
analysis_manifest.json
analysis_recipe.yaml
```

Record completion, table format, generated table names, resampling settings, and analysis identity.

### Step 3: load `report.yaml`

Validate its schema and resolve every requested section.

### Step 4: inventory available Parquet tables

Compare the package table list with the requested section sources.

### Step 5: resolve sections

Select source rows using metrics, coordinates, and filters. Do not average duplicate coordinates.

### Step 6: apply support and missing-value rules

Unsupported and absent values become gray or unavailable. Limited values remain visible with caution.

### Step 7: perform allowed report calculations

Apply percentages, within-row null subtraction, row selection, counts, and formatting. Do not calculate new estimators or aggregate estimates.

### Step 8: create figures and section tables

Use the same resolved data objects for Markdown, LaTeX, and PDF.

### Step 9: generate prose

Create statements from source-backed values. Missing quantities receive explanatory text.

### Step 10: validate stated numbers

Use the internal source ledger. Fail if any scientific number has no source row.

### Step 11: render all formats

Write Markdown and LaTeX, then compile the PDF.

### Step 12: visually inspect the PDF

Check title, provisional warning, page layout, figure labels, gray cells, units, and clipped content.

### Step 13: write `report_manifest.json`

Record the source analysis package and generated outputs.

### Step 14: run the implemented command

After copying and editing the example configuration, run:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run \
  --live-stream \
  -n MA-CC \
  mas-cc study report \
  --config /absolute/path/to/report.yaml
```

The first implementation supports `validation_summary` and
`state_budget_phase_suite` sections. The phase suite automatically retains
varying target semantics, persistence, social group size, task, and other
scientific coordinates as facets. It requires at least eight pre-existing
target-fraction bins spanning zero to one and never creates finer bins in the
report layer.

## 19. Recommended report order

A general report may use this order when the corresponding sources exist:

1. Executive summary.
2. Validation and completeness.
3. Scientific design and parameters.
4. Data realization and support.
5. Behavioral outcomes.
6. Primary information estimates.
7. Null comparisons.
8. Response and susceptibility.
9. Derived efficiencies and currents.
10. State-local results.
11. Game-specific diagnostics.
12. Interpretation.
13. Limitations and unavailable quantities.
14. Recommended next work.
15. Methods and source package.

`report.yaml` controls which sections are included and their order.

## 20. Implementation boundary for a future builder

The generic builder should contain reusable components for:

- package loading;
- `report.yaml` parsing;
- table and metric resolution;
- duplicate-coordinate checks;
- support masking;
- gray missing cells;
- row-level calculations;
- Markdown and LaTeX templates;
- figure generation;
- number-to-row validation;
- PDF compilation;
- report manifest writing.

Study-specific behavior should be expressed in `report.yaml` whenever possible.

A study-specific plugin is justified only when the aggregation package contains a specialized table that needs a genuinely different display, not when a standard metric merely has a different name.

## 21. Acceptance checklist

A completed report build should confirm:

```text
[ ] source analysis package opened successfully
[ ] validation status copied correctly
[ ] incomplete study warning shown when required
[ ] analysis manifest and recipe read successfully
[ ] report.yaml validated
[ ] all requested sources resolved or marked unavailable
[ ] no duplicate plot coordinates were silently averaged
[ ] no new descriptive aggregation was performed
[ ] unsupported and missing values are gray or unavailable
[ ] gray is explained as missing, not zero
[ ] every stated scientific number has a source row
[ ] Markdown generated
[ ] LaTeX generated
[ ] PDF compiled
[ ] PDF visually inspected
[ ] report manifest records the source analysis package
```

## 22. Final protocol rule

The agreed rule is:

> Aggregation produces the scientific estimates. Report creation selects, checks, reshapes, and explains them.

The report implementation may perform simple row-level arithmetic, but it must not replace missing estimators or aggregate existing estimate rows. Missing quantities remain visible as unavailable. Incomplete studies receive a provisional report. Every stated scientific number must be supported by a row in the source aggregation package.
