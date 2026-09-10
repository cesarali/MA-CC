# Questions for a Standard MA-CC Report Workflow

Choose one or more options for each question. Add a short note when none of the options is exact.

## 1. What should define the scientific calculations?

- [ ] A. `analysis.yaml` defines every estimator and uncertainty calculation.
- [ ] B. The report script may add missing calculations.
- [X] C. Both files may define calculations.
- [ ] D. Decide separately for each study.

## 2. What should happen when a requested quantity is missing?

- [X] A. Mark it as unavailable and explain why.
- [ ] B. Calculate it from `rounds.csv` during report creation.
- [ ] C. Stop and require a new aggregation.
- [ ] D. Omit the section without comment.

## 3. Which files should normally provide report results?

The current `study aggregate` command writes compressed Parquet tables under `analysis/tables/`.

- [ ] A. Only `primary_estimates.parquet` and `derived_observables.parquet`.
- [ ] B. All relevant aggregation Parquet tables, including optional diagnostic tables.
- [ ] C. Mainly the canonical `rounds.parquet` and `micro_slots.parquet` measurements.
- [ ] D. New report-ready Parquet views produced during aggregation.

## 4. How should report sections be selected?

- [ ] A. A reusable `report.yaml` file lists sections, metrics, and plots.
- [ ] B. The report script contains a fixed section list.
- [ ] C. `analysis.yaml` defines both calculations and report sections.
- [ ] D. The user selects sections through command options.

## 5. Which output formats should the standard report builder create?

- [ ] A. Markdown only.
- [ ] B. LaTeX and PDF only.
- [ ] C. Markdown, LaTeX, and PDF.
- [ ] D. PDF plus the source tables and figures.

## 6. How should incomplete studies be handled?

- [ ] A. Build the report with a clear provisional warning.
- [ ] B. Refuse to build any report.
- [ ] C. Build tables but not conclusions.
- [ ] D. Let `report.yaml` choose the behavior.

## 7. What descriptive aggregation should the report builder allow?

A descriptive aggregation summarizes existing estimates. It does not calculate a new estimator.

- [ ] A. No additional aggregation.
- [ ] B. Observation-weighted means only.
- [ ] C. Several declared weighting rules.
- [ ] D. Any aggregation that is clearly labelled.

## 8. How should unsupported or unvisited states appear in plots?

- [ ] A. Blank cells with different patterns for each reason.
- [ ] B. Gray cells for every missing value.
- [ ] C. Remove unsupported states from the axes.
- [ ] D. Let each report choose its own style.

## 9. How much traceability should each report retain?

Traceability means recording where every result came from.

- [ ] A. Record only the source analysis package.
- [ ] B. Record the source table and metric for each section.
- [ ] C. Record table, metric, filters, weighting, support rule, and analysis hash.
- [ ] D. Record all of option C plus the exact source rows.

## 10. What should make report validation fail?

- [ ] A. A missing required table or figure.
- [ ] B. A stated number without a source table row.
- [ ] C. A supported plot that includes unsupported estimator rows.
- [ ] D. Any of the above.

## Decisions and notes

Use this space to record the agreed choices:

```text
1.
2.
3.
4.
5.
6.
7.
8.
9.
10.
```
