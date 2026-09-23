# Full communication versus report-only report

Run from the repository root in the existing project environment:

```bash
python scripts/analysis/report_communication_comparison.py --config configs/reports/21-09-2026-full-vs-report-v0/report.yaml
```

The archive is extracted locally under `results/aggregation_results_local/`.
Output is `results/reports/21-09-2026-full-vs-report-v0/report.pdf`, with PNG
figures, Markdown, descriptive bar values as CSV, saved estimates as Parquet,
and a provenance manifest. This YAML uses the dedicated renderer, not the
standard `mas-cc study report` command. The original experiment config folder
referenced in archive provenance is absent from this checkout, so this report
configuration lives under `configs/reports/`.

The report emphasizes side-by-side communication-profile bars, separately for
no control, truth control and false control, budgets 3/12, and rho 0.75/1.00.
Start/end-round means and final-episode means are separate. Final bars show
individual episode dots. No budgets or arms are pooled. No-control means truth
share; false control means the recorded ALLOCATION_2 controller-target share.

Only canonical completed episodes are used: 97/100 expected, 970 rounds.
The report is provisional and excludes interrupted prefixes. Saved whole-cell
information/null tables and illustrative trajectories follow the bar pages.
The archive's causal-response effects are empty and are not reconstructed.

No aggregation, scientific estimators, bootstrap, permutations or provider calls
are run. Preparation computes descriptive arithmetic means and raw-minus-null
display differences only. The source archive remains unchanged.
