# Saved-results report

Source: `results/aggregation_results/report_only_authored_q3_q12_chatoss_v5_test_10x10_b6_b18_analysis.zip`.
All 300 episodes and 30 cells passed archive validation.

`report.yaml` is a **collection index**, not a standard single-report CLI config.
Its six `report_q*_*.yaml` files are standard report configs: q=3 and q=12,
each with no control, truth control, and false control. Each subset is selected
by physical cell identity and stored aggregate coordinates; estimator rows are
never averaged across q or arms. Per-part validation counts describe that subset.

From the repository root in the existing project environment:

```bash
python results/report_inputs/report_only_authored_q3_q12_chatoss_v5_test_10x10_b6_b18/prepare_report.py
python results/report_inputs/report_only_authored_q3_q12_chatoss_v5_test_10x10_b6_b18/build_report.py
```

Preparation filters saved tables and creates descriptive occupancy means only.
The builder runs the standard report command for each part, then joins the PDFs
with `pdfunite`. To rebuild one part, use `mas-cc study report --config` with its
`report_q*_*.yaml` file. No aggregation, estimators, bootstrap or permutations
are run. The source archive is unchanged.

Combined output:
`results/reports/report_only_authored_q3_q12_chatoss_v5_test_10x10_b6_b18/report.pdf`.
Individual PDFs, figures and manifests are under `q3_none/`, `q3_truth/`, etc.

Controlled parts contain stored null comparisons, budget curves, susceptibility
and available-mass causal susceptibility phase maps, lagged causal-response
curves, epistemic metrics and maps, three occupancy summaries, and examples.
Unnormalized x-binned propensity-weighted maps are not added because their
required tables are absent. Missing or unsupported requested metrics are marked.
No-control parts contain truth-share bars and epistemic/trajectory examples.

Start/end-of-round bars weight recorded rounds equally. Final bars weight each
completed episode equally. Budgets are pooled for bars within each q/arm only;
these descriptive means have no newly estimated confidence intervals. Examples
are selected by sorted episode ID, independently of outcomes. This is a small
10-episode-per-cell test; completeness does not imply precise estimates.
