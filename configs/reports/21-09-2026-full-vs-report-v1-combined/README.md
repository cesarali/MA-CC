# Combined saved-results comparison

Rebuild locally:

```bash
python scripts/analysis/report_communication_comparison.py --config configs/reports/21-09-2026-full-vs-report-v1-combined/report.yaml
```

Reads the v1 and v1-b6-b9-b15 extracted analysis archives directly. This is a
report collection, not a merged study for reaggregation. No estimator reruns,
permutations, simulations or uploads. Duplicate cell IDs or scientific plot
coordinates fail explicitly. Source hashes and validation reports are recorded.

Budgets 3/12/18 come from the original archive; 6/9/15 from the extension.
Original no-control cells are retained once. Total coverage: 2,717/3,120
expected episodes, 48/52 expected cells. Both sources are provisional.

IMPORTANT: original controlled config explicitly enables
`controller_allow_mixed_message_types: true`; the extension omits the option
(default false). Other game and prompt configs match; the budget grids differ.
The extension budgets carry an asterisk in phase maps. The full display is not
a strictly matched budget-only experiment. Do not attribute differences to
budget alone. The original report remains unchanged.

Bars show mean +/- sample episode SD. Phase diagrams keep arms and profiles
separate; rho-aggregated descriptive maps require both supported rho values
and give each weight 1/2. Missing data remains gray. No pooled scientific
estimate or new uncertainty interval is manufactured.
