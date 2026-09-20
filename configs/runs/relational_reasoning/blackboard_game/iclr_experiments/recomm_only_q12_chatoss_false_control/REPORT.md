# Report from saved results only

Source: `results/aggregation_results/recomm_only_q12_chatoss_false_control_frozen_aggregation_inputs_20260918.zip`.
The recorded study is complete: 540 episodes, nine cells, budgets 6/12/18,
persistence 0.70/0.85/1.00. The archive contains canonical trajectories and
completed whole-cell information-estimator fragments, with saved confidence
intervals and 1,000-permutation null summaries. It is not a finalized analysis
package.

`report.yaml` uses only those saved results. The report-only adapter
`results/report_inputs/recomm_only_q12_chatoss_false_control/prepare_saved_report.py`
verifies source checksums, concatenates completed fragments without averaging
their estimates, and selects recorded trajectory columns. Its only numerical
summaries are arithmetic means for descriptive occupancy bars and raw-minus-null
differences for display. It runs no estimators, bootstrap, permutations,
provider requests, or uploads.

The report contains:

- Stored whole-cell information and susceptibility vs budget, with retained CIs.
- Stored null comparisons for each cell.
- Start-of-round, end-of-round, and final-episode false-target share bars.
- Six recorded episode examples, including recorded epistemic indicators.
- Explicit unavailable markers for state-resolved and propensity-weighted maps.

The report does not use the unfinished outputs of the cancelled finalization
attempt. The complete status refers to the recorded episodes, not completion
of every analysis requested in `analysis.yaml`.

Report inputs: `results/aggregation_results_local/recomm_only_q12_chatoss_false_control_saved_report_data/`.
Output: `results/reports/recomm_only_q12_chatoss_false_control/report.pdf`.

Rebuild from the repository root, using the existing local project environment:

```bash
python results/report_inputs/recomm_only_q12_chatoss_false_control/prepare_saved_report.py
mas-cc study report --config configs/runs/relational_reasoning/blackboard_game/iclr_experiments/recomm_only_q12_chatoss_false_control/report.yaml
```
