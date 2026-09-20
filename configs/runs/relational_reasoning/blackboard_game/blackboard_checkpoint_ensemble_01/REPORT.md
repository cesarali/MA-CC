# Saved-results checkpoint report

Build using the existing local project environment from the repository root:

```bash
python scripts/analysis/report_saved_checkpoint.py --config configs/runs/relational_reasoning/blackboard_game/blackboard_checkpoint_ensemble_01/report.yaml
```

Input: extracted `blackboard_checkpoint_ensemble_01_frozen_aggregation_inputs_20260918.zip`
in `results/aggregation_results_local/`. Output:
`results/reports/blackboard_checkpoint_ensemble_01/report.pdf`.

This renderer verifies archived checksums and displays the existing diagnostic
estimates, descriptive means of recorded truth shares, and recorded branch
trajectories. It does not run aggregation, estimators, classifiers, bootstrap,
permutations, prefix recovery, or provider requests.

The report is provisional. It uses 100 completed canonical parent bundles
(900 complete ten-round paths), not the larger recoverable sample discussed
in the archived handoff. All nine branches remain separate within each cell.
The three bar pages distinguish start-of-round, end-of-round, and final truth
share; bars have no newly computed uncertainty intervals. The first sorted
completed parent in each cell supplies the trajectory examples.

Only sensor MAE, sensor MSE, and controller-action entropy are saved as
estimates. Dedicated checkpoint causal, classifier, and null results are not
present and are explicitly identified as unavailable. The source archive and
its bundled source code are not modified or executed.
