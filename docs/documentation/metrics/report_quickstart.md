# Building reports from saved results

Reports display the results already available in an archive. **A request to
build a report does not require rerunning aggregation, estimators, bootstrap,
permutations, or simulations.** Missing results are labelled unavailable.

## Where things live

| Item | Location |
| --- | --- |
| Original ZIPs | `results/aggregation_results/` (read-only bucket mount) |
| Local extracted inputs | `results/aggregation_results_local/` |
| Report settings | `report.yaml` beside the experiment's `study.yaml` |
| Experiment-specific instructions | `REPORT.md` beside `report.yaml` |
| Report preparation files, when needed | `results/report_inputs/<study>/` |
| PDF, figures, Markdown and provenance | `results/reports/<study>/` |

Keep the original archive unchanged. Extract locally and check its contents:

- **Finished analysis package:** usually has `analysis_manifest.json`,
  `validation.json`, and `tables/` containing estimates.
- **Frozen aggregation inputs:** may contain `input/` and `groups/`, sometimes
  under `frozen_generation/`. Only some calculations may have finished.
  A recipe requesting a metric does not mean its results are present.

## Build a standard report

Run from the repository root using the existing project Python environment:

```bash
mas-cc study report --config configs/runs/<experiment-folder>/report.yaml
```

The YAML selects the source directory, output directory, sections and metrics.
Paths in the YAML resolve relative to that YAML. The standard builder produces
`report.pdf`, `report.md`, `report.tex`, figures and `report_manifest.json`.
For frozen inputs, use the experiment's documented report-only adapter first;
there is no universal frozen-archive adapter yet.

## Rebuild the checkpoint report

This report uses a dedicated renderer because each parent has nine branches:

```bash
python scripts/analysis/report_saved_checkpoint.py \
  --config configs/runs/relational_reasoning/blackboard_game/blackboard_checkpoint_ensemble_01/report.yaml
```

Output: `results/reports/blackboard_checkpoint_ensemble_01/report.pdf`.
This renderer writes the PDF directly, plus Markdown, figures, saved diagnostic
tables and a manifest; it does not produce LaTeX.

It uses 100 completed canonical parent bundles, keeps branches separate, and
does not recover interrupted prefixes. Only three diagnostic metrics are saved
in this snapshot; checkpoint causal and null results are unavailable.

## What report preparation does

- Verify source checksums where supplied and preserve saved estimates and CIs.
- Format existing null summaries; never generate missing permutations.
- Compute simple descriptive means for bars, without new confidence intervals.
- Distinguish **start-of-round**, **end-of-round**, and **final-episode** shares.
- Select recorded trajectory examples by sorted identifiers, not outcomes.
- Mark incomplete samples and missing metrics, then inspect the PDF.

Adding missing causal or phase-map estimates is a **separate analysis task**.
In particular, `add_state_local_causal_response.py` runs estimators and bootstrap;
it is not a report-only preparation command.

For the full schema and section options, see [report_creation.md](report_creation.md).
For archive contents, see [study_aggregation_contract.md](study_aggregation_contract.md).
