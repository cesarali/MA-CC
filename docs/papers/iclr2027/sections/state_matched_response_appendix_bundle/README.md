# State-matched response appendix bundle

This bundle contains a self-contained paper appendix subsection for a one-round state-matched response experiment. It is written as an independent experiment and does not rely on or refer to any earlier population study.

## Main files
- `appendix_state_matched_response.tex` - LaTeX fragment for the paper appendix.
- `figures/state_matched_response.pdf` - publication figure used by the fragment.
- `figures/state_matched_response.png` - raster preview.
- `data/state_matched_response_summary.csv` - aggregate empirical results, bootstrap intervals, and exact q=1 comparator used in the figure.
- `data/state_matched_episode_outcomes.csv` - one row per one-round episode (120 rows).
- `data/state_matched_episode_pairs.csv` - one row per matched ADVOCATE/NOOP pair (60 rows), sufficient to recompute paired contrasts.
- `data/state_matching_effects_original.csv` - original aggregate analysis output.
- `scripts/plot_state_matched_response.py` - regenerates both figure formats from the summary CSV.
- `configs/task0001_resolved_base_config.yaml`, `configs/task0002_resolved_base_config.yaml` - resolved experiment settings.
- `preview_state_matched_response.tex` / `preview_state_matched_response.pdf` - standalone two-column compilation for layout verification.

## Reproduce the figure
From the bundle root:

```bash
python scripts/plot_state_matched_response.py
```

## Primary empirical quantity

`chi(x0) = mean(delta_x | ADVOCATE, x0) - mean(delta_x | NOOP, x0)`.

The reported 95% intervals use 1000 bootstrap resamples of matched episode pairs.

## Integrate in the paper
The fragment expects `graphicx` and assumes the bundle's `figures/` directory is copied with the same relative layout. Then use:

```latex
\input{appendix_state_matched_response.tex}
```
