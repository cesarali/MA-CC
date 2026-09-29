# Plot saved Santa Fe v3 simulations against reduced theory

The supplied theory runner is integrated as the importable
`src/santa_fe_theory` package. The original ZIP is no longer required at runtime;
the integration manifest hashes the extracted Python source files. The repository adapter is
`src/santa_fe/theory_integration.py`. Its inputs are existing v3 Parquet round
trajectories and the **exact YAML saved with those trajectories**. It does not
simulate new MICRO episodes or call an LLM.

The adapter selects physical cells and actual independent round-0 episodes,
projects each saved population and first front page with the runner's
`project_snapshot`, and exports strict `config.json`, `initials.json`,
`simulator.csv`, and `simulation_manifest.json` files. It calls the supplied
`python -m santa_fe_theory run`, `compare`, and `branch` commands for three
labeled variants:

| Variant | Message sampling | Day covariance |
| --- | --- | --- |
| `paper_reduced` | With replacement | Paper diagonal |
| `sampling_matched_reduced` | Without replacement | Paper diagonal |
| `sampling_clock_corrected_reduced` | Without replacement | Fixed update clock |

The solver is a three-variable **reduced Langevin theory** for truth share and
signed fact coverages. None of these variants is full heterogeneous mean-field
theory. The without-replacement option matches the simulator's category
sampling law, while identity-level acquisition remains exchangeable in the
reduced theory.

## Run on already saved cells

From the repository root, using the local project environment:

```bash
.venv/bin/python -m santa_fe.theory_integration \
  --config configs/santa_fe/v3_pilot.yaml \
  --out results/santa_fe_v3_theory_comparison_pilot \
  --cell-ids 0 1 --initial-blocks 4 \
  --theory-replicas 16 --substeps 24 \
  --branch-pairs 128 --branch-snapshots 2
```

Choose a **new empty output directory** for each invocation. The input pilot
contains rho=0.75, N=24, F+=7/F-=3, q=3, q_c=12 and b=0 or 6. A separate
`v3_theory_pilot.yaml` describes a larger 12-cell study; its trajectories must
exist before this adapter can analyze them.

Per cell, `simulator.csv` has one MICRO path per saved initial episode. Each
variant's `theory/trajectories.csv` has independently seeded reduced-theory
replicas. `comparison/overlays.pdf` shows means, mean uncertainty and signed
residuals for target share, both coverages, peer-board share, action frequency,
and actual posts. `comparison/comparison.csv`, `late_window_errors.csv`, and
`distribution_diagnostics.csv` contain the corresponding numbers.
`selected_distribution_bootstrap.csv` adds early/middle/final Wasserstein
distances, variance ratios, and paired initial-block bootstrap intervals. The root
`regime_pilot.pdf` shows categorical late-window target-share cells with common
support and zero-centered residual scales; missing coordinates stay missing.
For horizons below 41 rounds, the late window is the final ten rounds; for
longer runs it is rounds 41 through the horizon.

The adapter also exports genuine end-of-day, pre-action peer boards to
`pre_action.json`, verifies factual next-day MICRO replay, and runs forced
U=0/U=1 branches in both models. Each variant's `branch/branch_comparison.pdf`
shows continuous reduced-theory and discrete MICRO outcomes in the same display
bins, plus paired mean response. At b=0 both branches have exactly the same
population law. Branch JSD, T_pi and eta_IR are **not** calculated by this
adapter: continuous and discrete outcomes need a declared common partition,
matched conditioning, and an uncertainty/bias procedure before an information
comparison is meaningful.

Each run retains a source/version/seed manifest. Blocks have equal weight;
theory replicas are averaged within their initial block. The comparison
command checks parameter equality, initial projections, and complete
block-by-round panels before drawing a plot. It never rounds continuous theory
shares into artificial population counts.

## First saved-pilot interpretation

The completed local pilot is at
`results/santa_fe_v3_theory_comparison_pilot/pilot_report.md`. Four actual
initial blocks per cell and 16 theory replicas per block were used. For b=6,
late rounds 21–30 had MICRO target share 0.594; the three theory variants were
0.496, 0.499, and 0.487 respectively. With b=0, MICRO was 0.304 and theory
was 0.326–0.330. These are descriptive comparisons with only four initial
blocks. A separate 48-substep run for the b=6 cell changed the late target
bias by at most about 0.006 across the three variants, but this comparison also
contains theory Monte Carlo variation. The active cell had no recorded boundary
projections; the b=0 cell had a small number of daytime truth-share projections.

The reduced theory retains identity exchangeability, independent-binomial
coverage, vote/count independence, Gaussian night thinning, and independent
board generation. More blocks and rho/b coordinates are required before
claiming a validity range or a phase boundary. The plots already show the
observed pilot discrepancy rather than fitting the simulator coefficients to
remove it.
