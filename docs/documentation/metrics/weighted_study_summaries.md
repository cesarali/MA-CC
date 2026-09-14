# Weighted Task003 study summaries

All four Task003 q=3 Potsdam and q=12 DeepInfra truth/false recipes enable
`derived_study_aggregates`. Each study averages its three persistence cells
at fixed budget and target semantics. Providers, models, q values, and arms
remain separate. Separate truth/false roots do not become a joint study merely
by enabling this recipe.

## Estimands and weights

| Quantity | Within one cell | Across persistence cells |
| --- | --- | --- |
| Target CMI | Existing conditional MI | Equal cell weights |
| Original susceptibility | Identified-state occupancy weights | Equal cell weights |
| `eta_IF` | CMI / conditional action entropy | Ratio of weighted components |
| `eta_IR` | Occupancy-weighted Pinsker numerator / CMI | Ratio of weighted components |
| Propensity-weighted causal response, lags 1–3 | Mean eligible Horvitz–Thompson round score | Equal cell weights |
| Available-mass causal susceptibility | Sum of scores / sum of available mass | Ratio of equally weighted **cell mean** score and cell mean mass |
| Local causal response | Mean score in pre-action x bin | Observation weights |
| Local available susceptibility | Mean of score / exact row available mass in x bin | Observation weights |
| Symbolic epistemic summaries | Average rounds within each episode, then episodes equally | Equal cell weights |

Observation weights and all component ratios are recomputed inside every
bootstrap draw. Available-mass quantities exclude x=1; unnormalized causal
response retains those rows. More completed rounds do not give a cell extra
weight in the headline availability ratio.

The epistemic summaries cover all ten `epistemic_parameter_summary` metrics:
collective solvability, fragmentation, individual symbolic solvability, active
union fraction, active fact occurrences, holder redundancy, gold probability,
entropy, configured robustness, and reference robustness. Undefined robustness
remains missing: its mean is conditional on defined states, first within episodes.

Epistemically conditioned MI, symbolic causal surfaces, regression modulation,
joint drift, and censored capture timing retain their existing per-cell outputs.
This extension does not average regression coefficients or censored times, or
pool heterogeneous observations into a new MI estimator. Cross-cell versions
of those quantities require separately specified estimands.

## Susceptibility and efficiency

`propensity_weighted_causal_response` uses logged randomized activation
probabilities. Available susceptibility additionally normalizes by the remaining
opportunity to increase target support. Neither replaces the original chi
inside `eta_IR`; its Pinsker definition remains unchanged.

Efficiency numerator and denominator use the same contributing cells. CMI
null draws are combined across cells before calculating aggregate p-values.
P-values, confidence endpoints, and efficiency ratios are never averaged.

## Paired uncertainty and support

These recipes select `bootstrap: {unit: shared_initialization_block}`. Physical
initial-state or initialization-artifact hashes establish pairing; repetition
numbers alone do not. All rounds of an episode receive the same multiplicity.
Blocks are stratified by observed cell-membership pattern. Sampling within
those strata preserves cell episode counts and shared pairs even when some
repetitions are missing. Absent pairs are not imputed.

Inference is conditional on the observed completion pattern; this does not
correct nonrandom provider/validation failures. Incomplete packages remain
provisional. Missing/unsupported contributions are excluded with explicit cell
coverage and limited/unsupported labels. A partial point estimate describes
its contributing cells, not an unobserved full design. Empty bins remain missing.

Undefined positive-weight components invalidate a bootstrap draw; the ratio's
cell set is not independently renormalized on each side. Zero-occupancy cells
have zero weight in descriptive maps. Valid draw counts are retained. Causal
summaries require both observed actions in each contributing cell or slice.
Bootstrap multiplicities and raw null draws remain transient.

Stratum counts and the smallest stratum size are retained. A singleton stratum
cannot supply empirical between-block variance; summaries involving it are
marked limited. This matters especially for incomplete paired studies.

Epistemic intervals describe between-initialization uncertainty conditional on
the computed observables. Robustness Monte Carlo errors remain separate
diagnostics; the bootstrap does not rerun the symbolic Monte Carlo calculation.

## Tables and reporting

Compressed Parquet tables under `analysis/tables/`:

- `study_aggregated_metrics`: original chi, CMI, `eta_IF`, `eta_IR`.
- `state_local_aggregated_metrics`: their descriptive x-local maps.
- `causal_response_aggregated_metrics`: causal effects and availability ratios.
- `causal_state_local_aggregated_metrics`: raw and available causal x-local maps.
- `epistemic_aggregated_metrics`: ten equal-episode symbolic summaries.
- `state_local_reconstruction`: diagnostic local-versus-cell CMI comparison.

With this suite enabled, matching rho-only rows in the established
`rho_aggregated_state_local_maps` and `rho_aggregated_descriptive_summary`
report sources use the new weighted results. Their old means of efficiency
ratios are replaced. Labels remain `chi`, `T_pi`, `eta_IF`, and `eta_IR`.

New causal/epistemic rows retain units, weights, contributing cell IDs, coverage,
intervals, valid bootstrap counts, support, and analysis provenance.

The four recipes also request weighted budget plots for causal response,
available susceptibility, both efficiencies, and individual/collective symbolic
solvability. Confidence intervals remain available in the source tables.

Use the normal `mas-cc study aggregate --study-dir <root>`. Already prepared
SLURM generations freeze their recipe; prepare a new generation to use these
YAML edits. No simulation resubmission or provider call is needed.

Implementation: `src/mas_cc/studies/derived_aggregation.py` and
`src/mas_cc/studies/weighted_summaries.py`. Tests cover unequal cell sizes,
component ratios, paired covariance, incomplete pairs, unequal episode lengths,
saturation, unsupported bins, report aliases, and offline Parquet-to-ZIP execution.
