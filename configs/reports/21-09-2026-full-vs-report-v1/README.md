# Full communication versus report-only report

Run from the repository root in the existing project environment:

```bash
python scripts/analysis/report_communication_comparison.py --config configs/reports/21-09-2026-full-vs-report-v1/report.yaml
```

The archive is extracted locally under `results/aggregation_results_local/`.
Output is `results/reports/21-09-2026-full-vs-report-v1/report.pdf`, with PNG
figures, Markdown, descriptive bar values as CSV, saved estimates as Parquet,
and a provenance manifest. This YAML uses the dedicated renderer, not the
standard `mas-cc study report` command. The original experiment config folder
referenced in archive provenance is absent from this checkout, so this report
configuration lives under `configs/reports/`.

The report emphasizes side-by-side communication-profile bars, separately for
no control, truth control and false control, budgets 3/12/18, and rho 0.75/1.00.
Start/end-round means and final-episode means are separate. All bars show +/-1 sample standard deviation across episode values
(round means for start/end bars; final shares for final bars). No budgets or arms are pooled. No-control means truth
share; false control means the recorded ALLOCATION_2 controller-target share.

Only canonical completed episodes are used: 1,382/1,680 expected, 20,730 rounds across 24/28 cells.
The report is provisional and excludes interrupted prefixes. Saved whole-cell
information/null tables and illustrative trajectories follow the bar pages.
The archive's causal-response effects are empty and are not reconstructed.

No aggregation, scientific estimators, bootstrap, permutations or provider calls
are run. Preparation computes descriptive means, episode sample variances/SDs, and raw-minus-null
display differences only. The source archive remains unchanged.

Additional sections show stored T_pi, eta_IF, susceptibility, all saved whole-cell eta variants and their component metrics, and state-local eta_IR maps. Saved eta_th, eta_th_signed and eta_th_bounded rows have no finite estimates and are shown as unavailable. Confidence intervals are copied, never recomputed. Missing full-communication cells remain gaps.

State-by-budget phase diagrams also include saved T_pi and chi for each control
arm. Profiles and rho are separate panels. T_pi uses a shared sequential scale;
chi uses a shared zero-centered diverging scale. Missing cells and unsupported
bins are gray. These plots reshape existing eight-bin estimates only.

Error bars show variability, not confidence intervals. Their endpoints are not
clipped to 0–100%. The CSV retains sample variance in share-squared units and
percentage-points-squared units, SD, and episode counts.

Whole-cell budget curves are labelled as summaries over observed states, with
rho kept separate. Chi additionally includes the saved
`susceptibility_occupancy_weighted` metric (distinct from the primary
`round_target_susceptibility`). `aggregated_over_x_metrics.csv` exports saved
T_pi, eta_IF, occupancy-weighted chi and eta_IR/eta_th variants. These are not
arithmetic averages of plotted state-bin estimates. The archive has no
`study_aggregated_metrics` or rho-aggregated state-map tables; no cross-rho
summary is manufactured in the report.

## Rho-aggregated phase maps (explicitly requested)

`rho_aggregate_maps: true` adds descriptive equal-rho maps for T_pi, chi,
eta_IF and eta_IR. Each x-bin/budget value is the arithmetic mean of the saved
rho=0.75 and rho=1.00 estimates, with weights 1/2 each. Both must be finite
and supported; otherwise the bin is gray. Arms and profiles remain separate.
These are averages of estimates, not estimates from pooled trajectories, and
have no new confidence intervals. Eta_th has no saved state-local estimates
and is explicitly unavailable. `rho_aggregated_phase_maps.csv` retains both
input values, weights policy, coverage counts and the displayed mean.
