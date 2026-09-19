# Adaptive resampling: what it buys on b9/b15, and why it ships off

`resampling.adaptive` (off by default) stops a bootstrap once both interval
endpoints have moved by less than `tolerance × width` over the last
`check_every` draws, never before `min_resamples`. It applies to the
information stage's bootstraps (bits statistics and diagnostics, both
engines); null permutations stay fixed. Implementation:
`src/mas_cc/analysis/adaptive.py`; hooks in
`round_information_analysis`; realised draw counts land in the tables'
`bootstrap_resamples` column when the switch is on. With the switch off the
settings mapping carries no `adaptive` key, and hashes and tables are
byte-identical to a build without the module (tested).

## Measured (b9/b15 reference package, 16 CPUs, same node type)

Fixed 1000 draws (job 78) against `adaptive: {enabled: true, min_resamples: 200,
check_every: 100, tolerance: 0.02}` (job 86). Full table:
`adaptive-resampling-b9b15-2026-09-19.md`.

| | fixed | adaptive |
|---|---|---|
| information estimates compared | 132 | 132 |
| draws per interval | 1000 | mean 435, median 400, min 300 |
| intervals that stopped early | – | 99 % |
| endpoint shift, in units of the fixed-draw width | – | median 0.018, p95 0.06, max 0.17 |
| point estimates, p-values | identical | identical |
| `information_resampling` stage | 25.0 s | 16.0 s |
| whole finalize | 339.7 s | 330.1 s |

## Reading it

- **The saving is 9 s of 340 s.** After the index-array engines the information
  stage is no longer where the time goes (`derived_study_aggregates` is), so
  adaptive stopping has little left to save here. It would matter on a stage
  that still costs minutes per statistic; none does on this package.
- **The stop rule is not a bound on the error.** "Endpoints stable over the
  last 100 draws" is a plateau test; the median early-stopped endpoint sits
  1.8 % of a width from the 1000-draw one, but the tail reaches 17 %
  (`round_dual_action_event_fraction`, low end). Anyone enabling it should
  read `bootstrap_resamples` per row and treat sub-500-draw intervals as
  coarser, which is exactly the information the column now carries.
- **What is unchanged by construction.** Point estimates and permutation
  p-values do not depend on the bootstrap, and the draws that are made are the
  same draws (prefix of the fixed sequence), so an early-stopped interval is a
  quantile of a subset, not a different estimator.

## Recommendation

Leave it off for production packages. Keep it as an exploratory switch for
large studies where a fixed 1000 draws is the binding cost, and pair it with a
lower `tolerance` (0.01) and a higher `min_resamples` (400) if a tail like the
one above is unacceptable. The checkpoint family's parent bootstraps are not
covered yet; they follow once the pooled version of that module has landed.
