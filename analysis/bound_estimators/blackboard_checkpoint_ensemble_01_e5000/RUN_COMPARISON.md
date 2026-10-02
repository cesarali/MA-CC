# 5000-epoch rerun vs the 2026-09-21 run

Date: 2026-09-22. Offline; no simulations, no provider requests.

Config: `configs/analysis/bound_estimators/blackboard_checkpoint_ensemble_01_e5000.yaml`.
Changes from the original run, and nothing else:

- `max_epochs` 1000 -> 5000.
- early-stopping `patience` 10 -> 20 checks (evaluation every 10 epochs, so 200 stagnant
  epochs before stopping). Early stopping already existed; it runs on the inner folds only,
  and the outer refit uses the median inner-selected epoch count, per the specification.
- archive paths updated for the 2026-09-22 move to `paired_parent_exp`. Both SHA-256 hashes
  verify unchanged, and the rebuilt sample is identical: 160 parents, 1428 complete paths,
  the same 8 exclusions.
- the 96 label-swap jobs were copied from the original run. They use only the
  constant/linear/quadratic candidates, which are fitted by logistic regression and have no
  epochs, so rerunning them would return identical values.

480 jobs, 12 workers, 6205 s (1 h 43 m) of fitting under `caffeinate`.

## 1. The change did what it was meant to do

| | old (cap 1000) | new (cap 5000) |
| --- | --- | --- |
| mean fraction of neural refits hitting the epoch cap | 0.361 | 0.009 |
| jobs with any refit at the cap | 328 / 480 | 7 / 480 |

The old report's statement that the flat MLP and GRU critics were under-trained no longer
applies. The GRU memory critics, which were the most affected, gained about 0.25 nats.

## 2. Conclusions that did not change

Endpoint target information at h=10: mean absolute change 0.005 nats, maximum 0.024. Every
cell keeps its sign and its interpretation. Sensing cells remain 0.15-0.34 nats; always cells
remain within noise of zero.

Exploratory efficiency: supported in 15 of 48 (comparison, horizon) pairs before and after.
One cell lost support (q12/rho0.70/sensing/b3 at h=5), a different one gained it
(q3/rho0.70/always/b12 at h=10). The largest ratio moved from 0.144 to 0.113.

Memory: Y_0 plus the last round gives 1.92 nats against 1.96 for the last two rounds. As
before, nearly all of the controlled-versus-silent distinguishability is visible at the
endpoint.

## 3. The negative result that matters

More training did not fix the cost instability.

| | old | new |
| --- | --- | --- |
| cost cells with effective sample size <= 5 (of 32) | 20 | 20 |
| raw NWJ scores below zero | 5 | 6 |

Identical. One cell moved sharply in the wrong direction: q3/rho1.00/always/b3, target 0,
went from +0.96 to -0.72 nats. A negative lower bound on a divergence is not a value, it is a
symptom.

The original report recommended that "more silent parents are the binding constraint, not a
bigger model". That recommendation has now been tested directly by giving the models five
times the training budget, and the diagnostic did not move. The limit is 40 silent paths per
comparison. This is the quantitative case for the additional parents prepared in
`agents_control/paired_parent_exp_2` (60 per setting).

Secondary effect: with the sequence models now converging, the one-standard-error rule picks
the GRU in 4 cost problems instead of 2, and the plain linear model in 8 instead of 10. The
neural families win slightly more often, but not dramatically.

## 4. Read the cost as a range, still

Nothing here changes the reporting advice in the original report: give the cost as a range
across critic caps and training objectives together with the effective-sample-size
diagnostic, not as a single number, and give efficiency only for the q=12 sensing cells.
