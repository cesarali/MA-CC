# `bound_estimators` — control-efficiency bounds from learned critics

Implements Part II of
`agents_control/new_experiment_setup/results/CONTROL_EFFICIENCY_DERIVATION_AND_ESTIMATORS.md`.
Built 2026-09-21.

**Critic** means a small neural network trained to tell apart samples from two
distributions; its score gives a *lower bound* on the information between them.
That is the one idea the whole package rests on, and it is also where its
problems come from: a bound is only as good as the critic you managed to train.

## Code and how to run it

The library is **not here** — it is the installed package
[`src/bound_estimators/`](../../src/bound_estimators/), imported by
`src/rnd_init_metrics/` and `tests/bound_estimators/`.

```bash
.venv/bin/python -m bound_estimators.run \
    --config configs/analysis/bound_estimators/<run>.yaml --stage all
```

Jobs are cached under `results/<run>/jobs/` and the run is resumable. A full run
is about 50 minutes on 10 cores. Use `caffeinate` so the Mac does not sleep
mid-run.

## The runs

| directory | what it is |
|---|---|
| `blackboard_checkpoint_ensemble_01/` | the original study; critics trained 3000 epochs |
| `blackboard_checkpoint_ensemble_01_e5000/` | the same study re-fitted at 5000 epochs with early stopping, to check the estimates were not undertrained — see its `RUN_COMPARISON.md` |
| `new_rnd_init_experiment/` | observables and efficiencies for study `21-09-2026-full-vs-report-v1`, following `agents_control/shared_references/BLACKBOARD_OBSERVABLES_AND_EFFICIENCIES.md` |
| `synthetic/` | the estimators run against problems whose exact answers are known — the validation that says whether the machinery works at all |

Inside each run: `tables/` holds the findings as CSV, `report/main.tex` the
report source, `audit.json` the provenance record. The built PDFs for all three
are collected in [`report/`](report/).

## What was found

**Endpoint information is real.** Under *sensing* controller policies, endpoint
target information is 0.15–0.34 nats at horizon 10, and survives a
within-initialization target-label permutation test (`p = 0.005`).

**Under `always` policies it vanishes** — because both configured targets in fact
push the population toward `ALLOCATION_2`. The truthful-report controller steers
*away* from the correct answer. This was confirmed against the archive endpoint
table, and it is the same defect that
[`../preanalysis_task003_new_setup_and_coarsegraining/`](../preanalysis_task003_new_setup_and_coarsegraining/)
later traced to its cause: fact admission was evaluated once for the decoy
target and reused for both arms.

**Cost estimates are not trustworthy.** The NWJ cost bound gives 1–8 nats, but
the effective sample size of the baseline term is 1–5 out of 40 — the estimator
is dominated by a handful of samples. Efficiency ratios of 0.04–0.14 are
reported only where the diagnostics support them.

## Read this next to the preanalysis

The later study found that **trajectory** (path) KL estimated this way fails
every reliability check it was given, while **endpoint** mutual information
passes. Its replacement computes path divergence *exactly* on a 4-state
coarse-graining instead of bounding it with a critic. So:

- endpoint results here — usable;
- path/trajectory results here — superseded;
- anything comparing the two control directions — confounded by the pool defect.

## Scaled down relative to the specification

Known, deliberate: 1 fold repeat where the spec asks for 3; cost horizons
restricted to {1, 5, 10}; no full-refit bootstrap; permutation swaps run on the
cheap models only.
