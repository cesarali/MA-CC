# Observables and efficiencies: 21-09-2026-full-vs-report-v1

Computed 2026-09-23 from `/Users/rsanchez/Projects/agents_control/new_rnd_init_experiment`,
which was opened **read-only**; nothing in this analysis writes to `agents_control`.

Definitions follow `agents_control/shared_references/BLACKBOARD_OBSERVABLES_AND_EFFICIENCIES.md`.

## Start here

- `new_rnd_init_observables_report.pdf` — the full report: definitions, figures, tables,
  interpretation and limitations.
- `report/main.tex` — its LaTeX source; `report/figures/`, `report/tables/` the parts.

## Tables (`tables/`)

| file | contents |
| --- | --- |
| `effects.csv` | G+, G-, M, D and the target switch for allocations 0/1/2, every horizon, with 95% paired-initialization bootstrap intervals |
| `command.csv` | single-agent command channel I(Z;V_h), following F_h, posts, sensed votes, information per resource at lambda = 0, 0.1, 1 |
| `susceptibility.csv` | state-matched chi, IPW causal tau at lags 1/2/5, available-susceptibility row and cell ratios, weight diagnostics |
| `activation.csv` | one-round activation information, eta_IF, the Pinsker bound B_IR and eta_IR, assigned-policy information |
| `sensing.csv` | I(n_Z;S) using the exact hypergeometric sampling kernel |
| `evidence.csv` | mean proof coverage kappa and full-proof ownership phi by arm |
| `kmin.csv` | K_min(delta) by Legendre transform of the silent reward law, plus its Gaussian approximation |
| `trajectories.csv` | mean vote shares by arm and round |
| `estimates.csv` | fitted endpoint / path information and divergence scores, all candidates, all diagnostics |
| `efficiency.csv` | K^0, K^2, cost, the decomposition terms and eta_end / eta_ctl / eta_task |
| `swaps.csv` | within-initialization target-label permutation nulls |
| `dose_by_slot.csv` | how the controller's share of an agent's reading decays across the 24 update slots in a round |
| `dose_response.csv` | slope of target adoption on controller messages read: raw, population-adjusted, order-adjusted, with cluster bootstrap intervals |
| `board_facts.json` | message lifetime, board turnover and exposure rate |

## Structure of the calculation

* The **initialization** (`physical_initial_state_hash`) is the only independent unit; there
  are 60. Not the 1,382 trajectories, not the 24 votes inside a population, not the rounds.
  Every split and bootstrap resamples whole initializations with all arms attached.
* A **comparison** fixes communication profile, persistence and budget, and keeps the
  initializations with all three arms (silent, truth request, false request) complete.
* Closed-form observables are exact plug-in calculations. Information and divergence terms
  are cross-fitted lower-bound estimates; see the report for which is which.

## Reproduce

```
.venv/bin/python -m rnd_init_metrics.run \
  --config configs/analysis/rnd_init_metrics/new_rnd_init_experiment.yaml --stage all
```

Stages: `data`, `obs`, `fit`, `swaps`, `tables`, `report`. Fitted jobs are cached one pickle
per job under `jobs/` and skipped on re-run, so the pipeline is resumable.

The micro-level tables come from `rnd_init_metrics.micro`, which is not part of `--stage all`;
run it directly:

```
.venv/bin/python -c "from pathlib import Path; from rnd_init_metrics import micro; \
  micro.build(Path('results/bound_estimators/new_rnd_init_experiment'))"
```

Code: `src/rnd_init_metrics/`. `audit.json` records the source hashes and the validation
result for every trajectory.
