# Santa Fe v3 implementation and validation handoff

Prepared locally on 2026-09-26. This update adds an explicitly versioned
microscopic simulator. Historical Santa Fe YAMLs keep `santa_fe_legacy_v2`
semantics; the new [v3 pilot config](../../configs/santa_fe/v3_pilot.yaml)
selects `santa_fe_epistemic_feedback_v3`. No large beta/rho/budget sweep or
cluster job has been launched for v3.

## What changed

- `src/santa_fe/v3_game.py` implements **global once-per-round persistence**,
  a fixed front page throughout the day, a supporting fact in each eligible
  peer post, and fact-bearing controller posts chosen uniformly with
  replacement from the target-aligned pool.
- The per-round record retains agent fact IDs and `(r,s,v)` occupancy, before
  and after night persistence, all three boards, sensor IDs, action and
  controller fact IDs, signed coverage, and exact primitive probabilities.
  Micro-slot records retain sampled messages, acquisition, vote probabilities,
  selected facts and source attribution. A stored RNG boundary state permits
  exact factual next-day replay and paired forced-action branches.
- `src/mas_cc/games/hidden_bench/imitation_round_feedback/analysis.py`
  registers the new outcome and response recipes in the existing shared engine.
- `src/santa_fe/llm_parallel.py` adapts v3 `kappa_plus`, `kappa_minus`, and
  their joint state to the existing MA-CC round estimator. The shared engine
  has added named coverage-outcome CMI and signed-response recipes. It still
  supplies bootstrap intervals, policy-resampling and sensing nulls, and
  action-support diagnostics. No second MI/CMI estimator was added.
- `src/santa_fe/v3_validation.py` is a separate post-hoc command for exact
  one-step transition/emission checks, drift/diffusion coordinates, board
  mean/covariance, exact-vs-exchangeability acquisition, reduced closure,
  hypergeometric sensing MI, factual replay, paired response, Pinsker and
  information-response efficiency. Optional fixed-macrostate finite-size and
  sensing sample-size checks use the same saved or generated scientific
  coordinates. The [game guide](../documentation/games/santa_fe/santa_fe_game_mechanics.md)
  explains the v3 clock and each parameter.

## Pilot and measured limits

The local result root [results/santa_fe_v3_pilot](../../results/santa_fe_v3_pilot)
contains the exact resolved YAML, sealed per-cell data, a pooled and per-round
information table, and the generated
[validation report](../../results/santa_fe_v3_pilot/validation/validation_report.md).
The pilot used two cells (`b=0,6`), `rho=0.75`, 20 episodes per cell, `N=24`,
`F=10`, and 30 rounds. It produced 1,240 round rows and 28,800 microscopic
updates. The report is a smoke and approximation diagnostic, not a validation
of the full phase diagram.

| Check | Pilot result | Interpretation |
| --- | --- | --- |
| Semantic invariants | 11/11 passed | Global persistence, frozen board, aligned posts, sensor size and budgets held in saved records. |
| Exact one-step transition | 0/93 entries with expected count ≥1 exceeded three conditional SD | Sparse entries still have unstable normal z-scores; inspect the full table. |
| Exact peer emission | 0/73 supported entries exceeded three conditional SD | Cross-sign emissions remain impossible. |
| Factual boundary replay | 20/20 sampled states matched the saved next day | Paired branches use the same microscopic update clock. |
| Paired Pinsker check | 20/20 states satisfied the exact empirical-law bound | This does not prove the finite-sample plug-in estimator obeys it. |
| Reduced closure | Mean total variation 0.3503 | The low-dimensional binomial closure is visibly imperfect here. |
| Fact acquisition | Observed 0.1960; exact sampling 0.1964; exchangeability approximation 0.2399 | The exchangeability shortcut overpredicts acquisition at these settings. |
| Fixed-macrostate one-day scaling | Truth-share slope −0.389 [−0.578, −0.234]; `kappa_plus` −0.501 [−0.634, −0.363]; `kappa_minus` −0.512 [−0.675, −0.386] | Intervals are bootstrap estimates from 30 branches per `N=24,48,96,192`; all include −0.5. |
| Exact/reference sensing MI | 0.823/0.900 bits for `b=0`; 0.656/0.775 bits for `b=6` | Direct-counting sensing MI has positive finite-sample bias in this pilot. |

The pilot's `n=20` sensing subsample reuses every reference episode, so its
repeated estimate has zero resampling variance. It cannot answer whether 50
LLM episodes suffice. Drift/diffusion and multinomial board approximations
have tables and plots, but 12 sampled day starts do not justify a general
agreement claim. These distinctions remain explicit in the generated report.

## Run and inspect

From the repository root, with the local project Python and `PYTHONPATH=src`
if the package is not installed:

```bash
python -m santa_fe.cli --config configs/santa_fe/v3_pilot.yaml --dry-run
python -m santa_fe.cluster prepare --config configs/santa_fe/v3_pilot.yaml
python -m santa_fe.cluster run-cell --config configs/santa_fe/v3_pilot.yaml --cell-id 0
python -m santa_fe.cluster run-cell --config configs/santa_fe/v3_pilot.yaml --cell-id 1
python -m santa_fe.cluster aggregate-cells --config configs/santa_fe/v3_pilot.yaml
python -m santa_fe.cluster run-information-cell --config configs/santa_fe/v3_pilot.yaml --cell-id 0
python -m santa_fe.cluster run-information-cell --config configs/santa_fe/v3_pilot.yaml --cell-id 1
python -m santa_fe.cluster aggregate-information --config configs/santa_fe/v3_pilot.yaml
python -m santa_fe.v3_validation --config configs/santa_fe/v3_pilot.yaml --max-events 300 \
  --max-drift-states 12 --max-paired-states 20 --paired-repetitions 100 \
  --scaling-repetitions 30 --scaling-sizes 24 48 96 192 \
  --sensing-sample-sizes 10 20 --sensing-repetitions 20
```

The commands are resumable at sealed cells. Use a new output root for a changed
scientific YAML; the existing pilot result root rejects a changed recipe.

## Remaining theory input

The plan gives the exact microscopic clock and validation targets, but does
not specify a concrete reduced SDE drift, diffusion, boundary convention, or
numerical integration scheme. The validator therefore does **not** invent a
reduced-SDE trajectory or claim that its phase diagram is validated. The
heterogeneous one-step drift/diffusion tables and the measured closure error
show where a supplied reduced SDE can be compared. Once the exact reduced
SDE is provided, add a separately named trajectory comparison and report its
residuals before running the full v3 sweep. Keep the existing legacy beta/CMI
Cygnus handoff separate from this v3 pilot.
