# Is an epistemic coarse-graining Markovian? — test report

**2 October 2026.** Offline analysis of the archived
`21-09-2026-full-vs-report-v1` study. No simulations were run and no provider
requests were made.

Companion to
[`symmetric_controller_and_coarse_graining.md`](../20_controller_redesign/symmetric_controller_and_coarse_graining.md),
which proposed the coarse-graining this report tests.

Reproduce with:

```
python markov_test.py        # cross-arm transfer test  -> markov_results.json
python markov_test2.py       # same, with exposure      -> markov_results2.json
python markov_order.py       # within-arm order test    -> markov_order.json
python bin_sweep.py          # 2-6 bins, both schemes   -> bin_sweep.csv
```

## Summary

| question | answer |
|---|---|
| Does the epistemic state beat the vote count on the transfer test? | **No** — it is slightly worse |
| Does adding controller-exposure repair it? | **No** — it makes it worse |
| Was the transfer test valid? | **No.** Its premise is false; see §3 |
| On the *valid* within-arm test, is the state Markovian? | **At ρ=1.00 yes; at ρ=0.75 no** — see §5b |
| Does the number of bins matter? | **No** — 3/4/5 are statistically indistinguishable (§5c) |
| Is the coarse-graining usable for path KL? | **Yes at ρ=1.00**; an approximation at ρ=0.75 (§6) |

The headline is that the first test I proposed was mis-specified, the second
diagnosis explains why, and the third test — the one that is actually
identified — supports the coarse-graining.

## 1. Data

Only the **silent** and **A2-target** arms are used. The A0-target arm's recorded
dynamics are sound, but it is the confounded configuration — the controller
recommends `ALLOCATION_0` while quoting `ALLOCATION_2`-favouring evidence — so it
cannot inform a kernel meant to describe controlled dynamics. The A2 arm is
unaffected: its pool was built for its own target.

| source | round-transitions |
|---|---:|
| silent arm (passive throughout) | 3,360 |
| A2-target, gate-OFF rounds | 3,273 |
| A2-target, gate-ON rounds | 3,797 |
| *A0-target arm (excluded)* | *9,555* |

## 2. State variables

For agent *j* with active fact set *K*, the exact posterior
`e_j(t) = P(ALLOCATION_0 | K_j(t))`. Population state uses the **mean** over
agents, binned.

| name | definition | states |
|---|---|---:|
| **V** | binned truth-vote share, cuts at 0.33 / 0.66 | 3 |
| **M** | binned mean epistemic level, cuts at 0.33 / 0.50 / 0.75 | 4 |
| **MV** | the pair | 10–12 |

Bin edges follow the measured distribution of `e_j` across 28,800 agent-rounds
(mean 0.435, median 0.430, 10th/90th 0.191/0.708), not the originally proposed
33/66/99 split — under that scheme the top bin would hold 0.77% of observations.

## 3. Test 1 — cross-arm transfer. It fails, and the premise is wrong

**Design.** Fit the kernel on silent episodes; check it predicts gate-OFF rounds
of controlled episodes. Those rounds are passive by definition, so under a
sufficient state the kernels must agree. Gate-ON rounds serve as calibration:
the controller *is* acting there, so a large discrepancy is expected.

**Result.** `weighted_TV` is the share of probability mass the silent kernel
misplaces on held-out rounds; 0 would be perfect.

| variable | gate-OFF (should MATCH) | gate-ON (should DIFFER) |
|---|---:|---:|
| V | 0.146 | 0.166 |
| **M** | **0.162** | **0.112** |
| MV | 0.243 | 0.261 |

All three fail. Worse, for **M the "should match" case is larger than the "should
differ" case** — the test has no discriminating power at all.

**Why.** A gate-OFF round inside a controlled episode is not passive *in state*:

| | agents holding ≥1 controller-pool fact |
|---|---:|
| silent episodes | **0.541** |
| A2 episodes, gate OFF | **0.900** |
| A2 episodes, gate ON | 0.899 |

Gate-OFF and gate-ON are **identical** in exposure (0.900 vs 0.899) and both far
from silence. At ρ = 0.75 controller facts acquired in earlier rounds persist in
agent memory and keep being reactivated, so the population carries the
controller's accumulated influence whether or not it posts this round.

**The premise "gate-OFF rounds are comparable to silent rounds" is false.** The
test was mis-specified, not merely under-powered.

## 4. Test 2 — add exposure. Also fails, for a different reason

If exposure is the missing state, adding it should repair the agreement.

| variable | gate-OFF | gate-ON |
|---|---:|---:|
| M | 0.162 | 0.112 |
| M + exposure | 0.291 | 0.221 |
| M + V + exposure | 0.359 | 0.319 |

It gets **worse**. The cause is a positivity failure: silent episodes sit mostly
below the exposure cut and controlled episodes mostly above, so the transferred
kernel has little or no data in the states it is asked to predict. It
extrapolates rather than predicts.

**Conclusion: Markovianity cannot be tested by transferring a kernel between
these arms at all.** The arms do not share support.

## 5. Test 3 — within-arm Markov order. This one is identified

**Design.** Within a single arm, compare a first-order kernel `P(s' | s)` against
a second-order kernel `P(s' | s, s_prev)`, scored by **held-out log-likelihood
with episode-level splits** (70/30) and Dirichlet smoothing. If knowing the
previous state helps materially, the current state is insufficient.

| arm | variable | n test | order-1 | order-2 | **gain (nats/step)** |
|---|---|---:|---:|---:|---:|
| silent | V | 1,105 | −0.2249 | −0.2100 | +0.0149 |
| silent | **M** | 1,105 | −0.3419 | −0.3270 | **+0.0149** |
| silent | MV | 1,105 | −0.5504 | −0.5478 | +0.0026 |
| A2-target | V | 2,223 | −0.8070 | −0.7620 | **+0.0450** |
| A2-target | **M** | 2,223 | −0.5150 | −0.4966 | **+0.0184** |
| A2-target | MV | 2,223 | −1.2737 | −1.2593 | +0.0144 |

Three things to read here.

**The gains are small.** 0.015–0.045 nats per step is 1–4% of the predictive
log-likelihood. The state is not perfectly Markovian, but the memory it misses is
modest.

**M beats V where it matters.** In the controlled arm the vote state loses
0.045 nats/step to history; the epistemic state loses **0.018**, under half. The
epistemic variable is closer to sufficient exactly where control is acting —
which is the regime the analysis cares about.

**M also predicts better in absolute terms under control**: −0.515 against V's
−0.807 nats/step. It is both a better predictor and more nearly Markovian there.

MV has the smallest gain in the silent arm but the worst absolute likelihood
everywhere — the extra states cost more in sparsity than they buy in resolution.

### 5b. Splitting by persistence changes the conclusion

The §5 figures pool ρ = 0.75 and ρ = 1.00. Separating them (12 seeds, 4 bins):

| split | arm | rows | Markov gap (nats/step) |
|---|---|---:|---:|
| pooled | silent | 3,600 | 0.0138 ± 0.0050 |
| pooled | A2-target | 7,575 | 0.0168 ± 0.0030 |
| **ρ = 0.75** | silent | 1,800 | **0.0177 ± 0.0098** |
| **ρ = 0.75** | A2-target | 5,055 | **0.0155 ± 0.0027** |
| **ρ = 1.00** | silent | 1,800 | **−0.0050 ± 0.0029** |
| **ρ = 1.00** | A2-target | 2,520 | **0.0060 ± 0.0064** |

**At ρ = 1.00 the state is Markovian.** The silent gap is *negative* — a
second-order chain predicts held-out rounds **worse** than a first-order one,
the signature of a genuinely first-order process. At ρ = 0.75 the gap is a clear
0.016-0.018 in both arms.

Mechanism: at ρ = 1.00 nothing is forgotten, so an agent's fact set only grows
and the current epistemic level summarises its whole history. At ρ = 0.75 facts
are lost and reacquired, so the same mean can arise from different fact sets with
different futures.

The two persistence levels also occupy different regions of the state space
(M₃ is essentially unvisited at ρ = 0.75), so pooling them mixes two regimes.
**Kernels must be fitted separately per ρ.**

### 5c. The number of bins does not matter

Sweeping 2-6 bins under quantile and semantic schemes, across 12 episode splits:

| bins | silent | A2-target |
|---|---|---|
| 3 | 0.0121 ± 0.0043 | 0.0139 ± 0.0040 |
| 4 | 0.0138 ± 0.0052 | 0.0168 ± 0.0032 |
| 5 | 0.0120 ± 0.0049 | 0.0164 ± 0.0036 |

Five of six pairwise comparisons are indistinguishable; the sixth differs by
0.0019 nats/step. The residual is not a binning artifact — it comes from the
mean discarding *which* facts are held, which no resolution recovers.

**Warning:** absolute log-likelihood is not comparable across bin counts (it
falls from −0.23 at 2 bins to −0.68 at 6 simply because more states mean more
entropy to predict). Only the gap is comparable. An earlier single-seed run
appeared to favour 3 bins; that did not survive resampling.

## 6. What this means for the coarse-graining

**Use M, with 3 or 4 states — the data cannot separate them.** Take 4 if you
want M₀ ("believes against the truth", 27% of observations) as its own state,
which is where control plausibly acts. M beats the vote count on both criteria:
more nearly Markovian under control, and a better predictor.

**Fit separately per persistence level.** At ρ = 1.00 the chain is Markovian and
path KL, entropy production and MI can be computed directly. At ρ = 0.75 they are
approximations carrying a 0.016 nats/step residual.

**Do not use MV.** Ten to twelve states buys nothing and costs sparsity, which is
the failure mode that made the earlier trajectory-KL estimates unreliable.

**State-space budget.** With ~900 transitions per arm, a 4-state chain has 16
transition probabilities and roughly 55 observations each — workable for a path
KL. For comparison: the vote vector has 325 states, and a 4-bin *composition*
over 24 agents would have C(27,3) = 2,925. Both are hopeless at this sample size.

**Carry the caveat.** The +0.018 nats/step residual means a first-order chain on
M is an approximation. Any entropy production or path-KL figure computed from it
should be reported with that number attached, and ideally with a sensitivity
check against the second-order estimate.

**Do not attempt cross-arm kernel transfer.** Sections 3 and 4 show the arms do
not share support in any state variable that distinguishes them. Comparisons
between silent and controlled dynamics must be made as differences between
separately-fitted kernels, not by transferring one onto the other's data.

## 7. Honest notes on this analysis

- The first test was proposed in the companion design note and executed here. It
  does not work, and the reason is substantive rather than technical. Reporting
  it matters: it is the natural test to reach for, and it is invalid in this
  setting.
- The within-arm order test uses episode-level splits so the order-2 advantage
  cannot come from parameter count alone. It is still a comparison of two
  smoothed estimators, not a formal test, and the Dirichlet prior (α = 0.5)
  affects the absolute numbers though not the ranking.
- Bin edges were chosen from the pooled empirical distribution before any kernel
  was fitted, but they were not held out. A fully clean procedure would select
  edges on a subset disjoint from the evaluation episodes.
- All of this rests on one task, one model and 60 initializations.
