# Analysis plan — Simulation 1, runs of 7 October 2026

**Status: draft for review, written while the runs were still going and before
any of their results were read.** Anything we decide to look at after reading
results is labelled **exploratory** and must pass a confirmation run (§7)
before it is believed.

Runs covered (listed in [`../simulator/runs_index.csv`](../simulator/runs_index.csv)):

| run | what varies from `sim1_base` | cells |
|---|---|---|
| `2026-10-07_sim1_base` | — (argmax voting, supporting posts, abstain if unsupported) | 492 |
| `2026-10-07_sim1_pm` | probability-matching votes | 492 |
| `2026-10-07_sim1_pm_post_always` | probability-matching votes, random post when unsupported | 492 |
| `2026-10-07_sim1_uniform_post` | posts a random active fact, ignoring the vote | 492 |
| `…_no_proof_stop` (one per variant) | `silent_when_target_proved: false`; symmetric truth cells only | 4 × 120 |

Each 492-cell grid: setup {symmetric, nosolution} × arm {silent, truth, false}
× q {3, 6, 12} × qc {3, 6, 12, 24} × b {1, 2, 3, 6, 9} × ρ {0.75, 1}; the silent
arm is not crossed with qc and b. 1,000 episodes per cell, M = 30 days.

---

## 1. Principles

1. **Questions first, in a fixed order** (§4); each answered by comparisons
   that change one thing and hold the rest fixed.
2. **Never pool across setups or ρ.** They are different regimes.
3. **Paired comparisons.** Episode *e* uses the same random draws in the
   silent, truth and false arms, so differences are taken within an episode and
   the bootstrap resamples whole episodes, keeping an episode's arms together.
4. **Effect sizes with 95% intervals**, not p-values: with 1,000 episodes per
   cell, tiny effects are "significant" and meaningless.
5. **Time is part of every result**: trajectories over days 1–30 alongside the
   final day.

---

## 2. Data products (built once, reused by every question)

Built by a script from the run folders' `params.json` and Parquet files, never
by parsing folder names.

| table | one row per | columns |
|---|---|---|
| `cells` | cell × run | run, setup, arm, q, qc, b, ρ, voting rule, posting rule, `agent_post_always`, stopping rule, effective gate (§5.1) |
| `trajectories` | cell × run × episode × day | vote counts n₀ n₁ n₂, mean posteriors, proof rates (two ways), abstention, controller posts that night, cumulative facts posted |
| `nights` | cell × run × episode × night | v̂, true target share, decision, facts posted |
| `paired` | controlled cell × episode × day | the controlled trajectory minus the same episode's silent trajectory (§3) |

Written to `results/analysis/<date>/tables/` (not in git); small aggregate
tables are committed next to this plan.

---

## 3. Estimands

Notation: x_{a,t} = share of the 24 agents voting for allocation a at the end of
day t (t = 1…30). "Silent twin" = the silent cell with the same setup, q, ρ and
variant.

| name | definition | where from |
|---|---|---|
| **paired gain** G_t^{(a)} | mean over episodes of x_{a,t}(controlled) − x_{a,t}(silent twin, same episode) | observables reference §3 |
| **own-target gain** | G^{(0)} for truth cells, G^{(2)} for false cells | — |
| **shared part** M_t^{(a)} | (G_t^{(a)}[truth] + G_t^{(a)}[false]) / 2, same q, qc, b, ρ | reference §4 |
| **directional part** D_t^{(a)} | (G_t^{(a)}[truth] − G_t^{(a)}[false]) / 2; the full switch effect is 2D (both exported) | reference §4 |
| **delivered dose** C_t | facts the controller actually posted up to night t, mean per episode | reference §8 |
| **budget efficiency** | 24 · G_30 / C_29: extra supporting agents per fact posted | reference §8 |
| **time to 75%** | first day the target's share ≥ 0.75; episodes that never reach it count as "not by day 30" (censored), never dropped | reference §5 |
| **consensus split** | per day, the share of episodes whose majority is A0, A1 or A2 | — |
| **proof rates** | share of agents whose memory proves A0: from all facts, and from the agents' own evidence only | reference §5 |
| **abstention by vote** | share of agent-days with `post_reason` = abstained, split by the vote | — |
| **sensing bias** | v̂ − true target share, per night | — |

Units: shares are fractions of 24 agents; counts n_a = 24 · x_a.

---

## 4. Questions, in order

**Q0 — Sanity checks** (all cells)
- invariants at scale: posts only from memory; dose 0 or b per night; no
  controller after day 30; proof from own evidence ≤ proof from all facts;
- expected starting votes reproduce 7 / 6 / 11 (nosolution) and 9 / 7 / 8
  (symmetric) on day 1 under argmax (day-1 votes are before any controller);
- `sim1_base` silent cells reproduce the 6 October silent results: the final-night
  change cannot affect silent cells, and the new recording does not change any
  random draw. Any mismatch beyond rounding is a bug. Reference values (final-day
  mean vote shares, 1,000 episodes, from the 6 October run, whose files were
  deleted; values as printed in the working session):

  | setup | ρ | q | share A0 | share A2 |
  |---|---|---|---|---|
  | nosolution | 0.75 | 3 / 6 / 12 | .229 / .354 / .433 | .763 / .645 / .563 |
  | nosolution | 1.0 | 3 / 6 / 12 | .388 / .422 / .470 | .612 / .578 / .528 |
  | symmetric | 0.75 | 3 / 6 / 12 | .958 / .967 / .940 | .039 / .032 / .060 |
  | symmetric | 1.0 | 3 / 6 / 12 | 1.000 / .970 / .930 | .000 / .030 / .070 |

**Q1 — What drives the A2 drift?** Silent cells only, no controller.
Compare the four variants at fixed setup, q, ρ: vote trajectories and consensus
split over days. Hypothesis to test, not assume: argmax + vote-conditioned
posting drive it (a reviewing agent's diagnostic). Prediction if true: the
drift shrinks under `sim1_uniform_post` and `sim1_pm`.

**Q2 — Does control work, and how?** Every controlled cell against its silent
twin: G_t, own-target gain, M_t, D_t, dose C_t, efficiency, time to 75%.
Within each variant first; variants compared only through these paired
quantities.

**Q3 — Non-monotonicity in b and qc.** Along b at fixed qc, and along qc at
fixed b, everything else fixed. See the criteria in §6 before calling anything
non-monotonic.

**Q4 — Abstention and sensing bias.** The two probability-matching variants:
abstention by vote, and v̂ against the true share, as a function of day, qc and
the setup.

**Q5 — The stopping rule.** The 120 symmetric truth cells of each variant, rule
on (main run) against off (`_no_proof_stop` run), same settings: G_t, dose, proof
rates. The silent twin comes from the main run of the same variant.

**Q6 — Mechanisms** (after Q1–Q5): proof from own evidence against from any
facts; which facts spread, survive or die out; new facts against reactivations.

---

## 5. Known confounders and how each is handled

1. **qc and the gate.** With θ = 0.75 the controller stays silent at ≥ 3 of 3
   (qc = 3), ≥ 5 of 6, ≥ 9 of 12, ≥ 18 of 24 read posts. Only **12 vs 24**
   isolates reading amount; 3 and 6 are reported as "coarse sensing", and every
   qc figure marks the boundary.
2. **b is not the dose.** The gate often silences the controller. Every b
   result is shown against delivered dose C as well as nominal b.
3. **Variants change several things at once** (starting votes, abstention,
   board size, sensing). Compare variants through their own silent twins
   (Q1, then paired quantities), never raw shares across variants.
4. **The setups are two regimes**: never subtract their effect sizes.
5. **Floors and ceilings.** Where the target's share is near 1 by day 30, use
   trajectories and time to 75% rather than the final share.
6. **ρ is never pooled.**
7. **Abstention changes how many posts the controller can read**; `n_posts_read`
   is reported with every qc result.

---

## 6. What counts as a finding

- **An effect**: its 95% interval excludes 0, and it is large enough to matter
  (the threshold is stated in the figure, e.g. a gain of 1 agent = 0.042).
- **A non-monotonicity** (e.g. a larger b doing worse) requires all of:
  1. the reversal is larger than its interval;
  2. it appears in a neighbouring cell too (another q, or the other ρ);
  3. it is not explained by delivered dose (§5.2) or the gate (§5.1);
  4. it survives the confirmation run (§7).
- Uncertainty: bootstrap over episodes, 1,000 resamples, keeping each episode's
  silent, truth and false trajectories together; percentile intervals.

---

## 7. Confirmation runs

Any exploratory finding is rerun on just the cells involved, with
`seed: 2027` instead of 2026, in a new dated run. Our seeds fix all random
draws, so a quirk of the 2026 draws will not reappear. Cheap: minutes.

---

## 8. Figures

One family per question, fixed layout and colour scales within a family, the
silent twin shown in every panel, intervals always drawn.

**F1 — paired gains over time** (Q2, Q5). x = day 1–30, y = G_t (own target,
and all three answers), one line per b at fixed qc (and the reverse), panels
setup × ρ, one figure per variant and q.

**F2 — votes over time** (Q1, Q2). Mean n₀, n₁, n₂ per day with intervals, and
under it the **spread across episodes**: for each day, how many episodes have
each vote count, as a heat strip. A mean of 50/50 can hide half the episodes at
all-A0 and half at all-A2; the strip shows it. Plus the consensus split
(share of episodes with each majority) over days.

**F3 — phase diagrams of the final day** (Q2, Q3). A 4 × 5 grid of equal tiles,
x = qc (3, 6, 12, 24), y = b (1, 2, 3, 6, 9), categorical (not to scale), the
value printed in each tile. Panels: q (3) × ρ (2). One figure per setup × arm ×
variant, for:
- the own-target paired gain G_30 (diverging colours centred at 0; tiles whose
  interval includes 0 greyed);
- delivered dose C_29, next to it, since b is not the dose;
- the target's final share and A1's final share (shares sum to 1, so these two
  carry the full vote split).
A line separates qc ∈ {3, 6} (stricter effective gate) from {12, 24}. One
colour scale per family, shared across panels and variants.

**F4 — sensing** (Q4). v̂ against the true share; abstention by vote, over days.

All figures are written to `results/analysis/<date>/figures/` with an index
file listing each figure, its question and its cells. The report shows a
curated subset and links the index.

---

## 9. Out of scope for this round

The remaining observables of the reference (information quantities, path KL,
efficiencies beyond budget efficiency) come after Q0–Q6, starting with the
four Markov tests on the new data (`../coarse_graining/`).
