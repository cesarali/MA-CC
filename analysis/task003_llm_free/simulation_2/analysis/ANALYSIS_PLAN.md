# Analysis plan — Simulation 2, the main grid

**Status: agreed 9 October 2026** (size threshold, confirmation run and design decided by
the researcher; written while the grid was running, before any of its results were read).

**What had already been seen.** The bridge run `2026-10-09_sim2_bridge` (cell means only).
Its steps S4–S6 are, cell for cell, the grid's main cells at λc = 1 (same code, same seed;
tested). Hypotheses suggested by that look are listed in §5 and are decided on a
**confirmation run with seed 2027** (§7), not on the data that suggested them.

Runs covered (listed in [`../simulator/runs_index.csv`](../simulator/runs_index.csv)):

| run | what it is |
|---|---|
| `<date>_sim2_grid` | the main grid, 1,212 cells × 1,000 episodes ([`sim2_grid.yaml`](../simulator/configs/sim2_grid.yaml)) |
| `2026-10-09_sim2_bridge` | S0 = Simulation 1's rules in the new code (the S0 check); S1–S3 not analysed |
| `2026-10-07_sim1_pm` (Simulation 1) | the reference at the same q, qc, ρ, with b = 1 |
| `<date>_sim2_grid_confirm_seed2027` | the confirmation run (§7) |

Rules: [`../simulator/simulation_2_spec.md`](../simulator/simulation_2_spec.md) §14. The core:
agents and controller act at random times, memory fades continuously, the board fades
(48 most recent posts). Grid: setup × arm × q {3, 6, 12} × qc {3, 6, 12, 24} × ρ {0.75, 1}
× λc {0.5, 1, 2, 4, 8} × gate {on, off} × proof stop {on, off; separately only for
symmetric truth}. Budget 30 messages; control for t < 30; follow-up to t = 40.

---

## 1. Principles

1. **Questions first, in a fixed order** (§4); each comparison changes one thing.
2. **Never pool the two setups or the two ρ.**
3. **Paired comparisons.** Episode *e* uses the same keyed random draws in every cell
   with the same q and ρ: silent and controlled arms, every rate, gate and stop. So
   differences are taken within an episode, and the bootstrap resamples **episode
   numbers**, keeping all cells of an episode together.
4. **Comparisons with Simulation 1 are not paired** (different code, different draws):
   the two means are bootstrapped independently.
5. **Effect sizes with 95% intervals**, not p-values. Time is part of every result.

---

## 2. Data products (built once)

From each cell's `params.json`, the manifest's list of reused cells, and the Parquet files.

| table | one row per | columns |
|---|---|---|
| `cells` | cell (including reused ones, pointing to the cell that was run) | setup, arm, q, qc, ρ, λc, gate, stop, paired silent cell |
| `episode_values` | cell × episode | m_A0, m_A2, vote shares at t = 30 and 40; m_target averaged over [0, 30]; messages; reads; time the budget ran out |
| `trajectories` | cell × measurement time | means over episodes of m_A0, m_A1, m_A2, vote shares, share voted, proof rates, messages so far |
| `gains` | controlled cell | G(30), G(40), G(40) − G(30), average gain, vote gains, messages, efficiency, each with a 95% interval |

Written to `results/simulation_2/analysis/<date>/tables/` (not in git).

---

## 3. Estimands

m_k(t) = mean belief in allocation k over the 24 agents at time t. Target = A0 (truth),
A2 (false).

| name | definition |
|---|---|
| **gain** G(t) | mean over episodes of m_target(t) controlled − silent, same episode. **Primary: G(30).** |
| **persistence** | G(40) − G(30) |
| **average gain** | G over [0, 30], trapezoid rule on the 0.25 grid |
| **vote gain** | the same with the share of last votes for the target |
| **shared / directional parts** | M = (G_truth + G_false)/2 and D = (G_truth − G_false)/2, on m_A0 and m_A2 |
| **dose** | messages per episode; reads per episode; share of episodes that spend the budget, and when |
| **efficiency** | 24 · G(30) / messages (ratio of means, paired bootstrap) |
| **gate effect** | G(30) gate off − gate on, same episode; and the change in messages |
| **stop effect** | G(30) stop off − stop on, symmetric truth only |
| **sensing error** | v̂ − true current target share, per controller action |
| **core vs Simulation 1** | core (λc = 1, gate on, stop on) minus Simulation 1 (b = 1) at the same setup, q, qc, ρ: silent levels m_A0(30), m_A2(30), and G(30) |

**Size that matters: 1/24 ≈ 0.042**, one agent's worth of belief (decided).

---

## 4. Questions, in order

**Q0 — Sanity.** On every cell: messages ≤ 30; no controller action at t ≥ 30; agents act
in the follow-up; controller reads only agent posts. On the 52 full-table cells: no
self-reads, reads within the 48-post window, every fact read active afterwards,
replay of memories on 20 episodes per cell. The grid's main cells at λc = 1 equal the
bridge's S4–S6 exactly. The S0 check (done, passed).

**Q1 — Simulation 1 vs the core.** At every setup, q, ρ: the silent populations
(m_A0, m_A2 over time). At every setup, q, qc, ρ: G(30) in the core (λc = 1, gate on,
stop on) against Simulation 1 (b = 1). What does asynchrony do to the population, and
to control?

**Q2 — Control in the core.** G(30) over λc × qc, panels q × ρ, gate on and off, per setup
and target (phase diagrams), with the silent level and the dose next to it.

**Q3 — The controller rate.** G(30) along λc at fixed everything else, with messages
and the time the budget runs out. Is there a rate beyond which more is worse (the
counterpart of Simulation 1's large-b backfire)? Criteria in §6.

**Q4 — Persistence.** G(40) − G(30) across the grid; G(t) over time for the main cells.

**Q5 — The vote gate.** Gate off − gate on: gain and messages, across λc, qc and q.

**Q6 — The proof stop.** Symmetric truth: stop off − stop on.

**Q7 — Sensing.** v̂ against the true current share, across qc, λc and the gate.

**Q8 — Mechanisms** (exploratory, the 52 main cells): proof rates (own evidence vs
any), which facts are held and posted, the board window's composition, new facts vs
reactivations.

---

## 5. Hypotheses from the bridge's first look

Decided on the confirmation run (§7). Numbers in brackets are the first run's point
estimates (q 6, qc 12, ρ 0.75, λc 1).

| # | hypothesis | supported if, on the confirmation run |
|---|---|---|
| H1 | the effect persists: G(40) ≈ G(30) | in every main cell, the interval of G(40) − G(30) lies within ±0.042 |
| H2 | the core weakens control of nosolution compared with Simulation 1's rules [truth 0.42 → 0.29; false 0.27 → 0.19] | core − S0 < −0.042, interval below 0, each target |
| H3 | the core weakens the false controller on symmetric [0.22 → 0.13] | as H2 |
| H4 | the silent symmetric population believes less in A0 in the core [m_A0(30) 0.84 → 0.72] | core − S0 < −0.042, interval below 0 |
| H5 | gate off roughly triples messages on nosolution without raising the gain [≈ 12 → 28 messages] | messages rise; interval of the gate effect within ±0.042 |
| H6 | gate off raises the truth gain on symmetric [0.08 → 0.14] | gate effect > 0.042, interval above 0 |

H2–H4 compare with S0 of a seed-2027 bridge (S0 cells only). Anything else found is
**exploratory**, reported as such, and confirmed on seed 2027 before it is believed.

---

## 6. What counts as a finding

- **An effect**: its 95% interval excludes 0 and it is at least 0.042.
- **No effect**: the whole interval lies within ±0.042. Otherwise **inconclusive**.
- **A non-monotonicity** (e.g. a higher rate doing worse) requires: the reversal larger
  than its interval; the same reversal at a neighbouring q or qc; not explained by dose
  or the budget running out; and it survives the confirmation run.
- Bootstrap: 1,000 resamples of episode numbers, shared by all cells with the same q
  and ρ; percentile intervals.

---

## 7. The confirmation run

`sim2_grid_confirm_seed2027`: the 52 main cells (q 6, qc 12, ρ 0.75, every rate, gate and
stop) and the bridge's six S0 cells, with seed 2027. Further cells are added for any
exploratory finding outside the main cells. A few minutes.

---

## 8. Known confounders

1. **The silent baseline differs between core and Simulation 1** (Q1): every gain is shown
   with the controlled and silent levels.
2. **The budget**: at high rates it runs out early. Gains are always shown with messages,
   the share of episodes that spent the budget, and when.
3. **The gate makes the dose depend on the population.** Compare gate arms with messages.
4. **qc and the gate**: with θ = 0.75 the controller stays silent at ≥ 3 of 3 (qc = 3),
   ≥ 5 of 6, ≥ 9 of 12, ≥ 18 of 24; only 12 vs 24 isolates reading amount.
5. **Old votes on the board**: posts keep the vote of their author at posting time, so the
   gate reads old opinions (Q7 measures it).
6. **λc is not exactly Simulation 1's b**: b facts arrive together each night with no
   budget; λc facts arrive one by one with a budget of 30.

---

## 9. Figures

- **F1 — phase diagrams**: tiles λc (x) × qc (y), G(30) printed, intervals including 0 greyed;
  panels q × ρ; one figure per setup × target × gate (stop on); messages as a companion.
- **F2 — rate curves**: G(30) and messages against λc, one line per qc, panels q × ρ.
- **F3 — gains over time** for the main cells: G(t), 0–40, one line per rate.
- **F4 — Simulation 1 vs core**: silent trajectories, and G(30) per q, qc, ρ.
- **F5 — gate and stop effects**: gate off − on against λc; stop off − on.
- **F6 — sensing**: v̂ − true share.

Written to `results/simulation_2/analysis/<date>/figures/` with an index.
