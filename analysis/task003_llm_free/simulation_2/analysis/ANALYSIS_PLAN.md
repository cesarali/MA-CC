# Analysis plan — Simulation 2, the bridge study

**Status: draft for review, 9 October 2026.**

**What has already been seen.** Unlike Simulation 1's plan, this one is written
after a first look: the bridge run's `summary.csv` (cell means at t = 30 and 40,
no intervals) and the S0 check. Everything suggested by that look is therefore
listed as a **hypothesis** (§5) and is tested on a **fresh run with a new seed**
(§7), not on the data that suggested it. The first run is used for estimation
and figures; the fresh run decides which hypotheses hold.

Run covered (listed in [`../simulator/runs_index.csv`](../simulator/runs_index.csv)):
`2026-10-09_sim2_bridge`. Settings: [`../simulator/configs/sim2_bridge.yaml`](../simulator/configs/sim2_bridge.yaml);
rules: [`../simulator/simulation_2_spec.md`](../simulator/simulation_2_spec.md).

| step | change from the step before |
|---|---|
| S0 | none: Simulation 1's rules (b = 1) in the new code |
| S1 | board kept (readers sample the 48 most recent posts) |
| S2 | continuous forgetting instead of dawn forgetting |
| S3 | agents act at random times |
| S4 | controller acts at random times, λc = 1 |
| S5 | vote gate off |
| S6 | proof stop off (= the rate scan's cell at λc = 1) |

Each step: 2 setups × 3 arms (silent, truth = target A0, false = target A2);
S4–S6 reuse S3's silent cells. 1,000 episodes per cell; t from 0 to 40; the
controller acts only for t < 30; measured every 0.25.

---

## 1. Principles

The same as Simulation 1's plan, with one addition (4):

1. **Questions first, in a fixed order** (§4).
2. **Never pool the two setups.** They are different regimes.
3. **Paired comparisons with 95% bootstrap intervals**, not p-values.
4. **Pairing goes across steps too.** Episode *e* uses the same keyed random
   draws in every arm and every step, wherever the rules use the same draw. So a
   step-to-step difference is taken within episode *e*, and the bootstrap
   resamples **episode numbers**, keeping all arms and all steps of an episode
   together. This makes step differences more precise, not biased.
5. **Time is part of every result**: trajectories over 0–40, not only t = 30.

---

## 2. Data products (built once)

From each cell's `params.json` and Parquet files, never from folder names.

| table | one row per | columns |
|---|---|---|
| `cells` | cell | step, setup, arm, the six step settings, λc, silent cell it is paired with |
| `trajectories` | cell × episode × measurement time | m_A0, m_A1, m_A2 (mean beliefs), vote shares, share voted, proof rates (two ways), mean active facts, board size, messages and reads so far |
| `paired` | controlled cell × episode × time | controlled minus its silent cell, same episode (§3) |
| `controller` | cell × episode × controller action | v̂, true current target share, decision, budget |

Written to `results/simulation_2/analysis/<date>/tables/` (not in git).

---

## 3. Estimands

Notation: m_k(t) = mean belief in allocation k over the 24 agents at time t.
"Target" = A0 for truth cells, A2 for false cells.

| name | definition |
|---|---|
| **gain** G(t) | mean over episodes of m_target(t) controlled − m_target(t) silent, same episode. **Primary: G(30).** |
| **persistence** | G(40), and G(40) − G(30): how much of the effect is left 10 time units after the controller stops |
| **average gain** | G averaged over [0, 30] on the measurement grid (trapezoid rule; a grid approximation) |
| **step effect** Δ_k | G(30) in step S_k minus G(30) in S_{k−1}, same setup and target, same episodes |
| **shared / directional parts** | M = (G_truth + G_false)/2, D = (G_truth − G_false)/2, per step, measured on m_A0 and on m_A2 (as in Simulation 1) |
| **vote gain** | the same as G, with the share of last votes for the target instead of m_target |
| **dose** | controller messages per episode; reads per episode; share of episodes that spend the whole budget, and when |
| **efficiency** | 24 · G(30) / messages: extra "agents' worth" of belief per message (ratio of means, with paired bootstrap) |
| **sensing error** | v̂ − true current target share, per controller action |
| **silent level** | m_A0(t), m_A2(t) of the silent cells, per step |

**Size that matters:** 1/24 ≈ 0.042, one agent's worth of belief. An effect
smaller than that is reported but not called a finding.

---

## 4. Questions, in order

**B0 — Sanity at scale** (all 36 cells).
- the S0 check (done: passed, 1.9% of |z| > 2, max |z| = 2.72);
- invariants over every episode: messages ≤ 30; no controller action at t ≥ 30;
  agents act in the follow-up; no self-reads; reads only from the 48-post window;
  every fact read is active afterwards; each forgetting removes an active fact;
- replay of logged reads and forgetting reproduces the recorded memories, on
  50 episodes per cell;
- S6 equals S5 exactly in every cell except symmetric truth (only there can the
  proof stop act).

**B1 — The silent population across steps.** How each change alters the
population without a controller: silent m_A0(t), m_A2(t), proof rates and mean
active facts, S0 → S3. This is the baseline the gains are measured against.

**B2 — Step effects on control.** For each setup and target: G(30) per step
with intervals, and each step effect Δ_k. Which changes matter, in which
direction, and how much.

**B3 — Persistence.** G(t) over 30–40 per step: does the effect fade after the
controller stops?

**B4 — Dose and efficiency.** Messages, reads and efficiency per step. In
particular S4 → S5 (gate off): more messages, but more effect?

**B5 — Sensing.** v̂ against the true current share per step. S1 is where old
votes enter the controller's sample (spec §3); how large is the error, and does
it explain part of B2?

**B6 — Mechanisms** (exploratory, after B1–B5): for the largest step effects,
look at proof rates (two ways), which facts are held and posted, the board
window's composition, and new facts against reactivations.

---

## 5. Hypotheses from the first look, to be tested on the fresh run

Each is stated with what would count as support (interval and the 0.042 size
rule, §6). Numbers in brackets are the first run's point estimates.

| # | hypothesis | supported if, on the fresh run |
|---|---|---|
| H1 | the effect persists: G(40) ≈ G(30) | in every controlled cell, the interval of G(40) − G(30) lies within ±0.042 |
| H2 | keeping the board (S1) lowers the false controller's gain on symmetric [0.22 → 0.09] | Δ_1 < −0.042 with interval below 0 |
| H3 | continuous forgetting (S2) lowers both gains on nosolution [truth 0.41 → 0.33; false 0.25 → 0.22] | Δ_2 < 0 with interval below 0, for each target; called large only if < −0.042 |
| H4 | random agent timing (S3) lowers the nosolution gains further [0.33 → 0.29; 0.22 → 0.19] | as H3, for Δ_3 |
| H5 | random controller timing (S4) changes nothing | interval of Δ_4 within ±0.042 in all four setup × target |
| H6 | gate off (S5) roughly triples messages on nosolution without raising the gain [≈ 12 → 28 messages] | messages rise; interval of Δ_5 within ±0.042 on nosolution |
| H7 | gate off (S5) raises the truth gain on symmetric [0.08 → 0.14] | Δ_5 > 0.042 with interval above 0 |
| H8 | the silent symmetric population loses belief in A0 with continuous forgetting and random timing [m_A0(30): 0.88 → 0.72, S1 → S3] | silent m_A0(30) falls by more than 0.042 from S1 to S3, interval below 0 |

Anything else found during the analysis is **exploratory** and goes to a
further confirmation run before it is believed.

---

## 6. What counts as a finding

- **An effect**: its 95% interval excludes 0 and it is at least 0.042.
- **No effect**: the whole interval lies within ±0.042. An interval that
  includes 0 but extends past ±0.042 is **inconclusive**, not "no effect".
- **A step effect is about this ladder**: it measures the change given the
  steps before it. It does not say what the same change would do from another
  starting point (§8.2).
- Bootstrap: 1,000 resamples of episode numbers, keeping every arm and step of
  an episode together; percentile intervals.

---

## 7. The confirmation run

**Before any of §4 is analysed**, rerun the whole bridge with `seed: 2027`
(new study `sim2_bridge_confirm_seed2027`, 36 cells, about 3 minutes). Its
draws are unrelated to the first run's, so H1–H8 are tested on data that did not
suggest them. Both runs are reported; a hypothesis holds only if it holds on the
confirmation run.

---

## 8. Known confounders and how each is handled

1. **The silent baseline moves between steps** (B1). A change in G can come from
   the silent level moving toward a floor or ceiling. Every gain is shown with
   the controlled and silent levels next to it.
2. **Order of the ladder.** Each step effect is conditional on the steps before
   it. If a step effect is large (H2, H3), an exploratory check adds that one
   change directly to S0 (e.g. S0 + continuous forgetting only), to see if the
   effect depends on the order. Labelled exploratory.
3. **Dose differs between steps.** The gate makes the number of messages depend
   on the population (S0–S4), and turning it off (S5) changes the dose. Gains are
   always shown with messages and efficiency.
4. **S1 couples two changes**: kept evidence and old votes in the gate's sample.
   B5 measures the second.
5. **Days vs random timing.** Same average activity per time unit, but more
   spread: some agents act twice in a unit, some not at all. B1 reports the share
   of agents that have voted, and mean active facts.
6. **Setups are never pooled**, and their effects are never subtracted.

---

## 9. Figures

Fixed layout and colour scales within a family; intervals always drawn; the
silent level shown wherever a gain is.

- **F1 — the ladder.** x = step S0 … S6, y = G(30) with intervals, G(40) as a
  second marker; panels setup × target. The figure that answers B2 and B3 at a glance.
- **F2 — gains over time.** G(t) for t = 0 … 40, one line per step, a vertical
  line at t = 30; panels setup × target.
- **F3 — silent populations.** m_A0(t), m_A2(t) per step S0–S3; panels per setup.
- **F4 — dose and efficiency per step**, with the share of episodes that spend
  the whole budget.
- **F5 — sensing.** v̂ − true share per step, over time.

Written to `results/simulation_2/analysis/<date>/figures/` with an index file.

---

## 10. After the bridge: the rate scan

Planned separately once B1–B6 are read, following the source document's §8 and
§10: S6's settings with λc ∈ {0.5, 1, 2, 4, 8}, both setups; primary outcome
G(30); messages, reads and the time the budget runs out reported together;
never conditioning on episodes that spent the budget; any "best rate" is best
among the five scanned and is rechecked on fresh seeds.
