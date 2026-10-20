# Step 0: data audit (Simulation 2)

**Date:** 20 October 2026. **Scope:** Simulation 2 (continuous time) only, setups `task003-symmetric` (proof-bearing)
and `task003-nosolution` (no proof). Simulation 1 is out of scope. **Source plan:** [`00_handoff_2026-10-20.md`](00_handoff_2026-10-20.md).

**In one paragraph.** There are 10,154 Simulation 2 settings on disk, all with 1,000 episodes and all ending at t = 40.
Every setting has light records: vote shares, mean beliefs and proof rates every 0.25 time units, plus every
controller action. A few settings also have full records: each agent's remembered facts, and every post with its
author. Random draws are shared across arms, q, ρ and every other setting. So paired comparisons across q are valid.
Both tasks have all 8 memory values. The two real gaps are these:
- full records exist only at q = 6 (one exception: board posts at three q values for no-proof at ρ = 1), and only at ρ ≥ 0.75;
- no run goes beyond t = 40.

Reproduce the inventory with [`scripts/s0_inventory.py`](scripts/s0_inventory.py). It writes
`results/key_phenomena/inventory_cells.csv` and `inventory_summary.csv`.

## 1. Inventory matrix

Recording levels:
- **light**: `snapshots` (every 0.25 time units: vote shares over all agents, share who have voted, mean beliefs, proof rates, mean remembered facts, board size, messages and reads so far), `controller` (one row per controller action: time, posts read, estimated and true target share, decision, posted fact), `episodes` (totals).
- **posts**: light + `posts` (every post: time, author, fact, author's vote).
- **full**: posts + `actions` (every agent action: memory before and after as a fact bitmask, posts read, belief, vote, posted fact), `agent_snapshots` (every agent's memory and last vote every 0.25), `forgetting`.

All runs: B = 30 unless stated, W = 48 unless stated, t_control = 30, t_end = 40, 1,000 episodes per setting,
seed 20261007 unless stated.

| run (`results/simulation_2/`) | setup | level | ρ | q | q_c | λc | gate | other |
|---|---|---|---|---|---|---|---|---|
| `2026-10-15_sim2_symmetric_fine` | symmetric | light | 1, .75, .5, .25 | 3, 6, 12 | 3, 6, 12, 24 | 0.5, 1–8 (9) | on, off | 1,740 settings |
| `2026-10-16_sim2_symmetric_fine_extra_rho` | symmetric | light | .875, .625, .375, .125 | 3, 6, 12 | 3, 6, 12, 24 | 9 rates | on, off | 1,740 settings |
| `2026-10-16_sim2_nosolution_fine` | nosolution | light | all 8 | 3, 6, 12 | 3, 6, 12, 24 | 9 rates | on, off | 3,480 settings |
| `2026-10-18_sim2_<setup>_budget100` | both | light | 1, .75, .5, .25 | 3, 6, 12 | 12 | 9 rates | on, off | B = 100; 444 each |
| `2026-10-18_sim2_<setup>_window24` | both | light | 1, .75, .5, .25 | 3, 6, 12 | 12 | 9 rates | on, off | W = 24; 444 each |
| `2026-10-18_sim2_<setup>_window240` | both | light | 1, .75, .5, .25 | 3, 6, 12 | 12 | 9 rates | on, off | W = 240; 444 each |
| `2026-10-09_sim2_grid` | symmetric | **full** | .75 | **6** | 12 | 0.5, 1, 2, 4, 8 | on, off | false and silent arms; 240 more light truth-arm settings at ρ 1, .75 |
| `2026-10-12_sim2_main_rho1_full` | both | **full** | 1 | **6** | 12 | 0.5, 1, 2, 4, 8 | on, off | false, truth and silent arms |
| `2026-10-09_sim2_grid_confirm_seed2027` | both | **full** | .75 | **6** | 12 | 0.5, 1, 2, 4, 8 | on, off | seed 2027 |
| `2026-10-09_sim2_rho1_confirm_seed2027` | nosolution | **posts** | 1 | **3, 6, 12** | 6, 12 | 0.5, 1, 2, 4, 8 | on, off | seed 2027 |
| `2026-10-09_sim2_bridge`, `_s0_confirm_seed2027` | both | full | .75 | 6 | 12 | 1 | on, off | partial models (steps S0–S6 that add the core rules one at a time); not the full core |

Built tables (`results/gold_standard/`): `2026-10-16/` (both setups, 8 ρ, all q_c; it also contains
Simulation 1 rows, so filter `sim == "sim2"`) and `2026-10-18/` (the variants, column `case`).

## 2. What the code guarantees

- **Pairing.** Every random stream is `keyed_rng(seed, episode, purpose, ...)`, where the purpose is something like
  `"agent_clock"`, `"reads"`, `"vote"` or `"life"` ([simulation_2.py:121](../simulator/sim2/simulation_2.py#L121)).
  The key never contains the arm, q, ρ, q_c, rate, window or budget. So episode e shares its clocks, starting order and
  random lifetimes across all settings of the same seed. Paired episode-level differences are therefore valid
  **across q** too, not only between arms. Over time the trajectories diverge, so the pairing weakens but stays unbiased.
  Measured gain from pairing at t = 30: errors are 1.00–1.63 times smaller (median 1.15).
- **Forgetting.** Each remembered fact has its own lifetime, drawn from Exponential(γ) with γ = −ln ρ. Re-reading the
  fact renews it. **An agent's own starting fact is forgotten too**: its lifetime starts at t = 0 (spec §5). An agent
  that stops reading therefore drifts to an empty memory and the prior (1/3, 1/3, 1/3).
- **Window.** Posts are never deleted. Every reader samples from the W newest posts it may read. Agents skip their own
  posts; the controller reads agent posts only.
- **Horizon.** Every run has t_end = 40. No data exist beyond 40. `t_end` is a config setting, so longer runs need no
  code change.
- **Last controller post t\*.** Available in every run, light ones included: it is the largest `t` in
  `controller.parquet` with `post_id >= 0`. **Distinct facts delivered** come from the same table (`posted_fact`).
- **Fact classes** (from the setup files): *agent-origin* facts are those the agents start with (18 symmetric, 15
  no-proof). *Controller-only* facts are those in the A2 pool that no agent starts with: 10 of 12 in symmetric, all 12
  in no-proof.

## 3. Corrections to the handoff

1. **Both tasks have all 8 memory values** (1, 0.875, …, 0.125), at every q, q_c, rate and gate. The handoff says
   only no-proof does. The symmetric extra values come from `2026-10-16_sim2_symmetric_fine_extra_rho`.
2. **Pairing across q is verified from the code** (section 2).
3. **Full records are thinner than an earlier oral summary claimed.** They exist at **q = 6 only**, at ρ = 0.75 and 1,
   for both tasks. The one exception: no-proof at ρ = 1 has board posts (not memories) at q = 3, 6, 12 (seed 2027).
   Nothing has full records at ρ ≤ 0.5.
4. **Campaign B's branch (ii) ("communication interrupted") has a predictable outcome.** Agents' own facts are
   forgotten too, so without reading every memory empties at rate γ and both arms converge to the prior. The branch is
   a useful passive reference, but it cannot identify the carrier.
5. **The q_c and gate conclusions hold, with one exception.** q_c changes the mean G over rates by at most 0.06
   (symmetric) and 0.05 (no-proof, ρ < 1). In **no-proof at ρ = 1, however, q_c moves G by up to 0.087**
   (0.083–0.170, highest at q_c = 3 and 24), and the gate by up to −0.083. Both effects sit in the backfire regime.
   The handoff's q_c means are confirmed (symmetric ρ = 0.5: 0.28, 0.33, 0.34, 0.31; no-proof: 0.30, 0.28, 0.29, 0.30).
6. The report figure numbers in the handoff refer to the 19 October versions. The 17 October no-proof report and the
   19 October gold-standard copy are in `simulation_2/analysis/reports/`.

## 4. Resolved / not enough data / open (before step 1)

| question | existing data | status before step 1 | what step 1 did |
|---|---|---|---|
| Reading reversal exists | 8 ρ × 3 q × both tasks, paired | resolvable from light data | **resolved** (02, §1) |
| Location and uncertainty of the crossover | as above; ρ step 0.125, q ∈ {3, 6, 12} | resolvable at this resolution | **resolved** (02, §1) |
| Reversal explained by dose (messages) | messages per setting; gate off gives equal dose | resolvable | **resolved: no** (02, §2) |
| Evidence integration (memory coverage, beliefs, entropy) vs q | full records at q = 6 only | **not enough data** | described at q = 6 (02, §3); q dependence needs Campaign A |
| Board composition vs q | posts at q = 3, 6, 12 for no-proof ρ = 1 only | partial | done for that slice (02, §3) |
| Persistence of the population effect | light data to t = 40; fast rates stop early | resolvable within about 35 time units | **resolved within the window** (02, §4) |
| Recirculation of controller facts after stop | full records q = 6, ρ ≥ 0.75 | partial | **shown at ρ ≥ 0.75** (02, §3–4); ρ ≤ 0.5 needs new runs |
| Carrier of persistence (facts vs state vs communication) | none: needs an intervention | **open** | Campaign B |
| Relaxation beyond t = 40 | none | **open** | Campaign B (longer t_end) |
| Dose saturation / backfire | rate × budget grids, controller tables | resolvable | **resolved descriptively** (02, §5) |
| q_c sensitivity | 4 q_c × all | resolved: small except no-proof ρ = 1 | no new runs |
| Gate sensitivity | gate on/off × all | resolved: small except no-proof ρ ≥ 0.875 | no new runs |
| Board window | W = 24, 48, 240 at 4 ρ | resolved: reversal and persistence survive | no new runs |

**Not needed** (unless later evidence contradicts this): new q_c, rate, budget, window or gate sweeps, η/I surveys,
N (swarm-size) scaling, or Simulation 1.
