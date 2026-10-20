# Step 1: reanalysis of existing data (Simulation 2, no new simulations)

**Date:** 20 October 2026. **Data:** the runs listed in [`01_data_audit.md`](01_data_audit.md). **Main setting**
unless stated: q_c = 12, λc = 1, gate on, B = 30, W = 48, false controller (target A2), t = 30, 1,000 paired episodes.

**Notation.**
- G = paired susceptibility of the false controller: the mean over episodes of the A2 vote share with the controller
  minus the same episode's A2 share without it (shares among agents who have voted).
- D(q₁→q₂) = G(q₂) − G(q₁), the change in G when agents read more posts. It is computed per episode, because
  episode e is paired across q (audit §2). 95% intervals are ±1.96 × its standard error.
- δ(s) = the paired effect s time units after the controller's last post t\*.

## Answers first

1. **The reading reversal is real, robust and has the same shape in both tasks.** Reading more helps the false
   controller when agents forget and protects the swarm when they remember. The crossover moves to *lower* memory as
   reading grows, and it sits about **0.33 higher in ρ without a proof**. No dose explains it.
2. **The reversal is also a reversal in time.** Early on, reading more almost always helps the controller. The
   protective effect builds up over tens of time units.
3. **Persistence is longest where memory is shortest.** At ρ ≤ 0.5 the effect does not decay at all for more than
   35 time units after the last post, which is 35–100 unread-fact half-lives. At higher memory it relaxes partly, to a
   nonzero plateau. With a proof and full memory it disappears within about 2 time units.
4. **Controller-only facts keep circulating after the controller stops.** 10 time units later they still make up
   the same share of readable posts (about 41–44% in no-proof, 27% in symmetric at ρ = 0.75). The controller's facts
   also crowd the agents' own facts out of memory and off the board.
5. **The backfire is specific to full memory without a proof.** More messages and more distinct facts pull beliefs
   back toward the silent swarm's.

Mechanism status: the "integration against selective reinforcement" hypothesis is consistent with everything below
and contradicted by nothing. But its key test, how memory composition changes with q at ρ ≤ 0.5, needs data we do not
have (Campaign A). The carrier of persistence needs an intervention (Campaign B).

Scripts are in [`scripts/`](scripts/). Large outputs go to `results/key_phenomena/` (not in git). Small tables are in
[`tables/`](tables/) and figures in [`figures/`](figures/).

---

## 1. The reading reversal ([`s1_reversal.py`](scripts/s1_reversal.py), [Fig. 1](figures/fig1_reversal_map.pdf))

G at t = 30, main setting, and the paired increments (± is the 95% half-width). From [`tables/reversal_t30.csv`](tables/reversal_t30.csv).

| ρ | symmetric G(3, 6, 12) | D(3→6) | D(6→12) | no-proof G(3, 6, 12) | D(3→6) | D(6→12) |
|---|---|---|---|---|---|---|
| 1 | 0.00, 0.00, 0.00 | 0 | 0 | 0.19, 0.14, 0.11 | −0.047 ± 0.009 | −0.035 ± 0.010 |
| 0.875 | 0.10, 0.04, 0.01 | −0.058 ± 0.009 | −0.028 ± 0.007 | 0.24, 0.24, 0.21 | −0.001 ± 0.012 | −0.030 ± 0.012 |
| 0.75 | 0.24, 0.11, 0.04 | −0.127 ± 0.014 | −0.072 ± 0.010 | 0.22, 0.25, 0.23 | +0.033 ± 0.011 | −0.017 ± 0.011 |
| 0.625 | 0.29, 0.22, 0.11 | −0.070 ± 0.016 | −0.111 ± 0.013 | 0.22, 0.28, 0.27 | +0.059 ± 0.012 | −0.011 ± 0.011 |
| 0.5 | 0.28, 0.33, 0.18 | +0.045 ± 0.016 | −0.146 ± 0.016 | 0.20, 0.29, 0.30 | +0.092 ± 0.012 | +0.009 ± 0.011 |
| 0.375 | 0.27, 0.41, 0.30 | +0.145 ± 0.015 | −0.115 ± 0.018 | 0.18, 0.30, 0.33 | +0.120 ± 0.012 | +0.030 ± 0.011 |
| 0.25 | 0.24, 0.43, 0.40 | +0.193 ± 0.014 | −0.036 ± 0.017 | 0.16, 0.30, 0.37 | +0.137 ± 0.012 | +0.069 ± 0.010 |
| 0.125 | 0.21, 0.44, 0.49 | +0.223 ± 0.013 | +0.049 ± 0.016 | 0.15, 0.30, 0.39 | +0.150 ± 0.011 | +0.090 ± 0.009 |

**Crossovers** (linear interpolation between neighbouring ρ; resolution ±0.06 in ρ):

| step | symmetric | no-proof | shift |
|---|---|---|---|
| D(3→6) changes sign | ρ ≈ 0.55 | ρ ≈ 0.87 | +0.32 |
| D(6→12) changes sign | ρ ≈ 0.20 | ρ ≈ 0.55 | +0.35 |

- **The best reading level for the controller falls as memory grows.** Symmetric: q = 12 at ρ = 0.125, q = 6 at
  0.25–0.5, q = 3 at ρ ≥ 0.625. No-proof: q = 12 at ρ ≤ 0.5, q = 6 at 0.625–0.875, q = 3 at ρ = 1. The crossover is a
  sloped contour in the (ρ, q) plane, not a vertical line at one ρ. With a proof the contour sits about 0.33 lower in ρ.
- **The handoff's no-proof "+0.01" at ρ = 0.75 is resolved.** D(3→12) = +0.016 ± 0.012: a significant rise from 3 to
  6 (+0.033) followed by a fall from 6 to 12 (−0.017). G peaks at q = 6.
- **The signs are robust** ([`tables/reversal_sign_counts.csv`](tables/reversal_sign_counts.csv)). Across all
  72 combinations of q_c × rate × gate, every significant increment has the main setting's sign, except inside the
  crossover cells: symmetric D(6→12) at ρ = 0.25 (12 positive, 27 not significant, 33 negative); no-proof D(3→6) at
  ρ = 0.875 (12, 24, 36); no-proof D(6→12) at ρ 0.5–1 (mostly not significant or negative).
- **Window and budget move the contour only a little** (q_c = 12, λc = 1). The signs are the same in all four cases,
  except symmetric D(6→12) at ρ = 0.25: −0.036 (W = 48), −0.001 (W = 24), +0.030 (W = 240). So a longer window pushes
  that crossover to slightly higher ρ.

## 2. Dose does not explain the reversal

- With the gate on, reading changes how many messages the controller posts, and not always in the direction of G.
  Example, no-proof at ρ = 0.25: 22.2, 14.0, 9.3 messages for q = 3, 6, 12, yet G rises 0.16 → 0.37.
- **With the gate off, every q posts exactly 27.9 messages**, and the increments keep their sign wherever it is clear, changing by at most 0.04:

| ρ | symmetric D(3→6) on / off | D(6→12) on / off | no-proof D(3→6) on / off | D(6→12) on / off |
|---|---|---|---|---|
| 0.875 | −0.058 / −0.064 | −0.028 / −0.025 | −0.000 / −0.040 | −0.031 / −0.057 |
| 0.75 | −0.126 / −0.130 | −0.072 / −0.073 | +0.033 / +0.010 | −0.017 / −0.053 |
| 0.5 | +0.045 / +0.038 | −0.146 / −0.137 | +0.092 / +0.083 | +0.009 / −0.005 |
| 0.25 | +0.193 / +0.196 | −0.036 / −0.024 | +0.137 / +0.143 | +0.068 / +0.064 |

- At λc = 8 (B = 30) almost every q posts 22–30 messages, and the signs are again the same.

## 3. The reversal over time ([Fig. 2](figures/fig2_reversal_time.pdf))

- **At t = 2 every increment is positive, in both tasks**, except symmetric D(6→12) at ρ ≥ 0.5 (−0.002 to −0.016).
  At first, reading more just spreads whatever is on the board faster, the controller's facts included.
- **The protective sign develops over the next tens of time units.** Examples: no-proof ρ = 1, D(3→6): +0.052 at
  t = 5, −0.039 at t = 20, −0.047 at t = 30. Symmetric ρ = 0.75, D(3→6): −0.005 at t = 5, −0.108 at t = 20.
  Symmetric ρ = 0.375, D(6→12): +0.003 at t = 5, −0.115 at t = 30.
- So the "integration" side of reading needs time and memory to build up; the "spreading" side acts at once. This fits
  the two-function hypothesis of the handoff (§3.2). It does not prove it.

## 4. Memory and board composition ([`s1_mechanism.py`](scripts/s1_mechanism.py), [`tables/mechanism_t30_t40.csv`](tables/mechanism_t30_t40.csv))

Existing full records, q = 6, λc = 1, gate on; per-agent means, false arm against silent arm, t = 30.

| | symmetric ρ = 0.75 | symmetric ρ = 1 | no-proof ρ = 0.75 | no-proof ρ = 1 |
|---|---|---|---|---|
| agent-origin facts remembered | 4.7 vs 6.2 | 11.3 vs 12.2 | 3.8 vs 6.1 | 11.3 vs 12.6 |
| controller-only facts remembered | 2.0 | 2.2 | 2.1 | 3.9 |
| agents holding ≥ 1 controller-only fact | 90% | 100% | 92% | 100% |
| distinct agent-origin facts held by anyone (of 18 / 15) | 9.4 vs 11.2 | 18 vs 18 | 7.8 vs 10.4 | 15 vs 15 |
| mean belief in A2 | 0.32 vs 0.19 | 0.00 vs 0.00 | 0.67 vs 0.48 | 0.64 vs 0.50 |
| share proving A0 | 15.5% vs 12.0% | 99.9% vs 99.9% | — | — |
| readable posts with a controller-only fact | 29% | 10% | 41% | 44% |
| distinct agent-origin facts among readable posts | 8.0 vs 10.4 | 4.8 vs 6.3 | 6.4 vs 10.0 | 6.2 vs 9.8 |

- **The controller crowds out the agents' evidence.** At ρ = 0.75 agents remember 1.5–2.3 fewer of their own facts,
  and the swarm as a whole holds 1.8–2.6 fewer distinct agent-origin facts. This is the "selective reinforcement" side.
- **With full memory everyone ends up holding every agent fact** (coverage 18/18 and 15/15 in both arms). In symmetric
  that is a proof, and the controller's effect is zero. In no-proof, agents also hold 3.9 controller-only facts, and
  their belief in A2 relaxes from 0.64 at t = 30 to 0.61 at t = 40, toward the silent 0.50. This is the "integration"
  side: what remains is a slowly closing gap.
- **The board does not show why reading protects at ρ = 1** (no-proof, posts at q = 3, 6, 12, seed 2027): the share of
  readable posts carrying controller-only facts *rises* with q (0.42, 0.44, 0.45 at t = 30), and distinct agent facts
  on the board fall (6.6, 6.2, 6.0). Yet G falls with q (0.19 → 0.11). So the protection must sit in memory (agents who
  read more hold more of everything and integrate it), not in the board. **That needs full records at several q:
  Campaign A.**
- Curiosity: in symmetric at ρ = 0.75, slightly *more* agents prove A0 with the false controller than without it
  (15.5% vs 12.0%). The A2 pool shares 2 facts with the agents, and other pool facts may help complete proofs. Not
  pursued.

## 5. Persistence after the controller stops ([`s1_persistence.py`](scripts/s1_persistence.py), [Fig. 3](figures/fig3_persistence.pdf), [`tables/persistence.csv`](tables/persistence.csv))

δ(s) is the paired effect s time units after each episode's own last post t\* (same shift for the silent arm),
averaged over the episodes still observed at t\* + s ≤ 40. Only s with ≥ 100 episodes are used. Fast controllers
(λc = 8, B = 30) stop early: median t\* ≈ 4–6 in symmetric, 9–10 in no-proof, because the gate interrupts. This gives
about 30–36 observed time units after the last post.

| λc = 8, W = 48 | half-time (s) | δ(10)/δ(0) | δ(30)/δ(0) | unread-fact half-life |
|---|---|---|---|---|
| symmetric ρ = 0.125–0.5 | not reached (> 36) | 0.97–1.08 | 0.87–1.07 | 0.33–1.0 |
| symmetric ρ = 0.625 | 35 | 0.78 | 0.62 | 1.5 |
| symmetric ρ = 0.75 | 11.5 | 0.52 | 0.37 | 2.4 |
| symmetric ρ = 0.875 | 3.5 | 0.24 | 0.18 | 5.2 |
| symmetric ρ = 1 | 1.75 | 0.00 | 0.00 | ∞ |
| no-proof ρ = 0.125–0.625 | not reached (> 35) | 0.96–1.05 | 0.96–1.14 | 0.33–1.5 |
| no-proof ρ = 0.75 | not reached | 0.93 | 0.80 | 2.4 |
| no-proof ρ = 0.875 | not reached | 0.77 | 0.57 | 5.2 |
| no-proof ρ = 1 | 5.0 | 0.30 | 0.17 | ∞ |

- **Short memory, no relaxation.** For ρ ≤ 0.5 in both tasks the effect is flat for more than 35 time units after the
  last post, against an unread-fact half-life of 0.3–1 and a readable-post lifetime of about 2. This holds for
  W = 24 and W = 240 too (δ(30)/δ(0) = 0.75–1.2).
- **Longer memory, partial relaxation to a plateau.** For example symmetric ρ = 0.75 falls to about 0.37 and no-proof
  ρ = 1 to about 0.17, both still flat at the end of the window. These are plateaus within 36 time units. We cannot say
  whether they are permanent or very slow transients: there are no data beyond t = 40.
- **A proof erases the effect quickly**: symmetric ρ = 1 returns to zero within 2 time units.
- So the conditions that let agents integrate evidence (long memory, a proof, a long window) are the same ones that
  let the effect relax. This links the two priorities: one competition, seen as a reversal in q and as persistence in
  time.
- **Fixed clock:** δ(40) − δ(30) is between −0.03 and +0.00 for every setting at λc = 1 (paired errors about 0.005).
- **Recirculation at ρ ≥ 0.75** (full records): from t = 30 to t = 40 the share of readable posts carrying a
  controller-only fact stays at 0.41 → 0.41 (no-proof ρ = 0.75) and 0.44 → 0.45 (no-proof ρ = 1), and falls only from
  0.29 → 0.27 (symmetric ρ = 0.75) and 0.10 → 0.07 (symmetric ρ = 1). After t = 30 only agents post, so these are
  agents reposting the controller's facts. Agents' memories keep about 2 controller-only facts each.
- **What this does not settle:** whether the facts are the carrier (fact persistence) or only travel along with a
  biased state (state persistence). Recirculation is shown only at ρ ≥ 0.75, where the effect partly relaxes. The
  non-relaxing regime (ρ ≤ 0.5) has no full records. Both need Campaign B.

## 6. Backfire (secondary) ([`s1_backfire.py`](scripts/s1_backfire.py), [`tables/backfire.csv`](tables/backfire.csv))

No-proof, q = 6, gate on, t = 30.

| ρ | budget | λc 0.5 → 8: messages | distinct facts | G | mean belief A0 (false arm; silent) |
|---|---|---|---|---|---|
| 1 | 30 | 7.6 → 29.3 | 4.0 → 5.8 | 0.20 → 0.05 | 0.30 → 0.45 (0.50) |
| 1 | 100 | 7.6 → 90.6 | 4.0 → 6.6 | 0.20 → 0.02 | 0.30 → 0.47 (0.50) |
| 0.75 | 100 | 6.9 → 65.8 | 3.4 → 5.6 | 0.26 → 0.22 | 0.22 → 0.25 (0.38) |
| 0.5 | 100 | 7.2 → 65.0 | 3.5 → 5.2 | 0.28 → 0.30 | 0.23 → 0.23 (0.35) |

- With full memory, every extra message and extra distinct fact pulls the false arm's mean belief in A0 back toward
  the silent arm's 0.50. G falls almost to zero even though the controller posts 90 messages.
- With shorter memory the backfire is much weaker (ρ = 0.75: 0.26 → 0.22) or absent (ρ = 0.5: G is flat in messages).
- A likely reason, not tested: with full memory agents keep *all* the facts they meet. The more of the pool they hold,
  the closer they get to the posterior of everything available, (0.75, 0, 0.25), which favours A0. With short memory
  they hold only the most recent few, and those are the controller's picks.
- Secondary, as the handoff says. No new runs proposed for it.

## 7. Hypothesis assessment

| hypothesis | status |
|---|---|
| Reading has two competing effects: spreading the controller's selection (fast) and integrating complementary evidence (slow, needs memory) | **consistent** with the sloped contour (§1), the time reversal (§3), the crowding-out (§4) and the persistence pattern (§5). **Not yet tested directly**: needs memory composition against q at ρ ≤ 0.5 (Campaign A). |
| Proof reconstruction is required for protective reading | **rejected**: no-proof shows significantly protective reading at ρ = 1 (D(3→6)) and ρ ≥ 0.75 (D(6→12)), with no proof possible. |
| The reversal is a dose effect | **rejected** (§2). |
| A proof shifts the crossover | **consistent**: about 0.33 lower ρ with a proof. But the two worlds differ in more than the proof (handoff §2.2), so this is comparative, not causal. |
| Persistence needs a destroyed proof | **rejected**: no-proof persists as much or more. |
| Persistence is carried by recirculating controller facts | **supported at ρ ≥ 0.75** (§5); **open** for ρ ≤ 0.5 and as a causal claim (Campaign B). |
| A finite collective relaxation time exists | **open**: censored beyond 35 time units at ρ ≤ 0.5; plateaus at higher ρ; nothing beyond t = 40. |

**Paper framing:** one integrated paper still fits. Both phenomena appear in both tasks and look like two faces of one
competition between spreading and integrating. The backfire is a third face, at full memory without a proof.

## 8. What this changes for steps 2 and 3 (proposal, not run)

**Step 2 (Campaign A) becomes a mechanism run, not a mapping run.** Existing data already place the crossovers at a
ρ resolution of 0.125 with q ∈ {3, 6, 12}, with paired errors, in both tasks. What is missing is memory and board
composition against q at ρ ≤ 0.5.
- **Core:** full-record re-runs of existing settings: both tasks × ρ ∈ {1, 0.75, 0.5, 0.25} × q ∈ {3, 6, 12} ×
  {false, silent}, λc = 1, q_c = 12, gate on, W = 48, B = 30, seed 20261007. That is 48 settings, 4 of which already
  exist. Because the seed is the same, the light tables must reproduce the existing runs exactly, which validates
  the new runs. About 10 minutes and under 1 GB.
- **Test:** does the sign of D(q₁→q₂) follow the change, from q₁ to q₂, of (a) distinct agent-origin facts held by
  the swarm, (b) controller-only facts held per agent, (c) belief entropy, (d) controller-only share of the readable
  board, including over time (§3)?
- **Optional:** q = 4 and 8 with light records at all 8 ρ, if the shape of the contour (smooth or steep) matters for
  the paper. 64 settings, about 5 minutes.

**Step 3 (Campaign B) targets the two regimes found in §5:** ρ = 0.25 (no relaxation) and ρ = 0.75 (partial
relaxation to a plateau). Both tasks; q = 6, λc = 1, q_c = 12, gate on, W = 48, B = 30. W = 24 is not needed: the
window does not change persistence (§5).
- Horizon t = 130 (100 time units after the controller stops), with full records.
- Branches from t = 30:
  - (i) normal;
  - (ii) no reading or posting: the passive reference, expected to decay at about the fact lifetime;
  - (ii′) frozen board: agents keep reading, but only the posts on the board at t = 30;
  - (iii) controller-only facts removed from the board and from memories (10 facts in symmetric, 12 in no-proof).

  Each branch is run with its matching silent arm. In total: 2 tasks × 2 ρ × 4 branches × 2 arms = 32 settings,
  about 10 minutes, about 1.5 GB.
- **No saved state is needed.** Every random draw is keyed by (seed, episode, purpose), so a new setting that changes
  the rules only after t = 30 produces runs identical up to t = 30 by construction. Branch (i) with t_end = 40 must
  reproduce the existing runs exactly, which validates the code.
- **Code needed:** one new setting in the simulator (`post_stop_mode`), plus the removal step for (iii). Defaults must
  leave every existing run unchanged.
