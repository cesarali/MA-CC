# Phase A — pre-registration (written 23 October 2026, before any Phase A quantity is compared with data)

**Plan:** the theory continuation agreed on 23 October (after the external review
`M4_theory_review_and_analytical_continuation.md`): Phase A uses stored data only (no new simulations), then Phase B
(matched M3/M4 comparison), then Phase C (theory note). This file fixes the Phase A cells, quantities, predictors and
decision rules. Changes after this commit are logged at the end with their date and reason.

## Notation

- Memory M_i: the set of facts agent i remembers (`active` bitmask in `agent_snapshots`), at t = 30 (grid index 120).
- Φ₂(M) = P(A2 | M), the exact posterior (`World.posterior`).
- Circulating facts U = the setup's agent facts ∪ the A2 pool's controller-only facts (pool facts no agent starts with).
- μ_C(M), μ_S(M): distribution of agent memories in the false (C) and silent (S) arms at t = 30, over agents and episodes.
- Episodes are paired across arms and across q (shared random streams), so all differences are computed per episode.

## Cells

Both tasks (task003: run `2026-10-20_sim2_reading_mechanism`; Object Placements: `2026-10-22_sim2_op_main`), both
setups (symmetric, no-proof), ρ ∈ {0.25, 0.5, 0.75, 1}, q ∈ {3, 6}, both arms: 64 cells, 1,000 episodes each.
Reading increment studied: D(3→6) at t = 30. That is 16 (task × setup × ρ) points.
The data sign of D is the resolved sign of the vote-based D (95% interval excludes 0), as in all earlier scorecards.

## A1 — belief versus vote

G_vote: the usual vote-based effect. G_belief = mean over agents of Φ₂(M_i(30)), false − silent. Report both and
D_vote, D_belief for the 16 points, and whether the signs agree. Votes are drawn right after reading, from the memory at
that moment; snapshot memories have decayed since. No prediction is made; this measures how far a memory-only theory
can get.

## A2 — a compact evidence map, built from the world alone

State of a memory: s(M) = (z, c, a)
- z = 1 if M proves A0 (P(A0 | M) = 1), else 0;
- c = number of distinct controller-only facts in M, capped at 4;
- a = number of distinct agent facts in M, capped at 6.

Response of a state: Φ̂₂(s) = mean of Φ₂(M) over memories M drawn **uniformly** among subsets of U with c controller-only
and a agent facts (20,000 draws per (c, a); within a draw the state's z is whatever that subset gives). Computed
from the world and setup only, before any memory data are read.

Reported on the stored memories: the mean |Φ̂₂(s(M)) − Φ₂(M)| over agents, and G computed with Φ̂₂ against G_belief.

## A3 — two predictors of the sign of D(3→6), fixed now

Both use only the t = 30 memories and the world. Neither uses votes, G or D.
1. **Marginal (survival-only) predictor.** Each fact's own pull, weighted by how much more often the false arm holds it:
   Ĝ_marg = Σ_{f∈U} (m_f^C − m_f^S) · (P(A2 | {f}) − 1/3), with m_f the share of agents holding f.
   D̂_marg = Ĝ_marg(q = 6) − Ĝ_marg(q = 3).
2. **Evidence-state predictor.** Ĝ_state = Σ_s (μ_C(s) − μ_S(s)) · Φ̂₂(s), with the A2 map. D̂_state = its q difference.
3. Reference (not a predictor; the exact identity): D_belief from A1.

Scoring: a predictor is right at a point if its sign matches the data's resolved sign, where |D̂| < 0.005 counts as 0.
Report the counts per task and setup, and episode-bootstrap 95% intervals (200 resamples, paired across arms and q).

**Expectations, stated in advance:**
- The marginal predictor fails on Object Placements, because single-fact pulls are about 0 there. It may do well on
  task003.
- The evidence-state predictor gets at least as many signs right as the marginal one in each task.
- If the evidence-state predictor does not beat the marginal one on Object Placements, the (z, c, a) closure is reported
  as insufficient. The map is then not tuned on these 16 points.

## A4 — the two solvable branches

Runs: `2026-10-20_sim2_persistence_branches` (task003), `2026-10-23_sim2_op_post_stop` (Object Placements); both
setups, ρ 0.25 / 0.75, q 6. Given the state at t = 30, agents in these two branches are independent of each other:
- **No communication** (no reading, no posting; agents still act at rate 1 and vote from their current memory): each
  fact of M_i(30) survives to t independently with probability e^{−γ(t−30)}. A vote cast at time u is drawn from
  Φ(M_i(u)). The predicted share of votes for A2 at t is
  e^{−(t−30)} · 1[last vote at 30 = A2] + ∫₀^{t−30} e^{−s} E[Φ₂(M_i(t − s))] ds,
  computed exactly by enumerating the subsets of M_i(30). Agents who have not voted are excluded, as in x_A2.
- **Frozen board** (agents read q posts from the board as it was at t = 30, own posts excluded; nobody posts): each
  agent is an independent Markov chain on memories. It is simulated with the simulator's own rules (exact posterior,
  probability matching), 1 chain per agent per episode, from M_i(30) and the agent's frozen window.
  Analytic check: each fact's long-run holding share is r_f / (r_f + γ), with r_f = 1 − C(W_i − k_f, q) / C(W_i, q)
  the chance of reading f at one action (k_f copies among the W_i posts the agent can see).

Prediction: δ(t) at t = 31, 40, 80, 130 and the half-time of the no-communication decay. **Expectation:** these
reductions are exact given the t = 30 state, so they should match the data within sampling error at every time. A
mismatch means a misunderstanding of the rules or a bug, not a failed theory.

## Outputs

Scripts and tables in `analysis/theory/phaseA/`. Results in `analysis/theory/phaseA/phaseA_results.md`, written after
the computations, with every expectation above marked met or not met.

## Change log
(none yet)
