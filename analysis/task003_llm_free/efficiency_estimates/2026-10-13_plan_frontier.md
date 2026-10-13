# Success against actual cost: the operational frontier — plan

**Status: proposed 13 October 2026, not started.** Source of the method:
`~/Downloads/blackboard_control_efficiency_handoff.md` (§3–§5, §7), adapted to our simulations.

## 1. Question

For each controller configuration, how much does it change the probability that the swarm's
collective decision is (a) its assigned target, (b) the correct answer, and how much does it spend?
Which configurations reach a given change most cheaply, and how fast do returns diminish?

## 2. Outcomes (run level, fixed in advance)

- **Collective decision Y_T**: plurality of the agents' last votes at time T (Simulation 2: agents
  who have not voted yet are left out). **Ties**: the success counts 1/k when k answers tie (the
  expected success under a random tie-break). Sensitivity: strict majority (more than 12 of 24 votes).
- **Target attainment** S^target = 1{Y_T = assigned target}; **task correctness**
  S^task = 1{Y_T = A0}. Kept separate throughout.
- **Baseline**: the silent arm of the same episodes, scored with the same target label.
- **Benefit** Δ^k = p^k_controlled − p^k_silent, for k ∈ {target, task}, at T = 30 (end of control;
  Simulation 1: day 30) and T = 40 (Simulation 2, persistence); curves every 5 time units.

## 3. Cost

- **Primary**: facts the controller actually posted up to T (Simulation 2: messages, one fact each;
  Simulation 1: facts posted on nights). Mean over episodes.
- **Secondary**: posts the controller read (sensing) up to T, as a separate axis, and a combined
  cost facts + λ·reads with λ ∈ {0.1, 1} as sensitivity.
- The allowed budget (30 messages; b per night) is a design factor, never the cost.

## 4. Policy class (what competes on a frontier)

A frontier compares only **controller** settings, with everything else fixed. One group per
(simulation, setup, ρ, q, target):

- Simulation 2: λc × qc × gate (× proof stop for symmetric truth) — 40 (80) configurations;
- Simulation 1 (`sim1_pm`): qc × b — 20 configurations.

Two frontiers per group: tested configurations only, and its lower convex hull (randomised mixtures
of configurations, chosen once per run). All cells have the needed data (snapshot and day tables).

## 5. Metrics

Per configuration: Δ^target, Δ^task (signed, 95% intervals); cost; **gain per fact** Δ/C;
**η_cost** = C*(Δ)/C and **η_performance** = Δ/Δ*(C) (only where Δ > 0 and resolved; harmful or
unresolved: reported as such, no efficiency); along each family (rate at fixed qc and gate; b at
fixed qc) the **marginal return** ΔΔ/ΔC between neighbours and the elasticity. Per group: the
least cost **C*(δ)** for fixed thresholds δ ∈ {0.05, 0.1, 0.2, 0.3} (or "not reached").

## 6. Uncertainty, selection, precision

- Bootstrap over episode numbers (1,000), the same resample for every configuration of a group
  (episodes are paired across configurations); the frontier and η recomputed in each resample.
- Selection bias: frontier chosen on half of the episodes, η evaluated on the other half; compare
  with the in-sample values.
- Precision at 50 / 100 / 300 paths (200 subsets), as before.

## 7. What we will look for

1. Does any setting dominate (e.g. slow rate with the gate on), or do all lie on the frontier
   (η_cost ≈ 1 by lack of alternatives)?
2. The shape of the frontier: diminishing or negative returns (Simulation 1, b = 9; Simulation 2,
   full memory, high rate).
3. Manipulation against correctness: for the false controller, the correctness lost per point of
   target attainment gained, and its cost.
4. Simulation 1 against Simulation 2 on the same axes (facts posted against success).
5. Memory (ρ) and the vote gate.

## 8. Checks before use

The handoff's synthetic tests 1–5 and 7 (degenerate η, dominated policies, saturation with
diminishing returns, harmful controller, zero/near-zero cases, monotone budget frontier) as unit
tests; plurality and tie rule checked on hand-built vote vectors.

## 9. Outputs

`frontier.py` (+ tests) here; results in `results/efficiency_estimates/2026-10-13/` (`frontier_points.csv`,
`frontier_thresholds.csv`, `frontier_downsample.csv`); figures (success change against facts posted,
frontier and hull, per group); a short findings note.
