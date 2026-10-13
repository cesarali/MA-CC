# Action information: a pilot — plan

**Status: agreed 13 October 2026; pilot computed from this plan.**

## 1. Question

How much of the controller's influence is carried by **its act at a given step**, and how much is
**standing** (facts already on the board or in memories, so that even a step without an act differs
from silence)? Unlike the target-based η, the variable here is the controller's action, a genuine
random variable, so the symmetry that made η ≈ 1 does not apply.

## 2. Variables (decided 13 October 2026)

- **Action U ∈ {0, 1}**: intervention or not in the step. Simulation 1: U_t = 1 if the controller
  posted on night t (decision `posted`, `posted_proved` or `fallback`), 0 if the vote gate or the
  proof stop kept it silent. Simulation 2: U_t = 1 if it posted at least one message in [t, t + Δ).
- **Agent state V = an agent's own current vote** ∈ {A0, A1, A2} (Simulation 2 adds "not voted yet").
- **Outcome V′ = the same agent's vote after the step.** Simulation 1: day t → day t + 1 (every agent
  votes every day). Simulation 2: last vote at t → at t + Δ, from the per-agent snapshots.
- **Step**: Simulation 1, one night (between days t and t + 1), t = 1 … 29. Simulation 2, window
  Δ = 0.25 (the snapshot spacing; check Δ = 1), windows inside the control period (t + Δ ≤ 30).
- **Steps without budget are excluded** (Simulation 2: 30 messages already posted before t): there
  U = 0 by construction, not by decision.

## 3. Kernels and quantities

Counts over all eligible (episode, step, agent) triples, per arm. For the controlled arm,
P_u(v′ | v) = P(V′ = v′ | V = v, U = u) and w(u | v) = P(U = u | V = v); for the silent arm, over
the same steps, K₀(v′ | v). Mixture K̄(v′ | v) = Σ_u w(u | v) P_u(v′ | v); occupancy ω(v) =
share of transitions starting in v under control.

> I = Σ_v ω(v) Σ_u w(u|v) KL(P_u(·|v) ‖ K̄(·|v))  = I(U; V′ | V), bits, ≤ H(U | V) ≤ 1
> R = Σ_v ω(v) KL(K̄(·|v) ‖ K₀(·|v))   (standing influence)
> C = I + R = Σ_v ω(v) Σ_u w(u|v) KL(P_u(·|v) ‖ K₀(·|v)),   **η_U = I / C**

Also reported: P(U = 1); H(U | V); the normalised **I / H(U | V)** (share of the action's uncertainty
visible in the next vote); and the signed **response** χ = Σ_v ω(v) [P_1(z | v) − P_0(z | v)], the
change in the probability that the agent's next vote is the controller's target z when it acts.

Interpretation: η_U → 1 when the influence lives in the acts (between acts the swarm evolves as if
silent); η_U → 0 when it is mostly standing. η_U is undefined when U does not vary (Simulation 1 with
the gate off: the controller acts every night).

## 4. Variants

1. **All agents** (primary).
2. **Only agents who voted in the window** (Simulation 2: time since last vote < Δ at t + Δ): the
   agents who could change their vote. Simulation 1: identical to 1.
3. **Also conditioning on a coarse swarm state** S (target vote share in 3 equal bins):
   I(U; V′ | V, S), as a check on hidden-state leakage.
4. **Proof-stop nights** (Simulation 1, symmetric truth): with and without nights where the proof
   stop silenced the controller.

## 5. Settings and data (recorded only)

- Simulation 1 `2026-10-07_sim1_pm`: q 6, qc 12, b 1 / 2 / 3 / 6 / 9, ρ 0.75 / 1, both setups, both
  targets (gate on: U varies).
- Simulation 2 full-table cells: q 6, qc 12, ρ 0.75 (`2026-10-09_sim2_grid`) and ρ 1
  (`2026-10-12_sim2_main_rho1_full`), every rate, gate on and off, proof stop on, both setups and
  targets. **Gate off is the causal case**: the controller's clock is independent of the swarm.
  Gate on: U depends on the posts the controller read, so I mixes effect and selection.
- Flag settings with P(U = 1) outside 0.05–0.95 (little variation in U).

## 6. Uncertainty and precision

Per-episode count tensors; 95% intervals by resampling episodes (500 resamples), all agents of an
episode together. Precision: 200 subsets of 50, 100, 300 episodes, as in `downsample.py`.

## 7. Outputs

Script `action_info.py`; results in `results/efficiency_estimates/2026-10-13/`: `action_info.csv`
(every setting, variant, quantity, with intervals) and `action_downsample.csv`; a short note with
the findings next to this plan.

## 8. Known limits

- An agent's own vote is not a Markov state (board, memory): part of the hidden state leaks into I
  and R; variant 3 checks this.
- Kernels are pooled over the control period; time-resolved kernels are a later step.
- Agent-level, not swarm-level: agents influence one another, so the swarm's response is not the sum.
- Path-level directed information (Σ_t I(U_t; V′_t | V_t)) needs time-resolved kernels; not in the pilot.
