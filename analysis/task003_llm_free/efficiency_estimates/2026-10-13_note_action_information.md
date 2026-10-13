# Action information pilot — findings

**13 October 2026.** Plan: [`2026-10-13_plan_action_information.md`](2026-10-13_plan_action_information.md).
Code: `action_info.py`. Tables (not in git): `results/efficiency_estimates/2026-10-13/action_info.csv`
(570 rows: setting × variant × window, every quantity with 95% intervals), `action_downsample.csv`.
Checked: C computed as I + R equals the direct Σ ω w KL(P_u‖K₀) to rounding.

## Results (q 6, qc 12; 1,000 episodes per arm)

**1. Causal case (Simulation 2, gate off): the act carries almost no information.** Per agent and
per window of Δ = 0.25, I(U; V′ | V) ≈ 0.00001–0.0001 bits while the standing term R ≈ 0.003–0.06,
so η_U = 0.0003–0.003 (ρ 0.75). Wider window (Δ = 1), only agents who voted, or also conditioning on
the swarm state: η_U ≤ 0.03 everywhere. The response χ (change in the probability that the agent's
next vote is the target when the controller acts) is +0.001 to +0.005 per window (Δ = 1: up to
+0.06 for agents who voted, +0.2 for the false controller at λc 8 on symmetric). Exception without
meaning: symmetric at ρ 1, where R ≈ 0 because every agent proves A0, so η_U is a ratio of two
near-zero numbers.

**2. Gated controllers measure sensing, not effect.** With the gate on (both simulations) I is
10–100 times larger and η_U reaches 0.02–0.48, but χ is **negative** (Simulation 1: −0.04 to −0.40;
Simulation 2: −0.01 to −0.04): the controller acts when the posts it read go against its target, so
U = 1 marks moments when agents are moving away. I(U; V′ | V) there is mostly information about the
controller's *reading of the swarm*. Conditioning on a coarse swarm state halves χ but does not remove
it (Simulation 1, nosolution, b 1: −0.13 → −0.06).

**3. Not estimable at 100 paths.** Causal case, Δ 0.25: at 100 paths the error of I (5 × 10⁻⁵) is
larger than I itself (2 × 10⁻⁵); η_U has error 0.004 against a value of 0.002. R is estimable (±0.002).

## Interpretation

The decomposition is valid and does not suffer the target-symmetry degeneracy, but in these systems
the controller's influence is almost entirely **standing**: it lives in facts already on the board
and in agents' memories, and builds up over many acts; a single act changes an individual agent's
next vote by a fraction of a percentage point. Per-act, per-agent η_U is therefore ≈ 0 everywhere —
a finding about how control works here (cumulative, not punctual), but not a usable efficiency:
flat at zero, and below the noise at 100 paths. With a vote gate the variable U is selected on the
swarm's state, which makes I a sensing measure unless the gate's acting probability is known and
corrected for.

## What would make an action-based measure informative

- A longer response: lagged or cumulative effects, I(U_t; V′_{t+ℓ} | V_t) for ℓ of a few time units,
  or the episode-level directed information Σ_t I(U_t; S_{t+1} | S_t) on a validated Markov state
  (coarse-graining plan).
- Larger acts (several facts per intervention) or a swarm-level outcome.
- Randomised acting (or a logged acting probability) wherever a gate is used.

Absent these, the operational frontier of the handoff (success against actual messages) remains the
better primary efficiency.
