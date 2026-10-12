# Coarse-graining on simulator data — plan

**Status: plan, 12 October 2026. Not started** (the researcher asked to review it first).

## 1. Goals

Both, as decided by the researcher:

1. **A dynamical model of control.** A small state of the swarm whose next value can be predicted
   from its current value and the controller's action, well enough to describe how the swarm
   responds to control, how it relaxes after the controller stops (Simulation 2, t ≥ 30), and
   how long it remembers.
2. **Path-level efficiency.** On such a state, compute the information about the target carried
   by the **whole trajectory**, I(Z; Γ), and the total change of the trajectory distribution,
   C_path = ½ KL(P_+(Γ)‖P_0(Γ)) + ½ KL(P_−(Γ)‖P_0(Γ)), and η_path = I(Z; Γ)/C_path; compare
   with the fixed-time versions in `../efficiency_estimates/` (which are lower bounds).

Both at the sample size of the LLM runs (about 100 paths per arm) as well as at 1,000.

## 2. What exists, and what changes

- `coarse_graining/` (2 October): archived **LLM** study, 60 populations, 15 rounds. State e =
  mean belief in A0, 4 fixed bins (0.33 / 0.50 / 0.75). Four Markov tests (order, Chapman–
  Kolmogorov, conditional mutual information, lumpability). Markovian only silent at ρ = 1;
  missing coordinates: knowledge spread (silence), survival of decisive facts (control).
  Code: `code/four_tests.py`, `code/estimate.py` (they read a deleted table; loaders to replace).
- `agents_control/task003_llm_free/coarse_graining_/implementation_recommendations.md`
  (8 October): two belief coordinates, extra candidates (disagreement D, memory fragility F,
  board evidence B), held-out predictive validation with lags, multi-step checks, branching,
  sample-budget study, no thermodynamic claims.

This plan reuses the four tests, takes the candidate states and validation from the notes, and
**uses only recorded data** (decided): no reruns, hence **no branching** (deferred).

## 3. Data (recorded only)

| simulation | cells | what is available |
|---|---|---|
| Simulation 1 (`2026-10-07_sim1_pm`, other variants later) | every cell | per agent per day: beliefs, active facts, facts read, posts; per night: controller reads and posts |
| Simulation 2 grid (`2026-10-09_sim2_grid`) | every cell | snapshots every 0.25: mean beliefs, vote shares, board size, messages; posts; controller actions |
| Simulation 2, full tables | q 6, qc 12, ρ 0.75 (grid) and ρ 1 (`2026-10-12_sim2_main_rho1_full`): 2 × 52 cells | additionally every agent action, forgetting event and per-agent snapshot |

**Observation times.** Simulation 1: after each night's controller action and board clearing
(state for the next day), as the notes recommend; days 1–29 as transitions. Simulation 2:
snapshots at Δ = 1 (checks at Δ = 0.5 and 2); the protocol phase (t < 30 or ≥ 30) and the
remaining budget are conditioning variables, never pooled.

**Controller action** in a step: Simulation 1, posted or not that night (and which decision);
Simulation 2, messages posted in the step (0, 1, ≥ 2).

## 4. Candidate states

Built per (simulation, setup, ρ, arm); never pooled across ρ or arms.

| id | state | cells | source |
|---|---|---|---|
| S1 | e = m_A0, 4 fixed bins (0.33 / 0.50 / 0.75) | all | old work |
| S2 | s = m_A0 − m_A2, 3 equal bins (as in the efficiency measures) | all | efficiency estimates |
| S3 | (m_A0, m_A2), 3 × 3 bins on the simplex | all | notes, M |
| S4 | S2 or S3 + disagreement (JSD, 2 bins at the training median) | Simulation 1; Simulation 2 full tables | notes, D; old "spread" |
| S5 | S2 + memory fragility F (r = 0.75), 2 bins | same (needs active facts) | notes, F; old "decisive facts" |
| S6 | S2 + board evidence B (belief from the visible window), 2 bins | Simulation 1; Simulation 2 full tables | notes, B |
| V | vote-based S2 (x_A0 − x_A2, 3 bins) | all | for the LLM runs, where only votes are seen |

Bin edges fixed before evaluation. Continuous versions (the notes' preferred first step) are
used in the predictive comparison of §5.2.

## 5. Tests

### 5.1 The four Markov tests (from `code/four_tests.py`, reloaded)

Per (simulation, setup, ρ, arm, state): order 1 vs 2 on held-out episodes; Chapman–Kolmogorov
gap at 2, 3, 5 steps (respecting the cutoff at t = 30); conditional mutual information
I(S_{t+1}; S_{t−1} | S_t) against a permutation null; lumpability by D, F, B, number of facts
held, and controller-fact exposure.

### 5.2 Predictive sufficiency (notes §5)

Predict the next (m_A0, m_A2) from (A) the current state plus protocol variables, (B) plus one
lag, (C) plus two lags; held-out log loss and mean-prediction error; episode splits 60/20/20,
all arms of an episode in the same split. Also multi-step occupancy, mean trajectories,
probability of reaching the target region, and relaxation after t = 30, predicted by the fitted
chain against the data.

### 5.3 What counts as adequate

A state is **adequate** if, on held-out episodes: the order-2 gain is within its uncertainty of
0 or below 0.005 nats/step; the Chapman–Kolmogorov gap at 5 steps is below 0.05; and predicted
final target share, reaching probability and post-cutoff relaxation are within 0.02 of the data
(the notes' engineering tolerance). Choose the **smallest adequate state**; if none is adequate,
say so and use finite-history models (order 2) for §6.

## 6. Path-level efficiency on the chosen state

- Fit, per arm, the transition kernel of the chosen state (time-dependent where the protocol is:
  before and after t = 30, remaining budget), plus the initial distribution.
- **C_path**: KL between controlled and silent path laws factorises over steps for Markov
  kernels: KL(P‖Q) = KL(p_0‖q_0) + Σ_t E_P[KL(P(·|s_t)‖Q(·|s_t))]. Transitions never seen under
  silence give infinite C_path (the controlled swarm goes where silence never goes); report how
  often, and the bounded Jensen–Shannon version, never an arbitrary smoothing.
- **I(Z; Γ)**: the mixture of two Markov chains is not Markov, so compute
  E_z KL(P_z‖½P_+ + ½P_−) by Monte Carlo over paths sampled from the fitted chains, with exact
  path likelihoods (forward products of kernel entries).
- **η_path = I(Z; Γ) / C_path**, and the same with the order-2 kernels as a sensitivity check.
- Compare with the fixed-time I(Z; X_t), C(t), η(t): the gap I(Z; Γ) − I(Z; X_30) is the
  information carried by *when* things happened, not only where the swarm ended.

## 7. Sample budget

Repeat §5–§6 on 200 paired subsets of 50, 100, 300 episodes per arm (refitting everything on
each subset): bias, spread, how often C_path is infinite, whether the chosen state stays
adequate, and whether settings keep their order.

## 8. Scope and order of work

1. Loaders: build the per-step state tables from recorded data (both simulations).
2. §5 on Simulation 2 full-table cells (all candidates), then Simulation 1 (q 6, qc 12, b 1 / 3 /
   9) — decide the state.
3. §6 on the chosen state: main cells first, then the phase plane (states S1, S2, S3, V only:
   they exist in every cell).
4. §7.
5. A dated report and a decision note here.

## 9. Known limits

- No branching: lumpability and lags are weaker evidence than cloning microstates (notes §6).
- Fragility and board evidence need private memories and the board, which LLM runs will not
  give; they are diagnostics, not candidates for the LLM state.
- A Markov state is not a thermodynamic one; no entropy-production or dissipation claims.
- Transitions are counted per episode; agents and steps are never treated as independent.
