# Control efficiency — short plan

**Status: agreed scope 12 October 2026; computations start from this plan.**

**Question.** How efficiently does a controller steer the swarm: how much of the change it
causes is *directed* (depends on which target it was given), and how uncertain does it leave
the agents? Measured on existing Simulation 1 and Simulation 2 data, then downsampled to the
100 paths per arm affordable with LLM agents.

## 1. Variables

- Arms a ∈ {+, −, 0}: controller with target A0 (+), target A2 (−), silent (0). Episodes
  e = 1 … n per arm, **paired**: episode e uses the same random draws in all three arms.
- **Command Z**: a fair coin chooses + or − (by design, so Z is a random variable).
- **State X(e, t)** at time t, two resolutions:
  - **agent level**: the vote (or belief) of one agent drawn uniformly at random;
  - **swarm level**: s(e, t) = m_A0 − m_A2 (difference of the mean beliefs; votes version:
    x_A0 − x_A2), cut into k equal bins on [−1, 1], k ∈ {2, 3, 4, 8}.
- **Beliefs**: p_ik(t) = P(A_k | facts agent i remembers at t), exact from the world engine;
  m_k = mean over the 24 agents. **Votes**: x_k = share of agents whose last vote is k
  (Simulation 2 adds "not voted yet").

## 2. Estimators

For each arm a, an estimated distribution P̂_a(x) of X at time t:

- agent level: P̂_a(v) = (1/n) Σ_e x_v^a(e, t)  (beliefs: m_v^a(e, t));
- swarm level: P̂_a(x) = (1/n) #{e : X^a(e, t) = x}.

With Q̂ = ½ P̂_+ + ½ P̂_−, KL(p‖q) = Σ_x p(x) log₂ (p(x)/q(x)), JS(p, q) = ½ KL(p‖r) + ½ KL(q‖r),
r = ½(p + q):

| quantity | estimate | meaning |
|---|---|---|
| directed information | I = ½ KL(P̂_+‖Q̂) + ½ KL(P̂_−‖Q̂) | how much X reveals which target was given (bits, ≤ 1) |
| total change | C = ½ KL(P̂_+‖P̂_0) + ½ KL(P̂_−‖P̂_0) | how far control moves X's distribution from silence |
| undirected change | R = KL(Q̂‖P̂_0); exactly C = I + R | change that happens whatever the target |
| **efficiency** | **η = I / C = I / (I + R)** ∈ [0, 1] | share of the change that is directed |
| bounded version | R_JS = JS(Q̂, P̂_0); η_JS = I / (I + R_JS) | finite even where silence never visits a state (§3) |
| three-command information | I₃ = mean over a of KL(P̂_a‖M̂), M̂ = mean of the three P̂_a | Z ∈ {+, −, 0}, uniform |

**Agents' own uncertainty** (per episode, then controlled − silent, same episode):
H̄ = (1/24) Σ_i H(p_i) (individual uncertainty, bits) and JSD = H(m) − H̄ (disagreement);
H(m) = H̄ + JSD exactly.

Uncertainty: bootstrap over episode numbers (all arms resampled together, 1,000 resamples).
Null for I: swap the + and − labels within each episode.

## 3. Known problem: infinite C

KL(P_+‖P_0) is infinite when the controlled swarm reaches states the silent swarm never
visits. This happens at swarm level (e.g. nosolution, 4 bins, truth controller: 467 of 1,000
episodes in s ∈ [0.5, 1], silent 0 of 1,000). Not patched by arbitrary smoothing: report
η where finite, η_JS everywhere, and the smoothed value (add ½ per bin) only as a labelled
sensitivity.

## 4. Data

- Simulation 2 grid `2026-10-09_sim2_grid`: both setups, q 6, qc 12, ρ 0.75 and 1, every
  rate, gate on/off, stop on; t = 0, 1, …, 40.
- Simulation 1 `2026-10-07_sim1_pm`: both setups, q 6, qc 12, b 1 / 3 / 9, ρ 0.75 and 1;
  end of days 1 … 30.
- Agents' uncertainty needs per-agent beliefs: Simulation 1 has them in every cell;
  Simulation 2 only in the full-table cells (ρ 0.75). For ρ 1, rerun the q 6, qc 12 cells with
  full tables (`sim2_main_rho1_full`, same seed, so identical to the grid cells, about 5 minutes).

## 5. Steps

1. Full-sample values (n = 1,000) with intervals, for every quantity, setting and time.
2. Downsampling: 200 random subsets of n = 50, 100, 300 episode numbers (paired across arms).
   For each quantity: bias and spread against n = 1,000; how often the ordering of
   neighbouring settings (rates, gate on/off) is preserved; how often C is infinite.
3. Decision note: the smallest X that keeps most of the directed information and is
   estimable at n = 100; the two measures to carry forward.

Outputs: `results/efficiency_estimates/<date>/` (not in git); scripts and the decision note here.
