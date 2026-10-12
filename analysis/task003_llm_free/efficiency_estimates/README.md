# efficiency_estimates — how efficiently does a controller steer the swarm?

Measures of control efficiency for **both** LLM-free simulations (Simulation 1, days and nights;
Simulation 2, continuous time), chosen so they can also be estimated from about 100 paths per
arm, the budget of the later LLM runs.

| file | what it is |
|---|---|
| [`PLAN.md`](PLAN.md) | the plan: variables, estimators (with equations), the infinite-C problem, steps |
| [`2026-10-12_decision_note.md`](2026-10-12_decision_note.md) | which measures were adopted and why; precision at 50 / 100 / 300 paths |
| `build_states.py` | gathers one row per (simulation, setting, arm, episode, time): mean beliefs m, vote shares x, and the agents' uncertainty H̄, JSD |
| `efficiency.py` | **the estimators**: `measures()` computes I, C, R, η, η_JS, I₃ from three distributions; `resolutions()` defines X; 1,000-resample paired bootstrap; label-swap null for I |
| `downsample.py` | precision study: 200 paired subsets of 50 / 100 / 300 episodes; bias, RMSE, how often C is infinite, how often two settings keep their order |
| `phase.py` | efficiency over the whole control plane (every qc × rate or b), point estimates at t = 30, for the phase diagrams |
| `make_report_figures.py` | figures and precision tables for the efficiency sections of the extended reports |
| `extend_reports.py` | builds the extended reports (12 October) from the dated originals: original text unchanged, plus a simulator paragraph and the efficiency section |

Outputs (not in git): `../results/efficiency_estimates/<date>/` — `states.parquet`,
`efficiency.csv`, `uncertainty.csv`, `null_I.csv`, `downsample.csv`, `ranking.csv`.
Run from the repository root: `build_states.py`, then `efficiency.py`, then `downsample.py`.

## Definitions (short; full version in PLAN.md §2)

Arms: + (controller targets A0), − (targets A2), 0 (silent); episodes paired across arms (same
random draws). **Z** = the target, a fair coin by design. **X_t** = the state at one fixed time t:

- agent level: the vote (or belief) of one agent drawn at random; per episode the vector of
  vote shares (or mean beliefs);
- swarm level: s = m_A0 − m_A2 (votes: x_A0 − x_A2) in k equal bins on [−1, 1]; per episode a
  one-hot vector.

**P̂_a = average over the n episodes of arm a of those per-episode vectors**, at that time t
only (not pooled over time). With Q̂ = ½(P̂_+ + P̂_−) and KL in bits:

| symbol | definition | meaning |
|---|---|---|
| I | ½KL(P̂_+‖Q̂) + ½KL(P̂_−‖Q̂) | **steering**: bits X reveals about the target |
| C | ½KL(P̂_+‖P̂_0) + ½KL(P̂_−‖P̂_0) | total change caused by control |
| R | KL(Q̂‖P̂_0); C = I + R | undirected change (happens whatever the target) |
| η | I / C ∈ [0, 1] | **efficiency**: share of the change that is directed |
| η_JS | I / (I + JS(Q̂, P̂_0)) | bounded efficiency; finite where C is infinite |
| ΔH̄, ΔJSD | controlled − silent, same episode, of H̄ = mean entropy of agents' beliefs and JSD = H(mean belief) − H̄ | **confusion check**: individual uncertainty, disagreement |

**Adopted (12 October 2026):** I at swarm level (+ one-agent companion); η at agent level, η_JS
at swarm level; ΔH̄ and ΔJSD. η reported only where C ≥ 0.01 bits.

**Not path-level.** All distributions are at a single time t, across episodes. Path-level
divergences would be at least as large (data-processing inequality): C(t) is a lower bound on
the divergence between whole trajectories, and η(t) is a fixed-time efficiency.
