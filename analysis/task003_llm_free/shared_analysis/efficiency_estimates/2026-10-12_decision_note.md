# Control efficiency — decision note, 12 October 2026

**Adopted by the researcher on 12 October 2026.**

Plan: [`PLAN.md`](PLAN.md). Scripts: `build_states.py`, `efficiency.py`, `downsample.py`.
Tables (not in git): `results/efficiency_estimates/2026-10-12/` — `efficiency.csv` (every
setting, time and resolution, with 95% intervals), `uncertainty.csv`, `null_I.csv`,
`downsample.csv`, `ranking.csv`. Data: Simulation 2 grid (q 6, qc 12, ρ 0.75 / 1, every rate,
gate on/off) and Simulation 1 `sim1_pm` (q 6, qc 12, b 1 / 2 / 3 / 6 / 9, ρ 0.75 / 1), 1,000 episodes
per arm. Numbers below are at t = 30 unless stated.

## 1. What can be estimated from 100 paths per arm

Median over settings (task003-symmetric at ρ 1 excluded: every agent proves A0 and no
quantity carries information there).

| quantity | resolution | error at n = 100 | infinite or undefined | verdict |
|---|---|---|---|---|
| directed information I | one agent (beliefs / votes) | ±5% / ±7.5% of its value | never | **usable** |
| I | swarm, s binned (k = 3 beliefs; k = 2–4 votes) | ±4–10% | never | **usable** |
| total change C, efficiency η = I/C | one agent | η ±0.02 (beliefs), ±0.03 (votes) | 2–4% of subsets | **usable** |
| C, η | swarm, k ≥ 3 | biased by −0.4 to −0.7 bits when finite | 34–98% of subsets | **not usable**: the controlled swarm reaches states silence never visits |
| bounded efficiency η_JS | every resolution | ±0.01–0.03 | never | **usable** |
| change in individual uncertainty ΔH̄ | per episode, paired | ±0.017 bits (effects ≈ 0.25) | — | **usable** |
| change in disagreement ΔJSD | per episode, paired | ±0.006 bits (effects 0.04–0.2) | — | **usable** |

Ordering of two settings at n = 100 (share of subsets agreeing with n = 1,000): differences
of at least 0.02 in agent-level I, η or η_JS are ordered correctly 90–100% of the time;
differences below 0.01 are close to a coin flip. Swarm-level I needs differences of about
0.05.

## 2. What the measures show (n = 1,000)

- **η catches wasteful control.** Simulation 1, nosolution, b = 9: η = 0.25 (ρ 0.75) and 0.06
  (ρ 1), against 0.92 and 0.86 at b = 1: most of what a large batch changes is undirected. In
  Simulation 2 with full memory, η falls with the rate (nosolution, gate on: 0.93 at λc 0.5 to
  0.67 at λc 8): the same "saying too much of a true pool" mechanism, now measured as
  undirected change. When agents forget, η stays high (0.89–0.94) at every rate.
- **The swarm carries far more of the command than one agent.** Nosolution, Simulation 2,
  ρ 0.75: I ≈ 0.8 bits from the swarm's position (3 bins) against 0.18 (beliefs) or 0.30
  (votes) from one agent.
- **Controllers make agents surer, but some split the swarm.** Individual uncertainty falls
  under almost every controller (ΔH̄ ≈ −0.25 bits in nosolution). Disagreement rises: in
  nosolution by +0.06 (target A2) and +0.11 (target A0) at ρ 0.75; in symmetric the false
  controller raises it by +0.11 to +0.20 while hardly changing individual certainty, whereas
  the truth controller lowers both. The false controller on symmetric confuses rather than
  convinces.

## 3. Decision (proposed)

1. **Steering: I(Z; X_t)**, the information the swarm's state carries about which target was
   given. Primary at swarm level (s = m_A0 − m_A2, 3 bins; votes: x_A0 − x_A2, 2–3 bins),
   with the one-agent value as the exact companion.
2. **Efficiency: η = I / C at agent level** (finite and precise), with **η_JS** wherever the
   swarm level is needed. KL-based η at swarm level is not estimable at 100 paths and is
   dropped.
3. **Confusion check: ΔH̄ and ΔJSD** (individual uncertainty, disagreement), paired with
   silence. Cheap and precise; reported next to I and η rather than combined into a ratio.

Rules: η is reported only where C ≥ 0.01 bits (otherwise 0/0 noise, e.g. symmetric at ρ 1);
bins are fixed in advance (equal bins on [−1, 1]); a difference below 0.02 is not
interpreted at 100 paths.

For the LLM runs, votes are what is observable: vote-based agent-level η (±0.03) and
swarm-level I with 2–3 bins (±5–10%) are the realistic pair.
