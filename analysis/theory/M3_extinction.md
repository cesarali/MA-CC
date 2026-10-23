# Model version M3: mean field plus finite-swarm fact extinction (partial success)

**Date:** 22 October 2026. **Code:** [`models/mean_field.py`](models/mean_field.py) (`closure="M3"`), runs with
`run_models.py --closure M3 --seeds 40 --replicas 30`; scorecards `../task003_llm_free/simulation_2/key_phenomena_hypothesis/held_out_validation/scorecard_M3*.csv`.

## 1. One change from M2

M2's deterministic population is kept (exact single-agent memories and rules, deterministic shared board, expected
controller). **Added:** each circulating fact has an integer number of holders in a swarm of 24 agents. Per time step,
each holder loses the fact and each non-holder gains it with the probabilities measured in the M2 population (binomial
draws on the integer counts); the population is then forced to the new count. A fact with no holder and less than half
a copy on the board is **extinct for good** (unless the controller can still post it). This is the only finite-swarm
ingredient: demographic noise in fact holder counts, absorbing at zero. 40 realisations per condition; 720-agent
population (replicas 30).

## 2. Interpretation: facts as alleles

Board posts are copies of facts; reading is reproduction (a post read by a non-holder creates a holder); forgetting is
death; the posting rule is selection (agents post facts that support their vote); the controller is immigration of
A2-favouring alleles. A rare fact's reproduction number is about

  R_f ≈ (π_f / γ) · q (1 − m_f) / p_post,

where π_f is the probability that a holder posts f at an action, γ = −ln ρ the forgetting rate, p_post the share of
actions that post, m_f the share of agents holding f (posts per holder lifetime × readers per post). With about 1.3
holders per fact at the start, **drift drives facts extinct**, and extinction is permanent. After immigration stops,
the composition changes only through drift and weak selection, so the controller's imprint persists. Long memory
(γ → 0) raises R_f and lets Bayesian selection act: the imprint relaxes.

## 3. Scorecard (pre-registered; evaluated once)

| | world | held-out resolved signs | data + | data − | G error (held-out) |
|---|---|---|---|---|---|
| M2 | symmetric / nosolution | 8/17 / 8/16 | 1/4 / 4/10 | 7/13 / 4/6 | 0.15 / 0.18 |
| **M3** | symmetric / nosolution | **13/17 / 13/16** | 1/4 / 8/10 | 12/13 / 5/6 | 0.12 / 0.12 |

Development: 4/5 resolved signs (miss: nosolution ρ = 0.75, 3→6, data +0.033, model −0.046).
Crossover brackets, data against M3: nosolution 3→6 0.75–1 against 0.625–0.75 (one ρ step), 6→9 0.5–0.875 against
0.5–0.625 (inside), 9→12 0.25–0.625 against none; symmetric 3→6 0.5–0.625 against 0.25–0.375 (two steps low), 6→9
0.125–0.375 against none. **Main miss: the amplifying side in symmetric** (1/4 positive cells).

**Post-stop** (`scorecard_M3_post_stop.csv`): no communication decays with the data's half-times (1.0/1.0, 2.5/2.5,
1.0/1.0, model 7.5 against 5.5); the frozen board stays flat as in the data; removal collapses within 0.5–0.75 (data
0.5–1) to a residual of the right sign in nosolution (−0.16 against −0.06 at ρ = 0.25; −0.10 against −0.09 at 0.75)
but the wrong sign in symmetric ρ = 0.25 (−0.03 against +0.05); normal continuation relaxes by 20–26% at ρ = 0.25
where the data stay flat. Magnitudes are about 60% of the data throughout. The ordering test fails in 5 of 8 cases,
every time by swapping two branches with nearly equal values (normal and frozen; no communication and removal near 0),
so it is a fragile metric here.

## 4. Assessment

Adding extinction to M2 turns chance-level held-out signs into 13/17 and 13/16 and reproduces the post-stop
mechanisms and timescales. What remains: magnitudes about 60% of the data, the symmetric amplifying side, and too much
relaxation under normal continuation. Candidate next change (one at a time): the board's own finite-size noise (all
readers share one realised 48-post board; M3 keeps it deterministic).

## 5. Scaling facts with agents (decided 22 October)

The task generator is fixed at 3 people, 2 jobs and 9 hidden numbers (99 propositions, 49 true). Scaling agents alone
raises copies per fact. Agreed design: **K independent projects with one split** (the same person works alone in
every project; the split's score is summed over projects): three answers, 9K hidden numbers, about 49K true facts,
24K agents. To be built for K = 2 and 4 (`../task003_llm_free/simulation_2/key_phenomena_hypothesis/finite_size/`). The allele picture predicts that with copies per fact
held constant the effect does **not** vanish as the swarm grows.
