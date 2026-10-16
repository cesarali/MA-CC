# Eight coarse-grained states: Markov tests and readability

**15 October 2026.** Simulation 2, `task003-symmetric`, recorded data only. Code:
[`code/markov_study.py`](code/markov_study.py). Results (not in git): `results/coarse_graining/2026-10-15/`
(`markov_tests.csv`, `readability.csv`, `markov_medians.csv`, per-arm state tables `states_*.parquet`).

## Design (fixed before running)

**Data.** The cells with per-agent tables: q 6, q_c 12, λ_c 1, gate on, proof stop on; ρ 0.75 (9 October
grid) and ρ 1 (12 October full run); silent, target A0, target A2; 1,000 episodes each; states at integer
times.

**Reasoner-free.** Every state is computed from the facts each agent remembers, using only the world's logic:
p_ik = P(A_k | K_i) is what agent i's facts imply. V alone uses the agents' last votes. The same states can
be computed in an LLM run whose harness logs each agent's facts.

| id | state | cells |
|---|---|---|
| FV | fact-implied votes: each agent's facts favour the allocation(s) with the largest p_ik (ties split); share(A0) − share(A2), 3 equal bins on [−1, 1] | 3 |
| S1 | m_A0 = mean p_i0, cuts 0.33 / 0.50 / 0.75 | 4 |
| S2 | s = m_A0 − m_A2, 3 equal bins on [−1, 1] | 3 |
| S3 | (m_A0, m_A2), each in 3 equal bins on [0, 1] | ≤ 9 (6 reachable) |
| S4 | S2 × disagreement D = (1/N) Σ_i Σ_k (p_ik − m_k)², split at the median | 6 |
| S5 | S2 × memory fragility F: mean over agents of the expected total-variation change in p_i if each remembered fact is kept with probability 0.75 (exact up to 8 facts, else 256 fixed-seed draws); split at the median | 6 |
| S6 | S2 × board evidence b = b_0 − b_2, b_k = mean over agents of P(A_k \| facts in that agent's window, the 48 most recent posts not its own); split at the median | 6 |
| V | votes: share voting A0 − share voting A2, 3 equal bins on [−1, 1] | 3 |

Medians per ρ, pooled over the three arms and t = 5…30, so the mapping is the same in every arm; the upper
bin is "≥ median". For b the median is 1, so S6's second coordinate is "the board proves A0" against "it
does not". At ρ 1 the median of D is 0, so S4 equals S2 there.

**Markov tests**, on t = 5…30 (every agent has voted; the controller acts), steps of 1 time unit,
25 transitions per episode, about 25,000 per arm:

| test | what it asks | pass if |
|---|---|---|
| order | does the previous state improve held-out prediction of the next? (order 2 against order 1, 10 random 70/30 episode splits) | gain < 0.005 nats per step |
| CK (Chapman–Kolmogorov) | does the one-step kernel, raised to the 5th power, match the observed 5-step kernel? (weighted total variation) | gap < 0.05 |
| CMI | is I(S_t+1; S_t−1 \| S_t) zero? (200 within-state permutations) | p > 0.05 |
| lump (lumpability) | within one state, do swarms that differ in a hidden feature (D, F, b, mean facts held, decisive facts still held by anyone) move on differently? (median split, weighted total variation) | largest gap < 0.05 |

A chain that sits in one state for ≥ 95% of the window is reported as **absorbed**: the tests say nothing
about it.

**Readability** I(Z; X): Z the target (A0 or A2, fair coin), X the state of one episode at time t, plug-in
from the 1,000 episodes of each controlled arm; 95% bootstrap interval; root-mean-square error of the
estimate from 100 paths per arm (200 subsets). Reference rows: the one-agent vote state of the earlier
reports, and its fact-implied analogue.

## Table 1. Which states pass each Markov test

Each cell: order / CK / CMI / lump (✓ pass, ✗ fail); "absorbed (x%)": the chain sits in one state x% of
the time.

| state | ρ 0.75, silent | ρ 0.75, target A0 | ρ 0.75, target A2 | ρ 1, silent | ρ 1, target A0 | ρ 1, target A2 |
|---|---|---|---|---|---|---|
| FV | absorbed (97%) | absorbed (100%) | ✗ ✗ ✗ ✗ | absorbed (100%) | absorbed (100%) | absorbed (100%) |
| S1 | ✗ ✗ ✗ ✗ | ✗ ✗ ✗ ✗ | ✗ ✗ ✗ ✗ | absorbed (100%) | absorbed (100%) | absorbed (100%) |
| S2 | ✗ ✗ ✗ ✗ | absorbed (100%) | ✗ ✗ ✗ ✗ | absorbed (100%) | absorbed (100%) | absorbed (100%) |
| S3 | ✗ ✗ ✗ ✗ | absorbed (97%) | ✗ ✗ ✗ ✗ | absorbed (100%) | absorbed (100%) | absorbed (100%) |
| S4 | ✗ ✗ ✗ ✗ | ✗ ✗ ✗ ✗ | ✗ ✗ ✗ ✗ | absorbed (100%) | absorbed (100%) | absorbed (100%) |
| S5 | ✗ ✗ ✗ ✗ | ✗ ✗ ✗ ✗ | ✗ ✗ ✗ ✗ | ✓ ✓ ✗ ✓ | ✓ ✓ ✗ ✓ | ✓ ✓ ✗ ✓ |
| S6 | ✗ ✗ ✗ ✓ | ✗ ✓ ✗ ✓ | ✗ ✗ ✗ ✗ | ✗ ✗ ✗ ✗ | ✗ ✗ ✗ ✗ | ✗ ✗ ✗ ✗ |
| V | ✗ ✓ ✗ ✗ | absorbed (100%) | ✗ ✗ ✗ ✗ | absorbed (100%) | absorbed (100%) | absorbed (100%) |

The values behind it: order gain (nats/step) / CK gap at 5 steps / CMI excess over the null (bits) /
largest lumpability gap (the hidden feature responsible).

| state | ρ 0.75, silent | ρ 0.75, target A0 | ρ 0.75, target A2 | ρ 1, silent | ρ 1, target A0 | ρ 1, target A2 |
|---|---|---|---|---|---|---|
| FV | — | — | 0.033 / 0.14 / 0.048 / 0.22 (b) | — | — | — |
| S1 | 0.027 / 0.17 / 0.040 / 0.23 (b) | 0.012 / 0.09 / 0.019 / 0.11 (F) | 0.029 / 0.11 / 0.042 / 0.30 (b) | — | — | — |
| S2 | 0.016 / 0.08 / 0.025 / 0.39 (b) | — | 0.030 / 0.16 / 0.042 / 0.21 (b) | — | — | — |
| S3 | 0.022 / 0.12 / 0.035 / 0.31 (b) | — | 0.036 / 0.16 / 0.053 / 0.28 (b) | — | — | — |
| S4 | 0.037 / 0.17 / 0.055 / 0.32 (b) | 0.011 / 0.08 / 0.017 / 0.12 (F) | 0.042 / 0.16 / 0.064 / 0.25 (b) | — | — | — |
| S5 | 0.040 / 0.20 / 0.061 / 0.39 (b) | 0.021 / 0.11 / 0.031 / 0.13 (D) | 0.037 / 0.16 / 0.056 / 0.33 (b) | 0.001 / 0.03 / 0.004 / 0.03 (D) | 0.001 / 0.02 / 0.004 / 0.02 (D) | 0.001 / 0.02 / 0.003 / 0.02 (D) |
| S6 | 0.042 / 0.12 / 0.064 / 0.05 (decisive alive) | 0.014 / 0.04 / 0.022 / 0.02 (facts held) | 0.033 / 0.12 / 0.048 / 0.12 (D) | 0.019 / 0.18 / 0.029 / 0.12 (facts held) | 0.020 / 0.17 / 0.029 / 0.07 (facts held) | 0.022 / 0.17 / 0.034 / 0.12 (facts held) |
| V | 0.015 / 0.04 / 0.020 / 0.33 (b) | — | 0.022 / 0.11 / 0.033 / 0.37 (b) | — | — | — |

## Table 2. Readability I(Z; X), in bits

| state | cells used | ρ 0.75, t 10 | ρ 0.75, t 30 | ρ 1, t 10 | ρ 1, t 30 | error at 100 paths (ρ 0.75, t 30) |
|---|---|---|---|---|---|---|
| FV | 3 | 0.133 [0.115, 0.153] | 0.236 [0.215, 0.257] | 0 | 0 | ±0.033 |
| S1 | 4 | 0.333 [0.299, 0.368] | 0.494 [0.455, 0.533] | 0 | 0 | ±0.061 |
| S2 | 3 | 0.198 [0.174, 0.221] | 0.297 [0.271, 0.324] | 0 | 0 | ±0.045 |
| S3 | 5 | 0.301 [0.273, 0.332] | 0.451 [0.416, 0.487] | 0 | 0 | ±0.052 |
| S4 | 6 | 0.443 [0.413, 0.480] | 0.612 [0.580, 0.648] | 0 | 0 | ±0.049 |
| S5 | 6 | 0.533 [0.502, 0.569] | 0.632 [0.605, 0.662] | 0.308 [0.272, 0.344] | 0.401 [0.361, 0.439] | ±0.048 |
| S6 | 5 | 0.208 [0.184, 0.234] | 0.303 [0.278, 0.330] | 0.008 [0.004, 0.017] | 0.092 [0.072, 0.118] | ±0.043 |
| **V** (votes) | 3 | 0.083 [0.069, 0.099] | 0.099 [0.085, 0.112] | 0 | 0 | ±0.022 |
| **one agent, votes** (earlier reports) | 3 | 0.034 [0.030, 0.039] | 0.053 [0.046, 0.059] | 0 | 0 | ±0.010 |
| one agent, fact-implied votes | 3 | 0.069 [0.063, 0.075] | 0.117 [0.109, 0.124] | 0.001 | 0 | ±0.013 |

## Findings

1. **No candidate is Markov at ρ 0.75**, in any arm. Every one fails the order test (gain 0.011–0.042
   nats/step, two to eight times the threshold) and the conditional-independence test. With about 25,000
   transitions per arm the CMI test detects very small violations; its excess (0.017–0.064 bits) is the
   better guide to size. The smallest violations are in V and S2 under silence and in S4 and S1 under the
   A0 controller.
2. **The missing coordinate at ρ 0.75 is the board.** In 13 of the 16 tested (state, arm) pairs at ρ 0.75
   that lack board evidence, b is the hidden feature that most changes what happens next (all of them
   under silence and under the A2 controller; under the A0 controller it is fragility or disagreement). S6, which
   includes it, passes lumpability under silence and the A0 controller. This matches the exact state
   (agents' facts **and** the board). S6 still fails the order test: a single "board proves A0" bit is too
   crude an encoding.
3. **At ρ 1 the belief- and vote-based states are absorbed** during t = 5…30: every agent proves A0 by
   t ≈ 10, so S1–S4, FV and V sit in one cell and say nothing. Only S5 and S6 move, through coordinates
   that keep changing (fragility, the board). S5 passes three of four tests there, but its chain is mostly
   a slow drift of the fragility bit, so this says little about control. The dynamics at ρ 1 happen during
   t < 10, a transient that a time-independent chain does not describe.
4. **Readability: fact-based states read the target far better than votes.** At ρ 0.75, t 30:
   - V reads 0.10 bits, one agent's vote 0.05;
   - the same states computed from facts read more: FV 0.24, one agent's fact-implied vote 0.12.

   Probability matching adds noise between what the facts imply and what is voted.
5. **Bin placement matters as much as the variable.** S1 (cuts 0.50 / 0.75) reads 0.49 bits against S2's
   0.30. The swarm sits near A0, where S2's equal thirds put almost everything in one bin. This is also
   why S2, S3, FV and V are absorbed under the A0 controller.
6. **Disagreement and fragility add the most readability** (S4 0.61, S5 0.63 bits), and S5 reads 0.40
   bits at ρ 1 where every belief is identical. They do not measure where the swarm leans. They measure
   how the controller reshaped the agents' memories:
   - under the A2 controller agents disagree more (D 0.14 against 0.06);
   - their proofs rest on more facts, including the controller's own true A2 facts (F 0.22 against 0.12
     at ρ 1).

   This is real and reasoner-free, but it is a different question from "did the swarm move toward the
   target".
7. **Precision at 100 paths**: ±0.02–0.06 bits for the swarm states, ±0.01 for one-agent states. The
   six-cell states (S4–S6) cost about ±0.05 at 100 paths.

## Caveats

One simulation, setup and setting (q 6, q_c 12, λ_c 1, gate on). The tests assume a chain whose rules do
not change over t = 5…30; the build-up before t = 5 and the relaxation after t = 30 are excluded. Median
splits are the crudest encoding of D, F and b. Plug-in readability; with ≤ 6 cells and 1,000 episodes per
arm its bias is below 0.003 bits.
