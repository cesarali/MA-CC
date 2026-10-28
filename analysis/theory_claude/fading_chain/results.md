# Fading chain: results of step 1 (28 October 2026)

Pre-registration: `../../theory_blackboard/2026-10-28_claude_12.md`. Code: `fading_chain.py` (pre-registered) and
`posthoc_survival.py` (post-hoc, added after seeing step 1). Outputs: `chain_summary.csv`, `chain_episodes.csv`,
`survival.csv`. Stored normal-continuation runs, q 6, 300 episodes per cell. Run time about 10 s.

Quantities marked "cond." are averaged over episodes with a controller post on the board at that time (≥ 99.7% of
episodes in both no-proof setups and in task003; 74% and 17% in Object Placements symmetric at ρ 0.25 and 0.75).

## The chain, t = 30 → 130 (relative change)

| cell | δ | b_C (exposure) | distinct C facts on board | C facts per agent | S (cond.) | S with t = 30 strengths |
|---|---:|---:|---:|---:|---:|---:|
| OP no-proof ρ 0.25 | 0.049 → 0.020 | −1% | −45% | −27% | −36% | +2% |
| OP no-proof ρ 0.75 | 0.045 → 0.005 | +5% | −32% | −20% | −80% | +5% |
| task003 no-proof ρ 0.25 | 0.122 → 0.118 | −2% | −25% | −11% | −4% | 0% |
| task003 no-proof ρ 0.75 | 0.190 → 0.194 | +3% | −15% | −7% | +5% | −2% |
| task003 symmetric ρ 0.25 | 0.169 → 0.170 | −2% | −29% | −13% | −6% | 0% |
| task003 symmetric ρ 0.75 | 0.128 → 0.111 | −3% | −27% | −19% | −1% | −1% |

Encountered (agent, controller fact) pairs not already held, at t = 30 → 130 (cond.):

| cell | share pushing toward A2 | mean positive θ | mean negative \|θ\| | already held |
|---|---|---|---|---|
| OP no-proof ρ 0.25 | 0.66 → 0.62 | 0.32 → 0.25 | 0.09 → 0.09 | 0.29 → 0.36 |
| OP no-proof ρ 0.75 | 0.53 → 0.53 | 0.45 → 0.41 | 0.35 → 0.44 | 0.57 → 0.66 |
| task003 no-proof ρ 0.25 | 0.96 → 0.99 | 0.66 → 0.63 | 0.19 → 0.10 | 0.32 → 0.35 |
| task003 no-proof ρ 0.75 | 0.86 → 0.93 | 1.00 → 1.05 | 0.36 → 0.36 | 0.67 → 0.70 |

## Pre-registered predictions

- **P1 (controller diversity loss at constant share): met.**
  - OP no-proof: b_C within ±10%.
  - Distinct controller facts on the board fall 1.8–2.1× as much as in task003 no-proof (1.5–1.7× against the average
    of both task003 setups at the same ρ).
  - Controller facts held per agent fall 2.5–2.8× as much.
  - **But task003 also loses 15–29% of its controller facts without fading.** Diversity loss is real but does not by
    itself produce fading.
- **P2 (the operating point does not explain the fall): met.** With strengths frozen at t = 30, susceptibility stays
  flat (+2%, +5%) where the measured one falls 36% and 80%. The receivers' log-odds moving along the sigmoid explains
  none of it.
- **P3 (magnitudes, not signs): failed.**
  - The positive share is stable, as predicted.
  - But the positive magnitude falls 22% and 8% (predicted > 25%).
  - At ρ 0.75 the negative magnitude grows 27%.
  - Already-held pairs rise by 7–9 points, so redundancy contributes too: a held-adjusted counterfactual falls 8% and
    17%.

## New structural fact

In task003, 86–99% of encountered controller-fact pairs push toward A2. In Object Placements only 53–70% do. An OP
controller fact's net persuasion is a small difference of pushes in both directions. It can therefore approach zero
without any change in exposure.

## Post-hoc: which controller facts survive? (`survival.csv`)

Within episode, controller facts on the board at t = 30 that are still on it at t = 130 compared with those lost. The
measure is the mean add-one persuasion at t = 30 over agents not holding the fact; the paired difference is survivors
minus lost, with its standard error.

| cell | paired difference | episodes |
|---|---:|---:|
| task003 no-proof ρ 0.25 | +0.044 ± 0.009 | 171 |
| task003 no-proof ρ 0.75 | +0.135 ± 0.007 | 128 |
| task003 symmetric ρ 0.25 | +0.033 ± 0.006 | 216 |
| task003 symmetric ρ 0.75 | +0.055 ± 0.003 | 223 |
| OP no-proof ρ 0.25 | −0.003 ± 0.004 | 288 |
| OP no-proof ρ 0.75 | **−0.026 ± 0.004** | 275 |
| OP symmetric ρ 0.25 | +0.005 ± 0.005 | 205 |
| OP symmetric ρ 0.75 | −0.004 ± 0.010 | 41 |

**Reading:**
- Selection among controller facts after the stop has **opposite signs** in the two tasks.
- task003 keeps its persuasive facts.
- OP no-proof at ρ 0.75 keeps its less persuasive ones; this is the cell with the strongest fading.
- At OP no-proof ρ 0.25 there is no selection by identity. Its 36% drop must come from strengths changing at fixed
  facts, which is not explained here.

## Candidate mechanism (engine-free; to be tested in step 2)

Agents repost a fact that supports their current answer.
- **task003:** a controller fact pushes almost every receiver toward A2. The agents who repost it are the agents it
  converted, so persuasive facts recruit their own reposters.
- **Object Placements:** a controller fact pushes some receivers toward A2 and others away. The majority does not vote
  A2, so it reposts the controller facts that, in its own context, support its answer. These are the facts that
  persuade least.

The sign of selection is set by the **coherence** c_f of the controller facts (the share of contexts in which f pushes
toward A2) together with which side holds the majority.
