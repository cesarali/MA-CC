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

# Step 2: the selection model (28 October 2026)

Pre-registration: `../../theory_blackboard/2026-10-28_claude_13.md`. Code: `selection_model.py` (pre-registered),
`coherence_world.py` (Q5, plus a post-hoc majority run), `posthoc_board_coherence.py` (post-hoc). Outputs:
`selection_grid.csv`, `selection_paths.csv`, `selection_majority_posthoc.csv`, `coherence_facts.csv`,
`coherence_summary.csv`, `board_coherence.csv`.

## The model in one paragraph

This is the fact-flow mean field, with each of the 4 controller facts tracked separately. Each fact has a coherence c_f:
the share of receivers' contexts in which it pushes toward A2 (+β_C in log-odds) rather than away (−β_C). Agents repost
a fact when it supports their current answer, with weight ε + (1 − ε)·(chance that the holder's answer is the one the
fact supports). A fact pushing toward A2 is reposted by A2 voters, and a fact pushing away by the others. When most
agents do not vote A2 (x < 0), facts of low coherence are reposted more. Exactly: g_f increases with c_f if and only if
σ(x + β) > 1 − σ(x − β), that is, if and only if x > 0.

## Pre-registered predictions

| | prediction | outcome |
|---|---|---|
| Q1 | ambivalent set (c = 0.9, 0.7, 0.5, 0.3), ε < 1: c̄ falls and retained δ ≥ 0.1 below retained b_C | **partly met:** c̄ falls in all four runs; the fading gap beyond exposure is 0.15 at ε = 0.5, but only 0.06–0.08 at ε = 0.75 |
| Q2 | coherent set (c = 0.97): retained δ within 0.05 of retained b_C | **met** (gap 0.00–0.03) |
| Q3 | ε = 1: c̄ changes < 0.01 | **met** (no change) |
| Q4 | prior h₀ = +ln 2: ambivalent c̄ rises | **failed as run:** the prior did not create an A2 majority (Φ₂ = 0.42 at most), so c̄ still fell. Post-hoc, with h₀ raised until Φ₂ > ½ at the stop, c̄ is nearly flat (−0.006, −0.003), not rising. The flip is guaranteed by the equation above only while x > 0, and the controlled arm sits barely above ½ and falls back after the stop. Inconclusive |
| Q5 | world coherence (silent memories, no votes): task003 facts > 0.85 with sd < 0.1; OP spread, sd > 0.15 | **failed:** task003 symmetric fits (sd 0.09, min 0.58), but **task003 no-proof is the most spread pool** (sd 0.38; 5 of 12 facts push *away* from A2 in almost every silent memory). OP: sd 0.23–0.24, as predicted |

Model susceptibility S(130)/S(30) at ε = 0.5: coherent 0.88 and 0.79, ambivalent 0.03 and 0.74 (ρ 0.25, 0.75). The
contrast is large at ρ 0.25 and small at ρ 0.75, where the simulator's contrast is largest. At ε = 0.75 the ambivalent
set falls less than the coherent one at ρ 0.75 (0.93 against 0.88). So the model gets the direction of the task
contrast, not its pattern across memory.

## Post-hoc, prompted by Q5's failure: coherence of the facts that actually circulate

World coherence weighted by each fact's share of the controlled arm's board (`board_coherence.csv`):

| cell | c̄ at t = 30 | c̄ at t = 130 | change |
|---|---:|---:|---:|
| task003 no-proof ρ 0.25 / 0.75 | 0.860 / 0.885 | 0.857 / 0.891 | −0.003 / +0.006 |
| task003 symmetric ρ 0.25 / 0.75 | 0.773 / 0.797 | 0.772 / 0.798 | −0.001 / +0.001 |
| OP no-proof ρ 0.25 / 0.75 | 0.532 / 0.457 | 0.524 / 0.428 | −0.008 / −0.029 |
| OP symmetric ρ 0.25 / 0.75 | 0.489 / 0.624 | 0.460 / 0.454 | −0.029 / −0.171 (83% of episodes have no controller post left) |

**Reading.** The controller circulates coherent facts when the pool has them. task003 no-proof contains facts of
coherence 1.00 and 0.98, even though its pool is spread overall. Object Placements' best fact has coherence 0.75, so OP
circulates facts near ½. Their net persuasion is a small difference of opposite pushes, and support-based reposting by a
non-A2 majority lowers their coherence further. The relevant parameter is therefore the **coherence of the circulating
controller facts** (in practice, the best the pool offers), **not the spread of the pool**. This is post-hoc and must
be tested on a setup not used to find it.

## What this explains and what it does not

- **Explained, qualitatively:**
  - why OP fades at constant exposure while task003 does not;
  - why susceptibility falls only in OP;
  - why survivors are more persuasive in task003 and less in OP no-proof ρ 0.75.
- **Not explained:**
  - OP no-proof at ρ 0.25: susceptibility falls 36% while c̄ falls only 0.008, and there is no identity selection. The
    strengths change at fixed facts, as receivers' contexts change.
  - the size of the fading at ε = 0.75 (model gap 0.06–0.08, simulator δ ratio 0.37 against flat exposure);
  - the sign flip under an A2 majority (not shown).
