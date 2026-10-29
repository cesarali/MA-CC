# 15-agent campaign: pre-registration

29 October 2026. **Author: Claude.** Written before any official run. Configs:
`configs/sim2_15agents_task003.yaml` and `configs/sim2_15agents_op.yaml`. Setups: `README.md` and the four setup folders.

## Design

| | |
|---|---|
| simulator | Simulation 2 (`task003_llm_free/simulation_2/simulator/`), unchanged except the backward-compatible additions of 29 Oct 2026 (setup registry, `read_mode`, `record.agent_snapshot_times`) |
| setups | task003 and Object Placements, each symmetric (agents prove A0; A0 and A2 pools) and uniform no-solution (agents and a single pool each about (1/3, 1/3, 1/3)); 15 agents, one distinct fact each |
| arms | false controller (target A2) and its paired silent arm (same keyed random streams). No truth arm. |
| agents | N = 15, rate 1, W = 30 (own posts excluded), probability-matching votes, support-based posting |
| q | 1, 2, 3, 4, 6, 8, 12, 16 (log scale) |
| ρ | 0.018, 0.135, 0.37, 0.61, 0.78, 0.88, 0.94, 1: mean memory lifetimes τ = −1/ln ρ ≈ 0.25, 0.5, 1, 2, 4, 8, 16, ∞ |
| reading | `posts` (standard) and `distinct_facts` (deduplicated: q distinct facts, each through its newest copy) |
| controller | rate 1, q_c = 8, budget 30, vote gate at 0.75, proof stop; stops at t = 30 |
| horizon | t = 130 (100 time units after the stop); snapshots every 0.25 |
| paths | 1,000 per cell; 512 cells per task, 1,024 in all |
| recording | posts, controller, snapshots, episodes; each agent's memory and last vote at t = 10, 20, 30, 31, 35, 40, 60, 80, 100, 130 |

## Measures (fixed now)

- **G(t):** paired controlled-minus-silent difference in the share of A2 among agents who have voted (`share_A2 /
  share_voted`), averaged over the 1,000 episode pairs. Standard error over pairs.
- **Reading increment:** D(q₁→q₂) = G(30; q₂) − G(30; q₁), for consecutive q in the grid.
- **Crossover:** the ρ interval in which D changes sign from positive (shorter memory) to negative (longer memory).
- **Persistence:** retained fraction G(130)/G(30), reported only where G(30) exceeds 2 standard errors.
- **Silent accuracy:** silent share of A0 among voters at t = 30. For no-solution, silent A2 share instead, since
  there is no correct answer the agents can reach alone.

## Predictions (standard reading unless stated)

- **P1 (reversal exists).** In each of the four setups, at least one consecutive pair among q = 1→2, 2→3, 3→4, 4→6
  shows D > 0 at some memory and D < 0 at a longer memory, each beyond 2 standard errors.
- **P2 (crossover moves to shorter memory with q).** Within a setup, the crossover interval does not move to longer
  memory as the q pair increases, for all pairs where a crossover exists. At most one violation per setup.
- **P3 (setup order).** In each task, the symmetric setup crosses at shorter or equal memory than the no-solution setup
  for q = 3→4 and 4→6. The basis: stronger counter-evidence makes the protective sensitivity term act earlier. Low
  confidence for Object Placements, where the old no-proof reversal was not explained.
- **P4 (persistence, task contrast).** At q = 6 and ρ ∈ {0.37, 0.61, 0.78}, task003 retains more of its effect than
  Object Placements in at least 4 of the 6 (setup type × ρ) comparisons. task003 retains at least 0.7 in each of its six.
- **P5 (deduplicated reading lowers the effect).** At q ∈ {3, 4, 6, 8} and ρ ≤ 0.61, G(30) is lower under
  deduplicated reading in at least 80% of the 4 setups × 4 q × 3 ρ = 48 cells, and by at least 15% on average.
- **P6 (deduplicated reading is not just ignorance).** In those same cells, silent accuracy under deduplication is not
  lower by more than 0.02 (symmetric: share A0; no-solution: no rise of A2 share above 0.02).
- **P7 (the defence keeps the reversal).** P1 also holds under deduplicated reading.

## Not predicted

How G changes with 15 instead of 24 agents. Two forces pull in opposite directions: fewer copies per fact favour the
controller, and q_c = 8 instead of 12 weakens it. Behaviour at q = 1 and at ρ = 0.018 beyond the predictions above.

## Analysis plan

One dated report per analysis session in `reports/YYYY-MM-DD_<topic>/`. Each report evaluates P1–P7 as met, partly met
or failed, with the numbers, before any exploratory analysis.
