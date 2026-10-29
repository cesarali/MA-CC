# object_placement / symmetric (15 agents): Agents can prove the answer; two controller pools

**Built 29 October 2026** by [`../../build_setups.py`](../../build_setups.py) (deterministic; rebuild rather than edit). World: MuSR Object Placements (where Casey looks for the keys; A0 = drawer is correct, A2 = desk is the false target).

The 15 agents' facts together prove A0 (P(A0) = 1), but no single fact or pair does. They are a subset of the 18 distinct agent facts of the original 24-agent setup, chosen to keep the proof, minimise overlap with the controller pools and balance single-fact leanings. The two controller pools are the original ones, unchanged: the A0 pool is used when the controller targets A0 (truth arm), the A2 pool when it targets A2 (false arm).

## Files

| file | what it is |
|---|---|
| `agents.json` | `agent_assignments` (15 agents, one fact each), every fact's text and strength, and checked properties |
| `controller_pool_A0.json` | the 12-fact pool used when the controller targets ALLOCATION_0 (`fact_ids` is the part code uses) |
| `controller_pool_A2.json` | the 12-fact pool used when the controller targets ALLOCATION_2 (`fact_ids` is the part code uses) |

## Properties

| | |
|---|---|
| agents, facts per agent, distinct facts | 15, 1, 15 |
| agents' joint posterior (A0, A1, A2) | (1.000, 0.000, 0.000): proves ALLOCATION_0 |
| agents' single-fact leanings A0 / A1 / A2 (neutral facts excluded) | 4 / 1 / 3 |
| neutral agent facts (no answer above 1/3 + 0.01 when read alone) | 7 |
| mean single-fact belief A0 / A1 / A2 | (0.360, 0.313, 0.327) |
| agent facts also in a controller pool | belief:1:0:0:1, belief:2:1:1:0, belief:3:2:0:1, item:2:0, missed:1:2 |
| `controller_pool_A0.json`: joint posterior | (0.977, 0.000, 0.023) |
| `controller_pool_A0.json`: leanings A0 / A1 / A2, neutral facts | 1 / 0 / 2, 9 |
| `controller_pool_A0.json`: new facts for the agents | 9 of 12 |
| agents + whole `controller_pool_A0.json` | (1.000, 0.000, 0.000): proves ALLOCATION_0 |
| `controller_pool_A2.json`: joint posterior | (0.018, 0.000, 0.982) |
| `controller_pool_A2.json`: leanings A0 / A1 / A2, neutral facts | 2 / 2 / 3, 5 |
| `controller_pool_A2.json`: new facts for the agents | 10 of 12 |
| agents + whole `controller_pool_A2.json` | (1.000, 0.000, 0.000): proves ALLOCATION_0 |
