# task003 / no_solution (15 agents): No proof; agents and the single controller pool are each approximately uniform

**Built 29 October 2026** by [`../../build_setups.py`](../../build_setups.py) (deterministic; rebuild rather than edit). World: task003 (allocation puzzle: Alice, Bruno, Chandra; A0 is correct, A2 is the false target).

The 15 agents' facts together give approximately (1/3, 1/3, 1/3): they favour no answer and prove nothing. The single 12-fact controller pool, used for both targets, is also approximately uniform. No fact is in both. Agents and pool together prove nothing. Found by a seeded joint search (`../../build_setups.py`).

## Files

| file | what it is |
|---|---|
| `agents.json` | `agent_assignments` (15 agents, one fact each), every fact's text and strength, and checked properties |
| `controller_pool.json` | the 12-fact pool used when the controller targets ALLOCATION_0 or ALLOCATION_2 (`fact_ids` is the part code uses) |

## Properties

| | |
|---|---|
| agents, facts per agent, distinct facts | 15, 1, 15 |
| agents' joint posterior (A0, A1, A2) | (0.325, 0.350, 0.325); max deviation from uniform 0.0167 |
| agents' single-fact leanings A0 / A1 / A2 (neutral facts excluded) | 5 / 5 / 5 |
| neutral agent facts (no answer above 1/3 + 0.01 when read alone) | 0 |
| mean single-fact belief A0 / A1 / A2 | (0.313, 0.332, 0.355) |
| agent facts also in a controller pool | none |
| `controller_pool.json`: joint posterior | (0.342, 0.329, 0.329) |
| `controller_pool.json`: leanings A0 / A1 / A2, neutral facts | 4 / 4 / 4, 0 |
| `controller_pool.json`: new facts for the agents | 12 of 12 |
| agents + whole `controller_pool.json` | (0.750, 0.000, 0.250): proves nothing |
