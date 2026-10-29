# object_placement / no_solution (15 agents): No proof; agents and the single controller pool are each approximately uniform

**Built 29 October 2026** by [`../../build_setups.py`](../../build_setups.py) (deterministic; rebuild rather than edit). World: MuSR Object Placements (where Casey looks for the keys; A0 = drawer is correct, A2 = desk is the false target).

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
| agents' joint posterior (A0, A1, A2) | (0.333, 0.333, 0.333); max deviation from uniform 0.0 |
| agents' single-fact leanings A0 / A1 / A2 (neutral facts excluded) | 3 / 0 / 5 |
| neutral agent facts (no answer above 1/3 + 0.01 when read alone) | 7 |
| mean single-fact belief A0 / A1 / A2 | (0.331, 0.309, 0.360) |
| agent facts also in a controller pool | none |
| `controller_pool.json`: joint posterior | (0.333, 0.333, 0.333) |
| `controller_pool.json`: leanings A0 / A1 / A2, neutral facts | 2 / 2 / 2, 6 |
| `controller_pool.json`: new facts for the agents | 12 of 12 |
| agents + whole `controller_pool.json` | (0.667, 0.000, 0.333): proves nothing |

**Constraint of this world.** Object Placements has 34 true facts, 18 of them neutral when read alone, and only 2 informative facts leaning A1. Since 27 distinct facts are needed, at least 11 must be neutral; the search splits them so that agents and pool carry the same neutral fraction. Here both A1-leaning facts ended in the pool.
