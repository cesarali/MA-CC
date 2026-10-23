# op_symmetric: agents can prove the answer; two controller pools

**Built 22 October 2026** by [`../../simulation_2/analysis/build_op_setups.py`](../../simulation_2/analysis/build_op_setups.py) (seeded;
rebuild rather than edit). Same design rules as `task003_symmetric`, on a MuSR Object Placements world.

## The world (MuSR Object Placements, [`../../world/world_op1.json`](../../world/world_op1.json))

Three people (Alex, Blair, Casey), two items (notebook, keys), three locations (desk, shelf, drawer), three moves; each person
other than the mover sees a move with chance 1/3 (MuSR's generator, reproduced exactly in `task003_llm_free/llmfree_core/op_world.py`).

- The notebook starts on the shelf.
- The keys start on the desk.
- Move 1: Casey moves the keys to the drawer; seen by Alex.
- Move 2: Alex moves the keys to the desk; seen by Blair.
- Move 3: Alex moves the notebook to the drawer; seen by Blair.

**Question:** where would Casey look for the keys? **A0 = drawer** (Casey's belief: correct),
A1 = shelf, **A2 = desk** (where the keys really are: the false controller's target).
The world has 34 true facts (catalog in `task003_llm_free/llmfree_core/op_world.py`); beliefs are exact over the generator's 617,472
possible stories weighted by their probability.

## Properties

| | |
|---|---|
| agents, facts per agent, distinct facts | 24, 1, 18 (6 held twice) |
| agents' joint posterior | (1.0, 0.0, 0.0): proves A0; no single fact and no pair proves it |
| A2 pool alone (12 facts, 10 new to the agents) | (0.018, 0.0, 0.982) |
| A0 pool alone (12 facts: the 6 remaining true facts + 6 held by agents) | (0.977, 0.0, 0.023) |
| agents + A2 pool | (1.0, 0.0, 0.0) |

**Differences from task003_symmetric:** the story has only 34 true facts (task003: 49), so the A0 pool brings 6 new
facts instead of 10; the A2 pool is stronger (P(A2) 0.98 against 0.89).

## Agent facts (fact, text, agents holding it)

| fact | text | agents |
|---|---|---|
| `belief:1:0:0:1` | After move 1, Alex thinks the notebook is on the shelf. | 1 |
| `belief:1:0:1:2` | After move 1, Alex thinks the keys are on the drawer. | 1 |
| `belief:1:2:1:2` | After move 1, Casey thinks the keys are on the drawer. | 2 |
| `belief:2:0:0:1` | After move 2, Alex thinks the notebook is on the shelf. | 1 |
| `belief:2:0:1:0` | After move 2, Alex thinks the keys are on the desk. | 2 |
| `belief:2:1:1:0` | After move 2, Blair thinks the keys are on the desk. | 1 |
| `belief:3:0:0:2` | After move 3, Alex thinks the notebook is on the drawer. | 1 |
| `belief:3:2:0:1` | After move 3, Casey thinks the notebook is on the shelf. | 1 |
| `dest:1:0` | Move 2 puts the item on the desk. | 1 |
| `dest:2:2` | Move 3 puts the item on the drawer. | 2 |
| `item:2:0` | Move 3 moves the notebook. | 1 |
| `missed:0:1` | Blair did not see move 1. | 1 |
| `missed:1:2` | Casey did not see move 2. | 2 |
| `missed:2:2` | Casey did not see move 3. | 2 |
| `mover:0:2` | Move 1 is made by Casey. | 2 |
| `saw:0:0` | Alex saw move 1. | 1 |
| `saw:1:1` | Blair saw move 2. | 1 |
| `saw:2:1` | Blair saw move 3. | 1 |

## A2 pool

| fact | text |
|---|---|
| `start:1:0` | The keys start on the desk. |
| `mover:1:0` | Move 2 is made by Alex. |
| `belief:3:0:1:0` | After move 3, Alex thinks the keys are on the desk. |
| `item:1:1` | Move 2 moves the keys. |
| `mover:2:0` | Move 3 is made by Alex. |
| `belief:1:1:1:0` | After move 1, Blair thinks the keys are on the desk. |
| `belief:1:2:0:1` | After move 1, Casey thinks the notebook is on the shelf. |
| `belief:3:1:0:2` | After move 3, Blair thinks the notebook is on the drawer. |
| `dest:0:2` | Move 1 puts the item on the drawer. |
| `belief:2:2:0:1` | After move 2, Casey thinks the notebook is on the shelf. |
| `missed:1:2` | Casey did not see move 2. |
| `belief:3:2:0:1` | After move 3, Casey thinks the notebook is on the shelf. |
