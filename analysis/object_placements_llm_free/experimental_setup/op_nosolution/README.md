# op_nosolution: agents cannot prove anything; one shared pool

**Built 22 October 2026** by [`../../simulation_2/analysis/build_op_setups.py`](../../simulation_2/analysis/build_op_setups.py) (seeded;
rebuild rather than edit). Same design rules as `task003_nosolution`, on a MuSR Object Placements world.

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
| agents, facts per agent, distinct facts | 24, 1, 15 (9 held twice); no fact alone is more than 45% sure of any answer |
| agents' joint posterior | (0.5, 0.0, 0.5): no proof |
| pool alone (12 facts, none held by agents; both targets) | (0.667, 0.0, 0.333) |
| agents + pool | (0.667, 0.0, 0.333): still no proof |

**Difference from task003_nosolution:** the pool alone favours the truth (task003's pool was balanced, 0.37/0.26/0.37).

## Agent facts (fact, text, agents holding it)

| fact | text | agents |
|---|---|---|
| `belief:1:0:0:1` | After move 1, Alex thinks the notebook is on the shelf. | 1 |
| `belief:1:0:1:2` | After move 1, Alex thinks the keys are on the drawer. | 2 |
| `belief:1:1:0:1` | After move 1, Blair thinks the notebook is on the shelf. | 2 |
| `belief:1:1:1:0` | After move 1, Blair thinks the keys are on the desk. | 2 |
| `belief:1:2:0:1` | After move 1, Casey thinks the notebook is on the shelf. | 1 |
| `belief:2:0:1:0` | After move 2, Alex thinks the keys are on the desk. | 2 |
| `belief:2:1:1:0` | After move 2, Blair thinks the keys are on the desk. | 2 |
| `belief:2:2:0:1` | After move 2, Casey thinks the notebook is on the shelf. | 2 |
| `belief:3:0:0:2` | After move 3, Alex thinks the notebook is on the drawer. | 2 |
| `belief:3:1:0:2` | After move 3, Blair thinks the notebook is on the drawer. | 2 |
| `dest:0:2` | Move 1 puts the item on the drawer. | 1 |
| `item:0:1` | Move 1 moves the keys. | 2 |
| `missed:0:1` | Blair did not see move 1. | 1 |
| `mover:0:2` | Move 1 is made by Casey. | 1 |
| `mover:2:0` | Move 3 is made by Alex. | 1 |

## Pool

| fact | text |
|---|---|
| `belief:1:2:1:2` | After move 1, Casey thinks the keys are on the drawer. |
| `dest:1:0` | Move 2 puts the item on the desk. |
| `item:2:0` | Move 3 moves the notebook. |
| `saw:1:1` | Blair saw move 2. |
| `item:1:1` | Move 2 moves the keys. |
| `mover:1:0` | Move 2 is made by Alex. |
| `saw:0:0` | Alex saw move 1. |
| `start:1:0` | The keys start on the desk. |
| `missed:2:2` | Casey did not see move 3. |
| `belief:2:1:0:1` | After move 2, Blair thinks the notebook is on the shelf. |
| `belief:2:0:0:1` | After move 2, Alex thinks the notebook is on the shelf. |
| `belief:3:2:0:1` | After move 3, Casey thinks the notebook is on the shelf. |
