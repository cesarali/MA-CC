# llmfree_core — code and data shared by every LLM-free simulation

Simulation 1 ([`../simulation_1/`](../simulation_1/)) and Simulation 2 (to come,
`../simulation_2/`) both import from here. Nothing here depends on one
simulation's rules.

| file | what it is |
|---|---|
| `world.py` | the task_003 world: exact posteriors P(A0), P(A1), P(A2) given a fact set, and proof checks |
| `setups.py` | loads a frozen setup from [`../experimental_setup/`](../experimental_setup/): the agents' facts and the controller pool for each target |
| `world_task003.json` | task_003's 14,388 possible worlds and 49 true facts, exported once |
| `export_world.py` | the one script that imports MA-CC code; it wrote `world_task003.json` and does not need to run again |
| [`runtime_rules.md`](runtime_rules.md) | the real game's rules the simulators copy, and where they live in MA-CC |

A **fact set** is stored as one integer: bit *i* set means fact *i* of
`world_task003.json` is in the set. Example: 33 = 0b100001 = facts 0 and 5.

Import it with the folder `analysis/task003_llm_free/` on the Python path:

```python
from llmfree_core.world import World
from llmfree_core.setups import load_setup
```

**Changing anything here changes every simulation.** The runners treat an
uncommitted change in this folder as uncommitted simulator code.
