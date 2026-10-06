# simulator — the LLM-free simulator

24 agents and a controller play `task_003` with **no language model**: agents are
exact Bayesian reasoners and exchange fact IDs such as `cf_x00_ge_x04`.

| file | what it is |
|---|---|
| [`runtime_rules.md`](runtime_rules.md) | the real game's rules the simulator copies, and where they live in MA-CC |
| [`simulation_1_spec.md`](simulation_1_spec.md) | **Simulation 1**: days and nights, per-round controller budget. Every rule |
| [`simulation_1_config.yaml`](simulation_1_config.yaml) | the first study grid: 156 cells × 1,000 episodes |
| `run_simulation_1.py` | runs the grid, or part of it |
| `llmfree/` | the code: `world.py` (exact posteriors), `setups.py` (loads the frozen setups), `simulation_1.py` (the rules) |
| `world_task003.json` | task_003's 14,388 possible worlds and 49 facts, exported once |
| `export_world.py` | the one script that uses MA-CC code; it wrote `world_task003.json` |
| `tests/` | checks that the rules hold |

## It does not need MA-CC installed

At run time the simulator reads only `world_task003.json` and the setup files in
`../experimental_setup/`. It needs Python 3.11+ with `numpy`, `pandas`,
`pyarrow` and `pyyaml`. Only `export_world.py` imports MA-CC code, and it has
already been run. So this folder can become its own repository by copying it
and pointing `paths.setups_dir` in the config at a copy of the setups.

## Running it

From the repository root:

```bash
# the tests (about 5 seconds)
.venv/bin/python -m pytest analysis/task003_llm_free/simulator/tests -q

# a quick check: 5 episodes of every cell
.venv/bin/python analysis/task003_llm_free/simulator/run_simulation_1.py --episodes 5 --out /tmp/sim1_check

# one cell
.venv/bin/python analysis/task003_llm_free/simulator/run_simulation_1.py \
    --only task003_nosolution__false__q6__qc12__b3__rho0.75

# the full grid on 12 cores (a few minutes on a laptop)
.venv/bin/python analysis/task003_llm_free/simulator/run_simulation_1.py --workers 12
```

The runner refuses to start if the config gives a different number of cells
from `cells.expected_total`.

## Output

Written to `../results/simulation_1/` (not in git). One folder per cell, named
like `task003_nosolution__false__q6__qc12__b3__rho0.75`
(setup, arm, q, qc, b, ρ; silent cells have no qc or b), plus `manifest.json`
with the settings, input-file hashes, git commit and run time.

A **fact set** is stored as one integer: bit *i* set means fact *i* of
`world_task003.json` is in the set. Example: 33 = 0b100001 = facts 0 and 5.

**`days.parquet`**, one row per episode per day, population averages after the day:

| column | meaning |
|---|---|
| `mean_p_A0`, `mean_p_A1`, `mean_p_A2` | mean posterior over the 24 agents |
| `share_A0`, `share_A1`, `share_A2` | vote shares |
| `proof_rate_A0` | share of agents whose active facts prove A0 |
| `mean_active_facts` | mean number of facts an agent remembers |
| `agent_posts` | agent posts that day |
| `controller_posts_on_board_this_morning` | controller posts the agents could read that day |
| `controller_posts_tonight` | facts the controller posted that night |

**`agents.parquet`**, one row per agent per day:

| column | meaning |
|---|---|
| `position` | the agent's place in that day's random order (0 = first) |
| `active_facts` | fact set it remembers after reading |
| `facts_read`, `n_posts_read` | fact set it read today, and how many posts it read |
| `p_A0`, `p_A1`, `p_A2` | its posterior |
| `vote` | 0, 1 or 2 |
| `posted_fact` | fact index it posted, or -1 for none |
| `proves_A0` | its active facts prove A0 |

**`nights.parquet`**, one row per night (controlled arms only):

| column | meaning |
|---|---|
| `n_posts_read` | agent posts the controller read |
| `v_hat` | share of those posts whose author voted for the target (empty if none read) |
| `p_target_given_read` | P(target \| facts it read) |
| `decision` | `proved` (target already proved), `gate` (enough votes already), `posted`, or `fallback` (target ruled out; posted the set strongest on its own) |
| `posted_facts`, `n_posted` | fact set posted, and its size (0 or b) |

## Not implemented yet

Options defined in the spec but refused by the code with a clear error:
`board_read_weighting: recency`, `controller_memory: accumulate`. Simulations
2–4 are not specified yet.
