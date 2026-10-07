# simulator — the LLM-free simulator

24 agents and a controller play `task_003` with **no language model**: agents are
exact Bayesian reasoners and exchange fact IDs such as `cf_x00_ge_x04`.

| file | what it is |
|---|---|
| [`runtime_rules.md`](runtime_rules.md) | the real game's rules the simulator copies, and where they live in MA-CC |
| [`simulation_1_spec.md`](simulation_1_spec.md) | **Simulation 1**: days and nights, per-round controller budget. Every rule |
| [`configs/`](configs/) | one config per study; the grid and every fixed setting |
| [`runs_index.csv`](runs_index.csv) | one line per official run: date, study, git commit, size, results folder |
| [`run_summaries/`](run_summaries/) | each official run's `summary.csv`, committed so results are visible without rerunning |
| `run_simulation_1.py` | runs a study |
| `summarize_simulation_1.py` | one row per cell, compared with its silent baseline (the runner calls it) |
| `llmfree/` | the code: `world.py` (exact posteriors), `setups.py` (loads the frozen setups), `simulation_1.py` (the rules) |
| `world_task003.json` | task_003's 14,388 possible worlds and 49 facts, exported once |
| `export_world.py` | the one script that uses MA-CC code; it wrote `world_task003.json` |
| `tests/` | checks that the rules hold |

## It does not need MA-CC installed

At run time the simulator reads only `world_task003.json` and the setup files in
`../experimental_setup/`. It needs Python 3.11+ with `numpy`, `pandas`,
`pyarrow` and `pyyaml`. Only `export_world.py` imports MA-CC code, and it has
already been run.

## Running it

From the repository root:

```bash
# the tests
.venv/bin/python -m pytest analysis/task003_llm_free/simulator/tests -q

# a quick check, written to a scratch folder and not indexed
.venv/bin/python analysis/task003_llm_free/simulator/run_simulation_1.py \
    --config analysis/task003_llm_free/simulator/configs/sim1_base.yaml --episodes 5 --out /tmp/sim1_check

# an official run on 12 cores
.venv/bin/python analysis/task003_llm_free/simulator/run_simulation_1.py \
    --config analysis/task003_llm_free/simulator/configs/sim1_base.yaml --workers 12
```

- The runner refuses to start if the config gives a different number of cells
  from `cells.expected_total`.
- An official run (no `--episodes`, `--only` or `--out`) writes to a new folder
  `../results/<date>_<study>/`, refuses to overwrite one, appends a line to
  `runs_index.csv`, and copies its summary to `run_summaries/`.
- It warns if `simulator/` has uncommitted changes, since the recorded git
  commit would then not fully describe the run. Commit before official runs.

## Output

```
results/<date>_<study>/
  config.yaml      exact copy of the config that ran
  manifest.json    settings, input-file hashes, git commit, start/finish times, per-cell timings
  summary.csv      one row per cell
  cells/<cell id>/ agents.parquet, nights.parquet, days.parquet, params.json
```

A cell id names the setup, arm, q, qc, b and ρ, e.g.
`task003_nosolution__false__q6__qc12__b3__rho0.75`; silent cells have no qc or b.

A **fact set** is stored as one integer: bit *i* set means fact *i* of
`world_task003.json` is in the set. Example: 33 = 0b100001 = facts 0 and 5.
A **post read** is stored as `author * 64 + fact`, with author 0–23 for agents
and 24 for the controller. Example: 209 = agent 3 posting fact 17.

**`agents.parquet`**, one row per agent per day:

| column | meaning |
|---|---|
| `position` | the agent's place in that day's random order (0 = first) |
| `memory_after_dawn` | fact set it remembers after the dawn forgetting, before reading |
| `known_before_reading` | fact set it has ever held, before reading |
| `posts_read` | every post it read today (list of codes, repeats kept) |
| `facts_read`, `n_posts_read` | fact set read today, and how many posts |
| `active_facts` | fact set it remembers after reading |
| `p_A0`, `p_A1`, `p_A2` | its posterior |
| `vote` | 0, 1 or 2 |
| `posted_fact` | fact index it posted, or -1 for none |
| `post_reason` | 0 in context, 1 fallback (read alone), 2 random (`agent_post_always`), 3 uniform (control), 4 abstained, 5 empty memory |
| `proves_A0` | its active facts prove A0 |
| `proves_A0_own_evidence` | its active facts **from the agents' original evidence** prove A0. Differs from `proves_A0` when the proof used controller facts |

New facts read today = `facts_read & ~known_before_reading`; reactivated =
`facts_read & known_before_reading & ~memory_after_dawn`; lost at dawn =
yesterday's `active_facts & ~memory_after_dawn`.

**`nights.parquet`**, one row per night the controller acts on (controlled arms; none after the last day):

| column | meaning |
|---|---|
| `n_agent_posts_on_board`, `n_posts_read`, `posts_read` | agent posts available, how many it read, which (codes) |
| `v_hat` | share of the posts it read whose author voted for the target (empty if none) |
| `true_target_share` | share of **all 24 agents** voting for the target that day |
| `target_share_among_posters` | the same among agents who posted. `v_hat` − `true_target_share` is the sensing bias |
| `p_target_given_read` | P(target \| facts it read) |
| `decision` | `proved` (target already proved, stays silent), `gate` (enough votes already), `posted`, `posted_proved` (proved but `silent_when_target_proved` is false), `fallback` (target ruled out; posted the set strongest on its own) |
| `posted_facts`, `n_posted` | fact set posted, and its size (0 or b) |

**`days.parquet`**, one row per episode per day, population averages after the day:
mean posteriors, vote shares, `proof_rate_A0` and `proof_rate_A0_own_evidence`,
mean facts remembered, agent posts, `abstention_rate`, controller posts on the
board that morning and posted that night.

## Not implemented yet

Options defined in the spec but refused by the code with a clear error:
`board_read_weighting: recency`, `controller_memory: accumulate`. Simulations
2–4 are not specified yet.
