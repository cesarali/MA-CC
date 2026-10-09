# simulation_2/simulator — Simulation 2 (asynchronous activation)

24 agents and a controller play `task_003` in continuous time, with **no language
model**: agents are exact Bayesian reasoners and exchange fact IDs. Every setting
that differs from Simulation 1 is an option, so the bridge study (Simulation 1's
rules changed one at a time) and the controller-rate scan run through one code path.

| file | what it is |
|---|---|
| [`simulation_2_spec.md`](simulation_2_spec.md) | every rule, the bridge steps S0–S6, what is measured |
| [`configs/`](configs/) | one config per study. [`sim2_bridge.yaml`](configs/sim2_bridge.yaml): the bridge |
| `run_simulation_2.py` | runs a study |
| `summarize_simulation_2.py` | one row per cell, with paired gains over its silent cell (the runner calls it) |
| `sim2/simulation_2.py` | the rules: one episode as a list of timed events |
| `sim2/checks.py` | start-up checks on the frozen setups (spec §1) |
| `tests/` | the checks of spec §13; they also compare the posting and controller rules with Simulation 1's code |
| `runs_index.csv`, `run_summaries/` | created by the first official run |

Shared code (`world.py`, `setups.py`, the exported world) is in
[`../../llmfree_core/`](../../llmfree_core/). The S0 check is
[`../analysis/s0_check.py`](../analysis/s0_check.py).

## Running it

From the repository root:

```bash
# the tests (about a minute)
.venv/bin/python -m pytest analysis/task003_llm_free/simulation_2/simulator/tests -q

# a quick check, written to a scratch folder and not indexed
.venv/bin/python analysis/task003_llm_free/simulation_2/simulator/run_simulation_2.py \
    --config analysis/task003_llm_free/simulation_2/simulator/configs/sim2_bridge.yaml --episodes 20 --out <scratch folder>

# an official run on 12 cores (bridge: about 4 minutes, about 3.5 GB)
.venv/bin/python analysis/task003_llm_free/simulation_2/simulator/run_simulation_2.py \
    --config analysis/task003_llm_free/simulation_2/simulator/configs/sim2_bridge.yaml --workers 12

# the S0 check on a run
.venv/bin/python analysis/task003_llm_free/simulation_2/analysis/s0_check.py <run folder>
```

- The runner refuses to start if the config gives a different number of cells from
  `cells.expected_total`, if a cumulative step changes more than one setting, or if a
  setup fails its start-up checks.
- An official run (no `--episodes`, `--only` or `--out`) writes to a new folder
  `../../results/simulation_2/<date>_<study>/`, refuses to overwrite one, appends a line to
  `runs_index.csv`, and copies its summary to `run_summaries/`.
- It warns if `simulation_2/simulator/` or `llmfree_core/` has uncommitted changes.
  Commit before official runs.
- Silent cells whose agent settings equal an earlier step's are not rerun;
  `manifest.json` lists them under `reused_silent_cells` (S4–S6 reuse S3).

## Output

```
results/simulation_2/<date>_<study>/
  config.yaml, manifest.json, summary.csv
  cells/<step>__<setup>__<arm>/
    posts.parquet  actions.parquet  forgetting.parquet  controller.parquet
    snapshots.parquet  agent_snapshots.parquet  episodes.parquet  params.json
```

A **fact set** is one integer: bit *i* set means fact *i* of `world_task003.json`.
Example: 33 = 0b100001 = facts 0 and 5. Times `t` are in time units (one expected
action per agent); author −1 is the controller; vote −1 means none.

| table | one row per | main columns |
|---|---|---|
| `posts` | post | `post_id`, `t`, `author`, `fact`, `author_vote` (the vote when posting) |
| `actions` | agent action | `active_before` (memory before reading), `known_before`, `facts_read`, `posts_read` (post IDs, join with `posts`), `window_size`, `active_after`, `p_A0..p_A2`, `vote`, `posted_fact`, `post_id`, `post_reason` (0 in context, 1 fallback, 4 abstained, 5 empty memory), candidate sets `candidates_in_context`, `candidates_fallback`, `candidates_final`, `proves_A0`, `proves_A0_own_evidence` |
| `forgetting` | fact lost | `t`, `agent`, `fact`, `spell_start` (when its lifetime began; empty for dawn forgetting) |
| `controller` | controller action that could act (t < 30, budget left) | `posts_read`, `facts_read` (R), `v_hat`, `true_target_share` (all 24 agents' last votes), `share_voted`, `p_target_given_read`, `decision` (proved, gate, posted, posted_proved, fallback), `posted_fact`, `budget_before`, `budget_after`, `cumulative_reads` |
| `snapshots` | measurement time (0, 0.25, …, 40) | `mean_p_A0..A2` (= m_k(t)), `share_A0..A2` (last votes over all 24 agents), `share_voted`, `proof_rate_A0`, `proof_rate_A0_own_evidence`, `mean_active_facts`, `board_size`, `agent_posts`, `controller_messages`, `controller_reads` (so far) |
| `agent_snapshots` | agent per measurement time | `active`, `last_vote`, `vote_age` |
| `episodes` | episode | `controller_messages`, `controller_reads`, `budget_exhausted_at` (empty if never), `agent_actions`, `agent_posts`, `facts_forgotten` |

New facts in an action = `facts_read & ~known_before`; reactivated =
`facts_read & known_before & ~active_before`.
