# object_placements_llm_free — a second task for the LLM-free simulator

The same swarm-and-controller experiment as [`../task003_llm_free/`](../task003_llm_free/), on a different reasoning
task: a MuSR **Object Placements** story. MuSR (Sprague et al., arXiv 2310.16049, MIT licence) is a benchmark of
reasoning puzzles; "Object Placements" asks where a person would look for an item after some moves they may or may not
have seen. Here the agents are exact Bayesian reasoners, as in task003, so no LLM is involved.

Purpose: an **out-of-sample test of the theory** in [`../theory/`](../theory/). The M3 and M4 predictions for this
task were committed (commits `c0d22bc`, `c9d66c2`) before the simulation data were opened.

## The task in one example

Start: the notebook is on the shelf, the keys are on the desk. Then:
1. Casey moves the keys to the drawer (Alex sees it).
2. Alex moves the keys back to the desk (Blair sees it; Casey does not).
3. Alex moves the notebook to the drawer (Blair sees it).

Question: *where would Casey look for the keys?*
- **A0 = drawer**: Casey's belief, the correct answer.
- **A1 = shelf**.
- **A2 = desk**: where the keys really are; the false controller's target.

Agents hold "facts", which are true statements about the story, for example "Casey did not see move 2" or "Move 1 is
made by Casey". The story has 34 true facts. An agent's belief is the exact probability of each answer over all
617,472 stories that MuSR's generator can produce, weighted by how likely each one is.

## Results in one line

The reading reversal appears here too. M4's pre-registered predictions get 11/16 (symmetric) and 12/15 (no-proof)
resolved increment signs right and G within 0.02–0.03; M3 fails (5/16, 3/15). Details: the theory report,
[§2](../theory/reports/2026-10-22_theory_review/theory_review_report.md#section-2-musr-object-placements-second-task-pre-registered).

## What is here

| path | what |
|---|---|
| [`world/world_op1.json`](world/world_op1.json) | the story and question (the world). `world_op1.classes.npz` next to it is a cache, rebuilt on first load, not in git |
| [`experimental_setup/op_symmetric/`](experimental_setup/op_symmetric/) | agents can prove A0 together; two controller pools. README lists every fact |
| [`experimental_setup/op_nosolution/`](experimental_setup/op_nosolution/) | agents cannot prove anything; one shared pool |
| [`simulation_2/configs/sim2_op_main.yaml`](simulation_2/configs/sim2_op_main.yaml) | the simulation run: both setups, 8 memory values ρ, q 3/6/9/12, false and silent controller, 1,000 paired episodes per cell |
| [`simulation_2/analysis/`](simulation_2/analysis/) | `build_op_setups.py` (builds world and setups), `test_op_world.py` (engine against MuSR's own generator), `op_predictions_and_score.py` (model predictions, data table, scores), `preregistered_M{3,4}.csv`, `op_data.csv`, `scorecard_op_M{3,4}*.csv` |
| `results/simulation_2/2026-10-22_sim2_op_main/` | the raw run, 5.6 GB, not in git |

## Shared with task003 (not copied here)

- **Simulator:** [`../task003_llm_free/simulation_2/simulator/run_simulation_2.py`](../task003_llm_free/simulation_2/simulator/run_simulation_2.py).
  It reads the world, setups and output folder from the config, so it runs this task unchanged. Its runs index and run
  summaries (`runs_index.csv`, `run_summaries/2026-10-22_sim2_op_main_summary.csv`) stay with the simulator.
- **Story engine:** [`../task003_llm_free/llmfree_core/op_world.py`](../task003_llm_free/llmfree_core/op_world.py)
  lists all stories and computes the exact beliefs.
- **Theory models:** [`../theory/`](../theory/).

## How to rerun

From the repository root:

```
# engine tests (needs a local MuSR clone; set MUSR_CLONE to its path)
.venv/bin/python -m pytest analysis/object_placements_llm_free/simulation_2/analysis/test_op_world.py -q
# the simulation (writes a new dated folder under results/simulation_2/)
caffeinate -i .venv/bin/python analysis/task003_llm_free/simulation_2/simulator/run_simulation_2.py \
    --config analysis/object_placements_llm_free/simulation_2/configs/sim2_op_main.yaml --workers 7
# data table and scores
.venv/bin/python analysis/object_placements_llm_free/simulation_2/analysis/op_predictions_and_score.py data
.venv/bin/python analysis/object_placements_llm_free/simulation_2/analysis/op_predictions_and_score.py score --tag M4
```

## History

Built on 22 October 2026 inside `task003_llm_free/` and moved here on 23 October 2026 (with `git mv`; `git log
--follow` shows the history). Only paths changed: five episodes rerun from the moved config are identical to the
stored run. Text inside the world and setup JSON files still names the old locations. It is left as built, so the
files keep the hashes (fingerprints of their exact bytes) recorded in the run's manifest. The run's own `config.yaml`
also shows the old relative paths.
