# task003_llm_free — the two control setups and the LLM-free simulator

Everything here is about one world, `task_003`: three people, two jobs, three
candidate allocations. `ALLOCATION_0` (A0) is correct; `ALLOCATION_2` (A2) is the
wrong answer a false controller steers toward. 24 agents each hold one true fact;
a controller posts true facts on a shared board to steer them.

## The two setups (frozen 5 October 2026)

| | **task003-symmetric** | **task003-nosolution** |
|---|---|---|
| can the agents together prove A0? | **yes**: they hold the 6 decisive facts | **no**: none of them |
| controller pools | **two**: the A0 pool when targeting A0, the A2 pool when targeting A2 | **one**, for both targets |
| folder | [`experimental_setup/task003_symmetric/`](experimental_setup/task003_symmetric/) | [`experimental_setup/task003_nosolution/`](experimental_setup/task003_nosolution/) |

Start at [`experimental_setup/README.md`](experimental_setup/README.md).

## Folders

| folder | what it holds |
|---|---|
| [`task_and_facts/`](task_and_facts/) | the world: its 49 true facts, the exact posterior engine (`engine.py`), the proofs. Read [`task_003_analysis.md`](task_and_facts/task_003_analysis.md) for what "eligible", "decisive" and "proof" mean |
| [`experimental_setup/`](experimental_setup/) | one folder per frozen setup (agents, pools, build script, README), plus the plan for the LLM runs |
| [`simulator/`](simulator/) | the LLM-free simulator: agents are exact Bayesian reasoners. [`runtime_rules.md`](simulator/runtime_rules.md): the real game's rules to copy. [`simulation_1_spec.md`](simulator/simulation_1_spec.md) and [`simulation_1_config.yaml`](simulator/simulation_1_config.yaml): Simulation 1, agreed, not yet implemented |
| [`coarse_graining/`](coarse_graining/) | the 4-state summary of the swarm (mean belief in A0, binned) and the four tests of whether it is Markovian. The results inside are from the archived LLM study; they will be **rerun on the new simulator's data** |

## Recovering deleted material

On 6 October this folder was cleaned down to what the two frozen setups need.
Everything removed that was in git can be recovered from commit **`ba16739`**,
where this folder was called `analysis/preanalysis_task003_new_setup_and_coarsegraining/`.
For example:

```bash
git show ba16739:analysis/preanalysis_task003_new_setup_and_coarsegraining/50_next_experiment/notes/protocols_2026-10-02.md
git checkout ba16739 -- analysis/preanalysis_task003_new_setup_and_coarsegraining/60_exact_simulation
```

| removed | old path under that folder |
|---|---|
| the controller-redesign README (its design note moved to `experimental_setup/task003_symmetric/pool_design_2026-10-02.md`) | `20_controller_redesign/` |
| path divergence and mutual-information results on the archived study | `40_information_estimates/` |
| scripts that processed the archived study (the Markov-test and estimator code moved to `coarse_graining/code/`) | `scripts/` |
| the 8-page report on the archived analysis | `report/`, `preanalysis_task003_report.pdf` |
| the old LLM-free simulator, its PDFs and results | `60_exact_simulation/` |
| dated planning notes: protocol options, what Darius built, the merge, the 4 October summary, research ideas | `50_next_experiment/notes/` |
| superseded designs: the shared-pool task003-symmetric, the ineligible task003-nosolution, the 2 October draft | `50_next_experiment/archive/` |
| the 1 October pool audit from another session | `10_task_and_facts/task_003_redesign.md`, `task_003_pool_comparison.json`, `audit_target_pools.py` |

**Not recoverable** (never in git, deleted 6 October): the second LLM-free study
(`70_codex_simulations/`) and the gitignored `results/` data, which the
coarse-graining rerun will regenerate.
