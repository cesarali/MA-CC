# task003_llm_free — the two control setups and the LLM-free simulations

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

## Where to look for what

**Start here for current results:** the gold-standard report,
[`simulation_2/analysis/reports/2026-10-16_report1_symmetric/`](simulation_2/analysis/reports/2026-10-16_report1_symmetric/)
(Simulation 2, symmetric setup), its mirror for Simulation 1 in
[`simulation_1/analysis/reports/2026-10-16_report1_symmetric/`](simulation_1/analysis/reports/2026-10-16_report1_symmetric/),
and the reviewer guide next to the gold standard (`REVIEW_GUIDE.md`): every metric, where the data is, how to
rebuild every number.

| folder | what it holds | look here for |
|---|---|---|
| [`task_and_facts/`](task_and_facts/) | the world: its 49 true facts, the exact posterior engine, the proofs; [`task_003_analysis.md`](task_and_facts/task_003_analysis.md) defines "eligible", "decisive", "proof" | what the task is and why A0 is correct |
| [`experimental_setup/`](experimental_setup/) | one folder per frozen setup (`task003_symmetric/`, `task003_nosolution/`: agents, pools, build script, README), plus the plan for the LLM runs | who holds which fact; the controller pools |
| [`llmfree_core/`](llmfree_core/) | code shared by both simulators: exact posteriors (`world.py`), setup loading (`setups.py`), the exported world, [`runtime_rules.md`](llmfree_core/runtime_rules.md) | how beliefs are computed from facts |
| [`simulation_1/`](simulation_1/) | **Simulation 1** (days and nights; the controller posts b facts per night). `simulator/`: spec, configs, runner, tests, runs index, run summaries. `analysis/`: the 7 Oct plan, Q0–Q6 reports, reviewer guide, and `reports/` (dated PDF reports; the 16 Oct one mirrors the gold standard) | Simulation 1 rules and runs; its reports |
| [`simulation_2/`](simulation_2/) | **Simulation 2** (asynchronous; the controller posts one fact per message at rate λc). Same layout as Simulation 1; `analysis/reports/` holds the gold standard and the sham-baseline report; **`key_phenomena_hypothesis/`: the current study (reading reversal, persistence), from 20 Oct on Simulation 2 only** | Simulation 2 rules and runs; the gold standard; the current study |
| [`shared_analysis/`](shared_analysis/) | analysis code used by both simulations: `gold_standard/` (**the pipeline behind the current reports**: tables, measures, figures, sham baseline), `efficiency_estimates/` (12–14 Oct measures, superseded), `coarse_graining/` (Markov studies, paused). See its README | how a number in a report was computed |
| `results/` | data, not in git. `results/simulation_1/<date>_<study>/` and `results/simulation_2/<date>_<study>/`: raw runs (one folder per cell); `results/simulation_<n>/analysis/<date>/`: tables of the 7/9 Oct analyses; `results/gold_standard/<date>/`: tables of the current reports; `results/efficiency_estimates/`, `results/coarse_graining/`: outputs of those folders | the numbers behind every figure |
| [`TODO.md`](TODO.md) | deferred work | open items |

**Runs in use (light records, 1,000 episodes per cell):**
`results/simulation_1/2026-10-15_sim1_pm_symmetric_fine`, `results/simulation_2/2026-10-15_sim2_symmetric_fine`
(symmetric, ρ 1/0.75/0.5/0.25), with their `*_extra_rho` and `*_nosolution_fine` companions of 16 October
(ρ 0.875/0.625/0.375/0.125 for symmetric; all 8 ρ for nosolution). Older runs that were deleted are listed in
`results/simulation_*/2026-10-1*_deletion_list*.txt`.

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
