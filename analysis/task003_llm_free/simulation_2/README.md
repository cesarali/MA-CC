# simulation_2 — asynchronous activation

**Status: implemented and tested 9 October 2026; the bridge study is ready to run.**

| folder | what it holds |
|---|---|
| [`simulator/`](simulator/) | [`simulation_2_spec.md`](simulator/simulation_2_spec.md): the bridge from Simulation 1, then the controller-rate scan; plus the code, configs, tests ([`simulator/README.md`](simulator/README.md)) |
| [`analysis/`](analysis/) | [`ANALYSIS_PLAN.md`](analysis/ANALYSIS_PLAN.md) (draft): questions, hypotheses and the confirmation run; [`s0_check.py`](analysis/s0_check.py): does S0 reproduce Simulation 1? (spec §9) |

Results will go to `../results/simulation_2/` (not in git), in runs named `sim2_…`.
Shared code: [`../llmfree_core/`](../llmfree_core/).
