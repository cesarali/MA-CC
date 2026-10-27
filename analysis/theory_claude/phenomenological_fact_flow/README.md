# phenomenological_fact_flow — a minimal model of the reading reversal and persistence

Two evidence classes (agents' A, controller's C), each with a memory density m and a board density b. Agents vote
through a sigmoid and repost by an eligibility rule. No Bayesian engine, no task world, no M1–M4 code.

- **Start here:** [`phenomenological_theory_report.md`](phenomenological_theory_report.md)
- **Equations and derivations:** [`model.md`](model.md)
- **Every decision, including failures:** [`decision_log.md`](decision_log.md)

**Current best version:** revision 3 (softened vote gating ε). Both phenomena hold for ε ≳ 0.75 at P0.

Run, from this folder (each script writes to `results/`, takes seconds):

```
../../../.venv/bin/python stageA_scan.py         # reversal at P0 and robustness to each parameter
../../../.venv/bin/python stageA_mechanism.py    # dose-against-sensitivity split and its approximation
../../../.venv/bin/python stageD_persistence.py  # four continuations at P0, selective and neutral reposting
../../../.venv/bin/python rev1_scan.py           # revision 1: evidence per controller fact
../../../.venv/bin/python rev2_grid.py           # revision 2: reposting advantage x evidence strength
../../../.venv/bin/python rev3_grid.py           # revision 3: softened vote gating
../../../.venv/bin/python stageE_figures.py      # figures 1-5
```

Parameters: see `model.md` §1. ε = 0 is the plan's closure (5); ε = 1 is neutral reposting.
