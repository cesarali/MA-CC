# Santa Fe rho phase sweep on Cygnus

Submitted 2026-09-26 from `/shared/home/cesar/LanguageGames/MA-CC`.

The independent recipe is `configs/santa_fe/rho_phase_sweep.yaml`. It retains
the five beta pairs, nine controller budgets, 100 episodes per cell, 30 rounds,
all other simulation settings, and the four CMI conditioning variants from
`beta_cmi_sweep.yaml`. The rho axis is 11 equally spaced values from 0.4 to
1.0 inclusive, in increments of 0.06. This yields 495 cells, 49,500 episodes,
and 1,534,500 expected canonical round rows. No provider calls are made.

Results are under `/shared/home/cesar/work/results/studies/santa_fe_rho_phase/`.
The finalizer writes cell-level tables and rho-by-budget heatmaps for every
beta regime under `plots/beta_cmi/rho_phase_diagrams/`. The heatmaps include
target and truth share, susceptibility, action support, observed CMI, and CMI
minus its null mean. CMI is undefined at zero budget.

The plan and generated shell script passed dry-run and syntax checks; the
focused Santa Fe tests passed. The four Slurm stages are:

| Stage | Job ID | Array and resources |
| --- | ---: | --- |
| Simulation | 547 | 495 tasks, throttle 8, 4 CPUs, 8G, 1 hour |
| Cell aggregation | 548 | 2 CPUs, 32G, 2 hours |
| Information | 549 | 495 tasks, throttle 8, 4 CPUs, 12G, 8 hours |
| Finalization and phase diagrams | 550 | 2 CPUs, 32G, 2 hours |

Each later stage has an `afterok` dependency on its predecessor. All four stages
completed with exit code 0. The final outputs have 495 simulation seals,
495 information seals, 1,534,500 round rows, 1,980 beta/CMI detectability
rows, and 60 rho-by-budget phase diagram PNGs. The complete result report is
archived in `docs/theory/santa_fe_cygnus_20260926/`.
