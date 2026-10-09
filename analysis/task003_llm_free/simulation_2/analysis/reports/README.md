# Short reports on Simulation 2 (for human readers)

| report | PDF | about |
|---|---|---|
| Report 1 | [`report1_symmetric/report1_symmetric.pdf`](report1_symmetric/report1_symmetric.pdf) | task003-symmetric: without rounds the swarm holds its proof less firmly; the false controller is weaker; fast controllers overshoot; gate; help with the truth replaces own proofs; sensing |
| Comparison | [`comparison_sim1_sim2/comparison_sim1_sim2.pdf`](comparison_sim1_sim2/comparison_sim1_sim2.pdf) | one page: what changes from Simulation 1 to Simulation 2, and what carries over |
| Report 2 | [`report2_nosolution/report2_nosolution.pdf`](report2_nosolution/report2_nosolution.pdf) | task003-nosolution: both controllers tip the swarm and the effect stays; weaker than in Simulation 1 when agents forget; with full memory a faster controller steers less (distinct pool facts); gate; sensing |

Rebuild, from the repository root:

```bash
.venv/bin/python analysis/task003_llm_free/simulation_2/analysis/reports/make_report_figures.py
cd analysis/task003_llm_free/simulation_2/analysis/reports/report1_symmetric && latexmk -pdf report1_symmetric.tex
cd ../report2_nosolution && latexmk -pdf report2_nosolution.tex
cd ../comparison_sim1_sim2 && latexmk -pdf comparison_sim1_sim2.tex
```

The figure script reads the analysis tables in `results/simulation_2/analysis/2026-10-09/tables/`
(produced by the scripts one folder up) and copies two phase diagrams per report from
`results/simulation_2/analysis/2026-10-09/figures/`. The full analysis behind every number is in
`../2026-10-09_report.md`; a reviewing agent should start at `../REVIEW_GUIDE.md`.
