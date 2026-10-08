# Short reports on Simulation 1 (for human readers)

| report | PDF | about |
|---|---|---|
| Report 1 | [`report1_symmetric/report1_symmetric.pdf`](report1_symmetric/report1_symmetric.pdf) | task003-symmetric: a swarm that owns a proof can still lose it; sensing; own-evidence proofs |
| Report 2 | [`report2_nosolution/report2_nosolution.pdf`](report2_nosolution/report2_nosolution.pdf) | task003-nosolution: probability matching as the main case; the A2 drift as a special case and how to fix its start; the large-budget backfire; sensing |

Rebuild, from the repository root:

```bash
.venv/bin/python analysis/task003_llm_free/simulation_1/analysis/reports/make_report_figures.py
cd analysis/task003_llm_free/simulation_1/analysis/reports/report1_symmetric && latexmk -pdf report1_symmetric.tex
cd ../report2_nosolution && latexmk -pdf report2_nosolution.tex
```

The figure script reads the analysis tables in `results/simulation_1/analysis/2026-10-07/tables/`
(produced by the Q0–Q6 scripts one folder up) and copies four phase diagrams from
`results/simulation_1/analysis/2026-10-07/figures/`. The full analysis behind every number is in
the Q0–Q6 reports one folder up; a reviewing agent should start at `../REVIEW_GUIDE.md`.
