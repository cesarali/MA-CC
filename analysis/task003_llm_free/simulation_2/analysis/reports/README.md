# Short reports on Simulation 2 (for human readers)

| report | PDF | about |
|---|---|---|
| **Report 1 (15 Oct)** | [`2026-10-15_report1_symmetric/2026-10-15_report1_symmetric.pdf`](2026-10-15_report1_symmetric/2026-10-15_report1_symmetric.pdf) | task003-symmetric on the finer 15 October runs (memory ρ 0.25–1, all strengths, gate on and off); mirrors the Simulation 1 report: votes, paired susceptibility, phase diagrams, marginal susceptibility, gain per fact, proofs, readability and efficiency; appendices: gate off, other states, strong control |
| **Report 1 (14 Oct)** | [`2026-10-14_report1_symmetric/2026-10-14_report1_symmetric.pdf`](2026-10-14_report1_symmetric/2026-10-14_report1_symmetric.pdf) | task003-symmetric, rewritten to mirror the Simulation 1 report section by section: vote shares over time, gain phase diagrams, readability $I$(target; vote) phase diagrams, pair efficiency $\eta$, success against facts posted; progress, focus and price removed |
| **Report 2 (14 Oct)** | [`2026-10-14_report2_nosolution/2026-10-14_report2_nosolution.pdf`](2026-10-14_report2_nosolution/2026-10-14_report2_nosolution.pdf) | the same for task003-nosolution |
| **Report 1, extended (12 Oct)** | [`2026-10-12_report1_symmetric/2026-10-12_report1_symmetric.pdf`](2026-10-12_report1_symmetric/2026-10-12_report1_symmetric.pdf) | the 2026-10-09 report unchanged, plus a paragraph on how the simulator works and a section on the efficiency of control |
| **Report 2, extended (12 Oct)** | [`2026-10-12_report2_nosolution/2026-10-12_report2_nosolution.pdf`](2026-10-12_report2_nosolution/2026-10-12_report2_nosolution.pdf) | the same for task003-nosolution |
| Report 1 | [`2026-10-09_report1_symmetric/2026-10-09_report1_symmetric.pdf`](2026-10-09_report1_symmetric/2026-10-09_report1_symmetric.pdf) | task003-symmetric: without rounds the swarm holds its proof less firmly; the false controller is weaker; fast controllers overshoot; gate; help with the truth replaces own proofs; sensing |
| Comparison | [`2026-10-09_comparison_sim1_sim2/2026-10-09_comparison_sim1_sim2.pdf`](2026-10-09_comparison_sim1_sim2/2026-10-09_comparison_sim1_sim2.pdf) | one page: what changes from Simulation 1 to Simulation 2, and what carries over |
| Report 2 | [`2026-10-09_report2_nosolution/2026-10-09_report2_nosolution.pdf`](2026-10-09_report2_nosolution/2026-10-09_report2_nosolution.pdf) | task003-nosolution: both controllers tip the swarm and the effect stays; weaker than in Simulation 1 when agents forget; with full memory a faster controller steers less (distinct pool facts); gate; sensing |

Rebuild, from the repository root:

```bash
.venv/bin/python analysis/task003_llm_free/simulation_2/analysis/reports/make_report_figures.py
cd analysis/task003_llm_free/simulation_2/analysis/reports/2026-10-09_report1_symmetric && latexmk -pdf 2026-10-09_report1_symmetric.tex
cd ../2026-10-09_report2_nosolution && latexmk -pdf 2026-10-09_report2_nosolution.tex
cd ../2026-10-09_comparison_sim1_sim2 && latexmk -pdf 2026-10-09_comparison_sim1_sim2.tex
```

The figure script reads the analysis tables in `results/simulation_2/analysis/2026-10-09/tables/`
(produced by the scripts one folder up) and copies two phase diagrams per report from
`results/simulation_2/analysis/2026-10-09/figures/`. The full analysis behind every number is in
`../2026-10-09_report.md`; a reviewing agent should start at `../REVIEW_GUIDE.md`.

The extended reports are built from the originals by `../../../efficiency_estimates/extend_reports.py`
(it checks that the original text is unchanged); their efficiency figures come from
`../../../efficiency_estimates/make_report_figures.py`.

The 14 October reports are self-contained `.tex` files; their new figures come from
`../../../efficiency_estimates/make_2026-10-14_report_figures.py` (it also copies the figures it reuses
from the 12 October folders). Rebuild: run that script, then `latexmk -pdf` in each 2026-10-14 folder.

The 15 October report is built from `../../../symmetric_fine/build_tables.py` (summary table) and
`../../../symmetric_fine/make_report_figures.py` (figures); plan: `../../../symmetric_fine/2026-10-15_plan_reports.md`.
