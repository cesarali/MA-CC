# Short reports on Simulation 1 (for human readers)

| report | PDF | about |
|---|---|---|
| **Report 1 (14 Oct)** | [`2026-10-14_report1_symmetric/2026-10-14_report1_symmetric.pdf`](2026-10-14_report1_symmetric/2026-10-14_report1_symmetric.pdf) | task003-symmetric, rewritten to mirror the Simulation 2 report section by section: vote shares over time, gain phase diagrams, readability $I$(target; vote) phase diagrams, pair efficiency $\eta$, success against facts posted; progress, focus and price removed |
| **Report 2 (14 Oct)** | [`2026-10-14_report2_nosolution/2026-10-14_report2_nosolution.pdf`](2026-10-14_report2_nosolution/2026-10-14_report2_nosolution.pdf) | the same for task003-nosolution |
| **Report 1, extended (12 Oct)** | [`2026-10-12_report1_symmetric/2026-10-12_report1_symmetric.pdf`](2026-10-12_report1_symmetric/2026-10-12_report1_symmetric.pdf) | the 2026-10-07 report unchanged, plus a paragraph on how the simulator works and a section on the efficiency of control |
| **Report 2, extended (12 Oct)** | [`2026-10-12_report2_nosolution/2026-10-12_report2_nosolution.pdf`](2026-10-12_report2_nosolution/2026-10-12_report2_nosolution.pdf) | the same for task003-nosolution |
| Report 1 | [`2026-10-07_report1_symmetric/2026-10-07_report1_symmetric.pdf`](2026-10-07_report1_symmetric/2026-10-07_report1_symmetric.pdf) | task003-symmetric: a swarm that owns a proof can still lose it; sensing; own-evidence proofs |
| Report 2 | [`2026-10-07_report2_nosolution/2026-10-07_report2_nosolution.pdf`](2026-10-07_report2_nosolution/2026-10-07_report2_nosolution.pdf) | task003-nosolution: probability matching as the main case; the A2 drift as a special case and how to fix its start; the large-budget backfire; sensing |

Rebuild, from the repository root:

```bash
.venv/bin/python analysis/task003_llm_free/simulation_1/analysis/reports/make_report_figures.py
cd analysis/task003_llm_free/simulation_1/analysis/reports/2026-10-07_report1_symmetric && latexmk -pdf 2026-10-07_report1_symmetric.tex
cd ../2026-10-07_report2_nosolution && latexmk -pdf 2026-10-07_report2_nosolution.tex
```

The figure script reads the analysis tables in `results/simulation_1/analysis/2026-10-07/tables/`
(produced by the Q0–Q6 scripts one folder up) and copies four phase diagrams from
`results/simulation_1/analysis/2026-10-07/figures/`. The full analysis behind every number is in
the Q0–Q6 reports one folder up; a reviewing agent should start at `../REVIEW_GUIDE.md`.

The extended reports are built from the originals by `../../../efficiency_estimates/extend_reports.py`
(it checks that the original text is unchanged); their efficiency figures come from
`../../../efficiency_estimates/make_report_figures.py`.

The 14 October reports are self-contained `.tex` files; their new figures come from
`../../../efficiency_estimates/make_2026-10-14_report_figures.py` (it also copies the figures it reuses
from the 12 October folders). Rebuild: run that script, then `latexmk -pdf` in each 2026-10-14 folder.
