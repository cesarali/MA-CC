# shared_analysis — analysis code used by both simulations

| folder | status | what it holds |
|---|---|---|
| [`gold_standard/`](gold_standard/) | **current** | the pipeline behind the 15–16 October reports: summary tables from the light runs (`build_tables.py`), slopes with paired errors (`build_slopes.py`), readability and efficiency (`make_report_figures.py`, `mi_table`), the gold-standard figures (`make_report_figures_2026-10-16.py`), the sham-baseline measures (`sham_baseline.py`, `sham_baseline_figures.py`), and the report plan. Outputs in `../results/gold_standard/<date>/`. See its README. |
| [`efficiency_estimates/`](efficiency_estimates/) | superseded (12–14 Oct) | earlier efficiency measures (target-pair η and its variants, progress / focus / price, the action-based η_U pilot, the operational frontier and its LaTeX draft, precision at 50/100/300 paths). Reads the old 7/9 Oct runs, part of which have been deleted. The gold standard redefines the measures it uses. |
| [`coarse_graining/`](coarse_graining/) | paused | Markov tests of coarse-grained swarm states: the 2 Oct study on the archived LLM data, the 12 Oct plan, the 15 Oct eight-state study and fact prevalence (Simulation 2, per-agent cells). |

Each script states its inputs, outputs and definitions in its docstring and runs from the repository root.
