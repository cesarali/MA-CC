# gold_standard — the pipeline behind the current reports

Turns the light-record runs into the tables and figures of the 15–16 October reports (Simulations 1 and 2).
The metrics, the data, the rebuild order and the figure-to-code map are in the reviewer guide:
[`../../simulation_2/analysis/reports/2026-10-16_report1_symmetric/REVIEW_GUIDE.md`](../../simulation_2/analysis/reports/2026-10-16_report1_symmetric/REVIEW_GUIDE.md).

| file | writes |
|---|---|
| `build_tables.py` | `results/gold_standard/<date>/cells_time.parquet` (per cell and time: votes, beliefs, proofs, facts posted, G, state distributions) |
| `build_slopes.py` | `slopes.parquet` (slope of G between neighbouring strengths, paired errors) |
| `make_report_figures.py` | `mi_table.csv`, `mi_smoothing_sensitivity.csv`; figures of the 15 October reports |
| `make_report_figures_2026-10-16.py` | figures of the 16 October reports (gold standard and its Simulation 1 mirror) |
| `sham_baseline.py`, `sham_baseline_figures.py` | `sham_table.parquet`; figures of the sham-baseline report |
| `2026-10-15_plan_reports.md` | the plan of the 15 October reports |

Moved here from `analysis/task003_llm_free/symmetric_fine/` on 16 October 2026; outputs moved from
`results/symmetric_fine/` to `results/gold_standard/`.
