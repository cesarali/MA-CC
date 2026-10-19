# gold_standard — the pipeline behind the current reports

Turns the light-record runs into the tables and figures of the 15–16 October reports (Simulations 1 and 2).
The metrics, the data, the rebuild order and the figure-to-code map are in the reviewer guide:
[`../../simulation_2/analysis/reports/2026-10-16_report1_symmetric/REVIEW_GUIDE.md`](../../simulation_2/analysis/reports/2026-10-16_report1_symmetric/REVIEW_GUIDE.md).

| file | writes |
|---|---|
| `build_tables.py` | `results/gold_standard/<date>/cells_time.parquet` (per cell and time: votes, beliefs, proofs, facts posted, G, state distributions) |
| `build_slopes.py` | `slopes.parquet` (slope of G between neighbouring strengths, paired errors) |
| `build_info.py` | `info.parquet` (readability I, cost C, undirected part R, efficiency η from the one-agent vote; same numbers as `mi_table.csv`) |
| `check_overlap.py` | nothing; prints whether new runs reproduce the older runs on the cells they share |
| `explorer.py`, `explorer.ipynb` | nothing; notebook with sliders over every gold-standard figure, all 8 ρ values, both setups, plus Fig. 8b (fitted derivative of G in the rate, Simulation 2) |
| `make_report_figures.py` | `mi_table.csv`, `mi_smoothing_sensitivity.csv`; figures of the 15 October reports |
| `make_report_figures_2026-10-16.py` | figures of the 16 October reports (default) and of the 17 October nosolution reports (`--setup task003_nosolution --data-date 2026-10-16 --report 2026-10-17_report2_nosolution`) |
| `make_report_figures_2026-10-19.py` | figures of the 19 October variant reports (Simulation 2: B = 100, W = 24, W = 240 against the gold standard, both setups) |
| `sham_baseline.py`, `sham_baseline_figures.py` | `sham_table.parquet`; figures of the sham-baseline report |
| `2026-10-15_plan_reports.md` | the plan of the 15 October reports |

Table dates: `--date 2026-10-15` = the two symmetric runs of 15 October (ρ 1, 0.75, 0.5, 0.25);
`--date 2026-10-16` = those plus the 16 October `*_extra_rho` (ρ 0.875, 0.625, 0.375, 0.125) and `*_nosolution_fine` runs,
with a `setup` column; `--date 2026-10-18` = the six Simulation 2 small-grid variant runs of 18 October.
All tables built since 19 October also have a `case` column: `standard` (B 30, W 48), `budget100`, `window24`, `window240`. Rebuild the 16 October tables: `build_tables.py --date 2026-10-16`, `build_slopes.py --date 2026-10-16`,
`build_info.py --date 2026-10-16`. The notebook needs `ipykernel` and `ipywidgets` in `.venv`.

Moved here from `analysis/task003_llm_free/symmetric_fine/` on 16 October 2026; outputs moved from
`results/symmetric_fine/` to `results/gold_standard/`.
