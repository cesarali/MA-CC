# Review guide — Simulation 2 analysis (for a reviewing agent)

**Your task:** review the analysis of Simulation 2 for errors in code, logic, numbers
and interpretation. Work read-only unless asked otherwise. Report every problem with
the file, the line or table, and why it is wrong. Prefer recomputing a number from
the data over trusting the report.

All paths below are relative to `analysis/task003_llm_free/`.

---

## 1. Read these first, in order

| # | file | why |
|---|---|---|
| 0 | `README.md` | the folder layout: `llmfree_core/` (shared code), `simulation_1/`, `simulation_2/`, `results/` |
| 1 | `experimental_setup/README.md` | the two setups (task003-symmetric, task003-nosolution): agents, controller pools |
| 2 | `simulation_2/simulator/simulation_2_spec.md` | every rule. **§14 is the main design**; §1–§13 define the rules and the first bridge study |
| 3 | `simulation_2/simulator/README.md` | how runs are organised, and every output column |
| 4 | `simulation_2/analysis/ANALYSIS_PLAN.md` | questions Q0–Q8, estimands, hypotheses H1–H6, criteria, written before the grid's results were read |
| 5 | `simulation_1/analysis/REVIEW_GUIDE.md` | only if you need Simulation 1, the reference point |

## 2. The runs

All listed in `simulation_2/simulator/runs_index.csv` (date, study, config, git commit,
uncommitted changes flag). Configs in `simulation_2/simulator/configs/`.

| run | what it is | used for |
|---|---|---|
| `2026-10-09_sim2_bridge` | the first, step-by-step bridge S0–S6 | S0 = Simulation 1's rules in the new code (the S0 check); S4–S6 equal grid cells exactly |
| `2026-10-09_sim2_grid` | **the main grid**, 1,212 cells × 1,000 episodes | Q0–Q8 |
| `2026-10-09_sim2_grid_confirm_seed2027` | the 52 main cells (q 6, qc 12, ρ 0.75), seed 2027 | H1, H2–H6 |
| `2026-10-09_sim2_s0_confirm_seed2027` | S0 cells, seed 2027 | H2–H4 |
| `2026-10-09_sim2_rho1_confirm_seed2027` | nosolution, ρ 1, q 3/6/12, qc 6/12, seed 2027 | the exploratory rate reversal (confirmed) |
| `2026-10-07_sim1_pm` (Simulation 1) | `results/simulation_1/` | the reference in Q1 |

**Raw results** (not in git, about 30 GB): `results/simulation_2/<run>/cells/<cell id>/`.
Each cell has `params.json` and Parquet tables; `manifest.json` lists `reused_cells`:
cells not run because they are the same simulation as another (silent cells across
controller settings; proof stop off where it cannot act). If you cannot access them,
regenerate any run from the repository root with
`.venv/bin/python analysis/task003_llm_free/simulation_2/simulator/run_simulation_2.py --config <config> --workers 12`;
runs are deterministic. **Tests:** `.venv/bin/python -m pytest analysis/task003_llm_free/simulation_2/simulator/tests -q`.

## 3. The questions and where each is answered

Reports: `simulation_2/analysis/2026-10-09_report.md` (all questions, for agents) and the two
short PDFs in `simulation_2/analysis/reports/` (for people). Tables and figures:
`results/simulation_2/analysis/2026-10-09/` (`tables/`, `figures/`, `index.csv`).

| question | scripts | key tables |
|---|---|---|
| **Q0** sanity at scale | `q0_sanity.py`; the S0 check `s0_check.py` | `q0_invariants.csv`, `q0_bridge_identity.csv`; `results/simulation_2/2026-10-09_sim2_bridge/s0_check.csv` |
| **Q1** Simulation 1 vs the core | `q1_sim1_vs_core.py` | `q1_silent.csv`, `q1_gains.csv` |
| **Q2–Q4** control, rate, persistence | `build_tables.py`, `make_figures.py` | `gains.csv`, `parts.csv`, `rate_steps.csv`, `gain_trajectories_main.csv` |
| **Q5** vote gate | `build_tables.py` | `gate_effect.csv` |
| **Q6** proof stop | `build_tables.py` | `stop_effect.csv` |
| **Q7** sensing | `q7_sensing.py` | `q7_sensing.csv` |
| **Q8** mechanisms | `q8_mechanisms.py` | `q8_mechanisms.csv` |
| **H1–H6** | `confirm.py` | `confirm_hypotheses.csv` |

Shared helpers: `simulation_2/analysis/common.py` (cells table from `params.json` and the
manifest, per-episode values, bootstrap, verdict rule).

## 4. What deserves the most scrutiny

1. **Pairing and the bootstrap.** Gains subtract the silent cell of the same setup, q and
   ρ, episode by episode (`build_tables.py`; the pairing is in each cell's
   `params.json`, field `silent_cell`). The bootstrap resamples episode numbers with
   one index matrix per run (`common.boot_index`). Check that reused cells point to
   the right simulation (`cells.csv`, column `folder`).
2. **The verdict rule** (`common.verdict`, plan §6): effect = interval excludes 0 and
   |estimate| ≥ 1/24; no effect = interval within ±1/24; otherwise inconclusive.
3. **The comparison with Simulation 1** (Q1) is not paired: two codes, independent
   bootstraps. Simulation 1's day 30 (`day == 29`, counted from 0) is compared with
   t = 30. Check `q1_sim1_vs_core.sim1_cell` picks the right cells (b = 1).
4. **The rate reversal at ρ = 1** (found after looking; confirmed on seed 2027,
   `rho1confirm_rate_steps.csv`): explained by the number of distinct pool facts posted
   (`report2_distinct_facts.csv`, made by `reports/make_report_figures.py`). The explanation
   is a correlation across settings; check whether something else that varies with the
   rate (messages, timing of the budget) explains it as well.
5. **The sensing bias** (v̂ above the true share, growing with the rate) is new relative to
   Simulation 1; check `q7_sensing.py` compares like with like (posting-time votes vs last
   votes of all 24 agents).
6. **Numbers quoted in the reports** against the tables they cite.

## 5. Known limitations (stated in the reports; no need to report them)

- One world (task_003), 24 agents, exact-Bayesian agents, probability-matching votes.
- The controller has no memory and picks the single best fact for what it just read;
  it is not an optimal controller.
- A budget of 30 messages in every cell; at high rates it runs out early by design.
- Votes on posts are the author's vote at posting time; the gate reads old opinions.
