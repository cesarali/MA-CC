# Review guide — Simulation 1 analysis (for a reviewing agent)

**Your task:** review the analysis of Simulation 1 for errors in code, logic, numbers
and interpretation. Work read-only unless asked otherwise. Report every problem with
the file, the line or table, and why it is wrong. Prefer recomputing a number from
the data over trusting the report.

All paths below are relative to `analysis/task003_llm_free/`.

---

## 1. Read these first, in order

| # | file | why |
|---|---|---|
| 1 | `experimental_setup/README.md` | the two setups (task003-symmetric, task003-nosolution): agents, controller pools |
| 2 | `simulator/simulation_1_spec.md` | every rule of the simulation |
| 3 | `simulator/README.md` | how runs are organised, and every output column |
| 4 | `sim1_analysis/ANALYSIS_PLAN.md` | the questions, estimands, confounders and criteria, fixed before the results were read |

## 2. The questions and where each is answered

| question | report | scripts | key tables |
|---|---|---|---|
| **Q0** sanity checks | `sim1_analysis/2026-10-07_Q0_Q1_report.md` §Q0 | `q0_sanity.py` | `tables/q0_first_votes.csv`, `tables/q0_reference.csv`, `tables/q0_posting_reasons.csv` |
| **Q1** what drives the A2 drift (silent arms) | `sim1_analysis/2026-10-07_Q0_Q1_report.md` §Q1 | `q1_drift.py` | `tables/q1_final.csv`, `tables/q1_unanimity.csv` |
| **Q2** does control work, and how | `sim1_analysis/2026-10-07_Q2_report.md` | `q2_compute.py`, `q2_figures.py` | `tables/q2_final.csv` |
| **Q3** non-monotonicity in b and qc, and its seed-2027 confirmation | `sim1_analysis/2026-10-07_Q3_report.md` | `q3_scan.py`, `q3_confirm.py` | `tables/q3_nonmonotone.csv`, `tables/q3_confirmation.csv` |
| **Q4** sensing accuracy | `sim1_analysis/2026-10-07_Q4_Q5_report.md` §Q4 | `q4_sensing.py` | `tables/q4_sensing.csv` |
| **Q5** the stopping rule `silent_when_target_proved` | `sim1_analysis/2026-10-07_Q4_Q5_report.md` §Q5 | `q5_stopping.py` | `tables/q5_final.csv` |
| **Q6** mechanisms | `sim1_analysis/2026-10-07_Q6_report.md` | `q6_mechanisms.py` | `tables/q6a_*.csv`, `tables/q6b_*.csv`, `tables/q6c_*.csv` |

Shared helpers for all scripts: `sim1_analysis/common.py` (`cells_table()` builds the
table of every cell from each run's `params.json`; its `purpose` column separates
`main`, `no_proof_stop` and `confirmation` runs).

Curated figures quoted in the reports: `sim1_analysis/figures/`. The full set
(about 150 figures) and every table: `results/analysis/2026-10-07/`, listed in its
`index.csv`.

## 3. The data

- **Runs:** `simulator/runs_index.csv` lists all 11 official runs (date, study, config,
  git commit). Each run's config: `simulator/configs/<study>.yaml`.
- **Per-run summaries** (committed): `simulator/run_summaries/`.
- **Raw results** (not in git, about 45 GB): `results/<date>_<study>/cells/<cell id>/`
  with `agents.parquet`, `nights.parquet`, `days.parquet`, `params.json`. If you
  cannot access them, regenerate any run with
  `.venv/bin/python analysis/task003_llm_free/simulator/run_simulation_1.py --config <config>`
  from the repository root; runs are deterministic (same seed, same results).
- **Tests:** `.venv/bin/python -m pytest analysis/task003_llm_free/simulator/tests -q`.

## 4. What deserves the most scrutiny

1. **The pairing.** Paired gains subtract the silent run of the *same episode*. Check
   that `q2_compute.py` matches each controlled cell with the silent cell of the same
   variant, setup, q and ρ, and that the bootstrap resamples the arms together.
2. **The A2 drift explanation** (Q1, Q6): lock-in under argmax with vote-supporting
   posts, plus a 7 / 6 / 11 expected starting vote. Check the starting-vote numbers
   in `q0_sanity.py` (`first_votes`) and the unanimity table.
3. **The large-b fall** (Q2 §3, Q3): attributed to the pool's few facts per target
   plus all-or-nothing posting. Check the posted-fact composition and the seed-2027
   replication.
4. **The own-evidence proof result** (Q5, Q6c): `proves_A0_own_evidence` uses only the
   agents' original facts. Check its definition in `simulator/llmfree/simulation_1.py`
   and whether the posting-substitution explanation follows from the table.
5. **The size criterion in Q3** (one agent = 1/24): it decides that no rise-then-fall
   survives. Check that it is applied as the plan states.
6. **Numbers quoted in the reports** against the tables they cite.

## 5. Known limitations (already stated in the reports; no need to report them)

- One world (task_003), one population size (24), exact-Bayesian agents.
- The controller maximises P(target | sampled board facts + its posts), a proxy.
- Some effects depend on specific facts of this fact assignment (e.g. the A0 lean
  under uniform posting comes from two A2-side facts).
