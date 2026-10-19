# Reviewer guide — the gold-standard report (16 October 2026)

**The report:** [`2026-10-16_report1_symmetric.tex`](2026-10-16_report1_symmetric.tex) (PDF next to it).
It covers Simulation 2 (asynchronous activation) on the `task003-symmetric` setup.
**Read the `.tex` first**: it states every result with its number and the figure it comes from. This guide says
how each of those numbers was produced and how to check it.

- **Mirror for Simulation 1:** [`../../../../simulation_1/analysis/reports/2026-10-16_report1_symmetric/`](../../../../simulation_1/analysis/reports/2026-10-16_report1_symmetric/). Same sections, figures and equations.
- **Same structure for task003-nosolution (17 Oct):** [`../2026-10-17_report2_nosolution/`](../2026-10-17_report2_nosolution/) and its Simulation 1 mirror. Data: tables in `gold_standard/2026-10-16/` (filter `setup == "task003_nosolution"`; information in `info.parquet`, one-agent vote only).
- **Variants (19 Oct):** [`../2026-10-19_report3_variants_symmetric/`](../2026-10-19_report3_variants_symmetric/) and [`../2026-10-19_report4_variants_nosolution/`](../2026-10-19_report4_variants_nosolution/): budget B = 100 and board windows W = 24, 240 against the gold standard; tables in `gold_standard/2026-10-18/` (column `case`).
- **Notebook:** `shared_analysis/gold_standard/explorer.ipynb`, every figure with sliders, all 8 ρ values, both setups.
- **Companion:** [`../2026-10-16_sham_baseline_symmetric/`](../2026-10-16_sham_baseline_symmetric/). The same data, scored against a sham controller instead of no control.

All paths below are relative to `analysis/task003_llm_free/` unless stated. Commands run from the repository root
with `.venv/bin/python`.

---

## 1. The simulation

| what | where |
|---|---|
| rules (full specification) | `simulation_2/simulator/simulation_2_spec.md` |
| event engine | `simulation_2/simulator/sim2/simulation_2.py` |
| runner | `simulation_2/simulator/run_simulation_2.py` |
| tests | `simulation_2/simulator/tests/` |
| config of the run used | `simulation_2/simulator/configs/sim2_symmetric_fine.yaml` |
| world and posteriors (shared) | `llmfree_core/world.py`, `llmfree_core/world_task003.json` |
| setup: agents and pools (frozen) | `experimental_setup/task003_symmetric/` (`agents.json`, `controller_pool_A0.json`, `controller_pool_A2.json`, README) |

**In short (the report's first paragraph says the same):**
- **Agents.** There are 24 of them, each starting with one true fact. Each reasons as an exact Bayesian over the three allocations.
- **Acting.** Agents act at random times, on average once per time unit. Each action: read q posts from the 48 most recent, vote by probability matching, post one fact supporting the vote.
- **Forgetting.** A remembered fact survives one time unit with probability ρ on average; reading it again renews it.
- **The controller.** Its target is A0 (truth) or A2 (false). It acts at rate λc until t = 30 and reads q_c recent agent posts. It is silenced by the vote gate (at least 75% of the posts it read come from agents voting its target) or by the proof stop. Otherwise it posts the pool fact that best supports its target, at most 30 messages per episode.
- **Episodes** run to t = 40. Each is run three times, silent, target A0 and target A2, with the same random draws.

**Facts at the start (report Table 1):**
- the agents: 18 distinct facts, which together prove A0;
- each pool: 12 facts, 10 of them new to the agents;
- the A0 pool alone gives P = (0.963, 0.037, 0); the A2 pool alone gives (0.114, 0, 0.886).

**The grid:**
- q ∈ {3, 6, 12};
- q_c ∈ {3, 6, 12, 24};
- λc ∈ {0.5, 1, 2, …, 8};
- ρ ∈ {1, 0.75, 0.5, 0.25};
- gate on and off; proof stop on;
- seed 20261007; 1,740 cells × 1,000 episodes.

Cell folder names look like `task003_symmetric__false__q6__qc12__rho0.5__lc1__gateon`.

---

## 2. The data

| level | path (under `results/`, not in git) | content |
|---|---|---|
| raw run | `simulation_2/2026-10-15_sim2_symmetric_fine/cells/<cell>/` | `snapshots.parquet` (one row per episode every 0.25: vote shares over all 24 agents, share who have voted, mean beliefs, proof rates, controller messages and reads), `controller.parquet` (one row per controller action: posts read, v̂, true share, decision), `episodes.parquet`, `params.json` |
| summary table | `gold_standard/2026-10-15/cells_time.parquet` | one row per (simulation, cell, time): vote shares among voters `x0 x1 x2`, mean beliefs `m0 m2`, proofs `proof_any proof_own`, facts posted `facts`, paired susceptibility `G G_se`, state distributions `vote_*` (others: `plur_*`, `s1_*`, `s3_*`) |
| information table | `gold_standard/2026-10-15/mi_table.csv` | one row per (simulation, setting, time, state): `I C R eta` |
| slopes | `gold_standard/2026-10-15/slopes.parquet` | one row per (simulation, arm, setting, neighbouring strength pair, time): `slope slope_se` |
| sham measures (companion report) | `gold_standard/2026-10-15/sham_table.parquet` | `chi chi_se F1 eta_post I_X I_Y C_X R_X eta_info eta_goal` |

The summary tables also hold Simulation 1 (`sim == "sim1"`, run `simulation_1/2026-10-15_sim1_pm_symmetric_fine`).
The 2026-10-16 tables (`build_tables.py --date 2026-10-16` etc.) add a `setup` column, ρ 0.875/0.625/0.375/0.125 and the
nosolution runs; on the rows they share with the 2026-10-15 tables they are identical. `check_overlap.py` confirms the new runs
reproduce the older runs cell by cell where they overlap.

**Rebuild, in order** (the scripts are in `shared_analysis/gold_standard/`, a few minutes in total):

```bash
.venv/bin/python analysis/task003_llm_free/shared_analysis/gold_standard/build_tables.py            # cells_time.parquet
.venv/bin/python analysis/task003_llm_free/shared_analysis/gold_standard/build_slopes.py            # slopes.parquet
.venv/bin/python analysis/task003_llm_free/shared_analysis/gold_standard/make_report_figures.py      # mi_table.csv (+ 15 Oct figures)
.venv/bin/python analysis/task003_llm_free/shared_analysis/gold_standard/make_report_figures_2026-10-16.py   # the gold-standard figures
cd analysis/task003_llm_free/simulation_2/analysis/reports/2026-10-16_report1_symmetric && latexmk -pdf 2026-10-16_report1_symmetric.tex
```

`make_report_figures.py` reuses `mi_table.csv` if it is newer than `cells_time.parquet`. Delete it to force a
recomputation, which takes about 2 minutes.

---

## 3. The metrics (report §1) and how they are computed

| metric | definition | computed in |
|---|---|---|
| vote shares x_{a,t} (Eq. 1) | share of the agents who have voted whose last vote is a; set to 0 at t = 0 (nobody has voted) | `build_tables.load` (`share_Ak / share_voted`); the t = 0 point is added in the figure functions |
| paired susceptibility G_t (Eq. 2) | mean over episodes of x_target(controlled) − x_target(silent), same episode; target = the controller's own (A0 for target A0, A2 for target A2) | `build_tables.summarise` (`G`, `G_se`) |
| state X (Eq. 3) | the vote of a randomly chosen agent; P_s(x) = mean over episodes of the episode's vote shares | `build_tables.states` (`vote_*`) |
| readability I(Z;X) (Eq. 4) | Z = target (A0 or A2, fair coin); ½KL(P₊‖Q) + ½KL(P₋‖Q), Q = ½(P₊ + P₋); bits | `make_report_figures.mi_table` |
| cost C and efficiency η (Eqs. 5–6) | C = ½KL(P₊‖P₀) + ½KL(P₋‖P₀) = I + KL(Q‖P₀); η = I/C; undefined when C < 0.01 bits | same |
| pseudo-count (Eq. 7) | every arm's distribution: (E·P + 0.5)/(E + 0.5K) before any KL | same (`ALPHA = 0.5`); sensitivity in `mi_smoothing_sensitivity.csv` |
| gain per fact G_t/F_t | F_t = the controller's messages up to t (mean per episode) | figure function `fig_per_fact` |
| slope ΔG/Δλc (Eq. 8) | per-episode difference of the target share between neighbouring rates, divided by the step; mean ± 1.96 standard errors (paired) | `build_slopes.py` |
| proofs | share of agents whose remembered facts prove A0; from any facts / from the agents' original facts only | recorded by the simulator (`proof_rate_A0`, `proof_rate_A0_own_evidence`) |

**Figure → code:**

| report figure | file | function in `make_report_figures_2026-10-16.py` |
|---|---|---|
| Fig. 1 votes | `figures/fig_votes.pdf` | `make_report_figures.fig_votes` |
| Fig. 2 time series (G, I, η) | `figures/fig_timeseries.pdf` | `fig_timeseries` |
| Fig. 3 gain per fact and facts posted | `figures/fig_per_fact.pdf` | `fig_per_fact` |
| Figs. 4–6 phase diagrams | `figures/fig_phase_rho_q.pdf`, `fig_phase_rho_x.pdf`, `fig_phase_qc_x.pdf` | `phase_triple` |
| Fig. 7 G against the rate, with slope | `figures/fig_gain_strength.pdf` | `fig_gain_strength` |
| Fig. 8 time × rate map of G and its slope | `figures/fig_strength_time.pdf` | `fig_strength_time` |
| Fig. 9 proofs | `figures/fig_proofs.pdf` | `make_report_figures.fig_proofs` |

**Conventions:**
- **Main setting:** q = 6, q_c = 12, λc = 1 (Simulation 1: b = 1). This posts about as many facts as one fact per night does in Simulation 1.
- **Representative memory** in the q_c × λc diagram: ρ = 0.5.
- **Phase diagrams** are at t = 30, gate on, and show the susceptibility of the A2 controller.

---

## 4. Checks already done

| check | result |
|---|---|
| shared cells reproduce the earlier runs (same seed) | 486 Simulation 2 cells, all 1,000 episodes, every recorded table: identical |
| Eq. 5: C = I + KL(Q‖P₀) | holds to 4 × 10⁻¹⁵ bits |
| range of the measures | I between 0 and 1 bit; η in [0, 1] |
| slopes | equal the difference of the tabulated G to 2 × 10⁻¹⁶ |
| pairing | narrows slope errors by about 30% |
| pseudo-count sensitivity | α = 1 instead of 0.5 moves η by at most 0.004 (one vote); no pseudo-count, by at most 0.005 where C is finite |
| simulator tests | 108 pass |

---

## 5. Results in brief (details and numbers in the `.tex`)

1. **Memory decides whether a swarm that owns a proof can be steered.** With full memory every effect is gone by t ≈ 10. As memory shortens, the silent swarm holds A0 less firmly (100%, 85%, 71%, 58% voting A0 at t = 30) and the susceptibility rises (target A2 at t = 30: 0.09–0.15, 0.27–0.37, 0.37–0.44 for ρ = 0.75, 0.5, 0.25).
2. **Susceptibility and readability move together; η does not.** η stays at 0.76–0.94. It measures how mirror-like the two controllers' effects are, not their size.
3. **The 30-message budget sets the end point; the rate sets the speed.** At t = 5 every rate step pays. By t = 30 only the first (0.5 → 1) does, with slope 0.12 ± 0.03. Slow control is the most effective per fact.
4. **The posts each agent reads switch sides with memory**: they defend the proof at ρ = 0.75 and help the controller at ρ = 0.25.
5. **Every controller replaces part of the agents' own proofs.**

---

## 6. Things to look at critically

- **η is a symmetry ratio.** Read it next to I and G, never alone. The sham-baseline report shows that the handoff's η_info and η_goal behave the same way.
- **Vote shares are among agents who have voted.** The 9–14 October Simulation 2 reports divided by 24 and used a belief-based gain. Two artefacts came from that: an apparent slow rise of votes, and an apparent upward bias of the controller's sensing. Neither is real. Numbers there differ from these for that reason.
- **Mechanism sentences marked "likely reason, not tested"** are hypotheses: the reversal with q, and the effect of reading on the gate.
- **The proof-stop-off arm was not rerun.** Its 9 October cells keep only small tables (`results/simulation_2/2026-10-15_deletion_list.txt`).
- **Caveats** are in the report's last paragraph:
  - one world;
  - exact-Bayesian agents;
  - one controller policy;
  - plug-in information from 1,000 episodes per arm.
