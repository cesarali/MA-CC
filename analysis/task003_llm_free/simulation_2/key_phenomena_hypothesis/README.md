# key_phenomena_hypothesis — guide for agents

**What this folder is.** The study of two phenomena of the LLM-free blackboard swarm in **Simulation 2 (continuous
time)**, on both setups, `task003-symmetric` (the agents together hold a proof of the truth, A0) and
`task003-nosolution` (no set of facts proves anything):

1. **Reading reversal.** Reading more posts (larger q) helps a false controller when agents forget, and protects the
   swarm when they remember.
2. **Persistence.** The controller's effect on votes outlives individual memory and the board's readable window by
   tens of time units.

A third phenomenon, **backfire** (more messages hurt the false controller at full memory without a proof), is
secondary.

**Scope rules.**
- Simulation 2 only. Simulation 1 is out of scope; do not use it, not even as a replication.
- Don't edit the frozen setups (`experimental_setup/`).
- Don't rerun q_c, rate, budget, window or gate grids unless a finding here requires it (see `01_data_audit.md` §4).

## Read in this order

| file | what it holds |
|---|---|
| [`00_handoff_2026-10-20.md`](00_handoff_2026-10-20.md) | the researcher's plan (verbatim): priorities, hypotheses, the two possible campaigns, interpretation rules |
| [`01_data_audit.md`](01_data_audit.md) | step 0: every Simulation 2 run on disk, what each records, what the code guarantees (pairing, forgetting, horizon), corrections to the handoff, a resolved / open table |
| [`02_reanalysis_existing_data.md`](02_reanalysis_existing_data.md) | step 1: findings from existing data only (reversal map, dose check, reversal over time, memory and board composition, persistence, backfire), hypothesis status, and the proposal for steps 2 and 3 |
| [`reports/2026-10-20_step1_findings/`](reports/2026-10-20_step1_findings/) | short LaTeX report of step 1: six figures, each with a few precise statements (figures read from `figures/`) |
| [`reports/2026-10-20_step2_reading_mechanism/`](reports/2026-10-20_step2_reading_mechanism/) | step 2 (Campaign A): G and D at q = 3, 6, 9, 12 and 8 ρ; memory composition; the belief shift split into a direct and an indirect part |
| [`reports/2026-10-20_step3_persistence_carrier/`](reports/2026-10-20_step3_persistence_carrier/) | step 3 (Campaign B): four post-stop rules to t = 130; the controller-only facts carry the persistence |

## Status (20 October 2026)

- Steps 0–3: **done**. Step 2 ran `2026-10-20_sim2_reading_mechanism` (128 settings, 5.7 GB; q = 3, 6, 12 reproduce
  the fine runs exactly). Step 3 ran `2026-10-20_sim2_persistence_branches` (32 settings, 3.9 GB; the normal
  rule reproduces the fine runs up to t = 40). Step 3 uses the simulator setting `post_stop_mode` (default `normal`
  leaves earlier runs unchanged; tests in `../simulator/tests/`).
- Results: the reading reversal is a balance between a direct push of the controller's facts (spreading) and the
  agents keeping their own evidence (integration); the persistence is carried by circulating controller-only facts.

## Terms used here

- **G** (paired susceptibility): for the false controller (target A2), the mean over episodes of [A2 vote share with
  the controller − A2 share in the same episode without it]. Shares count only agents who have voted.
- **D(q₁→q₂)**: G(q₂) − G(q₁), computed per episode. Episodes are paired across q because every random stream is
  keyed by (seed, episode, purpose), never by q (`../simulator/sim2/simulation_2.py`, `keyed_rng`).
- **t\***: the controller's last actual post in an episode. **δ(s)**: the paired effect s time units after t\*.
- **Agent-origin facts**: the facts the 24 agents start with. **Controller-only facts**: those in the A2 pool that no
  agent starts with (10 of 12 in symmetric, 12 of 12 in no-proof).
- **Main setting**: q_c = 12, λc = 1, gate on, budget B = 30, window W = 48, t = 30, 1,000 episodes.
- Light, posts and full records: see `01_data_audit.md` §1.

## Scripts (run from the repository root, in this order)

| script | does | writes |
|---|---|---|
| [`scripts/s0_inventory.py`](scripts/s0_inventory.py) | lists every Simulation 2 setting on disk | `results/key_phenomena/inventory_cells.csv`, `inventory_summary.csv` |
| [`scripts/s1_reversal.py`](scripts/s1_reversal.py) | G per q and paired D per q-pair, every setting and time | `reversal_G.parquet`, `reversal.parquet` (about 1 minute, 12 workers) |
| [`scripts/s1_mechanism.py`](scripts/s1_mechanism.py) | memory and board composition from the full records (q = 6, ρ 0.75 and 1); board against q (no-proof, ρ = 1) | `mechanism.parquet`, `mechanism_board_q.parquet` |
| [`scripts/s1_persistence.py`](scripts/s1_persistence.py) | effect after the last post t\*, with censoring | `persistence.parquet`, `persistence_curves.parquet` |
| [`scripts/s1_backfire.py`](scripts/s1_backfire.py) | no-proof gain against messages and distinct facts delivered | `backfire.parquet` |
| [`scripts/s2_mechanism.py`](scripts/s2_mechanism.py) | step 2: validation, G and D at four q, memory and board composition, direct/indirect belief split | `s2_validation.csv`, `s2_G.parquet`, `s2_D.parquet`, `s2_composition.parquet` |
| [`scripts/s3_branches.py`](scripts/s3_branches.py) | step 3: validation, δ(t) per post-stop rule, half-times, composition over time | `s3_validation.csv`, `s3_delta.parquet`, `s3_summary.csv`, `s3_composition.parquet` |
| [`scripts/s23_figures.py`](scripts/s23_figures.py) | figures `step2_*`, `step3_*` and small tables | [`figures/`](figures/), [`tables/`](tables/) |
| [`scripts/s1_figures.py`](scripts/s1_figures.py) | figures 1–6 (reversal map, reversal over time, persistence, dose, memory/board composition, backfire) and small tables | [`figures/`](figures/), [`tables/`](tables/) |

Command: `.venv/bin/python analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/scripts/<script>.py`.
Large outputs go to `analysis/task003_llm_free/results/key_phenomena/`, which is not in git. Figures and small CSV
tables here are in git.

## Where the data come from

- Raw runs: `analysis/task003_llm_free/results/simulation_2/<run>/cells/<setting>/` (`params.json` plus parquet tables).
- Tables built earlier (`results/gold_standard/2026-10-16/`, `2026-10-18/`) are used only by `s1_backfire.py`.
  The 2026-10-16 tables also hold Simulation 1 rows, so filter `sim == "sim2"`.
- Fact sets: `llmfree_core/world.py` (`World.posterior` gives the exact belief for a fact bitmask) and
  `llmfree_core/setups.py` (`load_setup` gives agent facts and controller pools).

## Interpretation rules (from the handoff, kept here)

- Call sign changes **crossovers**, not phase transitions. Don't claim an exact critical ρ.
- Don't assign a collective relaxation time where the data show a plateau or are censored.
- Comparisons between the two tasks are comparative, not causal: the worlds differ in more than whether a proof exists.
- The controller posts **true** facts, selectively chosen.
- Mark untested mechanisms as "likely reason, not tested".
