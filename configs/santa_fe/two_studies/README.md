# Santa Fe v3: two staged studies (Cygnus handoff)

Design source: `docs/tdd/santa_fe/TWO_STUDIES_AGENT_PROMPT.md`. This directory is a **launch design**. No study, Slurm job, or provider call has been launched from these configs. The synthetic process has no LLM/provider calls.

## Scientific coordinates

All cells use N=24, actual F+=7 and F-=3, initial fact redundancy 3, 60 rounds, target -1, theta=.55, policy beta=8. The late window is rounds 41–60. The full physical space is rho=[0,.4,.7,.85,.95,1], integer b=[0,2,4,6,12,24], integer q_c=[1,3,6,12,24], integer q=[1,3,6,12], and three beta regimes: evidence (2,.5), balanced (1,1.3), social (.5,2). `shared_physical_manifest.csv` lists every one of the 2,160 distinct coordinates and its stage membership. Sensing and budget fractions in YAML resolve to these integers at N=24; verify the saved integer values and realized fact split after execution.

| Stage | Config | Cells | Episodes per cell | Episodes | Retained micro events (N × 60 × episodes) |
|---|---|---:|---:|---:|---:|
| A pilot | `study_A_pilot.yaml` | 24 | 64 | 1,536 | 2,211,840 |
| A core | `study_A_core.yaml` | 108 | 256 | 27,648 | 39,813,120 |
| A interaction, excluding core overlap | `study_A_interaction.yaml` | 456 | 64 | 29,184 | 42,024,960 |
| A complete space, **manifest only** | `study_A_full_manifest_only.yaml` | 2,160 | 256 provisional | 552,960 | 796,262,400 |
| B trajectory estimator pilot | `study_B_trajectory_pilot.yaml` | 10 | 128 reference | 1,280 | 1,843,200 if materialized |

A core and A interaction cover 564 unique cells (24 cells of the conceptual 480-cell interaction grid occur in core). The A pilot coordinates may overlap later stages, but have their own seed namespace and result root; treat their episodes as a separate pilot, not silently as extra core replicates. B's ten-cell panel uses eight balanced q/q_c/rho corners at b=6 and two beta extremes. Expand B later to b=0/24 and crossover cells after a declared pilot selection rule. This initial B panel does **not** meet the full requested estimator power design.

## Separate workflows and outputs

- **Study A MICRO:** `santa_fe.cluster` stages simulation per cell, aggregates saved round/micro trajectories, computes existing per-cell information and aggregates it. Its three reduced theory variants use `santa_fe.theory_integration` on actual saved initial states. Run theory only after A's cell aggregation. Set theory replicas and substeps after timing the pilot; initial values 16 replicas per two initial blocks and 24 substeps are pilot values, not full-grid defaults. Theory output for each physical cell needs a distinct directory. The adapter's current plot is a per-cell comparison; a cross-cell Figure A1/A2 report still needs a separate aggregation/plot pass after the stage completes.
- **Study B trajectory estimator:** `santa_fe.sample_size_cmi` generates its own reference episodes per cell, then makes repeated episode subsamples at fixed physical coordinates and runs the shared MI/CMI, bootstrap and conditional null engine. Pilot sizes are [16,32,64,96,128], 30 repeated subsamples, 49 bootstrap and 99 null draws. Repeated subsamples from one reference bank are correlated; these are exploratory stability curves, not independent physical trials or final false-positive/power estimates. Do **not** submit this B config with `santa_fe.cluster`; use only its sample-size commands.
- **Study B fixed snapshot:** `santa_fe.fixed_snapshot_calibration` consumes saved A v3 round trajectories. At one complete pre-action state it separately samples the U=0 and U=1 next-day channels, with distinct random streams; the physical b=0 null is two independent U=0 datasets. It uses the shared contingency MI estimator on propensity-weighted target-count distributions, converts bits to nats, and records exact H(U|S). Its current pilot report splits null trials into threshold and evaluation halves; scale repetitions/reference bank substantially before any detection or sample-size claim. It does not estimate I_sens at one fixed state. A separate fixed-prior sensor-kernel calculation is required for I_sens calibration.

Keep the result roots `results/study_A_theory_validation/{pilot,core,interaction}` and `results/study_B_estimator_calibration/{trajectory_pilot,fixed_snapshot_*}` separate. Save config copies, execution plans, seed namespaces and the shared physical manifest alongside the outputs on Cygnus. Never treat the full-space YAML as a launch target.

## Cygnus launch preparation (commands for the Cygnus agent)

The Cygnus agent must first check the paths, Python imports (`santa_fe`, `santa_fe_theory`, `mas_cc`, `pandas`, `pyarrow`), free storage, and Slurm limits in `docs/handoff/cygnus-end-to-end.md`; use that site's existing project environment. The old LLM study launcher in that runbook is a different pipeline. These synthetic configs use the existing `santa_fe.cluster` and `santa_fe.sample_size_cmi` cell-array launch planners, which write reviewable `submit_*.sh` scripts. `plan-cygnus` **does not submit**; executing a generated script submits jobs. For each A stage in order, substitute verified absolute `PY`, `ROOT`, and config paths:

```bash
export PY=/shared/home/<user>/.local/share/mamba/envs/MA-CC/bin/python
export ROOT=/shared/home/<user>/MA-CC-cygnus
export PYTHONPATH=$ROOT/src MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$ROOT"
$PY -m santa_fe.cluster plan-cygnus --config $ROOT/configs/santa_fe/two_studies/study_A_pilot.yaml --python "$PY" --repository-root "$ROOT"
# Inspect results/study_A_theory_validation/pilot/submit_cygnus.sh and the execution_plan.json.
# Only after an explicit launch decision: bash <the inspected submit_cygnus.sh>
```

The generated A script creates a simulation cell array, a dependent cell aggregation, a dependent information cell array, and final information aggregation. Repeat with `study_A_core.yaml` or `study_A_interaction.yaml` only after the pilot resource and theory check. Once an A stage's cell aggregation is complete, a theory task for cell `i` is:

```bash
$PY -m santa_fe.theory_integration --config $ROOT/configs/santa_fe/two_studies/study_A_pilot.yaml --out $ROOT/results/study_A_theory_validation/pilot/theory/cell-0000 --cell-ids 0 --initial-blocks 2 --theory-replicas 16 --substeps 24 --branch-pairs 16 --branch-snapshots 1
```

The generic theory-array planner writes that Slurm array with one task per cell, 4 CPUs by default, logs under the A root and a dependency on the **cell aggregation** job. Use the `cells_job` ID printed by the A submit script (or recorded by Slurm):

```bash
$PY -m santa_fe.theory_slurm --config $ROOT/configs/santa_fe/two_studies/study_A_pilot.yaml --python "$PY" --repository-root "$ROOT" --afterok <cells_job_id> --throttle 8 --cpus 4 --memory 16G --time-limit 08:00:00
# Inspect results/study_A_theory_validation/pilot/theory/submit_theory_cygnus.sh.
# Only after an explicit launch decision: bash <the inspected submit_theory_cygnus.sh>
```

The agent should time one q=1 and one q=12 cell before setting the final wall limit and throttle. Do not launch a theory replica for every saved episode by default. Cells are numbered per config; match theory outputs back to `shared_physical_manifest.csv` using resolved cell coordinates, never by cross-config cell ID alone.

For B's trajectory estimator pilot, after checking its 10 cells and reference volume:

```bash
$PY -m santa_fe.sample_size_cmi plan-cygnus --config $ROOT/configs/santa_fe/two_studies/study_B_trajectory_pilot.yaml --python "$PY" --repository-root "$ROOT"
# Inspect results/study_B_estimator_calibration/trajectory_pilot/sample_size_cmi/submit_sample_size_cygnus.sh.
# Only after an explicit launch decision: bash <the inspected submit_sample_size_cygnus.sh>
```

This generates 10 reference-bank tasks, 50 sample-size tasks and one final aggregation, with dependencies. The fixed-snapshot command is a separate downstream task after A's saved round trajectories exist; select actual b=0 and nonzero-b cell IDs from a single A config and put outputs in distinct B directories:

```bash
$PY -m santa_fe.fixed_snapshot_calibration --config $ROOT/configs/santa_fe/two_studies/study_A_pilot.yaml --out $ROOT/results/study_B_estimator_calibration/fixed_snapshot_pilot --cell-ids <b0_id> <active_id> --per-arm 32 64 128 256 512 1024 --reference-per-arm 4096 --repetitions 100 --round 30
```

This last line is a **proposed calibration design**, not a measured runtime guarantee. It may be costly: benchmark one snapshot and one R first, then shard by cell/snapshot/size if needed. The 4096-per-arm reference is still finite Monte Carlo. Freeze conditioning, output partition (integer target count), state selection, reference uncertainty, and independent evaluation seeds before final B testing. The b=0 physical channel has exact T_pi=0 despite a positive finite-sample plug-in estimate. Do not call the null-adjusted value an established information efficiency.

## Scale gate and existing reuse

The existing local saved v3 pilot (`results/santa_fe_v3_pilot`) has two rho=.75, q=3, q_c=12 cells at b=0/6 (40 episodes, 30 rounds each); the local theory comparison pilot (`results/santa_fe_v3_theory_comparison_pilot`) has matching initial-state adapters and plots. These are mechanics/replay examples, not data for the new 60-round, three-beta design. A prior 40-episode pilot compressed round Parquet to 3.5 MB and micro Parquet to 1.0 MB. Linear extrapolation suggests roughly 6 GB of retained Parquet for A core and 6.5 GB for A interaction, excluding per-cell duplication, plots, theory branches, null/bootstrap workspaces and overhead. Micro event counts above are more reliable than storage extrapolation. Wall time and theory enumeration at q=12 remain **unmeasured**; time the A pilot on Cygnus before scaling. The complete design would retain ~796 million micro events and should not be submitted as configured.

Before core submission, the Cygnus agent should document pilot wall time, peak memory/storage, theory Monte Carlo error and q=12 cost, factual replay and b=0 branch null, the supported B sampling levels, and the intended multiple-testing policy. Study A information panels should be limited to B-supported regimes and sample sizes. Final deliverables are separate A/B reports plus a synthesis; the prepared configs alone do not establish agreement, estimator reliability or phase transitions.
