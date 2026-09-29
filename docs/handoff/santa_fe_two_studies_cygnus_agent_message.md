# Message for the Cygnus agent: Santa Fe v3 two-study handoff

Please prepare and, only when authorized by Cesar, launch the two staged **synthetic Santa Fe v3** studies from this checkout. Start by reading `docs/tdd/santa_fe/TWO_STUDIES_AGENT_PROMPT.md`, `configs/santa_fe/two_studies/README.md`, `docs/documentation/games/santa_fe/santa_fe_theory_integration.md`, and `docs/handoff/cygnus-end-to-end.md`. This is the frozen-board binary game; there are no LLM/provider calls. Use Cygnus's verified Python and paths, not Potsdam paths.

The runnable theory package is already extracted into `src/santa_fe_theory/`; its adapter is `src/santa_fe/theory_integration.py`. The original `src/santa_fe/santa_fe_theory_runner_v1.zip` is **not needed** for execution or provenance after this handoff. Its source files are hashed directly in each theory integration manifest. Ensure the extracted source directory accompanies the code transfer. In this checkout `src/santa_fe_theory/` is currently untracked by Git: a Git-based transfer must add and commit that directory first; a file-copy transfer must copy it explicitly. Verify `src/santa_fe_theory/core.py` exists on Cygnus before deleting the ZIP from the transfer source.

## Files and stages

- Shared 2,160-cell design: `configs/santa_fe/two_studies/shared_physical_manifest.csv`.
- Study A pilot: `study_A_pilot.yaml` (24 cells × 64 episodes). Run this first. The theory planner is `python -m santa_fe.theory_slurm`, depending on the A cell aggregation job. It schedules `python -m santa_fe.theory_integration` per physical cell, comparing the three named reduced variants to actual saved MICRO initializations and trajectories.
- Study A core: `study_A_core.yaml` (108 cells × 256 episodes; q=3, q_c=12). Launch only after pilot timing, storage, replay and theory checks.
- Study A communication/sensing interaction: `study_A_interaction.yaml` (456 new cells × 64 episodes; with the 24 overlapping core cells excluded). Combined core + interaction = 564 distinct physical cells.
- `study_A_full_manifest_only.yaml` defines all 2,160 cells; do **not** launch it as a study.
- Study B trajectory estimator pilot: `study_B_trajectory_pilot.yaml` (10 reference cells × 128 independent episodes). Use `python -m santa_fe.sample_size_cmi plan-cygnus` and its own generated submit script; do not submit it through the A simulation launcher. It plans 10 reference tasks and 50 sample-size tasks using the existing MI/CMI, bootstrap and conditional-null engine.
- Study B fixed-snapshot pilot: `python -m santa_fe.fixed_snapshot_calibration` on saved A pre-action states after A cell aggregation. This separately draws U=0/U=1 next-day arm outcomes, evaluates a complete-snapshot channel on integer target counts, uses independent no-actuation null arms, and records exact policy action entropy. See the proposed command and caveats in the README.

Result roots are `results/study_A_theory_validation/{pilot,core,interaction}` and `results/study_B_estimator_calibration/{trajectory_pilot,fixed_snapshot_*}`. The A and B seed namespaces and outputs are separate. Preserve the YAML, shared manifest, resolved cell/seed metadata, raw saved states, and output manifests. Never merge pilot episodes into core cells silently or mix q/q_c/beta cells in one estimator.

## Cygnus sequence

1. Transfer the current source/config/docs, including `src/santa_fe_theory/`. Verify imports and the resolved Cygnus paths and resource limits. No ZIP unpacking is needed.
2. From the repository root, call `python -m santa_fe.cluster plan-cygnus` on A pilot; review the generated four-stage submit script and `execution_plan.json`. Its four stages are simulation-cell array → aggregate cells → information-cell array → aggregate information. The planning call writes files only; executing the generated script submits jobs.
3. After A cell aggregation, call `python -m santa_fe.theory_slurm` with the cell-aggregation Slurm job ID as `--afterok`; review the generated per-cell theory-array script. Time q=1 and q=12 theory cells before assigning full resources.
4. Independently plan B trajectory calibration with `python -m santa_fe.sample_size_cmi plan-cygnus`; inspect its reference → sample-size → aggregation dependencies. Run fixed-snapshot calibration only after selected A states exist.
5. Decide whether to expand A core and interaction from observed wall time, storage, theory Monte Carlo error, factual replay, b=0 causal null, and B estimator support. Do not launch the complete 2,160-cell design by default.

At the end, provide separate Study A theory-versus-MICRO results and Study B estimator-reliability results, then a short synthesis that marks which A information comparisons B supports. The current configs are an exploratory pilot and staged design, **not completed phase diagrams or a sample-size recommendation**. Cross-cell A1/A2 figure assembly and final B power/coverage calibration still require work after pilot data exist; do not claim them as already generated.
