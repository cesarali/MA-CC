# Regression references for the finalizer (Cygnus)

`regress.sh <candidate> [name|all]` finalizes each reference study with the candidate tree
into `~/agg/e2e/<name>-<jobid>` and compares every published table with `compare_dirs.py`.
The references are packages produced by unmodified upstream `main` (8c99626a) on 2026-09-19.

| name | study root (login node, user todie) | reference tables | shape | notes |
|---|---|---|---|---|
| b9b15 | `~/MA-CC/results/studies/recomm_only_q12_chatoss_false_control_b9_b15_cygnus` | `…/analysis/tables` (job 40, strict, 6 cells) | 16 CPU / 44 G | the gate used for #9/#10; 3,779 s on main, 438 s with them |
| potsdam | `~/agg/iclr_false_arm_potsdam/recomm_only_q12_chatoss_false_control` | `…/analysis/tables` (job 42, strict, 9 cells) | 8 CPU / 44 G | relocated frozen bundle |
| checkpoint | `~/agg/checkpoint_ensemble_01/blackboard_checkpoint_ensemble_01` | `…/analysis-runs/parallel-tracked-20260918/output/tables` (job 46, provisional) | 32 CPU / 110 G, `big-0` | `--allow-incomplete`; `study_lineage.json` is set aside in that root |

`finalize_to.py` prints elapsed seconds and peak RSS (self and largest spawned child) so a
change's memory footprint is measured, not guessed; standard nodes give at most 44 G.

Interpreting a difference: `compare_dirs.py` names the table and the first differing
columns. Timestamp-like columns are ignored by default. A change that alters numbers on
purpose ships behind a flag that is off by default and documents the delta in its PR.
