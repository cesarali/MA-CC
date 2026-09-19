# Regression references for the finalizer (Cygnus)

`regress.sh <candidate> [name|all]` finalizes each reference study with the candidate tree
into `~/agg/e2e/<name>-<jobid>` and compares every published table with `compare_dirs.py`.
The references are packages produced by unmodified upstream `main` (8c99626a) on 2026-09-19.

| name | study root (login node, user todie) | reference tables | shape | notes |
|---|---|---|---|---|
| b9b15 | `~/MA-CC/results/studies/recomm_only_q12_chatoss_false_control_b9_b15_cygnus` | `…/analysis/tables` (job 40, strict, 6 cells) | 16 CPU / 44 G | the gate used for #9/#10; 3,779 s on main, 438 s with them |
| potsdam | frozen generation bundle `~/agg/recomm_only_q12_chatoss_false_control_frozen_aggregation_inputs_20260918.zip`, relocated per job | `~/agg/iclr_false_arm_potsdam/recomm_only_q12_chatoss_false_control/analysis/tables` (job 42, finalize role, 9 cells) | 8 CPU / 44 G | **finalize role only**: the bundle ships the information fragments computed on Potsdam, so this reference exercises every finalizer stage except `information_resampling`. The first version of this script ran `finalize_to.py` on it and failed at validation (no runs in a generation bundle; job 79). Expect `extra=4`: the reference predates the four incomplete-input diagnostic tables (`available_*_prefixes`, `interrupted_episode_*`) that every finalize now writes; job 84 was byte-equivalent on all shared tables with exactly those four extra. |
| checkpoint | frozen generation bundle `~/agg/blackboard_checkpoint_ensemble_01_frozen_aggregation_inputs_20260918.zip`, relocated per job | `~/agg/checkpoint_ensemble_01/blackboard_checkpoint_ensemble_01/analysis-runs/parallel-tracked-20260918/output/tables` (job 46, finalize role, incomplete inputs allowed) | 16 CPU / 44 G (reference used 32 / 110 G on big-0; measured peak RSS 6.8 GiB) | **finalize role** on the relocated bundle: every checkpoint stage runs, and `branch_round_metrics` runs the information engine on the branch rounds, so the diagnostic bootstraps are exercised. The first version of this script ran `finalize_to.py` on the study root and failed (`KeyError: parent_id`, jobs 87/88): the checkpoint columns exist only in the generation's canonical input, not in a local re-aggregation. |

`finalize_to.py` prints elapsed seconds and peak RSS (self and largest spawned child) so a
change's memory footprint is measured, not guessed; standard nodes give at most 44 G.

Interpreting a difference: `compare_dirs.py` names the table and the first differing
columns. Timestamp-like columns are ignored by default. A change that alters numbers on
purpose ships behind a flag that is off by default and documents the delta in its PR.
