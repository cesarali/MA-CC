# Santa Fe live-board v4: implementation and completed run

**Status, 29 September 2026:** the `santa_fe_live_board_v4` microscopic simulator, its post-processing, and the 28 September full-grid review package exist. The source and configuration changes are currently uncommitted. This page is the entry point for reviewing those changes before committing. It describes the **live-board v4 run**, not the earlier frozen-board Santa Fe/theory study or the LLM experiments.

For the scientific definitions and instructions for reading every table, see the [analysis guide](../../../handoff/santa_fe_live_board_analysis_guide_20260928.md). The [study specification](../../../tdd/santa_fe/SANTA_FE_LIVE_BOARD_STUDY.md) records the original design; the [launch log](../../../handoff/santa_fe_live_board_launch_20260927.md) records dated submission and storage decisions. The resolved executable configuration and result manifests take precedence over early progress counts in that log.

## Implementation map

| Repository path | Role |
| --- | --- |
| [`src/santa_fe/live_board.py`](../../../../src/santa_fe/live_board.py) | v4 episode clock, live board, sensing, voting, fact identities, overload, and saved round/micro records. |
| [`src/santa_fe/game.py`](../../../../src/santa_fe/game.py), `state.py`, `config.py`, `runner.py` | Model dispatch, v4 state/parameter definitions, semantic validation and grid resolution, independent episode execution. |
| [`src/santa_fe/cluster.py`](../../../../src/santa_fe/cluster.py) | Cygnus cell preparation/execution, seals, whole-grid aggregation, and command-line entry point. |
| [`src/santa_fe/llm_parallel.py`](../../../../src/santa_fe/llm_parallel.py) | Adapter from Santa Fe round records to the shared round-information estimator in `src/mas_cc/games/hidden_bench/imitation_round_feedback/analysis.py`. Reusing the estimator does not make this an LLM experiment. |
| [`src/santa_fe/live_board_metrics.py`](../../../../src/santa_fe/live_board_metrics.py), `live_board_postprocess.py` | Additional entropy, propensity-weighted response, state-local response, sensing, and efficiency outputs. |
| [`src/santa_fe/live_board_calibration.py`](../../../../src/santa_fe/live_board_calibration.py), `live_board_plots.py`, `live_board_package.py` | Episode-count sensitivity, phase-map PDF/CSV, and compact review ZIP. |
| [`src/santa_fe/live_board_retention.py`](../../../../src/santa_fe/live_board_retention.py), `repack.py` | Verified per-cell compaction and lossless Parquet repacking. |
| [`tests/santa_fe/test_live_board.py`](../../../../tests/santa_fe/test_live_board.py) | Focused mechanics and adapter checks. |

The implemented version is selected by `model_version: santa_fe_live_board_v4`; it does not replace the v2/v3 code paths. One v4 round records a pre-action population, samples current agent votes for the controller, draws its action, forgets active facts, posts controller messages at dawn when active, then performs 24 sequential focal updates. Peer messages become readable immediately; a focal agent cannot read its own posts. Fact IDs are retained, repeated reads do not create new distinct facts, and overload attenuates the whole voting logit above seven active facts. The [mechanics guide](santa_fe_game_mechanics.md) explains the older clocks and points to this version.

## Experiments and completed coverage

| Run | Resolved config | Coverage and purpose |
| --- | --- | --- |
| Full grid | [`configs/santa_fe/live_board_full.yaml`](../../../../configs/santa_fe/live_board_full.yaml) | **2,916 physical cells**, 128 independent 60-round episodes per cell: **373,248 episodes**. Physical simulation, information post-processing, extra analysis, retention, and whole-grid aggregation completed for all cells. |
| Initial pilot | [`configs/santa_fe/live_board_pilot.yaml`](../../../../configs/santa_fe/live_board_pilot.yaml) | 12 selected physical settings, 128 independent episodes each, for mechanics, runtime, and initial estimator checks. |
| Calibration bank | [`configs/santa_fe/live_board_calibration_512.yaml`](../../../../configs/santa_fe/live_board_calibration_512.yaml) | 12 selected settings, 512 new independent episodes each, for episode-count sensitivity. |

The full grid is `rho={0.25,0.4,0.55,0.7,0.85,1}`, active-round post budget `B={0,3,6,9,12,15,18,21,24}`, reading `q={3,12,24}`, sensing `q_c={3,12,24}`, balanced/evidence-weighted/social-weighted `(beta_E,beta_S)={(1,1.3),(2,0.5),(0.5,2)}`, and controller target `{+1,-1}`. Fixed settings are `N=24`, 12 unique facts (eight truth-favoring), one fact initially per agent with two holders per fact, empty initial board, 60 rounds, policy `sigmoid[4(0.5-Y/q_c)]`, and overload factor `1/[1+0.2 max(0,L-7)]`. The YAML stores budget and sensing as fractions of `N`; the resolved grid has the integer values above. These maps are **finite-horizon** summaries, not established phase transitions.

All full-grid cells have physical summaries, pooled observational information estimates, additional entropy/response tables, and retention seals. The primary physical outcome is the within-episode mean of outcomes `X_41` through `X_60`, then the mean across independent episodes. The 95% late-support half-width target was 0.02: **2,784 cells met it; 132 were flagged precision-insufficient** at 128 episodes. Completion of a cell therefore does not imply adequate precision for every metric.

The full-grid information analysis uses all 60 action-to-next-population transitions per episode, 99 episode-bootstrap draws, 99 null draws, and direct-counting estimates in bits. It includes sensing MI, action entropy, one-round observational `T_pi=I(U_t;K_{t+1}|K_t)`, CMI at several added epistemic conditioning levels, and action-arm matched response/susceptibility. Extra tables include lag-1 to lag-3 propensity-weighted response, state-local response, entropy decompositions, a known-kernel sensing diagnostic, and `eta_IR` components. These are **observational trajectory calculations**, not forced intervention/silence branches. No stochastic mean-field theory was run for v4, and no v4 theory-minus-simulation panels exist. The [analysis guide](../../../handoff/santa_fe_live_board_analysis_guide_20260928.md) gives exact conditioning, units, nulls, and limitations.

The calibration bank varies complete independent episodes in groups of 8 through 512; groups are disjoint within each sample size but reuse the bank across sizes. It tests raw and null-adjusted information/response estimates at 12 fixed conditions. It does not establish calibrated bias, interval coverage, false-positive rate, power, or a universal required episode count. It does not vary forced-branch rollouts. Sparse CMI and ratios need particular care.

## Where the outputs live

| Location | Contents |
| --- | --- |
| `/shared/home/cesar/work/results/santa_fe_live_board_v4_full/` | Canonical full-grid results, `execution_plan.json`, resolved `config.yaml`, cell seals, per-run and per-cell summaries, information and extra-analysis outputs, maps, logs. |
| `/shared/home/cesar/work/results/santa_fe_live_board_v4_pilot/` | Separate 128-episode, 12-condition pilot. |
| `/shared/home/cesar/work/results/santa_fe_live_board_v4_calibration_512/` | Separate 512-episode, 12-condition calibration bank. |
| `/shared/home/cesar/work/aggregation_results/santa_fe_live_board_full_review_20260928.zip` | Compact review package, 115,634,429 bytes; SHA-256 `d95f1ea19862e07db9af2681a14557c461e51eeec8fd6de79831f15731da474f`. |
| `/shared/home/cesar/work/aggregation_results/santa_fe_live_board_analysis_guide_20260928.md` | Standalone copy of the repository analysis guide. |

Start inside the ZIP with `ANALYSIS_GUIDE.md`, then `figures/live_board_maps_index.csv` and `figures/live_board_maps.pdf` for the phase maps. The corresponding numerical values are `figures/live_board_maps.csv`. `summaries/cell_metric_summary.csv`, `summaries/per_run_summary.parquet`, and `information/round_information_estimates.parquet` contain the compact full-grid evidence; `derived/` contains the additional response/entropy tables; `calibration/` contains episode sensitivity. The ZIP's `MANIFEST.json` records its members and code versions. The result root's `execution_plan.json` records the prepared configuration and code hashes. These result directories and the ZIP live **outside Git**.

Each full-grid cell retains all 128 per-episode **summary rows**, its estimator outputs and individual null draws, and eight deterministically chosen illustrative trajectories. Full per-round/per-micro data for most episodes were deleted only after cell-level analyses and seals completed, as requested to control storage. The retained examples are illustrations, not a replacement independent data bank. New estimates needing the deleted paths or new branch outcome histograms require new simulation data.

## Reproduction and commit scope

The main CLI is `python -m santa_fe.cluster` with `prepare`, `run-cell`, `run-information-cell`, `postprocess-cell`, `aggregate-cells`, `aggregate-information`, and `plot-live-board` operations. The Cygnus launch instructions and environment rules are in [`docs/handoff/cygnus-end-to-end.md`](../../../handoff/cygnus-end-to-end.md); use the repository's existing Cygnus environment. The generated `submit_cygnus.sh` in the result root is a historical launch template; the completed cells already have manifests and seals. Do not rerun those commands against the compacted result root expecting deleted raw trajectories to reappear.

For a source commit, include the changed `src/santa_fe/` modules, the three `configs/santa_fe/live_board_*.yaml` files, `tests/santa_fe/test_live_board.py`, and these Santa Fe documentation files. Review `git status --short` before staging. Keep `/shared/home/cesar/work/results/` and `/shared/home/cesar/work/aggregation_results/` outside the repository; their package and manifests preserve the run artifacts separately. The committed source documents and reproduces the implementation, while the result root's hashes identify what executed.
