# Santa Fe live-board study launch — 27 September 2026

**Historical launch log.** Counts and pending-job statements below are dated
snapshots. For completed coverage and current artifact locations, use the
[v4 implementation and results guide](../documentation/games/santa_fe/live_board_v4_implementation_and_results.md).

## Scope and resolved design

This is a new microscopic simulator, `santa_fe_live_board_v4`. It does not run or fit stochastic mean-field theory. The [study specification](../tdd/santa_fe/SANTA_FE_LIVE_BOARD_STUDY.md) defines the scientific questions; [the full YAML](../../configs/santa_fe/live_board_full.yaml) is the resolved executable grid.

| Axis | Values |
| --- | --- |
| Persistence | 0.25, 0.40, 0.55, 0.70, 0.85, 1.00 |
| Posts per active round | 0, 3, 6, 9, 12, 15, 18, 21, 24 |
| Agent reading / controller sensing | 3, 12, 24 each |
| Evidence/social coefficients | balanced (1, 1.3), evidence-weighted (2, 0.5), social-weighted (0.5, 2) |
| Controller target | truth +1 and false -1 |

The Cartesian grid has **2,916 physical cells**. Initial replication is **128 independent episodes per cell**, or **373,248 episodes** total. Each episode has 60 action–outcome transitions, 61 saved population states, and 1,440 saved micro-slots. Truth- and false-target arms share an initialization seed at matching other physical parameters. They are separate physical trajectories once actions differ. The support precision target is a 95% half-width of at most 0.02; information estimates that fail support or calibration checks must be flagged and topped up selectively. The first completed 12-cell bank had maximum late-support half-width about 0.011 at 128 episodes, but it does not guarantee that bound over the entire grid.

Other fixed settings: N=24 agents, F=12 unique facts with eight truth-favoring and four false-favoring IDs, two initial holders per fact and exactly one fact per agent, empty initial board, policy `sigmoid[4(0.5 - Y/q_c)]`, 24 focal slots per round with replacement, reading without replacement from the current live board with own messages excluded, and whole-logit overload `1/(1+0.2 max(0,L-7))`. At B=0, the randomized gate is logged but posts nothing. Target support is measured before each action; the late support summary uses **X_41 through X_60**, the outcomes of intervention rounds 41–60. These are finite-horizon maps.

## Execution and storage status

The full manifest is [execution_plan.json](/shared/home/cesar/work/results/santa_fe_live_board_v4_full/execution_plan.json). Raw per-cell files will be under `/shared/home/cesar/work/results/santa_fe_live_board_v4_full/cells/cell-NNNN/`: `rounds.parquet`, `micro.parquet`, and `cell_complete.json`. The seal checks the config, simulator code hash, row/episode counts, and raw file hashes. Streaming aggregation leaves these per-cell files canonical; it does not duplicate a whole-grid trajectory archive.

An initial **200-cell production batch** was submitted as Slurm array **7899** with 32 simultaneous tasks, four CPUs and 12 GiB per task. Its 200 cells and 25,600 independent episodes are sealed. The other **2,716 physical cells are planned, not completed**. No full-grid information job has been submitted yet.

**28 September update:** array **8099** completed cells 200–299, giving **300 sealed physical cells and 38,400 episodes** with zero failed tasks in arrays 7899 and 8099. The cluster now reports eight idle compute nodes with 224 CPUs in total; `/shared` expanded to **167 GB total and 77 GB free** at 07:15 UTC. Array **8235** is running cells 300–1299 as 1,000 Slurm indices with a +300 cell-ID offset. The remaining cells are still unsubmitted. Slurm `MaxArraySize=1001`; the prepared script now splits each large array stage into chunks of at most 1,000 indices. Check live seals and disk free space rather than relying on these dated figures.

Two separate 12-cell estimator banks are complete: `/shared/home/cesar/work/results/santa_fe_live_board_v4_pilot` at 128 episodes per cell and `/shared/home/cesar/work/results/santa_fe_live_board_v4_calibration_512` at 512. Their shared-engine information outputs, disjoint episode-group calibration files, and extra entropy/response files are separate from production. A 99-draw policy-null calibration has also run on disjoint groups of the 512-episode bank. These are calibration conditions, not coverage of the full grid.

The first bank measured 355 MB of canonical raw files across 1,536 episodes, or 29.6 MB per 128-episode cell on average; size ranges from 16.8 MB at B=0 to 40.7 MB in high-budget cells. A straight 128-episode full-grid projection is about **86 GB raw**, before analysis and headroom. `/shared/home/cesar/work/results` was a 98 GB NFS filesystem on 27 September. Deleting 2.947 GB of prior Santa Fe CSV/Parquet files increased free space from about 13 GB to 16 GB; old configs/manifests/logs and the 16 MB September 26 review ZIP remain. On 28 September the filesystem expanded to 167 GB. **Check `df -h /shared/home/cesar/work/results` before each additional batch**; the remaining grid plus analysis needs headroom beyond the straight raw projection.

## What the analysis pipeline does

The [prepared Cygnus script](/shared/home/cesar/work/results/santa_fe_live_board_v4_full/submit_cygnus.sh) has six dependent stages for the full study: simulation array; sealed-cell summary aggregation; shared information-engine array; information aggregation; v4 entropy/response post-processing array; and finite-horizon map generation. It was prepared but **not executed as a whole** while storage was insufficient. Array 7899 was a bounded simulation submission using the same `run-cell` command; already sealed cells are skipped if the full script later launches.

The shared information stage adapts all 60 pre-action transitions per episode to the existing MA-CC round estimator. It computes sensing MI, sensor–action MI, action and conditional action entropy, target/population/truth/order one-step CMI, exact memory and coarse epistemic CMI, state-matched susceptibility, and information fractions. It exports raw estimates, policy-redraw or sensor-permutation nulls, raw-minus-mean-null, episode bootstrap intervals, and overlap diagnostics. Full-grid pooled estimates use all episodes within each physical cell; per-round estimates are configured for four primary statistics. `T_pi` here is observational `I(U_t;K_(t+1)|K_t)`, with a one-round lag. The action is the virtual logged gate at B=0. It is not a forced-intervention branch estimate.

The extra post-processing stage reads the sealed v4 rows and exports `H(K)`, `H(U)`, `H(U|K,E)` at declared evidence levels, `I(U;K_next)`, conditional outcome entropies and their CMI difference, known-propensity action entropy, IPW response at lags 1–3, available susceptibility, state-local response tables, hypergeometric sensing information, and descriptive `eta_IR`. These outputs reside in each cell's `live_board_analysis/`. The cell summary contains late support/evidence/overload, terminal majority, intervention frequency, actual cumulative posts, and matched B=0 gain where a baseline exists. The map script writes a PDF and the underlying numeric CSV.

Current limitations are explicit: exact high-dimensional conditioning can be sparse; raw plug-in CMI has substantial positive finite-sample bias in the calibration banks; the shared percentile bootstrap interval for raw CMI sometimes excludes its own observed estimate, so it must not be treated as calibrated coverage without correction. The 99-draw disjoint-group calibration assesses null false positives but does not by itself provide an independent high-precision reference for all alternative conditions. Extra entropy and IPW outputs currently lack full episode-bootstrap intervals, and the map script covers scalar cell summaries and shared-engine information, not all requested trajectory/distribution/state-local figures. Those are outstanding analysis work, not completed outputs.

## Exact continuation checks

1. Verify new shared storage with `df -h /shared/home/cesar/work/results`; do not infer capacity from free-space claims alone.
2. Confirm `find /shared/home/cesar/work/results/santa_fe_live_board_v4_full/cells -name cell_complete.json | wc -l` and check Slurm `sacct -j 7899` for failures.
3. Submit remaining simulation cells with the same config/code hash and four-CPU `run-cell` tasks, bounded by available storage. A fully verified expansion permits executing the prepared six-stage script; sealed cells are idempotently skipped.
4. After all 2,916 cells are sealed, run the dependent aggregation and analysis stages, verify their seals, and report any precision-insufficient cells. Do not label the pilot or first 200 cells as a complete grid.

There are no provider calls or paid LLM requests in this study.

## 28 September, 07:38 UTC: resumed grid and online retention

The technician's expansion is visible: `/shared` is 167 GB total. At the latest check it had 42 GB free, and Slurm exposed eight healthy nodes with 224 CPUs total. Per-node free RAM was at least 63 GB; 150 CPUs were allocated to this user's 38 running jobs. These are point-in-time values; use `df`, `sinfo -N`, and `squeue` for current capacity.

Simulation arrays 7899, 8099, and 8235 finished cells 0–1299. Array 8723 is running cells 1300–2299; **9776** is queued for cells 2300–2915 after 8723. The matching information and extra stages for the final slice are **9777** and **9778**. At 07:38, 1,780 physical simulation cells were sealed. Job IDs and completed-cell counts are separate from independent episode counts.

New raw Parquet files use Zstandard level 6; older sealed SNAPPY files remain valid. A cell-level retention stage now waits for both the information seal and all five extra analysis files, computes and seals all 128 per-episode summary rows, and keeps eight deterministic illustrative episode paths (first four ordered seeds, two lowest and two highest final target shares, with duplicate slots filled by ordered seeds). It then removes the full `rounds.parquet` and `micro.parquet` for that cell. Information estimates, null draws, state-local response tables, per-run summaries, and their source hashes remain. The retention manifest is `cells/cell-NNNN/retained/retention_complete.json`. Those examples cannot support arbitrary re-estimation across all 128 episodes; the full raw paths have been deliberately discarded at the user's request after the calculated statistics were verified.

Retention arrays **10151**, **10152**, **10153**, and **10164** cover the four ranges 0–299, 300–1299, 1300–2299, and 2300–2915. Each starts only after its matching information and extra arrays have ended; each cell independently checks both outputs before deleting raw files. Array 8507 has two intentionally canceled duplicate tasks whose valid information seals came from diagnostic array 8396, so retention uses `afterany` plus per-cell seal checks. The full-cell and information aggregators accept verified retained summaries and original source hashes. The first production cell was compacted and resume checks returned `already_compacted`; the pilot cell aggregation also passed after the refactor.

The launch script at the result root remains a **prepared template**, not a command to execute now: manual arrays above already cover the grid. Final whole-grid aggregation, maps, precision audit, and review package are still pending.
