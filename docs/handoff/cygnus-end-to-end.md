# MA-CC on Cygnus, end to end

For Cesar (and Ramses). Written 2026-09-19 by Christian's infra agent after running
the ICLR b9/b15 extension, your Potsdam b6/12/18 frozen bundle and the checkpoint
ensemble through the cluster. Every command below was run this week on Cygnus;
paths are shown for the `todie` account, substitute your own.

## 0 · Where the code is

| Where | What | State |
|---|---|---|
| `cesarali/MA-CC` **#8** | Cygnus site launchers, gateway provider components, study-04 row | merged 2026-09-18 |
| `cesarali/MA-CC` **#9** | calibration bootstrap across cells in a process pool; `MA_CC_ANALYSIS_LAUNCHER`; Cygnus analysis launchers; `mas_cc.studies.relocate` | open, awaiting you |
| `cesarali/MA-CC` **#10** (stacked on #9) | block-statistics calibration bootstrap (28×), per-cell derived pool, exact fast engine for MI/CMI bootstraps and nulls; finalizer 3,779 s → 438 s, 57/57 tables byte-identical | open, awaiting you |
| `cesarali/MA-CC` branch **`cygnus-fork-preview`** | the whole fork as one branch: #9 + #10 + `cross_study`, `systemone`, `semantic_attribution`, the ICLR study configs as run, and this document | pushed for you to read; 25 files, +2,819 / −231 vs `main` |
| `cerebral-work/ma-cc-cygnus` (private) | our working fork; default branch `cygnus-main` == the preview branch | ours |
| `#6` (yours) | EXPERIMENT_DESIGN_MUSR.md | open, untouched by us |

**The question at the end of this document:** do you want the instrument features
(`cross_study`, `systemone`, `semantic_attribution`) delivered as a PR branch into
`main`, kept on the preview branch, or left on our fork?

## 1 · Prerequisites (once)

```bash
# on slurm-login (your account)
PY=/shared/home/cesar/.local/share/mamba/envs/MA-CC/bin/python    # the shared env; has numpy/pandas/pyarrow/matplotlib/yaml
# the login node has no git: fetch the branch as a tarball
mkdir -p ~/MA-CC-cygnus && curl -sL https://api.github.com/repos/cesarali/MA-CC/tarball/cygnus-fork-preview \
  | tar xz --strip-components=1 -C ~/MA-CC-cygnus
export ROOT=$HOME/MA-CC-cygnus PYTHONPATH=$HOME/MA-CC-cygnus/src MA_CC_REPOSITORY_ROOT=$HOME/MA-CC-cygnus
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# gateway: every launcher reads the per-user key file and the in-cluster batch pool
ls -l ~/.llm.key                                    # your LiteLLM key; spend is attributed to you
export LLM_BASE=http://llm-batch.llm.svc.cluster.local:4000
```

Cluster facts that bite: standard nodes give at most `--mem=44G` per job (MemSpecLimit);
`big-0` (`--nodelist=big-0`) gives 40 CPUs and up to `--mem=115G`. Never pass
`sbatch --export=...` (user-environment retrieval fails and the job is held);
export variables in the submitting shell instead. `unzip`, `jq`, `bc` and `git`
are absent on the login node; use `python3 -m zipfile`, `python3 -c`, and tarballs.

## 2 · Run a study

### 2.1 Config layout
A study is a directory with `study.yaml`, one or more arm configs and `analysis.yaml`.
The b9/b15 extension is the worked example:

```
configs/runs/relational_reasoning/blackboard_game/iclr_experiments/recomm_only_q12_chatoss_false_control_b9_b15_cygnus/
  study.yaml          # execution: mode auto, cells_per_shard 1, throttle 6, cpus 8, mem 12G, time 36h, provider_load_control redis_adaptive
  false_control.yaml  # the arm; initialization.artifact_dir points at the 60 paired seeds
  analysis.yaml       # the recipe (estimators, resampling 1000/1000, derived_study_aggregates, calibration, plots)
```

`initialization.artifact_dir` must be an absolute path on the shared filesystem
(`/shared/home/<user>/...`); a Potsdam path fails preflight. Budget guard for a new
arm lives in the arm config (`truth_control.yaml` carries 3000 USD / 6M requests).

### 2.2 Preflight
```bash
STUDY=$ROOT/configs/runs/relational_reasoning/blackboard_game/iclr_experiments/<study>
RESULTS=$HOME/MA-CC-cygnus/results/studies/<study>
$PY -m mas_cc.cli.main study preflight --config-dir $STUDY --output-dir $RESULTS/preflight
cat $RESULTS/preflight/config-0000/report.md | head -60     # PERMITTED / DENIED, cells, episodes, calls, USD estimate
```
b9/b15: 6 cells, 360 episodes, PERMITTED. Truth + no-control arms: 18 cells, 1,080
episodes, ~804,600 calls, ≈ $694 estimated, PERMITTED (staged, not submitted).

### 2.3 Submit
`execution.mode: auto` with `cells_per_shard: 1` resolves to the **cell array**;
the launcher must match (the config-array launcher fails instantly on a cell manifest):

```bash
$PY -m mas_cc.cli.main study submit --config-dir $STUDY --results-dir $RESULTS --throttle 6 \
  --job-script $ROOT/scripts/Cygnus/SLURM/run_study_cell_array.job
cat $RESULTS/submission.json            # records the exact sbatch command and job id
squeue -u $USER
```
What it submitted for b9/b15: `--array=0-5%6 --cpus-per-task=8 --mem=12G --time=36:00:00`.

### 2.4 Watch
```bash
tail -f $RESULTS/logs/slurm-<jobid>_0.out
find $RESULTS/runs -name cell_complete.json | wc -l      # sealed cells
```
Measured: 360 episodes in 5 h 48 min with 6 shards; per call ≈ 25.5 s client-side
against 8.6 s model time (the gap is runner-side). Episode runtime statistics and the
per-round latency breakdown are in the bucket next to the run.

## 3 · Aggregate

### 3.1 One allocation, local backend (recommended)
```bash
sbatch --job-name=agg-<study> --cpus-per-task=16 --mem=44G --time=04:00:00 \
  --output=$RESULTS/logs/aggregate-%j.out --error=$RESULTS/logs/aggregate-%j.err \
  $ROOT/scripts/Cygnus/SLURM/run_study_aggregate.job $RESULTS
```
With #9 + #10 the strict b9/b15 finalizer takes **7.3 min** on 16 CPUs (63 min on
`main`); the Potsdam 9-cell study took 85 min on `main`. Watch
`$RESULTS/analysis/progress.json` (`stage`, `replicate`). The package publishes
atomically into `$RESULTS/analysis/` (tables/, plots/, reports/, validation.json,
analysis_manifest.json, `<study>_analysis.zip`). `analysis_manifest.json →
performance.stages` has per-stage seconds; that profile is how #10 was targeted.

### 3.2 Detached Slurm graph (prepare → per-cell groups → finalizer)
```bash
export MA_CC_ANALYSIS_LAUNCHER=$ROOT/scripts/Cygnus/SLURM/run_study_analysis.job
$PY -m mas_cc.cli.main study aggregate --study-dir $RESULTS --backend slurm
```

### 3.3 Incomplete studies
`--allow-incomplete` on either path produces an explicitly provisional package
(`validation.json: complete=false, allow_incomplete=true`). Label it as such wherever
it goes; the checkpoint-ensemble package in the bucket is one.

### 3.4 Run cost in the manifest

`mas-cc study submit` snapshots the submitting key's gateway spend (`GET
/key/info` on `LLM_BASE`, key from `LLM_KEY` / `LLM_KEY_FILE` / `~/.llm.key`)
into `<study>/run_cost.json`; `study aggregate` takes a second snapshot and
writes the delta into `analysis_manifest.json` under `run_cost` (`usd`,
`status`). Statuses: `ok`, `unavailable` (no gateway in reach, or no key),
`no_start_snapshot` (relocated bundles), `key_changed`. Nothing here makes a
model call and nothing raises: a missing snapshot is a field, not a failure.
Verified from the login node with the Slurm per-user key against the in-cluster
gateway (`http://llm.llm:4000`, status `ok`); the tailnet hostname does not
resolve from inside the cluster, so keep `LLM_BASE` on the in-cluster Service.

## 4 · Finalize a Potsdam frozen bundle on Cygnus

Your `*_frozen_aggregation_inputs_*.zip` bundles carry absolute Potsdam paths. One command
rewrites them, rebuilds the generation workspace where the manifest expects it and
re-verifies every hash and group seal (this replaced an afternoon of hand edits):

```bash
mkdir -p ~/agg/<study> && cd ~/agg/<study>
python3 -m zipfile -e /path/to/<study>_frozen_aggregation_inputs_<date>.zip bundle/
$PY -m mas_cc.studies.relocate --bundle bundle --study-root ~/agg/<study>/<study_id>
#   prints {"valid_groups": N, "problems": []} and the manifest path; --study-root must be empty
sbatch --cpus-per-task=16 --mem=44G --time=04:00:00 \
  $ROOT/scripts/Cygnus/SLURM/run_study_analysis.job finalize <printed manifest path>
```
Two things learned the hard way, both handled by `relocate`: (1) a bundle that ships
`study_lineage.json` without an `extensions/` tree would push the finalizer into
lineage mode and fail in 3 s ("study lineage has no target manifest"), so the file is
set aside; (2) checkpoint-ensemble bundles use the `analysis-runs/<run>/` layout and
**publish to `analysis-runs/<run>/output/`**, not `analysis/`. The checkpoint
ensemble also wants the big node (`--nodelist=big-0 --cpus-per-task=32 --mem=110G`);
it took 2 h 16 min, mostly in `checkpoint_paired_response`, `activation_response` and
`branch_round_metrics`, which are unprofiled and the next target for that family.

## 5 · Combine studies (joint paired aggregation)

When two packages were run on the same paired initializations they can be aggregated
jointly rather than concatenated. `cross_study` unions the canonical `cells`/`rounds`,
reports the pairing, rebuilds events with the finalizer's own builder and runs the
finalizer's `derive_study_control_aggregates` with a `PairedBootstrap` on the union:

```bash
sbatch --cpus-per-task=8 --mem=44G --time=02:00:00 --wrap "$PY -m mas_cc.analysis.cross_study \
  --package b9_b15=$RESULTS/analysis \
  --package b6_b12_b18=~/agg/recomm_only_q12_chatoss_false_control/analysis \
  --output ~/agg/cross/combined_false_control_b6_b18 --workers 8"
```
Real run: 15 cells, 27,000 events, 60/60 blocks shared (one stratum), 6 minutes.
Output: `tables/combined_study_aggregated_metrics.parquet` (the four derived metrics by
budget with joint paired CIs and permutation p-values), state-local versions,
`tables/all_packages_*.parquet` (per-study descriptive tables with `source_package`),
`plots/*_by_budget.png`, `cross_study_manifest.json` (input hashes, pairing report).
The b9/b15 points reproduce the single-study package exactly.

One observation from checking it, present in your own tables too: for the plug-in CMI
the percentile bootstrap interval sits above the point estimate (b9: 0.2023 vs
[0.2435, 0.2926]); the `null_adjusted_bits` columns are the self-consistent ones. A
basic or bias-corrected interval would fix the headline columns. Your call.

## 6 · Publish to the bucket

Bucket `agent-swarm-control-research`. Conventions used so far: your zips and ours in
`aggregation_results/<study>_analysis.zip`; unpacked packages and raw runs under
`ctodie/<study>/…` with `ctodie/README.md` describing each prefix. Christian mints
scoped, short-lived credentials (`cf r2 temporary-credentials create --bucket
agent-swarm-control-research --permission object-read-write --prefixes <your prefix>/
aggregation_results/ --ttl-seconds 43200`) and `rclone` from the login node does
~74 MB/s (`~/bin/rclone --config <conf> sync <dir> r2tmp:<bucket>/<prefix>`). Ask him
for a `cesar/` prefix and a credential when you want to push from your side.

The publish step is a command now, and it verifies what it sent:
```bash
RCLONE_CONFIG=~/.rclone-r2tmp.conf $PY -m mas_cc.cli study publish \
    --study-dir <study> --remote r2tmp:agent-swarm-control-research --prefix ctodie
```
It copies the package zip to `aggregation_results/`, mirrors the analysis directory
to `<prefix>/<study>/analysis`, reads the remote back (`rclone size`, `rclone lsjson`)
and writes `<study>/publish_receipt.json` with local and remote file counts and
byte totals; `verified: false` is a failed publish whatever rclone's exit code said.
Checkpoint-ensemble studies (`analysis-runs/<run>/output`) are found automatically.
To publish straight from the aggregation job, export `MA_CC_PUBLISH_REMOTE`
(and `RCLONE_CONFIG`) in the shell that submits `run_study_aggregate.job`; only a
complete aggregation is published.

## 7 · Optional: semantic attribution with System One

`mas_cc.analysis.semantic_attribution` reads `dashboard_semantic.jsonl`, builds one
bounded state per agent-posted message and asks TypeSafe System One four atomic typed
questions (cites the controller / stance / pressure toward target / provides
evidence). It is opt-in, cached by content hash, and records `provider_calls`, usage
and model in its manifest. It is not part of the finalizer and its question set (q1)
is not calibrated against hand labels yet.

```bash
# select and record a sample on the cluster without calling anything
$PY -m mas_cc.analysis.semantic_attribution --study-root $RESULTS/runs --output ~/agg/semantic/<study> --sample 300 --seed 1 --dry-run
# judge it (needs a gateway key allowed on /typesafe/v1/systemone — today only Christian's; ask him)
MA_CC_SYSTEMONE_KEY_FILE=~/.llm.key $PY -m mas_cc.analysis.semantic_attribution --messages ~/agg/semantic/<study>/semantic_attribution.parquet --output ~/agg/semantic/<study>-judged
```
The 300-message probe on b9/b15 cost 217k input tokens; sequential it took 58 s, with
`--workers 8` 8.5 s (answers identical by construction). The gateway showed no errors up to
32 concurrent requests (~50 requests/s). `--batch-size` above 1 folds several messages into
one request and is cheaper (17 % fewer tokens) but **changes the judgments**: measured
against single-message answers, stance label agreement 82.7 % and mean pressure score
0.45 -> 0.98. Leave it at 1 for anything that will be quoted. Its output is in the bucket
as `ctodie/semantic_attribution_probe_b9b15_q1/` with a README that says what it is.
`--max-usd` / `--max-input-tokens` refuse to send once the projected spend would cross the
cap; `MA_CC_SYSTEMONE_CACHE` shares the answer cache across runs and users.

### 7.1 Experimental: System One as a ballot provider

Provider type `typesafe` (`llm_runtime/providers/adapters/typesafe.py`,
component `configs/components/cygnus/gateway_typesafe_jev.yaml`) answers a
ballot with one calibrated typed Choice over the presented letters instead of
generated text; the ballot contract receives `vote`, a machine-written
`private_reason` with the probability, and a `NONE` public message, and the full
probability vector lands in `raw_response`. It is a different agent from
gpt-oss, not a drop-in: an arm using it is compared on the same paired seeds.
No study config references it. Probe: `scripts/Cygnus/analysis/typesafe_ballot_probe.py`
(20 real ballots, 0.18 s median per call, result in `docs/handoff/typesafe-ballot-probe-2026-09-19.json`).

## 8 · Verify a performance change before trusting it

Every change in #9/#10 was gated on this: finalize into a side directory and compare
every published table against the package the unmodified code produced.

```bash
# both scripts are in scripts/Cygnus/analysis/ on the preview branch
sbatch --cpus-per-task=16 --mem=44G --time=03:00:00 --wrap "\
  $PY $ROOT/scripts/Cygnus/analysis/finalize_to.py $RESULTS ~/agg/e2e/<study>-candidate && \
  $PY $ROOT/scripts/Cygnus/analysis/compare_dirs.py $RESULTS/analysis/tables ~/agg/e2e/<study>-candidate/tables"
# expect: RESULT: byte-equivalent tables; extra=0
```
Reference switches for A/B: `bootstrap_engine="rows"` (calibration) and
`MA_CC_INFORMATION_ENGINE=rows` (information stage).

## 9 · Gotchas, in the order they cost us time

1. A Python script that calls into a spawn `ProcessPool` needs `if __name__ == "__main__":`
   or every worker re-runs the script (BrokenProcessPool at submit). Cost two jobs.
2. Wrong launcher for the execution mode fails instantly (`run_config_array.job` on a
   cell manifest).
3. `sbatch --export` = job held with "user env retrieval failed".
4. Std nodes refuse `--mem` > 44G; `big-0` refuses > 115G.
5. The finalizer entered lineage mode on a relocated bundle (see §4).
6. Checkpoint studies publish to `analysis-runs/<run>/output/` (see §4).
7. The Slurm per-user gateway keys are not allowed on the System One pass-through route
   (LiteLLM 403); in-cluster judging waits on that allowlist.
8. Bucket uploads over the tailnet from a pod run at ~100 KB/s; from the login node
   with rclone, ~74 MB/s.

## 10 · What we would like from you

1. Review #9 and #10 (each byte-identical by construction, reference paths kept).
2. Tell us where the instruments should live: a PR branch into `main`, the
   `cygnus-fork-preview` branch as is, or our fork only.
3. Whether the interval convention in §5 should change.
