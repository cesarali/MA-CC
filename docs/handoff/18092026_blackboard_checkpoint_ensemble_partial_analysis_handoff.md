# Blackboard checkpoint K=40 recovery and partial-analysis handoff

Date: 2026-09-18  
Repository: `/home/ojedamarin/Projects/LanguageGames/MA-CC`  
Study result root: `/work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/blackboard_checkpoint_ensemble_01`

## Purpose and current user priority

The user urgently needs paper-ready analysis from the data already collected for:

```text
configs/runs/relational_reasoning/blackboard_game/blackboard_checkpoint_ensemble_01
```

The user explicitly accepts a visibly provisional/incomplete analysis using the
current data. Do not launch another provider study merely to make the dataset
strictly complete. Do not replace the checkpoint experiment analysis with only
generic study summaries. The required analysis is the dedicated analysis in:

```text
docs/tdd/features/games/16092026_BLACKBOARD_CHECKPOINT_EXPERIMENT_IMPLEMENTATION_PLAN.md
src/mas_cc/analysis/checkpoint_ensemble.py
```

The generic `mas-cc study aggregate` command is still the correct packaging and
execution vehicle because `analysis.yaml: checkpoint_ensemble_outputs` invokes
the dedicated checkpoint engine. Run it with `--allow-incomplete` for this
explicitly authorized provisional analysis.

No more provider calls are required for the requested partial analysis.

## Mandatory Potsdam environment

Read `.codex/skills/ma-cc-study-workflow/SKILL.md` and
`.codex/skills/ma-cc-study-workflow/references/potsdam.md` before acting.

Use this environment for every command:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run --live-stream -n MA-CC ...
```

Do not use system Python or create another environment. Results and analysis
must remain under `/work/ojedamarin/Projects/LanguageGames/MA-CC/results`.

## Scientific design in plain language

There are four `(q, rho)` cells and 40 independent prepared parents per cell:

```text
(3, 0.70), (3, 1.00), (12, 0.70), (12, 1.00)
```

Each parent has one immutable checkpoint after `L=2` preparation rounds and
nine `M=10` continuation paths:

1. `none`;
2. `always_truth`, `b=3`;
3. `always_truth`, `b=12`;
4. `always_false`, `b=3`;
5. `always_false`, `b=12`;
6. `sensing_truth`, `b=3`;
7. `sensing_truth`, `b=12`;
8. `sensing_false`, `b=3`;
9. `sensing_false`, `b=12`.

Thus the target is 160 parents and 1,440 continuation paths. The parent—not a
branch, scheduler task, or individual round—is the independent resampling block.

## Exact saved-data inventory

The detailed round and micro-slot records are much more complete than the
generic episode-compaction status suggests.

| Cell | Parents with all 9 paths | Complete paths |
|---|---:|---:|
| `q3_rho070` | 40/40 | 360/360 |
| `q3_rho100` | 40/40 | 360/360 |
| `q12_rho070` | 40/40 | 360/360 |
| `q12_rho100` | 38/40 | 348/360 |
| **Total** | **158/160** | **1,428/1,440** |

The partial-analysis recovery gate, run against the actual study lineage, found:

```text
parents_observed: 160
parents_with_complete_branch_coverage: 158
complete_paths: 1428
excluded_incomplete_paths: 12
retained continuation round records: 14280
retained checkpoint + continuation round records: 14440
retained complete-path micro-slot records: 342720
```

The two partial parents are both in `q12_rho100`:

- `run-0026`
- `run-0032`

Each lacks these six complete paths:

- `always_false`, `b=3`
- `always_false`, `b=12`
- `sensing_false`, `b=3`
- `sensing_false`, `b=12`
- `sensing_truth`, `b=3`
- `sensing_truth`, `b=12`

Both retain complete `none`, `always_truth/b=3`, and
`always_truth/b=12` paths. `run-0032` has one saved round of
`always_false/b=3`, but the partial-analysis selector excludes that entire path
because it does not have all horizons `1..10`. Missing paths are never replaced
with zeros.

The earlier count of only 100 generic “complete episodes” referred to compact
episode Parquet artifacts. It did not mean the branch trajectories were lost.
Rich `round_trajectory.jsonl` and `micro_slot_trajectory.jsonl` records exist
for the 1,428 complete paths.

## Execution history

- `1882011`: initial submission; failed immediately because offline pricing
  could not price the university model.
- Pricing mode was changed from offline to live; the provider reports a zero
  proxy price for this model.
- `1882015`: first extension attempt; failed because the cell worker rejected
  ordinary non-grid configs.
- `1882019`: real 24-hour run; one cell completed and three tasks timed out.
- `1882308`: 24-hour recovery; all three scheduler tasks completed in 3.6–6.5
  hours. It produced 1,428 sealed paths but exposed generic resume-compaction
  errors (`non-contiguous interactions`) and left the two partial parents above.

Provider state for `1882308` finished with concurrency limit 80, zero active
leases, no provider error, and no retries. The recovery made 39,596 acquire and
release transactions. No provider work is currently running.

## Why restart duplicates are safe to resolve

Interrupted branches appended a failed prefix, and a later retry appended a
complete replacement attempt to the same trajectory file. The established
canonical rule is to keep the last physical record at each scientific
coordinate. For checkpoint paths the relevant coordinates are:

```text
parent, policy, posting budget, copy, post-branch horizon
```

For micro-slots, round and within-round slot coordinates are also included.
After selecting the last record, a path is eligible only if its round horizons
are exactly `1..10`. This produces 1,428 complete paths and excludes exactly 12.

## Dedicated analysis required by the implementation plan

The final partial package should contain the checkpoint-specific outputs already
implemented in `src/mas_cc/analysis/checkpoint_ensemble.py`:

- canonical parent-by-branch endpoints;
- controlled-minus-`none` paired effects at `h=1` and `h=10`;
- truth-support and false-target effects in fractions and percentage points;
- `h=0..10` trajectories;
- 2,000-replicate whole-parent block bootstrap and effective `K`/missingness;
- assigned-policy `I(A; m_h | n_0)` through the existing CMI engine;
- unsmoothed, Jeffreys, Miller–Madow, and configured uniform-smoothing variants;
- five-fold parent-grouped cross-fitted classifier scores;
- repeated grouped splits as sensitivity, not confidence intervals;
- paired whole-record label-swap null with classifier refitting;
- sensing activation response;
- branch-local round metrics and nulls;
- controller/resource reporting;
- checkpoint validation, plots, reports, provenance, and ZIP package.

### Permutation-null semantics

The user specifically asked about this. The implementation at
`classifier_analysis()` is paired correctly:

- analysis is separate for each fixed `(q, rho, policy, budget, target, h)`;
- each row contains one parent’s controlled endpoint and its own paired `none`
  endpoint;
- each permutation independently decides whether to swap those two endpoints
  within each parent;
- no endpoint is moved to another parent;
- the full grouped classifier is refitted after every permutation;
- cross-validation folds keep both records from a parent together.

The recipe requests 1,000 label-swap/null permutations and five repeated
grouped splits. This is CPU-heavy but requires no provider calls. A realistic
estimate is approximately 2–6 hours, though the current implementation may
need parallelization or a longer finalizer limit because the classifier null is
sequential.

## Current uncommitted implementation changes

Do not discard the dirty worktree. It contains user work unrelated to this
study. In particular, do not modify or revert the dirty ICLR configs or the
untracked Potsdam scripts unless separately requested.

Relevant changes made for this recovery are:

1. `configs/.../blackboard_checkpoint_ensemble_01/analysis.yaml`
   - changed `theoretical_reference` from `single_affinity_revised` to `none`;
   - this is required because finite-memory board mode and finite epistemic
     persistence have no valid single-affinity theoretical reference;
   - the checkpoint estimators themselves remain configured unchanged.
2. `src/mas_cc/analysis/checkpoint_ensemble.py`
   - added `prepare_checkpoint_ensemble_inputs()`;
   - qualifies game-local parent IDs by scientific cell;
   - merges completed trajectories with explicitly available interrupted
     records;
   - selects the last record at each coordinate;
   - retains only paths with complete `h=1..M` coverage;
   - reports recovery diagnostics.
3. `src/mas_cc/studies/aggregation.py`
   - routes recovered checkpoint rounds/micro-slots to every dedicated
     checkpoint estimator;
   - writes recovered checkpoint canonical tables into the analysis package.
4. `src/mas_cc/studies/extension.py`
   - restores target cells omitted from a retry-delta execution manifest;
   - this is essential because the completed `q3_rho070` cell was absent from
     the latest three-cell retry manifest.
5. `tests/mas_cc/test_blackboard_checkpoint_phase2.py`
   - adds recovery selection, cell-qualified identity, incomplete-path
     exclusion, and restart-deduplication coverage.
6. `src/mas_cc/studies/analysis_slurm.py`
   - added `_frozen_config_inputs()`;
   - freezes each resolved config path exactly once using its current hash;
   - avoids requiring one current file to match multiple historical hashes
     carried by earlier submission and retry entries.
7. `tests/mas_cc/test_analysis_slurm.py`
   - adds a regression test proving duplicate historical hashes for one config
     path produce one current frozen config input.

Focused tests passed:

```text
pytest -q tests/mas_cc/test_blackboard_checkpoint_phase2.py \
  -k 'partial_checkpoint_analysis or strict_coverage or existing_cmi or grouped_classifier'
4 passed

pytest -q tests/mas_cc/test_study_extensions.py \
  tests/mas_cc/test_blackboard_checkpoint_phase2.py \
  -k 'partial_checkpoint_analysis or extension or retained or completed_checkpoint'
11 passed

pytest -q tests/mas_cc/test_analysis_slurm.py \
  tests/mas_cc/test_blackboard_checkpoint_phase2.py \
  -k 'generation_config_inputs or generation_freezes or partial_checkpoint_analysis or strict_coverage or existing_cmi or grouped_classifier'
6 passed
```

## Current analysis status

The original analysis attempt for generation `8a446c14c8ef23967882` failed in
preparation because its manifest contained old and current hashes for the same
four config paths:

```text
prepare:    1883073 FAILED after 16 seconds
array:      1883074 CANCELLED without running
finalizer:  1883075 CANCELLED without running
```

The failure was:

```text
ValueError: study config changed after planning: .../q3_rho070.yaml
```

This provenance/indexing defect is now fixed. The repaired generation manifest
contains exactly four unique config inputs, one for each current config path.
The same CPU-only provisional analysis was resubmitted as:

```text
generation: 8a446c14c8ef23967882
prepare:    1883079 COMPLETED in 7 seconds
array:      1883080 COMPLETED, all 4/4 groups, no failures
finalizer:  1883081 RUNNING
```

Latest observed progress on 2026-09-18 at approximately 09:31 CEST:

```text
completed_groups: 4
pending_groups: 0
failed_groups: []
published: false
stage: information_resampling
```

The four standard information-estimator groups are complete. The dedicated
checkpoint finalizer is now computing the plan-specific paired effects,
parent-block bootstraps, assigned-policy information, classifier sensitivity,
1,000 paired label-swap/null refits, plots, reports, and package. No provider
calls are involved. The finalizer has a six-hour SLURM limit and the working
estimate is 2–6 hours because the classifier-null refits are CPU-heavy and
currently sequential.

Progress path:

```text
/work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/blackboard_checkpoint_ensemble_01/analysis/.work/8a446c14c8ef23967882/progress.json
```

No final archive was published as of this update.

## Resume here

1. Monitor finalizer job `1883081`, not the cancelled first-attempt jobs.
2. Inspect both `sacct` and the generation `progress.json`; scheduler completion
   alone is insufficient.
3. If `1883081` fails or reaches its six-hour limit, inspect its finalizer log
   before changing analysis settings. Do not reduce 2,000 bootstrap or 1,000
   null permutations without explicit user authorization. Prefer parallelizing
   the existing parent-grouped calculations or increasing only the operational
   finalizer time limit.
4. Confirm the real checkpoint recovery gate remains exactly:

   ```text
   160 parents observed
   158 parents with all nine branches
   1428 complete paths
   12 incomplete paths excluded
   14280 continuation round rows
   342720 micro-slot rows
   ```

5. After publication, validate every checkpoint-specific Parquet table and
   report effective `K`:
   comparisons involving the six absent branches at `q=12, rho=1.00` should
   have `K=38`; unaffected comparisons should normally have `K=40`.
6. Confirm validation records 160 observed parents, 158 full-coverage parents,
   1,428 complete paths, and 12 excluded incomplete paths.
7. Clearly label the output provisional/incomplete and list the two partial
    parents and 12 excluded paths in the paper-facing summary.

Expected final archive path:

```text
/work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/blackboard_checkpoint_ensemble_01/analysis/blackboard-checkpoint-ensemble-01_analysis.zip
```

## Important cautions

### Progress instrumentation added after finalizer launch

The user requested counters but explicitly instructed: **do not stop anything**.
The running finalizer `1883081` was not cancelled or restarted.
`classifier_analysis()` now accepts an optional progress callback and reports
the active scientific comparison, comparison count, split-repeat count,
permutation count, total completed classifier refits, elapsed time, and a
rate-based ETA. Aggregation connects this callback to the existing atomic
progress writer/heartbeat, and has distinct checkpoint stage names for paired
response, assigned-policy information, classifier, activation response, and
branch-round metrics.

These edits apply to future invocations; the existing Python process has already
loaded its code and cannot acquire these counters automatically. No counters
were fabricated for that process. Three focused tests passed, including a
regression proving instrumentation does not change outputs/seeds. This change
adds reporting only; it does not yet add resumable comparison checkpoints.

Subsequently, future classifier-null launches were parallelized: aggregation
passes `SLURM_CPUS_PER_TASK` (default 1) to `classifier_analysis(workers=...)`.
Independent permutation refits use a spawn-based process pool, with nested BLAS
threads limited to one per worker. Parent-specific swap masks are drawn in the
coordinator in the original serial RNG order; result order and refit seeds are
unchanged. Three focused classifier tests passed, including exact serial versus
two-worker output equality using the actual classifier. With four allocated
CPUs a future finalizer can use four permutation workers. The already-running
job remains untouched; no cancellation, restart, or new launch was performed.
Only the permutation refits are parallelized, not every analysis stage. No
resumable comparison checkpoint mechanism has yet been added.

- Do not rerun the full provider study for this user request.
- Do not use missing branches as zero outcomes.
- Do not mix parents across `(q, rho)` cells; parent IDs are game-local unless
  qualified by `cell_key`.
- Do not average CMI estimates computed on execution shards.
- Do not implement a replacement CMI estimator.
- Do not freely shuffle labels across parents. The null is a within-parent
  controlled/none swap, followed by full classifier refitting.
- Do not claim the provisional package is the preregistered complete K=40
  result. It is a 1,428/1,440-path analysis with branch-specific effective K.
- Preserve raw results and all existing checkpoint/branch seals.

## Isolated tracked parallel analysis launched 2026-09-18

The user explicitly authorized a second analysis with separate working/output
paths and the same frozen scientific data. Generic analysis orchestration now
provides `fork_generation` and `submit_frozen_aggregation`; no study-specific
SLURM job was added. Source generation `8a446c14c8ef23967882` was copied,
not regenerated, into:

`/work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/blackboard_checkpoint_ensemble_01/analysis-runs/parallel-tracked-20260918`

- Preparation: **1883361**, initially pending priority.
- Finalizer: **1883362**, initially pending preparation dependency.
- Four CPUs, 16G memory, **24-hour** safety limit; four classifier permutation
  workers once that stage starts. Other stages can remain serial.
- All four completed information/support groups were copied and validated;
  no information array was needed and no provider calls are made.
- Progress: `<workspace>/progress.json`.
- Working finalization: `<workspace>/final`.
- Isolated published results: `<workspace>/output`, containing
  `blackboard-checkpoint-ensemble-01_analysis.zip` on success.
- Logs: `<study-root>/logs/analysis-8a446c14c8ef23967882-parallel-tracked-20260918/`.
- Original **1883081** remained RUNNING at elapsed 2:55:53 when checked.
  It was not stopped, cancelled, resubmitted, or modified.

Important fix discovered during isolation: the frozen-snapshot loader previously
loaded only cells/episodes/rounds/micro_slots, ignoring its frozen recovery-prefix
tables. The new invocation loads every table listed in canonical_inputs. The
old process cannot acquire this fix while running; audit its scientific counts
before using or comparing its final results. Both use the same frozen files,
but the new invocation correctly includes complete branches recovered from
those saved prefixes.

Independent inventory check on the copied frozen inputs confirmed:
160 observed parents, 158 complete-coverage parents, 1,428 complete paths,
12 excluded incomplete paths, 14,440 retained round records, and 342,720 retained
micro-slot records. This is authorized provisional/partial analysis, not strict
completion of the full study. No scientific configuration was changed.

Verification: all nine analysis-SLURM tests passed. Six focused submission,
isolation, counter, and parallel-classifier tests passed after final changes;
`git diff --check` passed. The broader phase-2 suite has a stale design assertion
expecting K=120 while the authorized configurations contain K=40; this assertion
failed and was not used as a reason to change the scientific design.

Do not submit the ordinary aggregate command against the original generation
to monitor/restart this isolated job. Inspect the isolated manifest and progress
paths above. The original publication can replace its own main analysis tree,
but the isolated workspace is outside that tree and publishes only to its own
output destination.

## Developer briefing: what aggregation calls and how to improve it

### What the supplied ZIP is

`blackboard_checkpoint_ensemble_01_frozen_aggregation_inputs_20260918.zip`
is the saved experimental data plus analysis inputs, NOT the finished statistical
results package. It contains all eight frozen Parquet inputs, validation,
four completed information/support groups, scientific configs, the analysis
recipe, implementation-plan document, and relevant code snapshots. Its README
explains recovery selection and relocation of absolute Potsdam paths.
It contains no credentials and requires no further LLM/provider calls.

There are 160 observed parent checkpoints, 158 parents with all nine
continuations complete, and 1,428 complete continuations out of 1,440.
Always describe the resulting analysis as provisional/partial.

### Public command versus the already submitted isolated job

The normal user-facing entry point for this authorized partial dataset is:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run --live-stream -n MA-CC \
  mas-cc study aggregate \
  --study-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/blackboard_checkpoint_ensemble_01 \
  --allow-incomplete
```

This is a description of the standard interface, NOT an instruction to execute
another aggregation now: the standard destination can collide with other work.
Do not run `study submit`; that launches scientific/provider collection, not
offline analysis. Strict aggregation without `--allow-incomplete` is unsuitable
for the current incomplete dataset.

The isolated job was created through Python APIs
`mas_cc.studies.analysis_slurm.fork_generation(...)` followed by
`submit_frozen_aggregation(manifest_path)`. Its generic SLURM launcher calls:

```text
python -m mas_cc.studies.analysis_worker finalize <isolated-execution-manifest>
  -> analysis_slurm.finalize(manifest_path)
  -> aggregation._aggregate_study_local(
       study_dir,
       allow_incomplete=True,
       analysis_output_dir=<workspace>/final,
       canonical_snapshot_dir=<workspace>/input,
       information_fragments_dir=<workspace>/groups,
       information_fragments_analysis_hash=<frozen-analysis-hash>,
       progress_path=<workspace>/progress.json)
  -> package compact tables/reports/plots
  -> publish ONLY to <workspace>/output
```

`scripts/Potsdam/SLURM/run_study_analysis.job` establishes the repository runtime
directory and MA-CC environment and limits nested numerical-library threads.
The finalizer job is 1883362; preparation is 1883361. Inspect current scheduler
state rather than treating submission as proof that computation has started.
No new job was launched while writing this briefing.

### Scientific pipeline and responsible code

The recipe is
`configs/runs/relational_reasoning/blackboard_game/blackboard_checkpoint_ensemble_01/analysis.yaml`.
The scientific requirements are in
`docs/tdd/features/games/16092026_BLACKBOARD_CHECKPOINT_EXPERIMENT_IMPLEMENTATION_PLAN.md`.
The checkpoint-specific implementation is
`src/mas_cc/analysis/checkpoint_ensemble.py`, called by
`src/mas_cc/studies/aggregation.py`; it is not merely a generic average of runs.

| Stage | Function / work |
|---|---|
| Frozen input validation | `analysis_slurm.prepare`: file/config/recipe checksums |
| Complete-data recovery | `prepare_checkpoint_ensemble_inputs`: merge saved rounds and prefixes, deduplicate physical coordinates, exclude incomplete branches |
| Branch endpoints | `endpoint_table`: qualified parent/branch/horizon outcomes |
| Paired response | `paired_response`: controlled-minus-none effects at h=1 and h=10; 2,000 parent-block bootstrap resamples, 95% intervals |
| Assigned-policy information | `assigned_policy_information`: recipe smoothing variants 0, 1, 12.5 |
| Classifier inference | `classifier_analysis` -> `cross_fitted_classifier_score`: five repeated parent-grouped splits and 1,000 paired label-swap refits per comparison |
| Additional scientific outputs | `resource_report`, activation-response analysis, `branch_round_metrics` and null summaries |
| Publication | Compact Parquet estimates, validation, methods/summary reports, checkpoint plots, provenance and final ZIP |

Generic configured estimators are sensor MAE, sensor MSE, and controller action
entropy. Their four per-cell information/support groups are already complete
and reused; remaining work is predominantly in the checkpoint-specific finalizer.

Classifier comparisons are kept separate by scientific conditions, including
q, rho, controlled policy, posting budget, target and horizon. Outer validation
is five-fold parent-grouped; inner selection is three-fold parent-grouped,
considering constant models and logistic penalties C=0.01, 0.1, 1, 10.
Repeated nested fitting is the suspected major computational bottleneck;
profile it to establish timings rather than assuming an exact percentage.

The classifier null swaps complete controlled/none records WITHIN each parent
for a fixed comparison and refits the classifier. It does not shuffle labels
across parents, cells, or unrelated branches. Parent IDs must be cell-qualified
to prevent train/test leakage. Missing outcomes are never filled with zeros.

### Existing improvements and safe next optimization targets

Already implemented for future/new processes:

- Load every frozen input table, including recovery prefixes. The old timed-out
  process had loaded the earlier code; do not infer its coverage from the new
  inventory check.
- Parallel permutation refits through a spawn-based `ProcessPoolExecutor`.
  Worker count comes from `SLURM_CPUS_PER_TASK`; this job has four workers.
  Each worker limits nested BLAS threads to one.
- Draw swap masks in the coordinator in the original serial RNG order and
  consume results in deterministic order, keeping seeds/results reproducible.
- Machine-readable progress after completed refits: comparison counts,
  repeat/permutation counts, refit counts, elapsed time, and rate-based ETA.
  Counters start when their analysis stage executes, not while SLURM queues.

Recommended developer work, NOT already implemented or authorized to launch:

1. Profile representative comparisons; separate input preparation, model
   fitting, process startup, serialization, and plotting from queue delay.
2. Reuse one worker pool across comparisons instead of starting a pool for
   each comparison, if profiling shows meaningful startup overhead.
3. Prepare compact numeric inputs once rather than repeatedly sending a
   DataFrame and rebuilding identical deterministic features for every refit.
   Validate that folds, nested model selection and null outcomes are unchanged.
4. Add invocation-local resumable comparison checkpoints with identity/hash
   validation and atomic writes. Currently a finalizer restart recomputes the
   classifier work; counters are NOT resumable scientific checkpoints.
   Remove transient refit/null draws from successful final packages.
5. Test serial/parallel and interrupted/resumed equivalence with fixed seeds,
   including cell-local duplicate parent IDs and incomplete branches.
6. If distributing independent comparisons across SLURM tasks, extend the
   generic analysis infrastructure rather than creating a checkpoint-specific
   job script. Scientific comparison identity must remain independent of shards.

Do not improve speed by silently reducing the 1,000 permutations, 2,000 bootstrap
resamples, folds, repeats, comparisons or sample coverage, replacing nested
validation with a different procedure, or using a different null. Those are
scientific changes requiring explicit agreement. CPU optimization also cannot
remove SLURM queue delay; the 24-hour allocation can constrain backfill windows.

The existing frozen-data ZIP contains the earlier handoff snapshot. Send this
updated Markdown document alongside that ZIP for this developer briefing.
