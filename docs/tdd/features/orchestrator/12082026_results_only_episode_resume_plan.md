# Results-only Orchestrator Artifacts and Episode-boundary Resume

**Date:** 2026-08-12
**Status:** implementation plan
**Scope:** `mas_cc` experiment/grid orchestration, local artifact retention, resume state,
configured analysis, and local-to-Comet reporting boundaries.

## Goal

Add a local `results_only` artifact profile that keeps the scientific outputs of a run
small and easy to copy without changing what the master process reports to Comet.

The target operating pattern is:

- one grid cell is commonly active at a time;
- approximately ten episodes run concurrently within that cell;
- the job may be left unattended for many hours;
- after interruption, completed episodes must not be paid for and run again;
- episodes that were in flight may restart from the beginning;
- after the cell finishes, the local directory should contain CMI results,
  order-parameter plots, a small prompt sample, compact scientific data, and only the
  metadata needed to understand and resume the run;
- Comet must continue receiving the same progress, aggregate metrics, plots, and
  information-analysis results it receives today.

This is a local-retention change, not a reduction in remote observability.

---

## 1. Why this is needed

The inspected completed cell under
`results/completed-comet-cell-0000-results-plots-data/` contains:

| Quantity | Observed value |
|---|---:|
| Completed episodes | 100 |
| Local size | 387 MiB |
| Repository-produced files | 3,129 |
| Prompt Markdown files | 2,000 |
| Size projected over 28 comparable cells | about 10 GiB |
| Files projected over 28 comparable cells | about 87,000 |

The extracted directory also contains 3,129 `:Zone.Identifier` sidecars. Those are
Windows transfer metadata rather than experiment artifacts, but they double the visible
file count after transfer.

The principal local costs are not the final CMI table or PNGs. They are repeated
per-episode operational records:

- `budget_events.jsonl`;
- `trajectory.jsonl` with rich decisions, responses, and repeated histories;
- `events.jsonl`;
- `experiment.log`;
- `usage_cost.jsonl`;
- `api_call_status.jsonl`;
- `metrics/streaming.csv`;
- prompt Markdown written once per configured round **per episode**.

In the inspected cell, the small result bundle consisting of the aggregate plots,
information estimate table/report, diagnostics, resolved cell config, cell summary, and
one prompt is approximately 0.27 MiB. A narrow scientific event table will add some data,
but should remain orders of magnitude smaller than the rich raw trajectory.

---

## 2. Executive design decision

Implement two independent controls:

1. **Artifact profile** — what is retained in the local results directory.
2. **Checkpoint mode** — the boundary at which completed work is durable and can be
   skipped on resume.

The requested configuration should read approximately:

```yaml
logging:
  comet: true
  options:
    show_metrics: true
    prompt_examples:
      count: 2
      scope: cell

storage:
  output_dir: results
  artifact_profile: results_only
  checkpoint_mode: episode
  overwrite: true
  wipe_and_recompute: false
```

`logging.comet` and the `observability.comet` section keep their current meanings. The
new storage settings do not disable, thin, rename, or reroute Comet data.

The `results_only + episode` combination is the first implementation target. A true
mid-episode/round resume mode is explicitly outside this task.

---

## 3. Required invariants

### 3.1 Comet is unchanged

For the same resolved run config, `full` and `results_only` must make equivalent calls to
the master Comet sink for:

- run parameters and tags;
- heartbeat/progress metrics;
- grid progress images;
- cell aggregate curves and scalars;
- requested aggregate metric PNGs;
- CMI estimates, confidence intervals, null summaries, and interpretability flags;
- the configured information report/assets.

Per-episode Comet experiments remain disabled. The master remains the only Comet writer.
No prompt, message, response, private evidence, or raw trajectory content is newly sent
to Comet.

Local cleanup must not depend on Comet being reachable. The compact local scientific
data and rendered outputs must be sufficient to retry aggregate reporting later. Comet
remains a view rather than the only store.

### 3.2 Resume is at the episode boundary

With `checkpoint_mode: episode`:

- a completed episode is durable and is skipped after restart;
- an incomplete episode has no valid completion marker and restarts from its original
  seed at round zero;
- with parallelism ten, a crash can repeat at most the episodes that were in flight,
  normally no more than ten;
- a completed cell is sealed and skipped as a whole;
- no raw prompt object is restored;
- no claim of mid-episode resume is made.

### 3.3 Scientific equivalence

Given the same completed episodes, the following must be numerically identical between
`full` and `results_only`:

- every configured information estimate;
- bootstrap intervals;
- permutation-null mean and interval;
- support diagnostics and interpretability flags;
- order-parameter aggregate curves and percentile bands;
- population-action-share and dominant-action-share aggregates currently sent to
  Comet;
- episode/cell inclusion and exclusion counts.

### 3.4 Cleanup is transactional

Raw or temporary data may be removed only after the compact replacement has been:

1. written to a temporary path;
2. flushed and atomically renamed;
3. read back successfully;
4. checked for expected episode identity and row counts;
5. recorded in a completion manifest.

If any step fails, the source data stays in place and the cell remains unsealed.

### 3.5 Config and seed compatibility are enforced

Resume must reject a checkpoint whose resolved-config hash, prompt-definition hashes,
cell coordinates, episode seed, scientific schema version, or pricing identity differs
from the requested run. It must never silently combine observations from two configs.

### 3.6 Budget accounting cannot move backward

A resumed run must not forget committed requests, input tokens, output tokens, or cost.
Reservations belonging to a dead process must not be restored as active reservations.
See Section 9 for reconciliation rules.

---

## 4. Current behavior that must be corrected

### 4.1 The current checkpoint is not a mid-episode resume point

`RunRecorder.record_interaction` writes a replaceable `.checkpoints/checkpoint.json`
containing game state, budget state, hashes, and `completed_rounds`.

The experiment orchestrator does not load that checkpoint into a game runtime. Its resume
path checks only the per-episode `manifest.json` and skips an episode only when the
manifest says `completed`.

Consequently, `storage.checkpoints: true` currently writes round snapshots but does not
resume an unfinished experiment episode from one. The new API must describe the resume
boundary honestly.

### 4.2 Prompt example count has episode scope

The current `prompt_examples.count` is passed to every episode observer. For 100 episodes
and `count: 20`, it creates 2,000 Markdown files. `scope: cell` must be introduced so two
examples means two examples for the whole cell.

### 4.3 Aggregation and analysis depend on verbose episode files

- Cell aggregation reads every episode's `metrics/streaming.csv`.
- HiddenBench imitation analysis recursively reads `trajectory.jsonl`.
- Grid summaries embed all episode outcomes in JSON.

The results-only implementation must teach these consumers to read a shared compact
scientific schema. It must not compute a different analysis merely to save space.

### 4.4 The recorder writes full operational history unconditionally

Even with detailed prompt audit disabled and per-episode Comet disabled, the recorder
still creates API status, usage, budget, event, log, trajectory, streaming metric,
checkpoint summary, and local metric files. Turning off Comet or lowering the log level
therefore does not solve the local retention problem.

---

## 5. Configuration contract

### 5.1 Artifact profile

Add a first-class `StorageConfig.artifact_profile`:

```text
full          current artifact behavior; default for compatibility
results_only  compact resumable scientific artifacts and final results
```

Reject unknown values during config loading and in the JSON schema. Export the resolved
value in `resolved_config.yaml`.

Do not implement this as a generic logging-level test spread across writers. Resolve the
profile once into a typed retention policy used by the recorder, orchestrator, analysis,
and finalizer.

### 5.2 Checkpoint mode

Add a first-class `StorageConfig.checkpoint_mode`:

```text
off       no resume guarantee
episode   completed episodes are durable; incomplete episodes restart
```

`episode` is the recommended mode for paid concurrent runs.

For backward compatibility, continue accepting the existing `checkpoints` boolean for
one transition period:

```text
checkpoints: false  -> checkpoint_mode: off
checkpoints: true   -> checkpoint_mode: episode
```

Rules:

- reject a config that specifies both fields;
- resolved configs emit only `checkpoint_mode`;
- document that historical `checkpoints: true` did not restore partial episodes;
- do not add a `round` value until a runtime can reconstruct the game and continue it
  without repeating or omitting a provider call.

### 5.3 Prompt sample scope

Extend the existing prompt example mapping:

```yaml
logging:
  options:
    prompt_examples:
      count: 2
      scope: cell       # episode | cell
```

Compatibility:

- omitted `scope` retains the existing `episode` behavior under `full`;
- `results_only` requires `scope: cell` or defaults it to `cell` with a visible resolved
  value;
- `count: 0` writes no prompt examples;
- examples are local only and remain excluded from Comet.

The sample must be deterministic, not whichever concurrent episode happens to finish
first. Each completed episode shard may temporarily retain at most `count` compressed
prompt/response candidates. At cell sealing, select the lowest-ID successful episode and
choose stable successful attempts from it, then discard every unselected candidate. For
`count: 2`, prefer an early and late attempt so the sample shows both the initial and
history-rich prompt shapes. This temporary bounded duplication is acceptable; thousands
of separate prompt files are not. Render all selected examples into one
`prompt_examples.md` per cell or one run-level file, not one file per prompt.

---

## 6. Compact scientific record

### 6.1 One schema serves analysis, aggregation, and resume

Create a versioned compact episode artifact, preferably Parquet because the repository
already uses Parquet compaction elsewhere and the data are tabular.

One completed episode shard must contain enough information to reproduce all configured
HiddenBench imitation CMI and all aggregate metrics currently requested by the run.

Minimum identity/provenance columns:

```text
schema_version
run_id
cell_id
episode_id
episode_seed
interaction_index
resolved_config_hash
prompt_definition_hashes_hash
dynamics_mode
control_mechanism
task_id
```

Minimum categorical/scientific columns:

```text
possible_answers
correct_answer
analysis_target
N_t
N_t1
Y_t
U_t
Z_t
Z_t1
Mtruth_t
Mtruth_t1
Morder_t
Morder_t1
Xf_t
Xf_t1
```

Retain or derive the fields needed for existing behavioral/support diagnostics, including
sensor error, focal target adoption, and action-conditional changes. Avoid duplicating a
field when it is an exact deterministic function of retained integer counts, provided the
reader performs that derivation centrally and tests it against the current analysis.

Do not store:

- compiled messages;
- rendered prompt blocks;
- raw model responses;
- reasoning text;
- repeated conversation histories;
- full budget snapshots on every row;
- arbitrary event dictionaries.

### 6.2 Initial state

Order-parameter plots include state zero. Each episode shard must therefore either:

- contain an explicit state-zero row; or
- contain enough fields in its first transition row to reconstruct state zero exactly.

Use one convention across games and document it in the schema. Do not infer state zero
from a later state.

### 6.3 Episode terminal metadata

Store a small terminal record alongside or within the shard:

```text
status
interaction_count
termination_reason
error_type/error_summary when failed
started_at
finished_at
usage totals
```

Only `completed` shards are valid resume checkpoints and analysis inputs. Failed or
partial shards must never be forward-filled into completed curves.

### 6.4 Atomic write protocol

For episode `E`:

1. Accumulate compact rows in memory while `E` runs.
2. Write `.resume/<cell-id>/<episode-id>.parquet.tmp`.
3. Flush/close it.
4. Read it back and validate schema, episode ID, seed, status, and expected interaction
   count.
5. Atomically rename it to `<episode-id>.parquet`.
6. Atomically update the cell state to mark `E` complete.

A `.tmp` file is never a completed checkpoint. On resume it may be deleted or replaced
after its explicit cell/episode path has been validated.

The game result must not be reported to the orchestrator as durably completed until step
five succeeds.

---

## 7. Runtime and resume lifecycle

### 7.1 Launch

Before sending provider calls:

1. Resolve the run and cell IDs exactly as today.
2. Load or create the run/cell checkpoint state.
3. Validate the config, prompt, pricing, and schema fingerprints.
4. Discover valid completed episode shards.
5. Reconcile budget state.
6. Schedule only missing/incomplete episode IDs, using their original derived seeds.

### 7.2 While episodes run

In `results_only`:

- scientific transition rows are accumulated in a bounded per-episode buffer;
- streaming metrics may be computed in memory and sent to the master as today;
- the recorder does not write verbose per-attempt/event histories;
- a concise failed-episode record is retained;
- the selected prompt sample collector retains only bounded candidates;
- committed global budget totals are checkpointed independently of episode success.

The memory bound is small for the target HiddenBench episodes. If a future game can have
an unbounded horizon, the compact writer may spill compact rows to a temporary shard; it
must not fall back to the rich trajectory schema.

### 7.3 Crash and resume

After a crash:

- valid completed shards are skipped;
- temporary/unsealed shards are ignored;
- originally assigned seeds are reused;
- unfinished episodes restart from round zero;
- dead-process reservations are reconciled as described in Section 9;
- the master progress count starts from the number of validated completed shards;
- completed cell artifacts are not recomputed unless requested or invalid.

For parallelism ten, the expected maximum repeated simulation work is ten episodes. A
process can have fewer or more active tasks than the configured parallelism during
shutdown, so correctness must rely on shard validity rather than on the number ten.

### 7.4 Cell finalization

When all expected episodes are terminal:

1. Validate the set of completed, failed, and skipped episode IDs.
2. Aggregate the cell directly from compact shards.
3. Render the aggregate plots.
4. Merge completed shards into `scientific_events.parquet.tmp`.
5. Read back and validate episode IDs and row counts.
6. Atomically rename to `scientific_events.parquet`.
7. Write `cell_summary.json` without embedding full per-episode histories.
8. Write `cell_complete.json` containing hashes of the compact file and outputs.
9. Notify the master monitor and attempt the same Comet reporting as today.
10. Remove the cell's `.resume` shards only after the local seal is durable.

Comet failure does not invalidate the local seal. Record Comet status in a small summary
and retain enough compact data to retry reporting.

### 7.5 Run/grid finalization

After all cells are terminal:

1. Merge or reference sealed cell scientific files.
2. Run configured CMI analysis from the compact schema.
3. Write the results-only output allowlist.
4. Upload the same configured CMI metrics, plots, and assets to Comet.
5. Write a final run manifest with artifact hashes and cell status counts.

For a one-cell experiment, avoid retaining both byte-identical cell-level and run-level
scientific tables. Promote or reference the cell file atomically.

---

## 8. Results-only final layout

The exact nesting may follow `results_run_dir`, but a completed one-cell run should be
equivalent to:

```text
<run>/
  manifest.json
  resolved_config.yaml
  run_summary.csv
  scientific_events.parquet
  information_estimates.csv
  information_estimates.md
  support_diagnostics.csv
  prompt_examples.md
  cell_complete.json
  comet_summary.json
  plots/
    information_estimates.png
    order_parameters.png
```

A grid may use:

```text
<run>/
  manifest.json
  resolved_base_config.yaml
  grid_summary.csv
  scientific_events.parquet
  information_estimates.csv
  information_estimates.md
  support_diagnostics.csv
  prompt_examples.md
  comet_summary.json
  cells/
    cell-0000/
      scientific_events.parquet
      cell_complete.json
      cell_summary.json
      plots/order_parameters.png
    cell-0001/
      ...
  plots/
    information_estimates_cell-0000.png
    information_estimates_cell-0001.png
```

During a partial grid, each sealed cell retains its own compact scientific file so that
its result survives before the rest of the grid finishes. Once the run-level merged file
has been validated and the run is sealed, the finalizer may remove byte-redundant
cell-level copies. Cell manifests then retain each contribution's row count and content
hash so the merge remains auditable.

Prefer one multipanel order-parameter image per cell containing `m_ctrl`, `m_truth`, and
`m_order`. Include other plots currently requested by `aggregation.cell_metrics` when
they are required for unchanged Comet reporting; they may be combined into sensible
multipanel figures locally while preserving the same remote metric series.

### 8.1 Files omitted after successful finalization

The results-only allowlist excludes:

```text
data/episodes/*/trajectory.jsonl
data/episodes/*/events.jsonl
data/episodes/*/experiment.log
data/episodes/*/budget_events.jsonl
data/episodes/*/usage_cost.jsonl
data/episodes/*/api_call_status.jsonl
data/episodes/*/metrics/streaming.csv
data/episodes/*/local_metrics.csv
data/episodes/*/checkpoint_manifest.json
data/episodes/*/comet_summary.json
data/episodes/*/prompts/round_*.md
analysis/event_metrics.csv
analysis/information_nulls.csv
analysis/option_share_trajectories.csv
```

The estimate table already retains the null mean and interval. The individual permutation
draws are not final results and need not be retained in `results_only`.

Do not emit a large `grid_summary.json` containing every outcome. Retain compact counts
and concise failure records. Full outcome detail remains available in the compact episode
terminal table when scientifically necessary.

---

## 9. Minimal budget checkpoint

The current budget log repeatedly serializes a large nested status object. Replace that
history in results-only mode with one atomically replaced run-level state:

```text
budget_state.json
```

It should contain at least:

```text
schema_version
resolved_budget_hash
pricing_snapshot_hash
committed_requests
committed_input_tokens
committed_output_tokens
committed_cost and unit
outstanding reservation ceilings at last durable update
updated_at
```

Update it under a process-local lock immediately after each provider reconciliation. Use
temporary-file + `fsync` + `os.replace`; concurrent episode tasks must never interleave
writes.

On resume:

1. Restore committed totals.
2. Do not restore old reservations as active.
3. If authoritative live spend is available, reconcile it before new calls.
4. If authoritative spend is unavailable, conservatively convert the last outstanding
   reservation ceilings into committed debit before clearing them.

This may overcount after an abrupt crash, but it must never undercount paid work and allow
the restarted run to exceed its approved ceiling.

Add a concise budget summary to the final manifest. Do not retain per-call budget history
unless the user selects `artifact_profile: full`.

---

## 10. Analysis and aggregation changes

### 10.1 Reader abstraction

Introduce one episode-scientific-data reader interface with implementations for:

- current rich `trajectory.jsonl` + `metrics/streaming.csv` artifacts;
- compact results-only Parquet shards/final files.

The CMI estimator and cell aggregator consume the normalized reader output rather than
testing paths themselves throughout the analysis.

Keep legacy readers so existing completed runs remain analyzable.

### 10.2 Results-only analysis writer

In `full`, preserve all current analysis files.

In `results_only`, write only:

- `information_estimates.csv`;
- `information_estimates.md`;
- `support_diagnostics.csv`;
- information estimate plot(s);
- compact analysis summary/manifest;
- any concise cell contrast table requested by the configured study.

Compute bootstraps and permutation nulls in memory. Do not persist every bootstrap or null
draw. Their seeds, counts, confidence level, summary intervals, and code/config hashes must
remain recorded.

### 10.3 Comet adapter

Do not branch Comet metrics on artifact profile. Both profiles pass the same normalized
aggregate and information rows into `MasterMonitor`/`CometMetricSink`.

A spy-sink integration test must compare names, values, steps, asset names, and image names
between the two profiles.

---

## 11. Existing-run compaction

Add an idempotent post-hoc command using the same compactors and validators as the live
orchestrator, for example:

```bash
python -m mas_cc.cli.main experiment compact \
  --run-dir <existing-run> \
  --profile results_only
```

Requirements:

- default to non-destructive preview or require an explicit `--delete-raw` for the first
  release;
- resolve and display the exact run directory before deletion;
- create and validate compact outputs before removing anything;
- refuse partial/corrupt cells unless explicitly limited to sealed completed episodes;
- never traverse outside the resolved run directory;
- be safe to rerun after interruption;
- print before/after byte and file counts;
- optionally create one copy-friendly archive excluding `:Zone.Identifier` sidecars;
- never modify historical runs automatically merely because a new version is installed.

This command is necessary to recover space from already completed paid experiments without
rerunning model calls.

---

## 12. Code map

Expected primary changes:

| Area | File(s) | Responsibility |
|---|---|---|
| Config model/schema | `src/mas_cc/config/models.py`, `loader.py`, `schema.py` | artifact profile, checkpoint mode, prompt scope validation |
| Orchestrator | `src/mas_cc/experiments/orchestrator.py` | shard discovery, scheduling, resume, cell sealing, finalization |
| Recorder | `src/mas_cc/observability/recorder.py` | profile-aware sinks; stop verbose local writes in results-only |
| Storage | `src/mas_cc/storage/` | atomic compact shard, state, manifest, hashes, cleanup |
| Budget | `src/mas_cc/llm_runtime/providers/budget.py` and orchestration adapter | compact durable budget state and restart reconciliation |
| Aggregation | `src/mas_cc/experiments/aggregation.py`, `src/mas_cc/metrics/aggregate.py` | aggregate normalized compact episodes |
| Imitation analysis | `src/mas_cc/games/hidden_bench/imitation/analysis.py` | read compact records; profile-aware output allowlist |
| Prompt examples | orchestrator/observer and prompt reporting modules | deterministic cell-scoped bounded sample |
| CLI | `src/mas_cc/cli/experiment.py`, `main.py` | existing-run compaction command |
| Documentation | config reference and launch guides | exact semantics and operational examples |

Keep the game dynamics and prompt construction code unchanged unless exposing a typed
compact transition view requires a small adapter.

---

## 13. Test-driven implementation sequence

### Phase 1 — Freeze contracts with failing tests

Add tests before implementation for:

1. Parsing/exporting `artifact_profile` and `checkpoint_mode`.
2. Backward aliases for `checkpoints: true/false` and rejection when both APIs appear.
3. Prompt `scope: cell` validation.
4. Exact compact schema from a known HiddenBench imitation transition.
5. Compact-to-analysis adaptation equality with the existing trajectory adapter.
6. Results-only final file allowlist.

Use a small deterministic mock-provider run. Do not use paid providers in tests.

### Phase 2 — Compact record and atomic shard writer

Implement:

- typed compact row/terminal models;
- Parquet serialization and schema versioning;
- read-back validation;
- atomic temporary rename;
- deterministic shard paths;
- corruption detection;
- cleanup restricted to a validated cell path.

Tests must inject failures before and after rename and prove source data are not lost.

### Phase 3 — Episode-boundary scheduling and resume

Implement shard discovery before task construction and schedule only missing episode IDs.

Test:

- completed episodes are skipped;
- incomplete temporary shards restart;
- seeds are unchanged;
- ten simulated in-flight tasks may restart while earlier completions do not;
- incompatible config/prompt/schema hashes fail before provider construction;
- a sealed cell is skipped as a unit;
- `wipe_and_recompute` remains an explicit override and never happens implicitly.

### Phase 4 — Minimal recorder and prompt sampling

Implement profile-aware recorder sinks. In results-only mode, keep compact scientific rows,
concise errors, budget state, and bounded prompt candidates without creating verbose
episode trees.

Test that 100 mock episodes with prompt sample count two produce two rendered examples for
the cell, not 200 files.

### Phase 5 — Aggregation and CMI from compact data

Run identical deterministic episodes through `full` and `results_only`; compare all
aggregate curves and information rows using exact equality where deterministic and tight
floating tolerances where serialization crosses NumPy/Pandas types.

Test failed episode exclusion explicitly.

### Phase 6 — Comet equivalence

Use a fake master sink and compare the complete recorded call stream between profiles:

- metric keys and values;
- steps;
- parameters/tags;
- image names;
- asset names;
- close summary.

The test must also show that prompt/raw content is absent from both remote streams.

### Phase 7 — Budget restart safety

Test atomic budget replacement under concurrent updates and the two crash cases:

1. authoritative live spend available;
2. live spend unavailable with outstanding reservation ceilings.

The restored available budget must never exceed the safe pre-crash amount.

### Phase 8 — Cell/run sealing and cleanup

Test successful compaction, failed compaction, interrupted compaction, idempotent rerun,
single-cell promotion, and multi-cell merge.

Assert an upper bound on final file count and confirm the omitted verbose basenames are
absent.

### Phase 9 — Existing-run compactor and operator documentation

Exercise the command against a copied fixture of the current legacy layout. Verify dry-run,
destructive opt-in, path containment, before/after counts, and analysis equivalence.

Document launch, interruption, resume, compaction, and expected final layouts.

---

## 14. Required tests and acceptance cases

### Configuration

- `full` is the default for old configs.
- Unknown artifact profiles and checkpoint modes fail with precise paths.
- The resolved config records the effective profile/mode/scope.
- No credential fields enter checkpoint or compact artifacts.

### Resume

- A run interrupted with 7 of 100 episodes complete resumes with exactly 93 scheduled.
- A run interrupted with 10 episodes active does not accept their `.tmp` files as complete.
- A completed episode is not sent to the provider twice.
- A complete sealed cell is not reopened merely to rebuild verbose artifacts.
- A changed seed/config/prompt definition is rejected.

### Scientific outputs

- Full and results-only produce identical CMI estimates and diagnostics.
- Full and results-only produce identical cell curve points and percentile bands.
- Initial order-parameter state is retained.
- Failed/partial episodes are excluded and visibly counted.
- Compact rows retain episode grouping for episode bootstrap and within-episode nulls.

### Comet

- `logging.comet: true` behaves identically across profiles.
- Results-only cleanup proceeds safely if Comet is unavailable.
- Compact data can replay the aggregate/information reporting path.
- Remote payloads remain aggregate-only and secret/prompt free.

### Local artifacts

- Two prompt examples per cell produce one bounded Markdown artifact.
- No per-attempt operational histories survive finalization.
- No raw trajectory survives a successfully sealed results-only cell.
- A failed validation leaves recoverable source/shards and no completion seal.
- Final manifests hash every retained scientific/result artifact.
- A sealed 100-episode cell retains no more than 30 files, excluding an explicitly
  requested archive and filesystem metadata not produced by the experiment.

### Budget

- Committed usage survives restart.
- Dead reservations are never restored as active.
- Unknown outstanding usage is conservatively charged.
- Resume cannot increase the remaining approved budget.

### Post-hoc compaction

- Dry-run mutates nothing.
- Destructive mode removes only allowlisted raw paths inside the selected run.
- Repeating the command is idempotent.
- Existing numerical results remain unchanged.

---

## 15. Definition of done

This feature is complete only when all of the following hold:

1. A run config can independently enable Comet, select `results_only`, and select
   episode-boundary checkpointing.
2. Interrupting a concurrent cell causes only incomplete episodes to rerun.
3. A completed results-only cell contains a compact scientific table, final summaries,
   CMI results, order-parameter plots, a bounded prompt sample, and small manifests.
4. The cell does not retain verbose per-episode operational histories after sealing.
5. Full and results-only modes produce equivalent scientific and Comet outputs.
6. Budget restoration is monotone and conservative.
7. Compaction and deletion are atomic, validated, path-contained, and idempotent.
8. Existing full runs can be compacted explicitly without new model calls.
9. Documentation explains that episode mode restarts incomplete episodes rather than
   resuming their last round.
10. The target HiddenBench configuration is updated to use the new profile only after the
    mock/integration equivalence tests pass.

---

## 16. Non-goals

- Do not reduce or remove existing Comet reporting.
- Do not send raw prompts or responses to Comet.
- Do not implement true mid-episode restoration in this task.
- Do not change HiddenBench imitation dynamics, controller semantics, estimators, or
  bootstrap/null definitions.
- Do not delete historical result directories automatically.
- Do not make successful local execution depend on Comet availability.
- Do not use compression alone as the solution; the file count and redundant schemas must
  also be reduced.
- Do not silently treat a partial episode as completed or forward-fill it into aggregate
  results.

The intended failure bound is simple: with episode-boundary checkpoints and concurrency
ten, an abrupt interruption may repeat the in-flight episodes, but it must preserve every
durably completed episode, every sealed cell, the approved budget boundary, and all
scientific outputs already finalized.
