# TDD: Graceful study drain and resumable safe points

**Date:** 2026-09-24
**Status:** proposed implementation plan; no authorization for provider calls or
SLURM submission
**Scope:** generic MA-CC experiment orchestration, generic Potsdam launchers,
checkpoint ensembles, and relational blackboard episodes used by the current
ICLR experiment folders

Related contracts:

- `docs/tdd/features/orchestrator/12082026_results_only_episode_resume_plan.md`
- `docs/tdd/features/orchestrator/22082026_TDD_standardized_study_submission_and_aggregation.md`
- `docs/tdd/features/games/16092026_BLACKBOARD_CHECKPOINT_EXPERIMENT_IMPLEMENTATION_PLAN.md`
- `docs/tdd/features/orchestrator/04092026_TDD_bounded_comet_shutdown_after_scientific_seal.md`

## 1. Problem statement

Remote-provider studies may need more than one scheduler allocation even when
their scientific state is fully resumable. Today, a Potsdam allocation that
reaches its wall-time is terminated abruptly. Completed episodes, parent
checkpoints, and sealed continuation branches survive, but work that is still
in flight may have to be repeated.

The immediate example is
`blackboard_checkpoint_ensemble_02`. Its first 24-hour allocation preserved
parent checkpoints and completed branch seals, but every active unsealed
continuation was exposed to abrupt termination. The same operational concern
applies to the ordinary 15- and 30-round relational episodes under
`configs/runs/relational_reasoning/blackboard_game/iclr_experiments/`.

Operators need to be able to:

1. request a graceful stop while an allocation is running;
2. stop the creation of new work immediately;
3. allow active work to reach the nearest supported durable boundary;
4. exit before the scheduler's hard wall-time;
5. resubmit the unchanged study to the same result root;
6. preserve completed scientific work without duplicate observations or
   repeated provider calls;
7. use ordinary 24-hour allocations when those are easier to schedule.

This feature is called **draining**. Draining is different from cancellation:
it is a cooperative transition to a resumable state, not an immediate process
kill.

## 2. Goals

### 2.1 Primary goals

- Add one generic, study-level drain protocol shared by local execution and the
  generic SLURM launchers.
- Add `mas-cc study drain` for a manual operator request.
- Allow SLURM to request the same drain automatically before wall-time.
- Stop scheduling new episodes once a drain is requested.
- Let each runtime stop at its strongest supported safe point.
- Record an explicit durable drain outcome that monitoring can distinguish
  from scientific completion, provider failure, and abrupt cancellation.
- Resume through the existing `mas-cc study submit` command and unchanged
  output roots.
- Preserve old parent checkpoints, branch seals, episode shards, and failure
  recovery checkpoints.

### 2.2 Supported safe-point levels

| Capability | Boundary | Intended coverage |
|---|---|---|
| `episode` | after a completed episode seal | every MA-CC game |
| `branch` | after a continuation branch seal | checkpoint parent bundles |
| `round_replay` | after a population-round recovery checkpoint | relational imitation/blackboard runtime |

The orchestrator must negotiate the strongest capability exposed by the
runtime. A game that implements only episode-level safety remains valid; it
must not be described as supporting round resume.

### 2.3 Current ICLR coverage

At the time of this plan, the experiment YAMLs under
`blackboard_game/iclr_experiments` all use
`relational_imitation_round_feedback` with ordinary, non-ensemble episodes.
They therefore require `round_replay` support for useful draining when all
configured repetitions start concurrently. Implementing the relational
adapter once should cover all current ICLR configurations without changing
their scientific design.

## 3. Non-goals

- Do not modify game dynamics, prompts, seeds, policies, budgets, estimators,
  bootstrap procedures, or analysis inclusion rules.
- Do not increase provider concurrency or scheduler throttle.
- Do not create a study-specific SLURM script.
- Do not run two workers concurrently against the same episode or branch
  output directory.
- Do not treat a partial episode, parent, or branch as scientifically complete.
- Do not make every game support direct round-state restoration in the first
  implementation.
- Do not automatically submit another paid allocation. Automatic resubmission
  is a separate policy and authorization concern.
- Do not delete incomplete artifacts during drain or resume.
- Do not use a stale drain request from an earlier job to stop a later
  submission.

## 4. Safety and scientific invariants

1. A drain changes execution timing only. Scientific identities and outcomes
   must be invariant to whether a run is uninterrupted or drained/resumed.
2. Completed episode shards, branch seals, parent checkpoints, and cell seals
   remain authoritative.
3. No completion seal is written until its existing completeness validation
   passes.
4. A drain marker is an operational artifact and is excluded from scientific
   hashes and analysis packages.
5. A drain request is scoped to one study root, submission attempt, SLURM job,
   and, where applicable, array task.
6. Writes use temporary file, flush, and atomic replace semantics.
7. Resume must reject incompatible task, seed, prompt, model, checkpoint,
   schema, or scientific protocol identities exactly as it does today.
8. Previously committed requests, tokens, and accounting units remain
   monotone across drain/resume.
9. Provider credentials, raw responses, and secrets never enter drain state.
10. Scheduler completion is not scientific completion. Monitoring continues to
    verify scientific seals after a drained job exits successfully.

## 5. Operator-facing contract

### 5.1 Manual drain

Add:

```bash
mas-cc study drain \
  --study-dir <study-result-root> \
  --job-id <active-job-id> \
  --reason manual
```

The command must:

1. resolve and print the exact study root;
2. verify that the requested job belongs to that study and is currently
   eligible to drain;
3. create one atomic, job-scoped drain request;
4. never send provider calls;
5. optionally wait and report per-shard acknowledgements when `--wait` is
   supplied;
6. never call `scancel` implicitly.

The operator may issue `scancel` only after every live shard reports `drained`
or when accepting the ordinary abrupt-interruption fallback.

### 5.2 Status

Extend study status/monitoring output with:

```text
drain requested at
request reason
target job/submission attempt
shards running
shards draining
shards drained
active episodes
active branches
last durable safe-point time
hard wall-time remaining
```

The status must clearly distinguish:

- `scientifically_complete`;
- `running`;
- `drain_requested`;
- `draining`;
- `drained_incomplete`;
- `failed`;
- `scheduler_timeout`.

### 5.3 Resume

Resume remains the normal command:

```bash
mas-cc study submit --config-dir <same-config-folder>
```

It must reuse the existing result root and skip all valid seals. A new
submission creates a new drain namespace and ignores requests from prior jobs.

## 6. Drain state and stale-request protection

Store operational state beneath:

```text
<study-root>/runtime/drain/job-<job-id>/
  request.json
  state.json
  shards/
    <array-index>.json
```

`request.json` contains only:

```text
schema_version
study_id
study_manifest_hash
submission_attempt
job_id
requested_at
reason: manual | walltime_signal | operator_signal
requested_by: cli | launcher
```

Each shard acknowledgement records its array index, process identity, state,
active work counts, last durable boundary, and timestamp. It must not contain
credentials, prompts, or responses.

Every write is atomic. A worker accepts a request only when all identity fields
match its current submission environment. A new submission never reuses a
prior job's drain directory.

The state machine is:

```text
running
  -> drain_requested
  -> draining
  -> drained_incomplete

running
  -> scientifically_complete
```

`drained_incomplete` is a successful operational shutdown, not a scientific
completion. It must not create a cell or study completion seal.

## 7. Generic orchestration behavior

### 7.1 Stop scheduling queued episodes

The experiment scheduler currently may create coroutine tasks before they
acquire an execution-parallelism slot. Drain support must distinguish queued
from started work.

After a matching drain request:

- do not acquire a slot for another episode;
- cancel only tasks that have not begun scientific execution;
- do not cancel an active provider request;
- let active episodes use their runtime-specific safe point;
- collect explicit `drained` outcomes rather than treating them as ordinary
  failures;
- close provider clients and observability sinks through bounded cleanup.

### 7.2 Typed drain control

Add a process-local `DrainController` shared by the study worker,
orchestrator, observers, and capable game runtimes. It exposes:

```text
requested
reason
requested_at
deadline
strongest_supported_safe_point
acknowledge(state, metadata)
raise_if_safe(boundary)
```

Polling the shared request file must be bounded and cheap. Signal handlers may
set an in-memory event but must not perform JSON serialization, locking, or
other unsafe work directly.

### 7.3 Outcome model

Add an explicit resumable outcome such as `drained` or
`interrupted_drained`. Do not overload `failed`, `completed`, or
`skipped_resumed`.

On a drained outcome:

- finalize recoverable operational metadata;
- do not publish a completed episode artifact;
- do not count the episode in scientific aggregation;
- retain any valid recovery checkpoint;
- return control to the shard supervisor;
- exit zero once all active work has drained successfully.

A zero scheduler exit is acceptable because scientific completion remains
seal-based. The shard's drain state must make the incomplete result explicit.

## 8. Safe point implementations

### 8.1 Universal episode boundary

All games receive this behavior:

1. completed episodes seal normally;
2. once drain is requested, no new episodes start;
3. active episodes that do not expose a finer boundary may finish normally;
4. if they cannot finish before hard wall-time, the existing abrupt-resume
   semantics apply.

This is useful when repetitions exceed parallelism. It is insufficient when
all long episodes start at once.

### 8.2 Checkpoint ensemble branch boundary

Extend `ParentBundleWorker` with drain awareness:

1. never interrupt validation or atomic publication of a branch seal;
2. check drain state before starting preparation, before starting each branch,
   and immediately after each branch seal;
3. after a request, finish the active branch, publish its valid seal, and stop
   before the next branch;
4. leave `parent_bundle_seal.json` absent unless every expected branch is
   complete;
5. on resume, validate and skip every existing sibling seal exactly as today.

If a parent has not reached its immutable preparation checkpoint, the
relational round-safe mechanism may drain it after a round. Otherwise the
two-round preparation may be allowed to finish and seal the parent.

Existing checkpoint-study artifacts must remain readable without migration.

### 8.3 Relational population-round replay

The ordinary ICLR configurations require a finer safe point. Use the existing
relational recovery ledger as the first implementation rather than inventing
direct state restoration immediately.

At the end of a fully completed population round, after trajectory and round
boundary recording:

1. detect the drain request;
2. atomically record validated agent decisions, controller choices, identity,
   budget state, and the completed round boundary;
3. record interruption type `graceful_drain`;
4. close the current episode as resumably interrupted, not complete;
5. exit through the typed drained outcome.

On resume:

1. reconstruct from the original episode seed;
2. replay recorded validated decisions and controller choices without provider
   calls;
3. rebuild state through the saved round boundary;
4. continue provider execution at the first unfinished decision;
5. write canonical output through a fresh/staged path so replay cannot append
   duplicate scientific rows;
6. remove the recovery checkpoint only after the episode completes and seals.

The recovery artifact must bind the run ID, cell ID, episode ID, episode seed,
resolved scientific identity, prompt hashes, task hash, game/runtime schema,
and replay version.

### 8.4 Later direct round restoration

Direct restoration from serialized game and RNG state may be added later if
replay time becomes material. It requires a dedicated versioned contract for:

- game state;
- all random generators and derived streams;
- controller cooldown/report-selection state;
- evidence and blackboard history;
- prompt/runtime state;
- trajectory writer append position;
- usage and budget state.

This larger optimization is not required for the initial drain feature.

## 9. SLURM integration

### 9.1 Early signal

Add an execution-only study setting, with a conservative Potsdam default for
long remote-provider jobs:

```yaml
execution:
  graceful_drain:
    enabled: true
    signal: USR1
    lead_time: "04:00:00"
```

The execution planner converts this to the generic `sbatch` signal option. The
launcher receives the signal, writes the same job-scoped request used by the
CLI, and lets the child worker drain. Do not put signal handling into a
study-specific job file.

For a 24-hour allocation and four-hour lead time:

```text
hours 0-20: normal execution
hours 20-24: drain window
```

The lead time must be reported in preflight and must be shorter than the
requested wall-time. Disabling automatic drain requires an explicit resolved
value.

### 9.2 Launcher supervision

The generic launcher must:

- forward ordinary termination semantics correctly;
- translate the configured early signal into a drain request;
- keep the Python child alive during the drain window;
- report drain progress in SLURM output;
- preserve the child's error status for genuine failures;
- exit successfully only after the worker acknowledges a clean drain or
  scientific completion;
- allow the scheduler hard limit to remain the final backstop.

### 9.3 Allocation strategy

This feature permits a long study to use repeated 24-hour allocations without
discarding durable work. It does not guarantee that one allocation completes
the study and does not authorize automatic resubmission.

On Potsdam, longer allocations remain legal subject to partition/QoS limits,
but operators may choose 24 hours to improve scheduling convenience. The
science and resume identity must be identical under either choice.

## 10. Provider and budget behavior

- A drain request does not cancel an HTTP request already in flight.
- No new logical request starts after a runtime reaches its selected safe
  boundary.
- Provider coordinator leases are released during ordinary client cleanup;
  dead leases still expire under the existing lease timeout.
- The new submission uses a fresh job-specific adaptive coordinator.
- Committed requests/tokens/accounting units survive through the existing
  durable budget state.
- Replayed decisions do not reserve or charge provider budget again.
- A failure while draining remains a failure; it is not converted into a clean
  drain merely because a request exists.

## 11. Storage and aggregation behavior

- Drain artifacts live only under `runtime/` and are not scientific inputs.
- Partial/replayed trajectories must never be double-counted.
- Strict aggregation continues to require every expected episode, parent, and
  branch seal.
- Partial aggregation, when explicitly requested, may report drain diagnostics
  as censored interruptions but must exclude them from completed-episode
  estimators.
- Successful resume supersedes its earlier interruption record.
- Final analysis ZIPs exclude runtime drain control files.

## 12. Configuration identity

`execution.graceful_drain` is operational and must be excluded from the
scientific protocol fingerprint. Changing wall-time, signal, or drain lead
time must not change episode IDs, seeds, checkpoints, branch IDs, or analysis
groups.

The implementation must still record these values in execution provenance and
the generated execution plan. The submitted source config hash may change when
operational settings change, but compatibility checks must compare the
versioned scientific protocol identity for reuse rather than silently relaxing
scientific fields.

## 13. Code map

| Area | Expected files | Responsibility |
|---|---|---|
| Drain state | `src/mas_cc/studies/drain.py` | schemas, atomic request/state/acknowledgement IO, stale request validation |
| Study CLI | `src/mas_cc/cli/main.py`, study command module | `study drain`, optional wait/status output |
| Planning/submission | `src/mas_cc/studies/execution.py`, `submission.py` | validate/report lead time and pass generic signal option |
| Experiment orchestration | `src/mas_cc/experiments/orchestrator.py` | stop queued episodes, collect typed drained outcomes, bounded cleanup |
| Recorder | `src/mas_cc/observability/recorder.py` | resumable drain finalization without scientific completion |
| Relational runtime | `src/mas_cc/games/relational_reasoning/imitation_round_feedback/runtime.py` | population-round drain checkpoint and decision replay |
| Checkpoint bundles | `src/mas_cc/games/relational_reasoning/imitation_round_feedback/checkpoint.py` | branch-boundary checks and incomplete bundle outcome |
| Generic launchers | `scripts/Potsdam/SLURM/run_config_array.job`, `run_study_cell_array.job` | early signal trap and child supervision |
| Monitoring | study status and dashboard adapters | drain phase, safe-point age, active work, wall-time remaining |
| Tests | `tests/mas_cc/` | state, signals, resume equivalence, launchers, aggregation exclusion |

Do not add a checkpoint-study-specific launcher.

## 14. Test-driven implementation sequence

### Phase 1 — freeze the generic contract

Write failing tests for:

1. atomic request and acknowledgement files;
2. request identity and stale job rejection;
3. the drain state machine;
4. typed drained outcomes;
5. queued episodes never starting after a request;
6. active episode completion under episode-only capability;
7. no cell seal for a drained incomplete cell.

Use only mock providers.

### Phase 2 — generic CLI and manual drain

Implement `mas-cc study drain` and status rendering. Test exact path
resolution, mismatched jobs, already completed jobs, idempotent repeated
requests, and `--wait` timeout behavior.

The command must never mutate scientific configuration or call a provider.

### Phase 3 — checkpoint branch draining

Add drain checks to `ParentBundleWorker` and test:

1. a request before the first branch starts no branch;
2. a request during a branch lets that branch seal;
3. no subsequent branch starts;
4. no incomplete parent bundle seal is written;
5. resubmission skips the sealed branch and finishes the remaining siblings;
6. final bundle hashes match an uninterrupted deterministic run.

### Phase 4 — relational round replay

Add the round-boundary recovery path and test:

1. drain after every possible round of a deterministic short episode;
2. resume without repeating provider calls for saved decisions;
3. identical final state and scientific trajectory versus uninterrupted
   execution;
4. identical controller communication choices and schedules;
5. identical episode/cell scientific hashes where the hash contract is meant
   to be execution-invariant;
6. no duplicate round or micro-slot rows;
7. recovery artifact rejection after seed, task, prompt, model, or protocol
   changes;
8. correct removal of recovery state after completion.

### Phase 5 — concurrency and budget safety

Run mock studies with multiple active episodes and injected drain timing.
Assert:

- no task begins after the request is observed;
- every completed boundary is represented once;
- provider call counts equal uninterrupted execution;
- replayed decisions add zero provider calls;
- committed usage never decreases;
- coordinator leases return to zero or expire normally;
- genuine provider/validation errors remain visible.

### Phase 6 — generic SLURM signal path

Test the generic launcher with a local/fake scheduler harness, then an
explicitly authorized tiny mock-provider SLURM smoke:

1. early `USR1` creates the correctly scoped request;
2. the shell does not kill the child during drain;
3. all shards acknowledge;
4. drained shards exit zero and remain scientifically incomplete;
5. a later submission ignores the old request and resumes;
6. a worker that exceeds the drain window is still terminated by scheduler
   wall-time and remains recoverable under ordinary abrupt semantics.

### Phase 7 — ICLR fixture coverage

Use downscaled copies or in-memory overrides of representative current ICLR
configs:

- one 15-round, parallelism-5 configuration;
- one 30-round, parallelism-30 configuration;
- one 30-round, parallelism-60 configuration.

Drain and resume with mock providers. Compare uninterrupted and resumed
canonical outputs and configured analysis inputs. Do not alter or launch the
real ICLR studies for this gate.

### Phase 8 — documentation and rollout

Document:

- manual drain procedure;
- automatic pre-walltime drain;
- how to wait for acknowledgements;
- when `scancel` is safe;
- normal same-root resubmission;
- capability differences between games;
- recovery after a hard timeout during the drain window.

Enable the feature by default for paid Potsdam studies only after the mock and
tiny scheduler gates pass.

## 15. Required failure-injection tests

- Process dies before writing a drain request.
- Process dies while atomically replacing a request or acknowledgement.
- Signal arrives during an HTTP call.
- Signal arrives while writing a branch seal.
- Signal arrives while writing an episode completion shard.
- Signal arrives during parent preparation.
- A second signal arrives during draining.
- CLI drain is requested after the job already ended.
- Old request exists when a new job starts.
- One array task drains while another finishes scientifically.
- One array task fails while others drain cleanly.
- Hard wall-time arrives before a non-capable game finishes its active episode.
- Recovery checkpoint exists but its identity or content hash is invalid.
- Resume completes after two or more drain cycles.

No injected failure may convert incomplete work into a valid scientific seal.

## 16. Acceptance criteria

The feature is complete only when:

1. An operator can request a drain with one study-level command.
2. The generic launchers request a drain at the configured time before
   wall-time.
3. No new episode or branch starts after the relevant request is observed.
4. Checkpoint bundles stop after sealing their active branch and resume without
   recomputing completed siblings.
5. Current relational ICLR episodes can stop after a complete population round
   and resume without repeating saved provider decisions.
6. Uninterrupted and drained/resumed mock executions are scientifically
   equivalent.
7. Drain cycles do not duplicate canonical rows or budget charges.
8. Stale requests cannot stop a new submission.
9. Strict aggregation still refuses incomplete studies.
10. The existing `blackboard_checkpoint_ensemble_02` artifacts resume without
    migration or deletion.
11. Monitoring visibly distinguishes scheduler success, clean drain, and
    scientific completion.
12. No study-specific SLURM file is added.

## 17. Operational example after implementation

For a 24-hour checkpoint-study allocation:

```text
submit unchanged study to the existing result root
  -> run normally for up to 20 hours
  -> manual request or automatic USR1 begins drain
  -> active branches finish and seal
  -> shards record drained_incomplete and exit
  -> inspect seals and errors
  -> resubmit the same study when authorized
  -> repeat until every scientific seal exists
  -> run strict aggregation
```

For an ordinary current ICLR relational study:

```text
drain requested
  -> stop queued episodes
  -> each active episode completes its current population round
  -> persist recovery ledger and drained outcome
  -> exit
  -> resubmit unchanged study
  -> replay saved decisions locally without provider calls
  -> continue at the first unfinished decision
```

## 18. Authorization boundary

Implementing and testing this feature with deterministic/mock providers does
not authorize a real provider call, SLURM submission, cancellation, or
resubmission. Production use still requires the normal import checks,
credential-free preflight, execution-plan review, valid output paths, and
explicit run authorization.
