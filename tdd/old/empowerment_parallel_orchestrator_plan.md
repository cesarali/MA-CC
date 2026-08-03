# TDD Plan: Single-Node and Slurm Empowerment Experiment Orchestrator

## Objective

Provide two execution backends over one manifest, task runner, checkpoint format, and
result hierarchy:

1. **Single-node backend (default)**: run several independent experiments as bounded
   asynchronous tasks in one Python process on one workstation or one allocated compute
   node. Each experiment may run several episodes concurrently. One configurable
   `max_parallel_api_calls` limit must cap aggregate API calls across all experiments in
   that process.
2. **Slurm backend (optional)**: submit independent experiment entries as a Slurm job
   array when separate allocations, queue-level retries, or stronger process isolation
   are useful.

The single-node backend is the expected normal path for this workload. The simulation
is primarily network/API bound, and roughly 20 CPUs on one node are sufficient for the
small amount of local scheduling, validation, Parquet writing, compaction, and analysis.
More Slurm jobs do not automatically make the provider answer faster and may make the
aggregate request rate harder to control.

Both backends provide two scientific-work parallelism levels:

- **Experiment parallelism**: independent manifest entries may run concurrently.
- **Episode parallelism**: independent episodes may run concurrently inside an
  experiment.

Do not parallelize pair interactions within one episode. `NamingConventionGame`
updates agent memory and scores after every pair, so those interactions are causally
ordered. The two LLM decisions belonging to one pair should remain concurrent as
they are now.

The design must provide failure isolation, resumability, and separate logs for every
experiment in both backends. A failed experiment task in the single-node event loop must
not cancel sibling tasks; a failed Slurm array task must not cancel sibling array tasks.

Do not introduce MPI, Dask, Ray, multi-node state sharing, or a distributed filesystem
protocol for the first implementation. They do not address the limiting resource.

---

## Current behavior and constraint

The current pilot uses:

```yaml
request_concurrency: 20
episode_concurrency: 2
```

`run_experiment()` already limits active episodes with an `asyncio.Semaphore`, but
the value can only be selected through the experiment YAML. Each active episode
runs one pair at a time, while the two decisions in that pair are requested with
`asyncio.gather`.

If several clients or processes use the same provider account, their client-side
semaphores do not normally coordinate. The orchestrator must consequently expose and
document an aggregate concurrency budget.

For a standalone experiment, approximate active requests are:

```text
requests ~= min(request_concurrency, 2 * episode_concurrency)
```

For the single-node backend, wrap each experiment client with a backend-owned shared
API-call limiter. This turns `max_parallel_api_calls` into a hard client-side cap for
all experiments using that provider in the process:

```text
single-node requests <= min(
  max_parallel_api_calls,
  sum(min(experiment_request_concurrency, 2 * experiment_episode_concurrency))
)
```

Keep a separate underlying client per experiment so its request statistics and lifecycle
remain attributable to that experiment. The lightweight wrapper owns no conversation
state. It acquires a per-experiment request permit first, then one shared API-call permit,
and delegates `complete()`; all orchestrated access must pass through the wrapper. This
ordering prevents one experiment from occupying the entire shared limit while waiting
for its own smaller limit. Internally this limit can be implemented with an
`asyncio.Semaphore`: a value of 10 means calls 1–10 run concurrently and call 11 waits
asynchronously until one finishes. It does **not** mean sequential execution. If a
client supports an additional capability such as
constrained completion, the wrapper must preserve that capability rather than silently
removing it.

Do not calculate `max_parallel_api_calls` from CPU count or the number of worker
processes. API calls spend most of their time waiting on the network, so one Python
process can manage many concurrent requests. Choose the value from the provider's known
capacity and observed throttling. A conservative starting value is 10; increase it only
after inspecting HTTP 429 responses, retry counts, latency, and provider guidance.

With `max_parallel_api_calls: 10`, three experiments that can each produce four calls
may still run together, but only ten calls are in flight and the other two wait. Because
one ordinary episode issues at most two simultaneous decisions, approximately five
active unforced episodes can fill a ten-call limit.

For Slurm, separate processes cannot share an in-memory semaphore. The array concurrency
cap and per-task request limits only provide an estimate:

```text
Slurm aggregate requests ~= sum(
  min(task_request_concurrency, 2 * task_episode_concurrency)
  for each active array task
)
```

Committee-forced decisions make no LLM request, retries add traffic, and provider-side
rate limits remain authoritative. CPU allocation must never be used as a proxy for the
safe API request rate.

---

## Proposed user interface

### One experiment

Allow CLI overrides without copying a YAML file merely to tune execution:

```bash
naming-game experiment \
  --config configs/empowerment_pilot_university.yaml \
  --output-dir /work/.../results/pilot-c4 \
  --episode-concurrency 4 \
  --request-concurrency 8 \
  --run-id pilot-c4
```

The YAML remains the scientific configuration. CLI concurrency overrides are
execution settings and must be recorded in run metadata.

### Several experiments: one manifest, two backends

Introduce a backend-neutral manifest. Execution defaults and backend-specific limits
are operational metadata; they do not alter scientific fingerprints or episode IDs.
For example:

```yaml
schema_version: 1
campaign_id: empowerment-parallelism-sweep

defaults:
  episode_concurrency: 2
  request_concurrency: 4

execution:
  provider_limits:
    university:
      max_parallel_api_calls: 10
    openai:
      max_parallel_api_calls: 10
  single_node:
    max_active_experiments: 3
    analysis_concurrency: 1
  slurm:
    array_concurrency: 2

experiments:
  - id: pilot-c1
    config: configs/empowerment_pilot_university.yaml
    episode_concurrency: 1
  - id: pilot-c2
    config: configs/empowerment_pilot_university.yaml
    episode_concurrency: 2
  - id: pilot-c4
    config: configs/empowerment_pilot_university.yaml
    episode_concurrency: 4
```

Provider API-call limits are keyed by the actual provider/account boundary used by the
repository. Experiments using distinct credentials or endpoints must not accidentally
share a limit key, while experiments sharing one account must not evade the shared cap
by using different models.

Add one orchestration command with a safe dry-run for both modes.

#### Option A: single-node execution (default)

```bash
# Inspect paths, task order, and request-budget calculations only.
naming-game orchestrate-experiments \
  --backend single-node \
  --manifest configs/empowerment_parallelism_sweep.yaml \
  --results-root /work/.../results/parallelism-sweep \
  --logs-root /work/.../logs/parallelism-sweep \
  --dry-run

# Execute all entries on the current workstation or allocated compute node.
naming-game orchestrate-experiments \
  --backend single-node \
  --manifest configs/empowerment_parallelism_sweep.yaml \
  --results-root /work/.../results/parallelism-sweep \
  --logs-root /work/.../logs/parallelism-sweep \
  --max-active-experiments 3
```

Omitting `--backend` should select `single-node`. This mode must not call `sbatch` or
spawn one background process per experiment. It should schedule bounded async experiment
tasks in one process, use a distinct client/statistics object per experiment, and wrap
those clients with the shared API-call limiter described above. Use
`asyncio.gather(..., return_exceptions=True)` or equivalent supervised-task handling so
one failure is recorded without cancelling siblings.

Long live-provider campaigns should run on a workstation intended for them or inside
one ordinary Slurm allocation, not on a cluster login node. That is still the
single-node backend: Slurm provides one allocation, while the Python orchestrator owns
all campaign parallelism inside it.

#### Option B: Slurm array execution

```bash
# Generate and inspect the array mapping and sbatch command.
naming-game orchestrate-experiments \
  --backend slurm \
  --manifest configs/empowerment_parallelism_sweep.yaml \
  --results-root /work/.../results/parallelism-sweep \
  --logs-root /work/.../logs/parallelism-sweep \
  --array-concurrency 2 \
  --dry-run

# Submit only after inspecting the dry-run.
naming-game orchestrate-experiments \
  --backend slurm \
  --manifest configs/empowerment_parallelism_sweep.yaml \
  --results-root /work/.../results/parallelism-sweep \
  --logs-root /work/.../logs/parallelism-sweep \
  --array-concurrency 2 \
  --submit
```

The Slurm backend creates one array task per selected manifest entry and uses an array
cap such as `--array=0-2%2`. Actual submission requires `--submit`; choosing the backend
alone is not permission to submit jobs. Separate tasks provide scheduler accounting,
cancellation, and retry isolation, but they do not share a hard provider semaphore.

Both backends must resolve the same deterministic task indices, call the same core task
runner, and write the same scientific results and status schema. Backend selection must
never change seeds, episode IDs, phase-space paths, or scientific results.

---

## Output and logging contract

Every campaign gets a stable root. Within it, results must be partitioned first by
the scientific phase-space cell and then by replicated episode. A user must be able
to inspect one configuration without loading or filtering the campaign-wide
Parquet files.

The required conceptual hierarchy is:

```text
campaign
  -> phase-space configuration
       -> replicated episode
```

Use explicit `key=value` path components for scientific dimensions. For example:

```text
<results-root>/<campaign-id>/
  cells/
    regime=neutral/
      committee_size=0/
        committee_policy=always_A/
          initial_condition=empty/
            replicate=0/
              episode.parquet
              interactions.parquet
              episode_status.json
            replicate=1/
              episode.parquet
              interactions.parquet
              episode_status.json

    regime=consensus_attack/
      committee_size=4/
        committee_policy=alternative/
          initial_condition=consensus_A/
            pulse_rounds=none/
              replicate=0/
                ...

    regime=pulse/
      committee_size=8/
        committee_policy=alternative_pulse/
          initial_condition=consensus_A/
            pulse_rounds=5/
              replicate=0/
                ...
```

The exact partition keys may differ by regime, but the path builder must be
centralized and deterministic. It must include every field required to uniquely
identify a scientific cell, while `replicate` and `episode_id` identify repeated
observations within that cell. Missing/not-applicable dimensions must use one
canonical representation such as `none`; they must not sometimes be omitted and
sometimes included.

Each phase-space cell should also have a small inspectable summary after its
replicates finish:

```text
<cell-path>/
  cell_config.json
  cell_status.json
  episodes.parquet
  interactions.parquet
  replicate=0/...
  replicate=1/...
```

`cell_config.json` contains the scientific values defining that point in phase
space. `cell_status.json` reports expected, completed, failed, and running
replicates. Cell-level compacted files contain only that cell's data and must be
written atomically.

For backward compatibility and convenient whole-campaign analysis, optionally
produce aggregate views at the campaign root:

```text
<results-root>/<campaign-id>/
  episodes.parquet
  interactions.parquet
```

These aggregate files are derived indexes, not the only authoritative storage.
They may be rebuilt deterministically from the episode files. Analysis code should
accept either the campaign root or one phase-space cell root.

Experiment tasks must never concurrently rewrite campaign-wide aggregate files. In
single-node mode, the parent orchestrator may rebuild campaign aggregates once all
selected tasks reach a terminal state. In Slurm mode, use a separate idempotent
`finalize-campaign` step after the array (optionally a dependency job), or run it
manually. A failed or partial campaign may still expose completed cell files without a
misleading “complete” campaign aggregate.

In addition to the scientific hierarchy, every independently scheduled experiment
task gets stable, non-overlapping operational paths:

```text
<results-root>/<experiment-id>/
  experiment_config.json
  execution_config.json
  run_status.json
  interactions.parquet              # after compaction
  episodes.parquet                  # after compaction
  .episode_shards/...               # resumable checkpoints
  analysis/...

<logs-root>/<experiment-id>/
  task.log
  events.jsonl
```

`task.log` is the backend-neutral human-readable log. Single-node tasks must not attempt
per-async-task redirection of process-global `sys.stdout`/`sys.stderr`. The parent may
have a campaign console/log stream. Slurm scheduler stdout/stderr remain separate files
whose concrete paths are recorded in task metadata.

If one manifest entry represents a full campaign, `<experiment-id>` and
`<campaign-id>` are the same. If a manifest entry represents one phase-space cell,
it writes only to its precomputed cell path beneath the shared campaign root. The
orchestrator must reject manifests in which two tasks could write the same cell or
replicate path.

`execution_config.json` should include:

- experiment ID and manifest path;
- source config path and content fingerprint;
- execution backend (`single-node` or `slurm`);
- requested episode concurrency, per-experiment request concurrency, campaign
  experiment concurrency, and applicable `max_parallel_api_calls` limit;
- host/node, process ID, and timestamps;
- Slurm job ID, array job ID, and array task ID when applicable, using one canonical
  `null`/not-applicable representation otherwise;
- provider/model identity without credentials;
- repository commit when available;
- exact command line.

`run_status.json` must be written atomically and transition through:

```text
pending -> running -> completed
                   -> failed
                   -> interrupted
```

Include the exit code, error type/message when available, completed episode count,
total episode count, last update time, and paths to the task/event logs plus optional
Slurm stdout/stderr. Never store API keys, `.env` contents, prompts containing secrets,
or authorization headers.

Write structured `events.jsonl` records for experiment start/end, episode
start/end/failure, retry warnings, compaction, and analysis. Continue to retain the
human-readable progress logs.

The existing experiment runner attaches a file handler to the root logger. That is not
safe when several experiments run concurrently in one process because records can leak
into every experiment log. Replace it with an experiment-scoped logger/event sink (for
example, a logger adapter carrying `experiment_id`) and close/remove handlers when a
task finishes. Tests must prove that concurrent single-node tasks do not cross-write
events or human-readable logs.

For single-node execution, the parent may write one campaign log, but each experiment
must still have its own stable task/event files. Concurrent writers need an async lock or
one queue-backed writer per destination so every JSONL line remains intact.

For Slurm execution, route scheduler logs directly to unique files such as:

```text
#SBATCH --output=<logs-root>/%x/%A_%a.out
#SBATCH --error=<logs-root>/%x/%A_%a.err
```

The array task may additionally create stable `stdout.log` and `stderr.log` links
or record the concrete Slurm paths in `run_status.json`. Avoid having several
processes append to one file.

---

## Implementation slices (tests first)

### Slice 1: execution-setting overrides

Add failing tests before implementation:

- CLI overrides replace `episode_concurrency` and `request_concurrency` without
  changing the input YAML on disk.
- zero or negative values are rejected.
- effective settings are recorded in `execution_config.json`.
- the experiment fingerprint remains a fingerprint of scientific inputs; execution
  concurrency does not create scientifically distinct episode IDs.
- resuming the same output directory with different execution concurrency is
  allowed and recorded.

Then add `--episode-concurrency`, `--request-concurrency`, and `--run-id` to the
experiment CLI. Prefer `dataclasses.replace()` on the loaded frozen config rather
than mutating it.

### Slice 2: observable episode execution

Add tests using `MockAsyncLLMClient` with controlled latency:

- active episode count never exceeds the configured episode concurrency;
- pair interactions inside one episode remain sequential;
- the two decisions in one unforced pair can overlap;
- a completed episode writes its checkpoint before another episode failure is
  reported;
- episode start, completion, and failure events contain experiment and episode IDs;
- event writes from concurrent episodes remain valid JSONL records.

Add a small event-sink abstraction rather than scattering file writes throughout
the simulation. Logging failures should be visible but must not silently invalidate
completed checkpoints.

### Slice 3: manifest parsing and validation

Create a typed manifest model and tests for:

- supported execution backends and a `single-node` default;
- unique, filesystem-safe experiment IDs;
- required config files and normalized absolute paths;
- default inheritance and per-entry overrides;
- duplicate output-directory rejection;
- positive concurrency values;
- positive `max_parallel_api_calls` limits keyed by a known provider/account boundary;
- single-node hard-cap and Slurm aggregate-concurrency calculations;
- budget warnings/errors that identify the contributing experiment entries;
- deterministic array-index-to-experiment mapping;
- identical index/task/path mapping under both backends;
- deterministic, collision-free phase-space cell and replicate paths;
- rejection when two manifest entries resolve to the same cell/replicate path;
- rejection of unknown manifest keys to catch spelling mistakes.

Manifest validation and either backend's dry-run must not submit jobs, create provider
clients, or call an LLM.

### Slice 4: backend-neutral task runner

Implement a Python task-runner function that executes exactly one resolved manifest
entry. Both backends call this function. Also expose a CLI wrapper for Slurm and focused
manual retries:

```bash
naming-game run-experiment-task \
  --manifest <path> \
  --index "$SLURM_ARRAY_TASK_ID" \
  --results-root <path> \
  --logs-root <path>
```

Tests must verify:

- the index resolves to the expected experiment;
- direct resolved-entry calls and index-based calls select identical work;
- paths cannot escape the configured roots;
- status transitions are atomic and correct for success, exception, and signal;
- an existing completed run is skipped unless explicitly forced;
- an interrupted or failed run resumes existing episode shards by default;
- one experiment never reads or writes another experiment's directory;
- one phase-space cell never reads or writes another cell's directory;
- completed replicates are discoverable directly from the human-readable hierarchy,
  without scanning opaque hash filenames;
- task cleanup closes only its own client, logger, and event sink; and
- scientific outputs are independent of the selected backend.

### Slice 5: single-node orchestrator

Implement the default backend as a supervised asynchronous scheduler in one process.
Do not shell out to one child Python process per manifest entry.

Tests must verify:

- `single-node` is selected when `--backend` is omitted;
- `--dry-run` prints the complete mapping, paths, effective concurrency, provider
  API-call limits, and worst-case request estimates without constructing a client;
- active experiment tasks never exceed `max_active_experiments`;
- active API calls across experiments sharing a provider never exceed the shared
  `max_parallel_api_calls` limit;
- each experiment retains its own underlying client statistics;
- experiments using separate provider-limit keys do not block one another;
- one failed experiment is marked failed while siblings continue to completion;
- cancellation or SIGTERM stops launching work, lets atomic checkpoint writes finish
  where safe, and marks remaining active tasks interrupted;
- experiment logs and events never cross-write;
- concurrent tasks use one campaign-level progress display or disable their individual
  `tqdm` displays so terminal output is not corrupted;
- completed tasks are skipped and failed/interrupted tasks resume by default;
- analysis/compaction concurrency is bounded separately from API concurrency; and
- no `sbatch`, subprocess, or Slurm environment variable is required.

Synchronous Parquet compaction and analysis can briefly block the event loop. Move
measured blocking work to `asyncio.to_thread()` and bound it with a small
`analysis_concurrency` semaphore. Do not add a process pool before profiling shows that
CPU work is material. Available CPU cores are capacity for this housekeeping, not a
reason to multiply API requests.

### Slice 6: Slurm orchestrator

Generate or submit an array job whose task invokes `run-experiment-task`.

Tests should mock the subprocess boundary and assert:

- dry-run prints the complete index mapping, output paths, log paths, concurrency
  estimates, and `sbatch` command without submission;
- selecting `--backend slurm` without `--submit` does not submit;
- submission uses the requested array concurrency cap;
- paths and exported values are shell-safe;
- the returned Slurm job ID is recorded in an orchestration summary;
- submission failure leaves a clear failed orchestration record;
- generated tasks use the same manifest indices and output paths as single-node mode;
- no credentials are passed through command-line arguments or written to metadata.

Prefer passing a small set of non-secret paths/IDs to the task and letting the task
read the manifest. Use the existing environment setup in the Potsdam job script.

### Slice 7: failure inspection and retry

Add read-only status and selective retry commands:

```bash
naming-game experiment-status --results-root <root>
naming-game retry-experiments --results-root <root> --state failed
```

Tests must cover mixed completed/running/failed states, stale `running` states, and
retry selection. Retry uses the explicitly selected backend: single-node retry schedules
the selected entries in the current orchestrator, while Slurm retry submits only those
array indices. Both resume checkpoints and must never delete partial results
automatically.

---

## Parallelism and backend acceptance tests

Use the mock client to run identical scientific configurations with episode concurrency
1, 2, and 4. Run the campaign through the single-node backend, and test the equivalent
Slurm mapping/submission boundary with `sbatch` mocked.

Required assertions:

1. All runs produce the same episode IDs and deterministic scientific outputs.
2. All runs contain the expected number of episode and interaction rows.
3. No episode has overlapping pair interactions or reordered state updates.
4. Measured episode concurrency does not exceed 1, 2, and 4 respectively.
5. With artificial request latency, concurrency 2 and 4 reduce wall time relative
   to concurrency 1 within a generous, non-flaky bound.
6. Killing a task after several checkpoints and rerunning it completes only the
   missing episodes.
7. Three concurrent single-node experiment tasks produce three separate task logs,
   event logs, statuses, and result locations.
8. One deliberately failing single-node task does not stop the other tasks.
9. A shared mock provider records no more active requests than
   `max_parallel_api_calls`, even when the sum of per-experiment limits is higher.
10. Each experiment's client statistics remain separately attributable.
11. Single-node execution calls no subprocess or Slurm command.
12. The mocked Slurm backend maps the same entries to the same scientific output paths.
13. Three Slurm array tasks have separate scheduler log paths, and one failing task does
    not imply cancellation of its siblings.
14. Every episode is stored under the correct deterministic phase-space cell and
    replicate path.
15. Cell-level compacted files contain only their own cell, while rebuilt campaign
    aggregates contain all completed cells exactly once.
16. Analysis produces identical estimates when reading the campaign aggregate and when
    recursively reading the partitioned cell hierarchy.
17. Scientific episode IDs, row content, and analysis results are identical between
    backends for the same manifest and seeds.

After mock acceptance, run a small live-provider smoke sweep with the **single-node
backend** before changing the full pilot. Start conservatively and inspect
throttling/retry rates:

```text
active experiments = 1, episode concurrency = 1
active experiments = 1, episode concurrency = 2
active experiments = 1, episode concurrency = 4
```

Only then test multiple simultaneous experiments while the shared limiter keeps actual
client-side concurrency within `max_parallel_api_calls`. Use the Slurm backend later
only if there is an operational reason—such as queue time limits, independent
allocations, or isolated retries—and begin with array concurrency 1. Do not use multiple
jobs merely to consume available scheduler slots.

---

## Operational safeguards

- Preserve the existing stable episode IDs and atomic Parquet checkpoint writes.
- Resume by default; require an explicit flag for a clean non-resumed run.
- Default to the single-node backend; require explicit `--backend slurm --submit` for
  scheduler submission.
- In single-node mode, enforce `max_parallel_api_calls` with one shared semaphore per
  actual provider/account boundary across all experiment wrappers.
- In Slurm mode, display that no hard cross-job request semaphore exists and show the
  worst-case aggregate estimate before submission.
- Keep `max_active_experiments`, `episode_concurrency`, `request_concurrency`, and
  `max_parallel_api_calls` separate in configuration and metadata. Do not silently
  derive one from allocated CPU count.
- Never share an output directory between concurrent experiment tasks.
- Treat per-replicate episode files as the authoritative checkpoints; cell and
  campaign aggregates must always be safely rebuildable from them.
- Sanitize partition values and reject path separators, traversal components, and
  ambiguous encodings in phase-space keys.
- Refuse duplicate active experiment IDs when a valid live status/lock exists.
- Use an atomic lock containing job/task identity; provide a read-only stale-lock
  diagnostic before any manual lock removal.
- Apply bounded retry/backoff for provider throttling and expose retry counts in
  events/status.
- Keep analysis local to its experiment, run it only after successful history
  compaction, and bound simultaneous analyses separately from API work.
- Use scoped loggers/event sinks in the single process; never attach a per-experiment
  handler to the root logger without removing and filtering it.
- On single-node shutdown, stop admitting new experiments, preserve completed shards,
  mark active tasks interrupted, and close every created client exactly once.
- Do not infer provider capacity from allocated CPUs: these experiments are mostly
  network/API bound.
- Do not increase concurrency for the currently running pilot. Let it finish, then
  validate the new runner with mock and smoke configurations.

---

## Definition of done

- One backend-neutral manifest can execute locally on one node or through a Slurm array.
- Single-node is the documented and CLI default backend.
- Single-node mode uses one orchestrator process, bounded experiment/episode tasks, and
  a hard shared `max_parallel_api_calls` limit.
- Slurm submission is optional and requires explicit `--submit`.
- Experiment-level and episode-level concurrency are independently configurable.
- Sequential within-episode semantics and simultaneous within-pair decisions are
  unchanged.
- Every experiment has separate results, a human-readable task log, structured events,
  and atomic status metadata; Slurm tasks additionally have scheduler stdout/stderr.
- Results are human-inspectable as campaign -> phase-space cell -> replicated
  episode, with no need to filter one opaque global file.
- Each cell has its own configuration, status, episode summary, and interaction
  history files.
- Campaign-wide Parquet files remain available as rebuildable compatibility views,
  not as the sole copy of the results.
- Failed/interrupted experiments resume without repeating completed episodes.
- Either backend's dry-run exposes exact work, output paths, and aggregate request
  calculations without calling an LLM or submitting work.
- Backend parity tests prove identical task mapping, IDs, and scientific outputs.
- Unit, integration, resume, failure-isolation, API-limit, and mock parallelism
  tests pass.
- The existing single-experiment command and current result schema remain backward
  compatible.
