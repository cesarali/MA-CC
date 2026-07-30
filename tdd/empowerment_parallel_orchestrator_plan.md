# TDD Plan: Parallel Empowerment Experiment Orchestrator

## Objective

Add explicit, independently configurable parallelism at two safe levels:

1. **Experiment parallelism**: run independent experiment configurations as separate
   Slurm jobs through a manifest-driven orchestrator.
2. **Episode parallelism**: run independent episodes concurrently inside each
   experiment process.

Do not parallelize pair interactions within one episode. `NamingConventionGame`
updates agent memory and scores after every pair, so those interactions are causally
ordered. The two LLM decisions belonging to one pair should remain concurrent as
they are now.

The design must provide failure isolation, resumability, and separate logs for every
experiment. A failed experiment must not cancel or corrupt other experiments.

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

For ordinary, non-forced pairs, the approximate request concurrency of one process
is therefore:

```text
min(request_concurrency, 2 * episode_concurrency)
```

If several experiment processes use the same provider account, their client-side
semaphores do not coordinate. The orchestrator must consequently expose and
document an aggregate concurrency budget:

```text
aggregate requests ~= active experiments * 2 * episodes per experiment
```

This is a limit estimate, not a guarantee: committee-forced decisions make no LLM
request, retries add traffic, and provider-side rate limits remain authoritative.

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

### Several experiments

Introduce a manifest, for example:

```yaml
schema_version: 1
defaults:
  provider_budget: 20
  episode_concurrency: 2
  request_concurrency: 4
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

Add an orchestration command or script with a dry-run mode:

```bash
naming-game orchestrate-experiments \
  --manifest configs/empowerment_parallelism_sweep.yaml \
  --results-root /work/.../results/parallelism-sweep \
  --logs-root /work/.../logs/parallelism-sweep \
  --dry-run
```

Submission should create a Slurm job array, with one array task per manifest entry.
Use an array concurrency cap such as `--array=0-2%2` to limit the number of active
experiments. Do not launch background children from one allocation: separate Slurm
tasks provide clearer accounting, cancellation, retries, and logs.

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
  stdout.log
  stderr.log
  events.jsonl
```

If one manifest entry represents a full campaign, `<experiment-id>` and
`<campaign-id>` are the same. If a manifest entry represents one phase-space cell,
it writes only to its precomputed cell path beneath the shared campaign root. The
orchestrator must reject manifests in which two tasks could write the same cell or
replicate path.

`execution_config.json` should include:

- experiment ID and manifest path;
- source config path and content fingerprint;
- requested episode and request concurrency;
- Slurm job ID, array job ID, array task ID, node, and timestamps;
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
total episode count, last update time, and paths to stdout/stderr. Never store API
keys, `.env` contents, prompts containing secrets, or authorization headers.

Write structured `events.jsonl` records for experiment start/end, episode
start/end/failure, retry warnings, compaction, and analysis. Continue to retain the
human-readable progress logs.

Slurm should initially route logs directly to unique files such as:

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

- unique, filesystem-safe experiment IDs;
- required config files and normalized absolute paths;
- default inheritance and per-entry overrides;
- duplicate output-directory rejection;
- positive concurrency values;
- aggregate concurrency calculation and budget warnings/errors;
- deterministic array-index-to-experiment mapping;
- deterministic, collision-free phase-space cell and replicate paths;
- rejection when two manifest entries resolve to the same cell/replicate path;
- rejection of unknown manifest keys to catch spelling mistakes.

Manifest validation and dry-run must not submit jobs or call an LLM.

### Slice 4: task runner

Implement a command that executes exactly one manifest entry by index:

```bash
naming-game run-experiment-task \
  --manifest <path> \
  --index "$SLURM_ARRAY_TASK_ID" \
  --results-root <path> \
  --logs-root <path>
```

Tests must verify:

- the index resolves to the expected experiment;
- paths cannot escape the configured roots;
- status transitions are atomic and correct for success, exception, and signal;
- an existing completed run is skipped unless explicitly forced;
- an interrupted or failed run resumes existing episode shards by default;
- one experiment never reads or writes another experiment's directory.
- one phase-space cell never reads or writes another cell's directory;
- completed replicates are discoverable directly from the human-readable hierarchy,
  without scanning opaque hash filenames.

### Slice 5: Slurm orchestrator

Generate or submit an array job whose task invokes `run-experiment-task`.

Tests should mock the subprocess boundary and assert:

- dry-run prints the complete index mapping, output paths, log paths, concurrency
  estimates, and `sbatch` command without submission;
- submission uses the requested array concurrency cap;
- paths and exported values are shell-safe;
- the returned Slurm job ID is recorded in an orchestration summary;
- submission failure leaves a clear failed orchestration record;
- no credentials are passed through command-line arguments or written to metadata.

Prefer passing a small set of non-secret paths/IDs to the task and letting the task
read the manifest. Use the existing environment setup in the Potsdam job script.

### Slice 6: failure inspection and retry

Add read-only status and selective retry commands:

```bash
naming-game experiment-status --results-root <root>
naming-game retry-experiments --results-root <root> --state failed
```

Tests must cover mixed completed/running/failed states, stale `running` states, and
retry selection. Retry should submit only selected experiments and resume their
checkpoints. It must never delete partial results automatically.

---

## Parallelism sweep acceptance test

Use the mock client to run identical scientific configurations with episode
concurrency 1, 2, and 4.

Required assertions:

1. All runs produce the same episode IDs and deterministic scientific outputs.
2. All runs contain the expected number of episode and interaction rows.
3. No episode has overlapping pair interactions or reordered state updates.
4. Measured episode concurrency does not exceed 1, 2, and 4 respectively.
5. With artificial request latency, concurrency 2 and 4 reduce wall time relative
   to concurrency 1 within a generous, non-flaky bound.
6. Killing a task after several checkpoints and rerunning it completes only the
   missing episodes.
7. Three array tasks produce three separate stdout, stderr, event, status, and
   result locations.
8. One deliberately failing task does not stop the other array tasks.
9. Every episode is stored under the correct deterministic phase-space cell and
   replicate path.
10. Cell-level compacted files contain only their own cell, while rebuilt campaign
    aggregates contain all completed cells exactly once.
11. Analysis produces identical estimates when reading the campaign aggregate and
    when recursively reading the partitioned cell hierarchy.

After mock acceptance, run a small live-provider smoke sweep before changing the
full pilot. Start conservatively and inspect throttling/retry rates:

```text
active experiments = 1, episode concurrency = 1
active experiments = 1, episode concurrency = 2
active experiments = 1, episode concurrency = 4
```

Only then test multiple simultaneous experiments while keeping the estimated
aggregate request concurrency within the provider budget.

---

## Operational safeguards

- Preserve the existing stable episode IDs and atomic Parquet checkpoint writes.
- Resume by default; require an explicit flag for a clean non-resumed run.
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
- Keep analysis local to its experiment and run it only after successful history
  compaction.
- Do not infer provider capacity from allocated CPUs: these experiments are mostly
  network/API bound.
- Do not increase concurrency for the currently running pilot. Let it finish, then
  validate the new runner with mock and smoke configurations.

---

## Definition of done

- A manifest can launch multiple isolated experiments through one Slurm array.
- Experiment-level and episode-level concurrency are independently configurable.
- Sequential within-episode semantics and simultaneous within-pair decisions are
  unchanged.
- Every experiment has separate results, stdout, stderr, structured events, and
  atomic status metadata.
- Results are human-inspectable as campaign -> phase-space cell -> replicated
  episode, with no need to filter one opaque global file.
- Each cell has its own configuration, status, episode summary, and interaction
  history files.
- Campaign-wide Parquet files remain available as rebuildable compatibility views,
  not as the sole copy of the results.
- Failed/interrupted experiments resume without repeating completed episodes.
- Dry-run exposes the exact work and aggregate request budget before submission.
- Unit, integration, resume, failure-isolation, and mock parallelism-sweep tests pass.
- The existing single-experiment command and current result schema remain backward
  compatible.
