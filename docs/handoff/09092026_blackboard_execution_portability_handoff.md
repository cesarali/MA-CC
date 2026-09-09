# Blackboard execution portability handoff

## Purpose

Set up the MA-CC blackboard study workflow on another compute system while
preserving the current scientific behavior. The operational goal is to keep the
provider's safe concurrency and token throughput well utilized. CPU allocation
supports that goal but does not define it because remote calls spend most of
their time waiting for network and model responses. Use the simplest available
compute architecture that can keep the provider busy and retain results
reliably. A single server with 20–30 CPUs may host many more logical episode
tasks; Kubernetes is useful when it is already the simplest available platform
or when one server cannot sustain the required load.

The current DeepInfra study is the reference deployment:

```text
/home/ojedamarin/Projects/LanguageGames/MA-CC/configs/runs/
relational_reasoning/blackboard_game/astra_task003_false_control_30x30
```

In the repository, the relative path is:

```text
configs/runs/relational_reasoning/blackboard_game/
astra_task003_false_control_30x30
```

This folder contains the experiment configuration, study execution settings,
analysis recipe, and README. Use it to understand the complete workflow rather
than treating an individual shell command as the specification.

This reference folder is the acceptance anchor. A new runner must resolve its
YAML files into the same cells, repetitions, episode seeds, provider/model,
retention settings, and analysis request. Scheduler topology may differ, but a
comparison must always begin from this resolved configuration.

## Portable scientific input bundle

The frozen Task 3 data and the 30 paired initialization artifacts used by the
reference config are packaged together here:

```text
/work/ojedamarin/Projects/LanguageGames/MA-CC/results/handoffs/
astra_task003_execution_inputs_20260910.zip
```

Archive SHA-256:

```text
69abb34d8ba00246d58ce4378671dcdb05a9a94e4dbac59c15dd2b9928a3aebb
```

The ZIP contains 67 files in two top-level directories:

```text
task_003/
astra_task003_false_control_30x30_initializations/
```

`task_003/` is the complete frozen task directory, including the task and
hidden-world definitions, generated fact text, 24-agent private assignment,
controller fact pool, controller budget subsets, and symbolic provenance. The
initialization directory contains `initialization_manifest.json` and the 30
deterministic episode-seed artifacts shared across parameter cells.

After extraction, configure `game.options.task_dataset_dir` to the directory
that directly contains `task_003/`. Configure
`game.options.initialization.artifact_dir` to the extracted initialization
directory. On a platform that can reproduce the reference filesystem layout,
the YAML can be used unchanged; otherwise only these storage locations need to
be adapted to the deployment. Do not regenerate the task or initializations
when validating execution equivalence.

Verify the archive checksum before extraction and keep all files together. The
reference YAML also records hashes for `task.json`, the private assignment, and
the controller pool; preflight must continue to validate those identities. The
bundle contains scientific inputs only: the engineer still needs the repository
at the selected Git revision and a DeepInfra credential supplied separately.

## What must remain stable

The scientific structure is independent of the machine that runs it:

```text
study
  -> scientific parameter cells
      -> repeated episodes
          -> rounds and microscopic events
```

- A **cell** is one fixed parameter combination, such as one `(rho, b)` point.
- An **episode** is one repetition of that cell with a deterministic identity
  and seed.
- The cell remains the unit of scientific completeness, validation, analysis,
  uncertainty summaries, and comparison even if its episodes execute on
  different workers or machines.
- Round and microscopic records are the canonical scientific observations.
- Aggregation groups episodes by their scientific cell, regardless of which
  CPU, process, node, or deployment ran them.

Game rules, task data, prompts, controller behavior, seeds, observables,
estimators, and analysis recipes must continue to come from the existing MA-CC
configuration and runtime. Deployment settings may control how much work runs
at once, but they must not alter scientific identities or results.

## Current reference workload

The reference DeepInfra study currently uses:

- 20 scientific cells;
- 30 episodes per cell, for 600 episodes total;
- one scheduler shard per cell;
- up to 9 active shards;
- 8 allocated CPUs per shard, or 72 allocated CPUs at full occupancy;
- up to 20 active episode/request slots per shard;
- a study-wide concurrency ceiling of 180 outstanding provider requests;
- a configured safety target of 1,000 dispatched requests per minute;
- DeepInfra model `deepseek-ai/DeepSeek-V4-Flash`.

The 72 allocated CPUs are a consequence of the current SLURM layout. They are
not a scientific requirement. Most episode time is spent waiting for remote
model responses, so a smaller machine can potentially maintain similar network
concurrency by managing many waiting episode tasks with fewer CPU processes.
Likewise, allocating 72 CPUs does not guarantee 180 live requests: the runner
must maintain enough ready episode work to refill provider slots immediately.

DeepInfra exposes the account limits at:

```text
GET https://api.deepinfra.com/v1/me/rate_limit
```

At the time of this handoff, the account reported a limit of 200 concurrent
requests per model and 1,100,000 tokens per minute. These values should be
queried during setup or preflight rather than treated as permanent constants.

## Throughput objective

Treat provider capacity as a runtime resource separate from CPUs:

- CPU capacity runs local game transitions, validation and serialization.
- Logical episode concurrency determines how many episodes can advance.
- Provider concurrency limits outstanding remote calls.
- Token throughput limits prompt and response volume per minute.
- An optional RPM target provides an additional operational safety bound.

For example, 180 continuously occupied request slots with a 20-second average
response time could yield approximately 540 requests per minute. A run with an
allowed concurrency of 180 but only 50 live requests will remain far below that
rate. The runner should measure both the configured ceiling and actual provider
occupancy.

The objective is sustained successful throughput within current account limits.
Maximum CPU utilization and a fixed process count are not objectives.

## Recommended first deployment

Start with one reliable server and the existing MA-CC Python application.
Use the CPUs and memory that are readily available, durable storage, and the
DeepInfra credential through the host's secret mechanism. A machine with
20–30 CPUs is a reasonable example, not a cap on logical episodes or remote
requests.

Run a shared pool of episode workers. Each worker takes the next missing episode
from a queue, executes it through the existing game runtime, writes it beneath
the correct scientific cell, and then takes another episode. Episodes from the
same cell may run on different workers because the episode already carries its
cell identity and deterministic seed.

The number of CPU processes and the number of outstanding provider requests are
separate controls. A useful initial arrangement is:

- enough processes to use the available CPUs for game logic and serialization;
- roughly 180–190 logical episode tasks when using the current DeepInfra account,
  adjusted after measuring memory and token demand;
- threads or asynchronous tasks for remote calls, because waiting calls do not
  require dedicated CPU cores;
- one global DeepInfra concurrency and rate controller shared by every worker.

Measure CPU, memory, response latency, request concurrency, tokens per minute,
successful responses, and completed episodes before increasing load. Keep
provider slots occupied when eligible work exists while retaining a safety
margin below the provider's current account limits.

## Execution behavior

The new runner should consume the same resolved study/episode plan used by the
existing workflow. A queued episode needs only:

- study and scientific cell identity;
- resolved experimental coordinates;
- repetition index and deterministic episode seed;
- resolved game/provider configuration;
- canonical output destination.

The shared queue may interleave episodes from many cells to avoid idle provider
capacity. Queue order and worker placement are operational choices. They must
never change a cell's resolved coordinates, expected repetition set, episode
identity, seed, or output ownership.

Workers must publish an episode atomically: write and validate its retained
records first, then publish its completion seal. A cell is complete when all of
its expected episode identities have valid completion seals. Cell completion
must be derived from the study manifest rather than from one worker, process,
pod, or scheduler job finishing.

On restart, the runner scans the expected episode plan, keeps valid completed
episodes, and schedules only missing or invalid episodes. Two workers must not
be able to claim the same episode simultaneously. This provides recovery from
process crashes, machine restarts, and later study extensions without repeating
expensive provider calls.

Aggregation begins from the declared scientific cells and joins completed
episodes by their stable identities. It must not group by process, host, pod,
scheduler shard, completion order, or storage partition. Per-cell bootstrap,
null, support, response, information and efficiency calculations therefore use
the same scientific population as the current workflow.

## Provider coordination

All workers must share one provider controller. It should enforce:

- account limits resolved from the provider before launch when available;
- the DeepInfra account concurrency limit with a safety margin;
- the account token-per-minute limit;
- an optional dispatch-rate safety target;
- short backoff and retry for retryable provider errors;
- fast reduction when provider errors increase;
- gradual recovery after stable responses;
- bounded retries within the current episode.

It should expose live provider occupancy, dispatched requests over the last
minute, token throughput, latency, successes, retryable failures, and the
current adaptive limit. This distinguishes unused capacity from provider
throttling and slow model responses.

For one server, shared local state and process-safe locking are sufficient. A
network service such as Redis is warranted only when workers span independent
machines that cannot coordinate through reliable shared state.

Validation failures from malformed model decisions are scientifically distinct
from HTTP rate-limit or provider failures. Preserve strict response validation
and the existing bounded correction behavior; do not coerce malformed decisions.

## Storage and analysis

Use the existing canonical hierarchy for studies, cells, episodes, rounds,
microscopic events, provenance, and completion seals. A single server can use a
durable local or mounted filesystem. A multi-machine deployment needs shared
storage or object storage with equivalent atomic publication semantics.

The existing aggregation command must remain the analysis entry point. It
should process completed canonical observations in exactly the same way whether
they were generated by:

- the new single-server runner;
- the existing Potsdam SLURM workflow;
- an optional Kubernetes backend.

Aggregation, estimators, plots, reports, and final packages must not import or
depend on scheduler-specific concepts. Potsdam must continue using its existing
generic SLURM launchers and dedicated `MA-CC` Conda environment.

## When Kubernetes is appropriate

Use Kubernetes when it is already the simplest supported compute platform or
when the deployment needs one or more of the following:

- workers spread across several independent machines;
- automatic replacement of failed machines or containers;
- elastic capacity that changes while the study is running;
- centralized operational monitoring already supplied by the organization;
- shared object storage and a network coordinator already available.

If Kubernetes is selected, one Job or pod should manage a bundle of episode
tasks. Creating one pod for every individual episode adds scheduling overhead
without improving the scientific model. The Kubernetes component should only
translate the canonical episode plan into work; it should not introduce another
study format. Pods may work on episodes belonging to different cells, but every
result must return to its declared cell identity and pass the same completion
checks.

## Five implementation steps

1. **Reproduce the environment.** Install MA-CC at a recorded Git revision,
   configure DeepInfra credentials externally, and verify the blackboard game,
   task loader, pandas, and PyArrow imports.

2. **Run one episode plan locally.** Use the current study expansion logic and
   existing episode runtime. Confirm deterministic identity, seed, output path,
   retained records, and completion seal.

3. **Add the shared worker pool and provider controller.** Schedule enough
   missing episodes across all cells to occupy safe provider capacity while
   enforcing one global concurrency, token, and dispatch policy.

4. **Verify interruption and aggregation.** Stop a small test midway, restart
   it, confirm that completed episodes are retained, and run the standard
   aggregation from the resulting canonical data.

5. **Scale using measurements.** Increase active logical episode tasks until
   provider concurrency or token throughput is well utilized, or CPU, memory,
   response reliability, or available infrastructure becomes the practical
   limit. Use the simplest deployment that reaches stable throughput.

## Acceptance test

Use a small frozen subset of the reference configuration without changing its
game semantics. Compare it with the current runner and require:

- identical resolved scientific cells and episode seeds;
- exactly the declared repetition membership for every cell, even when those
  repetitions ran on different workers or machines;
- no duplicate episode identities;
- compatible round, micro-event, provenance, and completion records;
- successful restart that schedules only missing episodes;
- unchanged final estimator values within established numerical tolerances;
- successful standard plots and analysis packaging;
- provider concurrency that never exceeds the resolved account limit;
- evidence that eligible queued work refills free provider capacity without
  mixing or pooling scientific cells;
- no regression in the existing Potsdam SLURM path.

Once this passes, the deployment can run larger studies by changing operational
capacity settings. The scientific configuration, task generation, and analysis
remain shared across every environment.
