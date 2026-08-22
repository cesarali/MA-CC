# SLURM jobs, grid cells, parallelism, and result aggregation

This document describes the execution and post-processing behavior that exists
in the repository today. It also gives a standard launch policy so that a new
experiment config does not automatically imply another study-specific `.job`
file.

## Short version

A `.job` file is infrastructure. A YAML config is scientific design. They
should normally vary independently.

The preferred default is:

1. Put all conditions that share a game type, provider, model, pricing policy,
   and budget into one YAML `grid:`.
2. Submit that config with a reusable grid job.
3. Let one `experiment run` process expand the grid, run episodes under one
   shared `execution.parallelism` limit, and aggregate every completed cell.
4. Add a study-specific analysis command only when the scientific estimand is
   not one of the configured generic metrics.

A new `.job` is justified only when the scheduler topology is genuinely new:
for example, a cell-sharded array, a row-sharded study, special resources, or a
mapping from array indices to several configs. A new experiment name, grid,
seed, or output label alone is not a reason for a new `.job`.

## The five levels that are easy to confuse

```text
SLURM submission
  array task = one OS process (`python ... experiment run`)
    run = one config and one run directory
      grid cell = one resolved combination of `grid:` axis values
        episode = one repetition with its own deterministic seed
          provider calls = concurrent async requests made during the episode
```

These levels have different controls and different outputs.

| Level | Created by | Parallelism control | Scientific identity |
|---|---|---|---|
| SLURM array task | `sbatch --array=...%T` | `%T` array throttle | None by itself; it is an execution shard |
| Python process/run | one command in the job | number of running array tasks/jobs | config, experiment name, seed, output root |
| Grid cell | Cartesian product of YAML `grid:` axes | no separate per-cell pool | stable `cell-NNNN`, overrides, resolved config |
| Episode | `execution.repetitions` per cell | one shared `execution.parallelism` semaphore per process | `cell-NNNN-RRRR` and derived seed |
| Provider request | game call plan | `llm_provider.request_concurrency`, additionally bounded by active episodes | request/accounting records |

There is no Python process pool in the normal experiment runner. Episode
parallelism is implemented with `asyncio` tasks and one semaphore. Blocking
cell aggregation is moved to worker threads with `asyncio.to_thread` so it does
not stall live episodes. Multiple Python processes appear only when SLURM runs
multiple jobs or array tasks.

## How a normal grid run works

Given:

```yaml
execution:
  repetitions: R
  parallelism: P

grid:
  some.axis: [a, b]
  another.axis: [x, y, z]
```

the engine constructs `2 x 3 = 6` cells in stable Cartesian-product order.
Each cell has `R` episodes, so the process creates `6R` episode tasks. All
`6R` tasks share one semaphore of size `P`. `P` is therefore **process-wide**,
not `P` per cell.

Consequences:

- Episodes from different cells may run at the same time.
- Completion order is not cell order and is not episode order.
- A cell is still respected: it has its own `cell-NNNN` directory, resolved
  config, override record, episode IDs, and aggregate.
- Grid axes do not create extra processes.
- The provider client, pricing quote, and runtime budget guard are shared by
  every cell in the process.

The grid loader deliberately forbids sweeping provider identity, model, game
type, budget, or pricing. Those fields require separate configs/runs because
the current runner shares those objects across the grid.

### Cell and seed stability

Cells are numbered from the Cartesian-product order of axes as written in the
config. Each episode seed is derived from the base execution seed, cell index,
and repetition index. Do not reorder grid axes or values and assume an old
output directory is still the same design.

`experiment.metadata.common_random_numbers_across_grid: true` is an explicit
exception: it holds the episode stochastic streams fixed across cells for
matched designs. It does not merge cells; only their random streams are
matched.

## The two current SLURM execution modes

### Mode A: one process per complete grid (preferred default)

The job invokes:

```bash
python -m mas_cc.cli.main experiment run \
  --config CONFIG \
  --output-dir OUTPUT_DIR \
  --no-progress
```

One process owns the complete grid. It writes the normal run tree and performs
cell aggregation during the run. Study 03, each Study 04 row, and each Study
05 task config use this mode.

Use it when:

- the complete config fits within one job's wall time and memory;
- one shared process-level concurrency/budget is desirable;
- downstream analysis expects all relevant cells in one run directory; or
- there is no measured need to shard cells across nodes.

### Mode B: one SLURM array task per original cell (exception)

Study 01 and Study 02 use
`scripts/experiment_design/run_classical_grid_cell.py`. The wrapper loads the
original grid, selects one original `GridCell`, and runs a one-cell view of it.
It preserves the original cell index, `cell-NNNN` ID, overrides, and seed
derivation. Each array task writes an independent shard run tree.

Use it when cells are long and independent enough that distributing them over
nodes materially improves wall time. It has important costs:

- every array task has its own provider client, request-concurrency limit,
  pricing check, and runtime budget guard;
- there is no single live grid-level aggregate across shards;
- a separate roll-up is mandatory;
- the SLURM array throttle must account for the multiplied provider traffic.

For a cell-sharded array:

```text
maximum active episode tasks = array throttle x execution.parallelism
rough provider-request ceiling = array throttle
                                  x min(execution.parallelism,
                                        llm_provider.request_concurrency)
```

The actual request rate also depends on the game's call plan and provider
latency. The conservative operational rule is to treat every running array
task as an independent client at its configured `request_concurrency` and
document the rate-limit arithmetic in the submission command.

### Row/config arrays are not cell sharding

Study 04 runs one array task per `q_c` row, and Study 05 runs one array task per
world config. Each task still runs a complete grid and produces a normal run
tree. This is config-level sharding: use it when the configs are scientifically
separate slices or cannot be represented as one legal grid.

Do not describe such an array as “one task per cell.” Its cells remain inside
each Python process.

## What is written, and when

A normal grid run has this conceptual layout:

```text
OUTPUT_DIR/
  <game>/<experiment>/<experiment>-<seed>/
    resolved_base_config.yaml
    manifest.json
    grid_summary.csv
    budget_state.json
    grid_progress.png
    sweep_metrics.json                 # only when sweep metrics are configured
    cells/
      cell-0000/
        overrides.json
        resolved_config.yaml
        cell_summary.json
        aggregate.json
        metrics/plots/
        data/episodes/...              # full profile, or compact scientific output
        .resume/<episode-id>/...       # durable episode state/checkpoint material
```

Exact retained episode files depend on `storage.artifact_profile`. With
`results_only`, completed data are compacted to the scientific tables needed
for analysis and aggregation; temporary `.resume` material supports recovery
while work is incomplete.

`storage.checkpoint_mode: episode` means the unit of resume is a completed
episode. The default CLI behavior is `--resume`. Before skipping an episode,
the runner validates its manifest/provenance, including resolved-config,
prompt-definition, pricing, and scientific schema identities. A changed config
is not silently mixed with old completed episodes.

Safe rules:

- Resume the same design into the same output root.
- Use a new experiment name/output root when the scientific design changes.
- Use `--no-resume` only when deliberately rerunning work; do not point it at a
  populated directory unless the intended storage policy permits that.
- Treat `storage.wipe_and_recompute` as destructive and exceptional.

## Built-in aggregation versus scientific post-processing

These are separate stages and should not be called by the same name.

### 1. Built-in cell aggregation

When the last episode belonging to a cell reaches a terminal state, the master
process reads the completed episode records and writes that cell's
`aggregate.json` and plots. Incomplete or failed episodes are excluded and
listed explicitly. Aggregation runs outside the episode semaphore, so it does
not consume an episode execution slot.

For configured sweep metrics, the master refreshes `sweep_metrics.json` after
every newly completed cell. Local files are written before optional Comet
publication. Comet is a monitoring view, not the authoritative result store.

Because each cell is aggregated at completion, a job killed at 80% can still
contain valid aggregates for every cell that finished.

### 2. Offline re-aggregation

Built-in metrics can be recomputed from disk without provider calls:

```bash
python -m mas_cc.cli.main experiment aggregate \
  --run-dir /work/.../<game>/<experiment>/<run-id>
```

By default this reads the recorded `aggregation:` section from the run's
resolved config. Passing `--config NEW_CONFIG` intentionally substitutes that
config's aggregation section, allowing new percentile bands, windows, fill
rules, or metric selections without rerunning episodes.

This command uses the same aggregation implementation as the live runner.

### 3. Shard roll-up

Cell-sharded arrays do not have one parent grid directory. After all shards are
complete, run:

```bash
python scripts/experiment_design/aggregate_grid_shards.py \
  --result-root /work/.../results/STUDY \
  --output-dir /work/.../results/STUDY_analysis \
  --expect-cells EXPECTED_CELLS \
  --expect-episodes EXPECTED_EPISODES \
  --zip
```

This is a read-only scientific roll-up of the shard sources. It writes design,
episode, round, and interaction tables plus hashes and integrity reports into a
separate output directory. It does not recreate dynamics or modify shards.

### 4. Study-specific analysis

Some estimands are deliberately outside generic aggregation. For example,
Study 05 disables generic information analysis because its deterministic
action cells have zero action entropy. Its matched contrast is computed by:

```bash
python scripts/relational_study05_state_matching.py analyze \
  --run-dir RUN_TASK0001 \
  --run-dir RUN_TASK0002 \
  --output-dir /work/.../results/relational_population_study05_analysis
```

This combines the two world runs, verifies matched always/never pairs, and
writes the requested effect table and figure. This command should run only
after both task runs complete successfully.

## Standard submission policy

Before writing a new `.job`, classify the run:

| Question | If yes | If no |
|---|---|---|
| Can conditions share game/provider/model/pricing/budget? | Put them in one YAML grid | Use separate configs/runs |
| Does the complete grid fit one allocation? | Use one generic complete-grid job | Consider row/config or cell sharding |
| Must downstream analysis pool cells in one run tree? | Keep that slice in one process | Sharding is allowed with an explicit roll-up |
| Are only config, label, time, memory, or output changing? | Pass parameters to a reusable job | A specialized job may be justified |
| Does an array index require study-specific mapping? | Keep a small study launcher or manifest | Use the generic job |

The desired reusable interface is one generic job that accepts at least:

```text
CONFIG
OUTPUT_DIR
```

and receives scheduler resources at submission time:

```bash
sbatch --job-name=LABEL --time=04:00:00 --mem=32G \
  --output=/work/.../logs/LABEL_%j.out \
  --error=/work/.../logs/LABEL_%j.err \
  scripts/Potsdam/SLURM/<generic-grid-job> CONFIG OUTPUT_DIR
```

An array of complete configs should preferably use a checked-in manifest
(`array index -> config -> output`) consumed by one reusable array job. That
keeps scientific mappings reviewable without duplicating proxy exports,
environment checks, Python paths, and the runner command in every study.

Until that reusable launcher is consolidated, use
`hidden_bench_imitation_round_feedback_grid.job` as the closest existing
parameterized example, but note that its result naming and allocation are
still partly hard-coded. The population-study jobs are therefore current
operational launchers, not the target abstraction.

## Completion and post-processing checklist

1. Check every expected SLURM task with `squeue` and then `sacct`; a missing
   task is not the same as a successful task.
2. Require `State=COMPLETED` and `ExitCode=0:0` for every array element.
3. Inspect `.err` files. Comet `INFO` lines are normal; Python tracebacks,
   provider errors, budget stops, and aggregation errors are not.
4. Inspect each run's `manifest.json`, `grid_summary.csv`, and cell summaries;
   compare completed/failed/skipped counts with the design.
5. For complete-grid runs, confirm each expected `cell-NNNN/aggregate.json`.
6. For cell-sharded arrays, run `aggregate_grid_shards.py` with explicit
   expected counts and retain its source manifest.
7. Run the study-specific analysis, if one exists, into a new analysis
   directory rather than writing into source runs.
8. Record the job ID, config commit, output roots, and analysis command in the
   study README or run log.

## Current status and recommended cleanup

The experiment engine already provides a coherent grid/cell/episode model,
shared in-process concurrency, durable episode resume, live per-cell
aggregation, and offline re-aggregation. The inconsistency is mainly in the
SLURM launch layer:

- many study jobs repeat the same environment and launch boilerplate;
- resource requests and output naming are embedded in scripts;
- some jobs run complete grids while others shard cells, but filenames and
  comments are the main way to discover which;
- study-specific post-processing commands are not exposed through one common
  completion workflow.

The next standardization step should therefore be launch-layer consolidation,
not a rewrite of the experiment engine:

1. one parameterized complete-grid job;
2. one parameterized original-cell array job;
3. one manifest-driven multi-config array job;
4. a submission/check script that validates configs, computes total effective
   concurrency, creates log/output directories, and prints the exact
   post-processing command;
5. small study manifests and analysis scripts retained only where the science
   requires them.

That structure would make “new config” mean “new submission arguments or one
manifest row,” not “copy and edit another `.job`.”
