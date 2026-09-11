# TDD: Single-command SLURM-parallel information resampling

**Date:** 2026-09-11  
**Status:** implementation plan only  
**Scope:** standardized study aggregation on SLURM, especially MI/CMI bootstrap
and null resampling

## 1. Problem

`mas-cc study aggregate --study-dir <study>` already gives the correct
scientific handoff, but information resampling can dominate wall time. In the
recent Potsdam blackboard aggregation, 17 independent information groups took
most of a 4 hour 25 minute allocation; table finalization and plotting followed
quickly.

The user-facing and delivery contracts are already correct and must not become
more complicated. A user should not submit resampling shards, merge estimator
files, run a second aggregation command, or assemble a ZIP manually.

The target remains exactly one command:

```bash
mas-cc study aggregate --study-dir <study>
```

That command may coordinate several SLURM jobs internally, but it must publish
the same canonical `analysis/` directory and the same final ZIP contract as the
current serial/local-parallel workflow.

## 2. Non-negotiable invariants

This work changes execution topology only.

- The scientific cell remains the physical estimation unit.
- Scheduler task IDs are never scientific coordinates.
- MI, CMI, estimator variants, bootstrap, permutation/randomization nulls,
  support diagnostics, seeds, and numerical definitions remain unchanged.
- A heterogeneous study-wide information estimate must not be manufactured.
- Execution order, worker count, array throttle, and node placement must not
  change estimator results.
- Completed-episode filtering and `--allow-incomplete` semantics remain
  unchanged.
- Canonical `cells`, `episodes`, `rounds`, and `micro_slots` tables remain the
  scientific source of truth.
- The final analysis layout, table schemas, plot names, reports, provenance,
  validation, and ZIP name remain unchanged.
- Persistent analysis caches, individual null draws, and individual bootstrap
  draws remain excluded from the final output and package.
- No study-specific SLURM script is introduced.

## 3. User-facing behavior

### 3.1 Normal invocation

On a SLURM-capable installation, the existing command should:

1. validate and plan one aggregation generation;
2. submit the required internal SLURM dependency graph;
3. record the aggregation job IDs and progress location;
4. return a concise submission/status message;
5. allow the jobs to finish without the user's terminal remaining attached.

The internal jobs then:

1. prepare a consistent canonical input snapshot;
2. calculate independent information groups in parallel;
3. finalize derived tables, plots, reports, validation, and the ZIP once;
4. atomically publish the completed generation;
5. remove transient staging after successful publication.

The user does **not** run a separate merge/finalize command.

### 3.2 Non-SLURM and already-allocated execution

The same command must retain a local bounded-process backend for developer
machines and for execution inside a single existing allocation. Backend choice
is operational and must not affect scientific output.

An explicit backend override may be useful for testing, but the standardized
Potsdam path should select the SLURM backend without requiring additional
scientific configuration.

## 4. Execution architecture

Conceptually, one command submits this internal dependency graph:

```text
aggregation preparation
        |
        v
information-resampling array
  [group 0] [group 1] ... [group N]
        |
        v
single lightweight finalizer
        |
        v
atomic analysis/ publication + one ZIP
```

The finalizer is not a new scientific "aggregation of aggregations." It only
checks and concatenates compact estimator rows produced for disjoint scientific
groups, runs the existing downstream derived/plot/report/package stages, and
publishes one delivery.

### 4.1 Planning and consistent input generation

The initial command should cheaply discover the stable scientific cell/group
identities and create an `aggregation_execution_manifest` before submitting
jobs. The manifest must include:

- aggregation generation ID;
- study identity and analysis recipe hash;
- canonical scientific input identities/hashes;
- stable group index and complete scientific grouping identity;
- estimator requests and existing deterministic seed assignment;
- expected compact output fragment paths;
- requested SLURM resources and dependency job IDs.

The preparation job builds or validates one immutable input generation under a
transient staging directory. Resampling workers must all read this same
generation; they must not read a mixture of files while simulation or another
aggregation is modifying the study.

When retained canonical Parquet tables already form a valid source generation,
preparation should reuse them by identity rather than duplicate the complete
dataset unnecessarily.

### 4.2 Resampling work unit

The default work unit is one complete physical information-estimation group,
normally one scientific cell for the established study recipes. One worker
calls the existing authoritative estimator for all requested information
statistics/variants for that group and emits only compact final rows:

- information estimates and confidence summaries;
- compact null summaries;
- support diagnostics;
- estimator/group provenance and completion metadata.

Raw bootstrap and null draws remain process-local and are discarded after the
compact summaries are atomically written.

If there are many tiny groups, the planner may bundle several stable group
indices into one array task. Bundling is an execution optimization and cannot
alter group boundaries or estimator inputs.

### 4.3 CPU and memory model

The system does **not** require one permanently allocated CPU per study cell.
It requires one CPU slot per concurrently active resampling worker unless the
authoritative estimator demonstrates a justified internal CPU requirement.

Examples:

- 17 groups with a 12-task throttle: 12 groups run first and 5 follow.
- 100 groups with a 20-task throttle: groups run in approximately five waves.
- A small allocation may run four workers without changing the result.

The planner should choose a bounded task throttle from available CPUs, memory,
site policy, and observed per-group memory. Each task should default to one
process and avoid nested multiprocessing/BLAS oversubscription. Large groups
may request more memory or be scheduled in a lower-memory-concurrency wave,
but they remain the same scientific group.

### 4.4 Collision-free shared output

All internal jobs may target the same study root, but they must never write the
same final file concurrently.

Use a transient generation layout similar to:

```text
analysis/.work/<aggregation-generation-id>/
    execution_manifest.json
    progress.json
    input/
    groups/
        <stable-group-hash>.information.parquet
        <stable-group-hash>.support.parquet
        <stable-group-hash>.complete.json
    final/
```

Each group owns unique hash-addressed paths and writes through temporary-file
plus atomic-rename publication. Only the finalizer writes consolidated final
tables and the ZIP.

The currently published `analysis/` handoff should remain readable until the
new generation is complete. Final publication should replace the relevant
generated outputs atomically or through an equivalent validated generation
swap, so a failed reaggregation does not leave a half-new package.

After success, remove `analysis/.work/<generation>` and any invocation-local
draws. The staging directory is never included in the ZIP. On failure, it may
remain temporarily to support safe resubmission of only unfinished groups; it
must be removed after successful recovery and must not become a persistent
scientific cache.

### 4.5 Deterministic results

Parallelism must reuse the current stable sorted group order and the exact seed
assigned to each group by the existing implementation. Do not derive seeds
from SLURM array IDs, job IDs, process IDs, completion order, or node names.

The authoritative per-group estimator should receive the same observations,
configuration, and seed it receives today. The finalizer sorts compact rows by
the existing canonical result keys before publication. Serial, local-process,
and SLURM backends must therefore be numerically equivalent within the already
established tolerance (preferably byte-equivalent for deterministic tables
after metadata normalization).

## 5. Scheduler orchestration

Add a generic analysis launcher family under the existing Potsdam SLURM
infrastructure, not under any study folder. A likely division is:

```text
scripts/Potsdam/SLURM/
    run_study_analysis_prepare.job
    run_study_information_group.job
    run_study_analysis_finalize.job
```

If one parameterized generic job can cleanly cover these roles, prefer fewer
files. None may contain scientific coordinates or Study 06/07/09-specific
logic.

The command submits jobs with dependencies equivalent to:

```text
prepare job
array job:  afterok:<prepare-job>
finalizer:  afterok:<array-job>
```

If an array group fails, the finalizer must not publish a successful final
generation. A repeated invocation should validate fragment hashes and resubmit
only absent/invalid groups, then run one finalizer. Successful fragments are
execution recovery artifacts only and are deleted after final success.

SLURM stdout/stderr must use absolute paths under:

```text
<study-root>/logs/analysis-<generation-id>/
```

They remain execution artifacts and are excluded from the analysis ZIP.

All Potsdam Python commands must use the dedicated `MA-CC` Conda environment.

## 6. Progress and observability

Maintain a small machine-readable progress record while the generation is
active. It should report:

- generation ID and recipe/input hashes;
- prepare, array, and finalizer job IDs/states;
- total, pending, running, completed, and failed groups;
- group identity for failed tasks;
- elapsed time and last update;
- final publication/ZIP status.

The normal CLI output should point to this record and the final ZIP location.
Progress is operational, not scientific provenance; remove the transient
record after success or retain only a compact completed aggregation entry in
`analysis_manifest.json`.

## 7. Final delivery contract

The final output must remain the established standardized tree. At minimum:

```text
analysis/
    validation.json
    validation.md
    analysis_manifest.json
    analysis_recipe.yaml
    tables/
        cells.parquet
        episodes.parquet
        rounds.parquet
        micro_slots.parquet
        primary_estimates.parquet
        information_estimates.parquet
        support_diagnostics.parquet
        derived_observables.parquet
        ... existing configured compact result tables ...
    plots/
        ... all configured plots ...
    reports/
        summary.md
        methods.md
    provenance/
        ... existing minimal provenance ...
    <study-name>_analysis.zip
```

There must be no backend-specific differences in the handoff. In particular,
the final tree and ZIP must not contain:

- `.work/`;
- group fragments;
- SLURM logs;
- raw bootstrap draws;
- raw null draws;
- persistent estimator caches;
- duplicated consolidated tables.

## 8. Implementation sequence

### Phase 1: extract and freeze the per-group boundary

1. Identify the current function that executes one information group.
2. Define a serializable work specification around that existing function.
3. Preserve the current group ordering, seed mapping, compact output schemas,
   and progress callbacks.
4. Prove that invoking work specifications separately reproduces the existing
   in-process result.

No scheduler work should begin until this equivalence gate passes.

### Phase 2: transient fragments and finalization

1. Add generation-scoped staging and atomic fragment writes.
2. Add strict fragment identity/hash validation.
3. Refactor final table assembly to consume the compact group results.
4. Keep all downstream observables, plots, reports, validation, and packaging
   on their existing paths.
5. Clean staging after success.

### Phase 3: SLURM backend

1. Add generic prepare/group/finalize entry points.
2. Add dependency-aware submission behind the existing aggregation command.
3. Derive resources/throttle from declared analysis execution policy and site
   limits.
4. Record job IDs and progress without adding them to scientific identities.
5. Support safe retry of incomplete groups.

### Phase 4: operational validation

1. Run a tiny mock-provider study through serial and SLURM backends.
2. Run a Study-09-like fixture with several cells and nontrivial null/bootstrap
   counts.
3. Compare tables, plots, reports, validation, manifest, and ZIP contents.
4. Benchmark elapsed time, peak per-worker memory, aggregate memory, staging
   size, and final package size.

## 9. Test plan

### 9.1 Scientific equivalence

- Serial and SLURM backends produce the same MI/CMI point estimates.
- Jeffreys, Miller–Madow, and unsmoothed variants remain equivalent.
- Bootstrap interval summaries remain equivalent.
- Null means, standard deviations, p-values, and permutation counts remain
  equivalent.
- Support diagnostics remain equivalent.
- Signed response, susceptibility, eta outputs, currents, effective affinity,
  and all other downstream results remain equivalent.
- Reordering array completion does not change any result.
- Changing array throttle does not change any result.

### 9.2 Scientific grouping

- One work unit receives exactly one complete physical estimator group.
- A split execution never averages shard-level CMI values.
- No cross-cell or cross-rho super-pool is introduced.
- Partial aggregation retains the existing visibly incomplete semantics.

### 9.3 Scheduler behavior

- One CLI invocation submits one valid prepare/array/finalize dependency graph.
- The finalizer cannot run after a failed prepare or array task.
- A retry submits only missing/invalid group work.
- Job IDs, array IDs, and completion order never enter estimator identities or
  seeds.
- Potsdam jobs invoke the `MA-CC` environment and write logs beneath `/work`.

### 9.4 Filesystem and package behavior

- Concurrent workers only write their own group paths.
- Fragment publication is atomic.
- A failed generation leaves the previous handoff intact.
- Successful publication contains the same canonical filenames and schemas.
- The final ZIP contains no staging, fragments, logs, caches, or draw tables.
- Successful aggregation removes transient staging.
- Reaggregation from retained canonical tables remains supported.

### 9.5 End-to-end acceptance fixture

For one deterministic multi-cell fixture:

1. aggregate with the existing serial reference path;
2. aggregate with the local bounded-process path;
3. aggregate with the SLURM array path;
4. compare normalized final analysis trees and ZIP inventories;
5. assert established numerical tolerances for every scientific table;
6. assert identical configured plots and reports are present.

## 10. Resource acceptance criteria

The implementation must report rather than hide:

- number of information groups;
- requested worker/task throttle;
- CPUs and memory per worker;
- total peak requested CPUs/memory;
- observed per-group duration and memory;
- preparation and finalizer durations;
- overall elapsed time.

Do not promise a fixed speedup. As an initial operational target, a study with
roughly 17 similarly sized independent groups and 12 usable worker slots should
finish its resampling stage substantially faster than the observed multi-hour
baseline, without increasing scientific approximation or losing support/null
outputs.

## 11. Acceptance criteria

This feature is complete when:

1. The user runs only `mas-cc study aggregate --study-dir <study>`.
2. SLURM parallel preparation/resampling/finalization is automatic on Potsdam.
3. No manual fragment merge or second aggregation command is required.
4. Workers may run across nodes and write collision-free outputs under one
   generation in the same study root.
5. The same physical groups, estimators, resampling counts, and seeds are used.
6. Serial and distributed scientific results pass equivalence tests.
7. The existing canonical analysis directory and ZIP delivery are preserved.
8. Failed distributed aggregation cannot corrupt the last valid handoff.
9. Temporary fragments and draws are absent after success and excluded from
   every package.
10. The implementation uses only generic study-analysis launchers.

## 12. Explicit non-goals

Do not:

- change estimator mathematics or reduce resampling counts for speed;
- assign one permanent node or CPU to every cell;
- require the user to understand or invoke a merge job;
- persist raw resampling draws;
- treat execution fragments as scientific outputs;
- place scheduler topology in experiment YAML scientific coordinates;
- modify simulation/provider execution;
- introduce a study-specific launcher;
- change the final handoff layout or naming.

