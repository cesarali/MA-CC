# TDD: Incremental Study Extension, Cell Reuse, and Run Provenance

**Date:** 2026-08-31  
**Status:** implementation plan; no implementation in this document  
**Scope:** standardized `mas-cc study` workflow

## 1. Objective

Add a general study launch mode that can extend an existing standardized study
without repeating valid scientific work.

The target operations are:

1. increase the repetition target for existing cells;
2. expand a parameter grid while reusing coincident cells;
3. do both at once;
4. aggregate the original and extension outputs as one scientifically coherent
   study;
5. retain exact provenance for every cell and episode.

Example:

```text
existing target: 4 x 4 cells, 20 repetitions per cell
new target:      8 x 8 cells, 40 repetitions per cell
```

For the 16 coincident cells, the planner must schedule only repetition indices
20--39. For the 48 new cells, it must schedule repetition indices 0--39. The
final strict aggregation must contain 64 cells and 2,560 episodes.

This is an extension of the existing standardized study architecture, not a
new experiment runner or SLURM topology.

## 2. Non-goals

This feature must not:

- reinterpret scientifically incompatible episodes as matched observations;
- use array indices, directory names, or SLURM job IDs as scientific identity;
- overwrite or relocate valid source episodes;
- introduce a study-specific SLURM job file;
- change game, controller, persistence, provider, estimator, or aggregation
  mathematics;
- duplicate completed episodes merely to obtain a uniform directory layout;
- treat a retry of failed work as a new scientific extension;
- allow an analysis-only change to trigger provider calls.

## 3. User-facing workflow

Proposed command:

```bash
mas-cc study extend \
  --study-dir <existing-study-results> \
  --config-dir <target-study-configs>
```

The target config folder describes the complete desired scientific design, not
only its delta. This keeps YAML authoritative and makes the final target easy
to inspect.

Useful options:

```bash
mas-cc study extend \
  --study-dir <existing-study-results> \
  --config-dir <target-study-configs> \
  --dry-run

mas-cc study extend \
  --study-dir <existing-study-results> \
  --config-dir <target-study-configs> \
  --throttle 4
```

`--dry-run` must resolve compatibility and write or print the proposed delta
without submitting provider work. A real invocation performs the same
preflight again immediately before submission.

After all extension shards finish:

```bash
mas-cc study aggregate --study-dir <existing-study-results>
```

Strict aggregation evaluates the latest declared target, discovers all source
runs and extensions, deduplicates by scientific identity, and packages one
canonical handoff.

## 4. Fundamental identities

The implementation needs four identities with separate purposes.

### 4.1 Study lineage ID

A stable UUID or content-bound identifier created with the original study and
retained by every extension.

```text
study_lineage_id
```

This says that several submissions belong to one extensible scientific study.

### 4.2 Scientific protocol fingerprint

A canonical hash of all settings that must agree before observations can be
reused, including where applicable:

- game type and version;
- population size, topology, rounds, task/dataset identity, prompts, and
  response contract;
- controller mechanism and all fixed controller semantics;
- model/provider identity and response-affecting generation options;
- sensing, persistence, receiver, message, and evidence semantics;
- base seed and common-random-number policy;
- retention/schema versions required by configured analysis.

It excludes operational and target-size fields such as:

- SLURM resources, throttle, node count, paths, and job IDs;
- logging and observability destinations;
- pricing/budget ceilings;
- `execution.parallelism`;
- `execution.repetitions`;
- the grid value lists themselves;
- descriptive metadata such as expected cell/episode counts.

The exact included/excluded field registry must be explicit and versioned. It
must not be implemented as a loose collection of ad hoc dictionary deletions.

```text
protocol_fingerprint_version
protocol_fingerprint
```

### 4.3 Scientific cell key

A cell key identifies one resolved point in the scientific parameter space:

```text
cell_key = hash(protocol_fingerprint + canonical resolved swept coordinates)
```

Coordinates must use typed canonical values so that `1`, `1.0`, and the string
`"1"` cannot accidentally collide. The cell key is independent of grid order,
config filename, `cell-0007`, array index, and extension index.

The original execution-local cell ID remains provenance, but is not used to
decide whether two cells coincide.

### 4.4 Episode key

Within a cell:

```text
episode_key = (cell_key, repetition_index)
```

The episode seed, task ID, and retained scientific identity must agree with the
expected record for that key. A duplicate key with unequal content is a hard
validation error, not a last-write-wins situation.

## 5. Extension/run index for provenance

Use the proposed run index as a monotonic **extension index**:

```text
extension_index = 0  original submission
extension_index = 1  first enlargement of the target design
extension_index = 2  second enlargement
...
```

The index is useful provenance but is never a scientific coordinate.

Retries do not increment it. A retry belongs to the same extension index and
receives a separate `submission_attempt`:

```text
(extension_index=2, submission_attempt=0)  initial submission
(extension_index=2, submission_attempt=1)  safe retry of missing episodes
```

Allocate a new extension index atomically under a study-level lock. Never
reuse or renumber an allocated index, even if submission fails. Record failed
or abandoned attempts explicitly.

Suggested layout:

```text
<study-root>/
    study_lineage.json
    extensions/
        extension-0000/
            target_manifest.json
            delta_manifest.csv
            submissions/
                attempt-0000.json
            runs/
        extension-0001/
            target_manifest.json
            compatibility_report.json
            delta_manifest.csv
            execution_manifest.csv
            execution_plan.json
            submissions/
                attempt-0000.json
                attempt-0001.json
            runs/
    analysis/
```

Existing studies without this layout are treated as `extension-0000` through
a small migration/index manifest. Their source trees remain in place.

## 6. Compatibility rules

Before planning any provider calls, classify every target cell as one of:

```text
COMPLETE_REUSABLE
NEEDS_ADDITIONAL_EPISODES
NEW_CELL
INCOMPATIBLE
CONFLICTED
```

Reuse requires:

1. the same versioned scientific protocol fingerprint;
2. the same cell key;
3. valid canonical records and completion seals;
4. matching task/world and response-affecting provider configuration;
5. no duplicate episode-key conflict;
6. retained fields sufficient for the target analysis recipe.

`INCOMPATIBLE` means the observation belongs to a different scientific
protocol. `CONFLICTED` means two retained records claim the same episode key
but disagree in seed, identity, status, or content hash. Both conditions stop
submission until resolved; they must never silently trigger recomputation over
the existing data.

Changing only the analysis recipe is compatible and should invoke
reaggregation, not extension execution.

## 7. Repetition and seed contract

Increasing repetitions from `R_old` to `R_target` schedules the missing
repetition indices in `[0, R_target)`. It must not generate a second local
sequence beginning at zero.

For an existing cell, additional episode seeds must be produced using the
original study's recorded seed derivation contract and original cell seed
namespace. This preserves exact replay compatibility.

For `common_random_numbers_across_grid: true`, repetition `k` must retain the
same episode seed across every compatible cell, including newly introduced
cells.

For legacy non-CRN studies whose seed derivation depends on the original grid
index:

- existing cells retain their recorded legacy cell index/seed namespace;
- extensions must not reassign those indices when the grid is reordered;
- new cells receive a durable seed namespace recorded in the cell registry;
- the chosen legacy/new derivation version is explicit in provenance.

Going forward, non-CRN cells should use a stable cell-key-derived seed
namespace rather than grid position. This is a versioned seed-contract change
for newly created lineages, not a silent migration of old studies.

Every planned episode row should record at minimum:

```text
study_lineage_id
extension_index
submission_attempt
protocol_fingerprint
cell_key
source_config
resolved_coordinates
repetition_index
episode_seed
episode_key
expected_output_dir
status
```

## 8. Delta planner

The extension planner performs these steps without provider calls:

1. load the original lineage and every existing extension manifest;
2. discover and validate completed canonical episodes;
3. resolve the complete target configs and grid;
4. compute target protocol fingerprints and cell keys;
5. join target episode keys against retained valid episode keys;
6. reject incompatible or conflicted reuse;
7. create a delta containing only missing episode keys;
8. group missing episodes into cell or cell-bundle execution shards;
9. run ordinary preflight/budget/resource planning on the delta workload;
10. report reuse and new-work totals before submission.

Required pre-submission report:

```text
target cells / episodes
reused cells / episodes
partially reused cells
new cells
missing episodes to execute
incompatible/conflicted records
nominal, expected, and conservative provider calls for delta only
token/cost estimates for delta only
throttle, episode slots, request concurrency, target RPM, wall time
extension index and submission attempt
```

An empty delta is successful and submits no SLURM job. It may still update the
target manifest and request reaggregation when the analysis recipe changed.

## 9. Execution

Reuse the existing generic cell-array worker and
`scripts/Potsdam/SLURM/run_study_cell_array.job`.

The execution manifest must be enhanced to describe explicit repetition
indices per shard. A worker must no longer assume that selecting a cell implies
running `0..execution.repetitions-1`; it should run only the episode keys listed
in its delta assignment while retaining ordinary safe resume behavior.

One practical manifest shape is:

```csv
array_index,extension_index,config_path,cell_key,source_cell_id,
resolved_overrides,episode_plan_path,output_dir
```

`episode_plan_path` points to a compact CSV/JSON document containing the exact
repetition indices and episode seeds for that shard. Large per-episode lists
should not be embedded in one CSV field.

Provider coordination, array throttling, CPU allocation, RPM planning, and
bounded in-episode retry behavior remain unchanged and operate only over the
delta workload.

## 10. Discovery and aggregation

Discovery must read the lineage/extension manifests and treat all extension
run roots as sources for one target study.

Canonical tables should add compact provenance columns:

```text
source_extension_index
source_submission_attempt
source_run_id
source_cell_id
source_episode_id
cell_key
repetition_index
episode_seed
```

Aggregation must:

1. select the latest target manifest;
2. require exactly the target episode-key set in strict mode;
3. deduplicate byte/content-identical retries;
4. reject disagreeing duplicates;
5. preserve the physical cell as the estimator unit;
6. recompute estimator summaries from the combined canonical observations;
7. package small lineage and extension manifests as provenance;
8. continue excluding run trees, logs, caches, and resampling draws from the
   scientific handoff ZIP.

The final analysis ZIP remains a snapshot of the latest target. Prior analysis
ZIPs may be retained outside the canonical `analysis/` directory or named with
their target/extension index, but only one latest package should be advertised
as canonical.

## 11. Failure and concurrency safety

- Take a study-level planning lock while allocating an extension index and
  publishing its target/delta manifests.
- Refuse two concurrently active extensions of the same lineage unless a
  future explicit merge protocol exists.
- Safe retries use the same extension index, episode keys, and output roots.
- A failed episode remains missing; it is not counted merely because its SLURM
  task exited successfully.
- Strict aggregation must wait for every target episode key to be complete.
- An interrupted planner must leave either no published extension or a
  clearly marked `PLANNED`, `SUBMISSION_FAILED`, or `ABANDONED` record.
- Never delete prior successful observations during extension.

## 12. Migration of existing standardized studies

Provide an inspection-first command or internal migration step:

```bash
mas-cc study index-existing --study-dir <existing-study-results> --dry-run
```

It should:

1. validate current manifests and canonical identities;
2. assign `study_lineage_id` and `extension_index=0`;
3. build cell and episode registries from existing source records;
4. record the legacy seed derivation version and original cell indices;
5. write only small index/provenance manifests;
6. leave all existing run and analysis data untouched.

This may also be performed automatically on the first `study extend`, but the
dry-run report must be available before any mutation or provider call.

## 13. Implementation map

Expected changes:

| Area | Responsibility |
|---|---|
| `src/mas_cc/cli/main.py` | Add `study extend` and optional indexing command |
| `src/mas_cc/studies/manifest.py` | Load target design and lineage manifests |
| `src/mas_cc/studies/identity.py` (new) | Versioned protocol fingerprints, cell keys, episode keys |
| `src/mas_cc/studies/extension.py` (new) | Compatibility join, delta planning, extension allocation |
| `src/mas_cc/studies/execution.py` | Explicit per-shard episode plans |
| `src/mas_cc/studies/cell_worker.py` | Execute selected episode indices only |
| `src/mas_cc/studies/discovery.py` | Discover original and extension sources |
| `src/mas_cc/studies/canonical.py` | Retain extension/run provenance columns |
| `src/mas_cc/studies/validation.py` | Validate target episode-key coverage and conflicts |
| `src/mas_cc/studies/aggregation.py` | Aggregate latest target across extensions |
| `tests/mas_cc/` | Identity, planning, execution, migration, and aggregation tests |

Do not change the generic SLURM launcher unless the existing manifest argument
cannot carry the enhanced execution plan. No scientific estimator module needs
to change.

## 14. Test plan

### Identity and compatibility

1. Grid reordering produces identical cell keys.
2. Config filename and SLURM array changes do not affect scientific identity.
3. Typed coordinate differences do not collide accidentally.
4. Operational resource changes remain compatible.
5. Model, prompt, task, rounds, controller semantics, or persistence changes
   are rejected as incompatible unless they are explicit target axes with new
   cell keys under an otherwise compatible protocol.
6. Fingerprint schema/version changes are explicit.

### Repetition extension

7. Extending 20 repetitions to 40 schedules exactly indices 20--39.
8. Holes such as `{3, 11, 17}` schedule exactly those indices.
9. Existing episode seeds are unchanged.
10. New CRN episode seeds match across cells.
11. No duplicate episode key is submitted.

### Grid extension

12. A 4x4 to 8x8 extension reuses the 16 coincident cells.
13. With 20 to 40 repetitions, the delta is exactly 2,240 episodes:
    `16*20 + 48*40`.
14. Grid order changes do not cause recomputation.
15. A genuinely new coordinate produces a new cell key and complete episode
    plan.

### Provenance and retries

16. Original work is `extension_index=0`.
17. A scientific enlargement increments the extension index once.
18. A safe retry increments only `submission_attempt`.
19. Every canonical row resolves to its source extension/run/cell/episode.
20. Concurrent extension allocation is locked and deterministic.

### Aggregation

21. Strict aggregation requires the latest complete target episode-key set.
22. Original plus extension observations reproduce a from-scratch equivalent
    fixture within established numerical tolerances.
23. Identical retry duplicates are deduplicated; disagreeing duplicates fail.
24. Estimators remain grouped by physical cell.
25. No persistent cache, raw null/bootstrap draws, run trees, or logs enter the
    package.
26. Reaggregation from retained canonical observations remains supported.

### Failure cases

27. Incompatible protocol changes fail before provider calls.
28. Insufficient retained scientific fields fail with a clear explanation.
29. Missing or corrupt completion seals are not counted as reusable episodes.
30. An empty delta submits no job.

## 15. Delivery phases

### Phase 1: identity and inspection

- implement versioned protocol/cell/episode identity;
- index existing studies without moving data;
- produce compatibility and delta reports;
- add comprehensive dry-run tests.

### Phase 2: repetition-only extension

- support explicit episode-index plans for existing cells;
- preserve original seed namespaces;
- submit missing repetitions through the generic cell-array launcher;
- aggregate original plus extension outputs.

This phase delivers the most common request with the lowest identity risk.

### Phase 3: grid extension

- add new-cell planning and durable seed namespaces;
- support coincident-cell reuse across reordered/enlarged grids;
- validate 4x4-to-8x8 behavior end to end.

### Phase 4: operational hardening

- locks, interrupted-submission recovery, retry attempts, migration tooling;
- documentation and Potsdam smoke test with a mock provider;
- downscaled real-provider validation only after explicit authorization.

## 16. Acceptance criteria

The feature is complete when:

1. an existing complete study can be extended to a larger repetition target
   without rerunning any valid episode;
2. an enlarged grid reuses all scientifically coincident cells independent of
   grid order;
3. deterministic seeds and CRN matching remain correct;
4. every retained observation has original/extension/submission provenance;
5. retries do not masquerade as scientific extensions;
6. strict aggregation validates the latest complete target and produces one
   canonical package;
7. numerical results match an equivalent clean from-scratch fixture;
8. no study-specific launcher or estimator implementation is introduced.

## 17. Recommended design decision

Adopt the extension index, but use it only as an immutable provenance axis.
Scientific reuse must be decided by the versioned protocol fingerprint,
scientific cell key, and repetition index. This gives the convenient run
history the user suggested without coupling scientific meaning to execution
order.
