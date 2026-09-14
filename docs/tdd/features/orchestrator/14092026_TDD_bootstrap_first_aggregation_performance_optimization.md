# TDD: Bootstrap-first standardized aggregation performance optimization

**Date:** 2026-09-14  
**Status:** implemented and benchmarked on retained q=3/q=12 studies, 2026-09-14  
**Scope:** standardized MA-CC study aggregation and post-processing  
**Primary command:** `mas-cc study aggregate --study-dir <study>`

Implementation evidence and operational settings are recorded in
[`aggregation_performance_results.md`](../../../../scripts/benchmarks/aggregation_performance_results.md).
Measured finalization fell from 14.0 to 5.7 minutes for q=3 and from 11.6 to
5.3 minutes for q=12. All 39 scientific tables and archive membership match
the baseline for each study. The design below remains the rationale; optional
extra SLURM stages were unnecessary after the measured improvements.

## 1. Outcome

Make complete Study-09-style aggregation predictably fast and memory-bounded
while preserving one command, one canonical analysis tree, and one final ZIP.

The scientific definitions, physical-cell boundaries, bootstrap laws, null
procedures, seeds, support rules, estimator variants, tables, plots, reports,
and package contract must remain unchanged. This is an execution and data-flow
optimization only.

The implementation should optimize in this order:

1. measure the complete pipeline on real retained studies;
2. finish the bootstrap conversion and verify it end to end;
3. optimize symbolic epistemic robustness;
4. remove repeated wide-table construction and copying;
5. parallelize only the remaining independently reproducible stages;
6. keep the final merge/report/package step small and deterministic.

## 2. Evidence motivating the work

Two finalizers were observed on 2026-09-14:

| Study | Canonical rounds | Canonical micro-slots | Runtime before cancellation | RSS before cancellation |
| --- | ---: | ---: | ---: | ---: |
| Potsdam q=3 false control | 16,200 | study canonical table | 1 h 19 min | about 58 GiB |
| DeepInfra q=12 false control | 12,270 | 294,480 | 1 h 19 min | about 45 GiB |

Both had already completed their SLURM-parallel information-estimator groups.
Their finalizers completed causal-response validation and then remained inside
the epistemic phase analysis. They were cancelled without publishing, and the
completed information fragments remained reusable.

The newly merged `_BlockBootstrap` implementation in
`src/mas_cc/analysis/epistemic_phase.py` replaces 1,000 duplicated bootstrap
DataFrames with bounded batches of shared-initialization multiplicities. The
checked-in isolated benchmark reports:

- 75.404 s to 0.390 s for drift plus modulation;
- approximately 193x speedup;
- 528.2 MiB to 148.1 MiB peak RSS;
- numerical agreement at `rtol=1e-10`, `atol=1e-12`.

This benchmark is strong evidence, but it excludes symbolic state building,
robustness Monte Carlo, canonical loading, causal-input preparation, plotting,
and packaging. A whole-study measurement is therefore the first required step.

Reference benchmark:

`scripts/benchmarks/epistemic_bootstrap_results.md`

## 3. Non-negotiable invariants

- The scientific cell remains the physical estimation unit.
- A scheduler shard, worker, process, or array index is not a scientific unit.
- Shared/common-random-number initialization blocks remain shared across cells.
- Bootstrap sampling remains at the currently defined episode or shared-block
  level for each estimator.
- Deterministic seeds must not depend on worker count, scheduling order, node,
  process ID, job ID, or completion order.
- MI, CMI, causal response, susceptibility, epistemic modulation, robustness,
  currents, effective affinity, compliance, and efficiency definitions remain
  authoritative and unchanged.
- Null permutations and bootstrap resamples retain their configured counts.
- Unsupported states remain unsupported; optimization must not fill or smooth
  unvisited regions.
- `--allow-incomplete` and strict aggregation retain their current meanings.
- Canonical `cells`, `episodes`, `rounds`, and `micro_slots` tables remain the
  source of truth.
- The final package remains compact: no raw bootstrap draws, null draws,
  persistent caches, or duplicated run trees.
- The user still invokes one aggregation command and receives one analysis
  directory and one ZIP.
- Temporary resumable fragments may exist only under the current generation's
  `analysis/.work/<generation-id>/` and are deleted after successful publish.
- Existing Potsdam environment and generic SLURM-launch rules remain in force.

## 4. Performance questions to answer first

The next implementation should not guess which stage is slow. It should obtain
the following real-study profile for both a complete Potsdam study and the
incomplete DeepInfra q=12 study:

| Stage | Required measurements |
| --- | --- |
| Canonical discovery/read | elapsed time, projected columns, input bytes, RSS delta |
| Information fragment validation/merge | groups, rows, elapsed time, RSS delta |
| Primary/current/affinity estimators | per-estimator elapsed time and peak RSS |
| Causal response | input construction, bootstrap, available susceptibility, micro audit |
| Epistemic state construction | inventory reconstruction, solver calls, unique states |
| Robustness Monte Carlo | states evaluated, draws, solver-mask calls, cache/dedup ratio |
| Epistemic bootstrap | drift, surface, regression elapsed time and bounded-batch size |
| Derived/state-map construction | table rows, joins, elapsed time, RSS delta |
| Plotting/reporting | figures, elapsed time, RSS delta |
| Parquet writing/ZIP | files, compressed bytes, elapsed time, RSS delta |

Use the existing transient progress mechanism, extending its substages where
needed. Record concise stage timing and peak-RSS measurements in
`analysis_manifest.json`; do not persist profiler traces in the final package.

Acceptance for this step:

- every stage above is visible in progress/status;
- a stalled finalizer identifies its active substage;
- measurements add negligible overhead when detailed profiling is disabled;
- no scientific table changes.

## 5. Phase A: validate the merged compact epistemic bootstrap

The merged `_BlockBootstrap` is the correct first optimization direction. It
must be validated against the literal historical implementation before further
refactoring.

Required checks:

1. Run the checked-in synthetic benchmark at several row, block, cell, and
   resample counts, including unequal block sizes and missing scores.
2. Compare all drift, susceptibility-surface, and modulation-regression fields,
   not only point estimates.
3. Verify shared block selections are identical for equal seeds.
4. Verify results do not depend on bootstrap batch size.
5. Run a complete retained-study comparison using a frozen pre-optimization
   implementation and canonical inputs.
6. Measure finalizer wall time and peak RSS after the change.

The compact implementation should become the common pattern for other
bootstrap code, but not a universal estimator abstraction until equivalence is
demonstrated estimator by estimator. Different analyses currently use different
sampling units and must not be silently homogenized.

Target:

- epistemic drift/modulation memory proportional to observations plus a bounded
  draw batch, not observations multiplied by resamples;
- no full resampled DataFrame retained per bootstrap draw;
- output equivalence within the established `rtol=1e-10`, `atol=1e-12` unless a
  stricter existing test applies.

## 6. Phase B: optimize symbolic epistemic robustness

After bootstrap optimization, the likely next major cost is
`build_epistemic_round_states()` and `_robustness()`.

For each solvable pre-intervention round state, the current recipe may evaluate:

- 500 configured-persistence survival draws; and
- 500 reference-persistence survival draws.

For 12,270 round states this can approach 12.27 million mask-solvability checks;
for 16,200 states it can approach 16.2 million checks, reduced by unsolvable and
rho=1 shortcuts.

Implement only optimizations that preserve the existing random experiment:

### B1. Inventory signatures and diagnostics

- Encode each active union and holder-count vector as a stable compact
  signature.
- Report total states, unique inventory signatures, immediately unsolvable
  states, rho=1 shortcuts, and actual Monte Carlo calls.
- Use these counts to determine whether deduplication is worthwhile.

### B2. Vectorized bit-mask evaluation

- Retain the same generated Bernoulli survival matrix and seed per round state.
- Convert surviving fact sets to compatible-world masks using vectorized or
  compiled bit operations rather than Python loops over every draw and fact.
- Preserve exact success counts for identical random survival matrices.

### B3. Safe reuse boundary

Do not cache one Monte Carlo answer merely because two states have the same
inventory. Current seeds include round identity, so such reuse would change the
randomization. Reuse is allowed only for deterministic components such as fact
mask lookup, holder counts, solvability of the unthinned state, and compiled
inventory representation. Any common-random-number redesign requires a
separate scientific decision and is outside this optimization.

### B4. Bounded CPU parallelism

If vectorization is insufficient, evaluate independent round-state robustness
chunks with a bounded process pool or SLURM stage. Seed each state from its
existing stable identity before dispatch. Sort outputs by canonical keys before
merging.

Prefer one multi-CPU finalizer allocation for this stage before introducing a
new scheduler layer. Escalate to a SLURM array only if measurements show a clear
wall-time benefit.

## 7. Phase C: remove repeated wide-table work

Several stages reconstruct similar causal and epistemic views from the same
canonical rounds. The optimized path should:

- project only required Parquet columns for each stage;
- avoid `rounds.copy()` over wide nested/object columns when only scalar
  estimator inputs are required;
- build common causal columns (`U_t`, propensity, target shares, lags,
  initialization block) once per generation;
- pass a narrow immutable prepared table to causal response, joint drift, and
  modulation rather than calling `build_causal_response_inputs()` repeatedly;
- reduce repeated `cells.to_dict(orient="records")` and repeated coordinate
  merges;
- keep micro-slot communication audits on a narrow projection containing only
  identity and exposure/acquisition fields;
- release large intermediate frames after their final consumer.

Prepared tables are transient invocation products. They may be written as
generation-local fragments for recovery, but they must not enter the final ZIP
or become persistent caches.

Acceptance:

- no estimator receives a different row set;
- canonical ordering and identity checks remain intact;
- peak RSS is measured before and after each projection;
- reaggregation from canonical tables alone still works.

## 8. Phase D: inventory and convert remaining bootstraps

Create a bootstrap inventory with one row per implementation:

| Estimator family | Sampling unit | Current representation | Candidate optimization |
| --- | --- | --- | --- |
| Information MI/CMI | episode within physical cell | per-cell worker-local draws | retain; already SLURM-parallel |
| Causal response | shared initialization block | compact block-count plan | retain and benchmark |
| Epistemic drift/modulation | shared initialization block | bounded `_BlockBootstrap` | validate and reuse pattern |
| Current | episode within physical cell | small per-cell bootstrap | vectorize only if measured |
| Effective affinity/compliance | episode within physical cell | small count summaries | vectorize count sampling if measured |
| Derived study aggregates | configured physical-cell/block semantics | transient component structures | replace stored draw objects with compact arrays if measured |
| Thermodynamic bundles | whole-episode joint bundle | joint bootstrap call | retain joint dependence; optimize internally only |

For each candidate, first add a literal-reference equivalence test. Replace
DataFrame replication with block multiplicities or sufficient-statistic arrays
only when the estimator can be expressed exactly that way. Do not combine
separately bootstrapped intervals when the established implementation uses a
joint bootstrap.

Raw draw arrays remain transient and should be freed immediately after compact
CI/null summaries are formed.

## 9. Phase E: make finalization a small dependency graph

The information estimator already uses:

```text
prepare -> per-cell information array -> finalizer -> atomic publish
```

If whole-study profiling still shows a long finalizer, extend the same generic
generation architecture, without exposing extra commands to the user:

```text
canonical preparation
      |
      +--> information groups by physical cell
      +--> causal/epistemic prepared-state groups
      +--> optional robustness groups by stable cell/chunk
      |
validated compact reducers
      |
single report/plot/package finalizer
      |
atomic analysis/ publication
```

Rules:

- parallel tasks own disjoint hash-addressed fragment paths;
- each fragment records generation identity, scientific cell/chunk identity,
  recipe hash, estimator version, seed specification, row count, and checksum;
- reducers validate every fragment before use;
- global/shared-block analyses preserve one deterministic draw plan across
  fragments;
- no task writes final tables or plots concurrently;
- retry submits only missing/invalid operational fragments;
- success removes the transient generation tree;
- package contents remain unchanged.

Do not create one CPU job per tiny table. Bundle work so scheduler overhead is
small relative to computation. The first implementation should support a
multi-CPU finalizer and only add new array stages when real measurements justify
them.

## 10. Phase F: output and packaging efficiency

After computation is fast, measure output costs separately:

- write each authoritative Parquet table once;
- avoid reconstructing the same coordinate-enriched table for multiple plots;
- render independent plots with a bounded local worker pool only if plotting is
  a measured bottleneck and Matplotlib process isolation is reliable;
- stream ZIP creation from final files instead of loading file contents into
  memory;
- preserve deterministic package membership and validation;
- never include `.work`, caches, draws, logs, run trees, or duplicate CSV
  mirrors.

Packaging optimization must not weaken atomic publication: the currently
published analysis remains readable until the new validated generation and ZIP
are complete.

## 11. Testing strategy

### Numerical equivalence

- literal versus compact bootstrap for every converted estimator;
- identical point estimates, CI endpoints, support status, counts, and
  identification flags within established tolerance;
- identical configured seeds and resample counts;
- equal results across batch sizes and worker counts;
- complete and `--allow-incomplete` fixtures;
- shared-block studies with unequal/missing cells;
- Study-09-like q=3 and q=12 fixtures.

### Operational correctness

- progress identifies every major substage;
- cancellation leaves canonical inputs and completed valid fragments reusable;
- retry does not rerun completed information groups;
- fragment hash/identity validation rejects stale generations;
- final publication is atomic;
- successful publication removes transient work;
- no bootstrap/null draws or caches enter the final package;
- aggregation from retained canonical tables reproduces results.

### Performance regression tests

Keep deterministic benchmarks outside ordinary unit-test timing assertions.
CI should verify structural memory properties, such as bounded draw batches and
absence of a list of full resampled DataFrames. Record benchmark results for:

- 4,000-row synthetic input;
- one medium retained fixture;
- one full Study-09-like retained study on Potsdam.

## 12. Success criteria

For a Study-09-scale package with 12,000--16,000 rounds and 1,000 resamples:

- epistemic drift/modulation completes in minutes rather than more than one
  hour;
- finalizer peak RSS remains comfortably below 16 GiB unless profiling proves
  another scientific stage intrinsically requires more;
- complete finalization after reusable information fragments targets less than
  20 minutes on one node, with a stretch target below 10 minutes;
- progress updates at least every two minutes during long substages;
- the finalizer can be cancelled and resumed without repeating completed
  per-cell information resampling;
- all final scientific outputs remain numerically equivalent;
- the final ZIP contract remains unchanged.

These are engineering targets, not reasons to reduce bootstrap/null counts or
alter scientific estimators.

## 13. Recommended implementation order

1. Run the newly merged compact-bootstrap benchmark and focused tests in the
   target environment.
2. Relaunch one cancelled finalizer using the reusable information fragments
   and collect complete substage timing/RSS evidence.
3. Profile symbolic robustness using counts and timings, not full retained
   traces.
4. Implement vectorized mask evaluation and deterministic preparation reuse.
5. Project narrow canonical columns and reuse causal prepared inputs.
6. Inventory and convert only measured remaining DataFrame-replicating
   bootstraps.
7. Add bounded multi-CPU execution for robustness or other proven hotspots.
8. Introduce additional SLURM fragment stages only if a single multi-CPU
   finalizer still misses the wall-time target.
9. Verify full numerical/package equivalence and regenerate both cancelled
   study archives.

## 14. Explicit non-goals

- reducing bootstrap or null-permutation counts;
- changing confidence levels or random seeds;
- changing MI/CMI or causal/epistemic estimator mathematics;
- pooling heterogeneous scientific cells;
- adding persistent analysis caches;
- adding a legacy/heavy package mode;
- requiring the user to manually finalize an aggregation;
- changing simulation, controller, game, provider, or checkpoint behavior;
- coupling scientific analysis to Potsdam, DeepInfra, SLURM, or Kubernetes.

## 15. Handoff checklist for the implementing agent

Before changing code, the implementing agent should read:

- `.codex/skills/ma-cc-study-workflow/SKILL.md`;
- `AGENTS.md`;
- `docs/tdd/features/orchestrator/22082026_TDD_standardized_study_submission_and_aggregation.md`;
- `docs/tdd/features/orchestrator/11092026_TDD_single_command_slurm_parallel_information_resampling.md`;
- `scripts/benchmarks/epistemic_bootstrap_results.md`;
- `src/mas_cc/analysis/epistemic_phase.py`;
- `src/mas_cc/analysis/causal_response.py`;
- `src/mas_cc/studies/aggregation.py`;
- `src/mas_cc/studies/analysis_slurm.py`.

The implementing agent should report:

1. measured baseline by stage;
2. exact algorithms optimized;
3. numerical-equivalence evidence;
4. before/after wall time and peak RSS;
5. SLURM resources used;
6. tests and benchmarks run;
7. confirmation that the final analysis tree and ZIP contract did not change.
