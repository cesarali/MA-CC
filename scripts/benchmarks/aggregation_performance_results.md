# Aggregation performance implementation — 2026-09-14

Implements the bootstrap-first proposal in
`docs/tdd/features/orchestrator/14092026_TDD_bootstrap_first_aggregation_performance_optimization.md`.
The baseline below already includes the previously merged compact epistemic
bootstrap. These measurements therefore quantify the additional improvements,
not the historical improvement from the cancelled 79-minute finalizers.

## Retained-study measurements

Potsdam `MA-CC` environment, same frozen canonical inputs and completed
information fragments. Baseline: commit `66a946f` plus the frozen working-tree
changes present at the start of this
implementation, including the existing fragment-hash correction. Each run used
one compute node; baseline allocations had two CPUs (the finalizer used one),
optimized allocations had four CPUs and 16 GiB. BLAS/OpenMP used one thread per
process. Timings are single measurements, not a statistical performance claim.

| Study | Rounds | Micro-slots | Baseline | Optimized | Speedup | Optimized parent peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Potsdam q=3, complete | 16,200 | 388,800 | 839.73 s | 340.93 s | 2.46× | 7.51 GiB |
| DeepInfra q=12, incomplete | 12,270 | 294,480 | 694.98 s | 317.34 s | 2.19× | 6.59 GiB |

SLURM batch MaxRSS was 7,907,112 KiB and 6,971,292 KiB respectively, both below
the 16 GiB allocation. Parent RSS is almost unchanged from the compact-bootstrap
baseline (7.52 and 6.64 GiB); this change primarily improves wall time.

| Stage | q=3 baseline → optimized (s) | q=12 baseline → optimized (s) |
| --- | ---: | ---: |
| Joint efficiency/derived estimation | 506.97 → 165.72 | 412.76 → 144.41 |
| Causal-input construction | 22.85 → 6.40 | 17.99 → 5.16 |
| Symbolic round-state construction, including robustness | 146.39 → 89.86 | 122.04 → 86.82 |
| Joint epistemic drift | 21.35 → 4.31 | 19.76 → 6.10 |
| Epistemic surface/modulation | 23.44 → 5.87 | 20.98 → 7.26 |
| Capture timing | 35.22 → 2.34 | 27.10 → 1.72 |
| Complete epistemic analysis | 245.06 → 103.53 | 205.11 → 103.22 |
| ZIP creation | 3.63 → 3.70 | 4.45 → 4.70 |

Nested stage timings must not be summed. Packaging was not a material timing
bottleneck; streaming bounds its temporary memory without claiming a speedup.

## Algorithms and invariants

- Symbolic robustness uses deterministic lookup tables for intersections of up
  to eight fact masks. Draws use the same Bernoulli survival matrices and the
  same round-identity seeds. Arbitrary-width Python integer masks preserve all
  worlds. No Monte Carlo answer is shared across states.
- Joint efficiency bootstrap precomputes four microscopic transition counts
  per episode. Each existing episode selection still chooses the rounds and
  counts together. All other ingredients are recomputed from the same round
  sequence; joint dependence and percentile intervals are unchanged.
- Independent physical-cell efficiency bundles use a bounded spawn process
  pool. Each sorted cell retains its original `seed + index`; completion order
  does not affect output ordering or scientific identity.
- Epistemic drift, modulation, occupancy and capture share one narrow prepared
  causal table. When causal communication already prepared that table, the
  epistemic stage joins its validated scalar fields by canonical round identity.
- Causal identity iteration, micro communication audits and affinity inputs
  project only the fields they consume. Full canonical fields remain in the
  published tables, so initial canonical loading still reads all required data.
- ZIP writing streams file contents. Stage timings, RSS and robustness counts
  are compact provenance; no draws, deterministic lookup tables, profiler traces
  or generation work directories enter the archive.

## Bootstrap inventory and conditional work

| Family | Sampling unit | Disposition |
| --- | --- | --- |
| MI/CMI | episode within physical cell | Existing parallel information workers unchanged; fragments reused |
| Causal response | shared initialization block | Existing compact plan retained; input construction optimized |
| Epistemic drift/modulation | shared initialization block | Compact implementation verified; prepared inputs reused |
| Current | episode within physical cell | Retained; not a measured bottleneck in these recipes |
| Affinity/compliance and thermodynamic efficiency | jointly sampled whole episodes | Exact transition-count summaries; physical-cell parallelism |
| Derived study aggregates | configured physical-cell/block semantics | Retained; not enabled in these measured recipes |

No additional robustness SLURM array, plot worker pool or estimator abstraction
was added: the four-CPU finalizer meets the proposal's under-ten-minute stretch
target. There is no change to simulation, provider calls, estimands, support
rules, seeds, bootstrap counts, null counts or scientific grouping.

## Validation

- Full baseline comparisons check all 39 scientific Parquet tables in each
  study, including canonical observations, CI endpoints, support flags and
  partial-study outputs, at `rtol=1e-10`, `atol=1e-12`.
- Archive membership and ZIP integrity are checked against the baseline.
- Focused tests verify literal versus lookup solvability, large integer masks,
  missing facts, contradictory masks, joint micro-bootstrap interval equality,
  one versus two workers, prepared versus independently built causal inputs,
  batch-size independence, streamed packaging and failure-stage visibility.
- The combined study/SLURM/causal/epistemic/affinity suite passed 113 of 120
  tests. Seven pre-existing failures reference absent legacy Study 06/07/08
  directories and the relocated controlled-smoke config. Subsequent targeted
  tests cover the additional batch-size and progress changes.
- The existing 4,000-row, 1,000-resample literal-bootstrap benchmark passed:
  178.09 s → 1.61 s, 525.16 → 145.52 MiB. A 500-row, three-cell, 13-block,
  137-resample variant also passed: 30.76 s → 0.45 s.
- The robustness kernel benchmark checked 100 × 500 identical survival draws:
  0.767 s → 0.216 s (3.55×), with exact Boolean results.

## Operations and reproduction

The command remains `mas-cc study aggregate --study-dir <study>`. The generic
SLURM launcher now honors the finalizer CPU allocation and establishes the
Potsdam repository working directory. Defaults are up to four finalizer CPUs
(bounded by cell count), 16 GiB, and the existing time limit. An explicit recipe
`slurm.finalizer_cpus`, `slurm.finalizer_memory`, or `slurm.finalizer_time_limit`
overrides those values; the existing `slurm.memory` remains a memory fallback.

Progress identifies long substages and writes a heartbeat every 30 seconds.
Detached finalization also updates the generation's advertised progress path.
Successful publication removes transient progress/work; failed finalization
retains the active stage and valid information fragments. Parent and completed
child RSS are reported separately; SLURM provides allocation-level accounting.
The manifest's packaging timing includes content compression up to the final
manifest write, excluding ZIP close/rename overhead.

Run these harnesses in the dedicated project environment and use a compute-node
allocation for full studies:

```bash
python scripts/benchmarks/benchmark_aggregation.py <execution_manifest.json> <new-output-directory>
python scripts/benchmarks/compare_aggregation.py <baseline-analysis> <candidate-analysis>
python scripts/benchmarks/benchmark_epistemic_robustness.py --tasks results/studies/musr_truthful_selective_task_calibration_01/tasks
```

Set `PYTHONPATH` to a frozen source tree to measure a baseline. Operational
measurements and comparison logs are retained outside the scientific packages
under `/work/ojedamarin/Projects/LanguageGames/MA-CC/aggregation-performance-20260914/`.
Baseline jobs: 1876685/1876686; final implementation benchmarks: 1876690/1876691.
After equivalence checks, the existing generic finalizer refreshed the q=12
archive in job 1876697 and the q=3 archive in job 1876699, reusing all 15 and 18
information fragments respectively. The q=12 result retains its explicitly
incomplete status. Publication checks compare every Parquet file's SHA-256 to
the validated candidate, verify ZIP integrity and manifest agreement, and
confirm removal of transient work.
No study-specific job file or replacement CMI implementation was introduced.
