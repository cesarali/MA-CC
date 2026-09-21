# Epistemic bootstrap benchmark — 2026-09-14

The compact implementation replaces duplicated bootstrap DataFrames with
shared initialization-block multiplicities, evaluated in bounded batches.
Score means use block sums and nonmissing counts. Regression uses square-root
row multiplicities and SVD least squares, retaining the duplicated-row rank
cutoff and the existing condition-number threshold.

## Local measurement

Synthetic prepared causal inputs: 4,000 rows, 2 cells, 30 shared initialization
blocks, 3 target-share bins, 2 epistemic bands, and 1,000 bootstrap replicates.
Both implementations run in fresh processes with one BLAS/OpenMP thread.
Python 3.11.15, NumPy 2.4.6, pandas 3.0.5; Linux peak RSS via `getrusage`.
Baseline: `a6300e1e7ee453f719cc1bb0cd3250089a23c6cb`.

| Measurement | Baseline | Compact |
| --- | ---: | ---: |
| Joint drift | 63.435 s | 0.125 s |
| Surface and modulation regression | 11.969 s | 0.264 s |
| Combined | 75.404 s | 0.390 s |
| Peak process RSS | 528.2 MiB | 148.1 MiB |

This single local measurement gives a 193× combined speedup and a 72% reduction
in peak process RSS. It is not a whole-study speedup estimate. Task loading,
symbolic robustness, causal-input preparation, plots, and packaging are excluded.
All three output tables match (`rtol=1e-10`, `atol=1e-12`), including confidence
intervals, identification flags, and support diagnostics.

## Reproduce

From the repository root, using the existing project Python environment:

```bash
git show a6300e1e7ee453f719cc1bb0cd3250089a23c6cb:src/mas_cc/analysis/epistemic_phase.py > /tmp/epistemic_phase_baseline.py
python scripts/benchmarks/benchmark_epistemic_bootstrap.py --baseline /tmp/epistemic_phase_baseline.py
```

The harness accepts `--rows`, `--cells`, `--blocks`, and `--resamples`. It checks
output equivalence and deletes its temporary output tables after comparison.
The baseline module is executable Python; use a trusted local source file.

## Validation and operational scope

- All 11 focused epistemic tests pass, including literal-bootstrap equivalence
  for unequal/shared blocks, missing scores, empty groups, zero resamples, and
  singular/ill-conditioned regressions, plus offline aggregation integration.
- The broader study test run has seven failures caused by absent legacy config
  paths (Study 06/07/08 and related smoke configs).
- Aggregation now exposes epistemic substage counts and elapsed times through
  its existing transient `analysis/progress.json`. Updates are throttled to two
  seconds, except stage transitions and completions; progress advances at round
  or estimator-group boundaries.
- Bootstrap and robustness draw defaults, seeds, scientific grouping, and
  estimator version are unchanged. Bootstrap draws remain transient. No
  study-specific launcher or replacement CMI estimator was introduced.
- Robustness optimization and CPU parallelism are deferred; this change targets
  bootstrap memory, repeated filtering, weighted regression, and visibility.
- No cluster jobs were restarted and no provider calls were made.
