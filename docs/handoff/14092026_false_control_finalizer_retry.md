# False-control finalizer retries — 2026-09-14

For aggregation on another cluster, see the
[portable input bundle handoff](14092026_portable_false_control_aggregation_inputs.md).

Both finalizers failed during Parquet table writing after calibration:
`ArrowTypeError: Expected bytes, got a bool object`, column `target_semantics`.
Legacy rho views retained YAML boolean `False`, while derived weighted views
used string `false`. Their coordinate merge failed to match equivalent labels,
then concatenated both types into one report column.

`weighted_rho_aliases` now uses the existing semantic-label normalizer on both
inputs before matching report coordinates. It preserves missing labels and
does not mutate source tables or change scientific estimators/resampling.

Verification:

- 25 tests passed across `test_weighted_summaries.py` and
  `test_derived_study_aggregation.py`.
- Regression coverage includes boolean/string labels, no duplicate bins,
  missing-bin preservation, Parquet round-trip, and full offline aggregation
  using boolean false with rho views enabled.
- Both failed jobs' saved tables reconstructed successfully: 160/192 state
  map rows and 20/24 descriptive estimator rows, unique coordinates, valid
  Arrow conversion. All 15/18 completed information fragments validated.

Only finalizers were resubmitted, using the existing generic analysis launcher:

| Study | Generation | Failed job | Retry job |
|---|---|---|---|
| astra_task003_false_control_q12_deepinfra_rho3 | 22d854a7ece3da79fa41 | 1876736 | 1878067 |
| astra_task003_false_control_30x30_potsdam_rho3 | 6017cf539f77f907945f | 1876753 | 1878068 |

Both retries retain 4 CPUs, 16 GB, six-hour allocation, frozen scientific
inputs/recipes, and 1000 bootstrap resamples. Generation manifests reference
the retry job IDs. The completed per-cell information analysis is reused;
in-memory finalizer/calibration work must recompute. No provider calls or
simulation reruns. DeepInfra false remains explicitly incomplete/provisional;
Potsdam false remains strict.

Study roots are under
`/work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/`.
Logs: `<study>/logs/analysis-<generation>/finalize-<retry-job>.{out,err}`.
Progress: `<study>/analysis/.work/<generation>/progress.json` until publication.
At handoff the retry jobs are submitted, not confirmed finished. The old
published ZIPs remain authoritative only for their older analysis generation
until successful atomic publication replaces them.
