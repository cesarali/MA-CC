# Potsdam execution and university-provider profile

Read this reference when commands will run on the Potsdam host or SLURM
cluster, or when a config uses the university provider.

## Compute-site rules

- Use `/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC` for
  every Python, `mas-cc`, pytest, preflight, submission, aggregation, and
  post-processing command on Potsdam.
- Before a real submission, perform a credential-free import check for
  `mas_cc`, `pandas`, and `pyarrow` in that environment.
- Use the generic launchers
  `scripts/Potsdam/SLURM/run_config_array.job` and
  `scripts/Potsdam/SLURM/run_study_cell_array.job`. Do not create a
  study-specific job file unless scheduler topology genuinely differs.
- Keep scientific results and SLURM logs under
  `/work/ojedamarin/Projects/LanguageGames/MA-CC/results`. Put the absolute
  destination in `study.yaml`, guard it with `require_results_under`, and
  verify every generated manifest output path.
- The launcher must establish
  `/home/ojedamarin/Projects/LanguageGames/MA-CC` as its runtime working
  directory so repository-relative inputs and local provider configuration
  resolve consistently. Output location and runtime working directory are
  separate.
- Use explicit CPU, memory, time, array throttle, and absolute log paths.
  Confirm conservative shard duration fits the requested time limit.

These absolute paths are specific to Potsdam. Do not reproduce them on another
cluster.

## University-provider rules

The adapter type is `university`. It resolves credentials from
`POTSDAM_API_KEY` and its endpoint from `BASE_POTSDAM_LLM_URL`. Never print,
copy into YAML, or commit either value.

Query live model availability, model information, pricing/accounting units,
RPM, and TPM during preflight. Dated documentation and previous execution
plans are comparison anchors, not permanent limits.

For the current Task003 q=3 anchor, the proven conservative topology is:

- model `gwdg/openai-gpt-oss-120b`;
- one cell per shard, up to three active shards;
- 20 episode/request slots per shard;
- study-wide concurrency ceiling 60;
- target 600 RPM;
- 8 CPUs and 12 GiB per shard;
- 16-hour shard limit.

Do not copy these numbers to another model or study without rerunning
preflight and recalculating provider load. Keep the shared adaptive provider
coordinator enabled. Treat published RPM as a ceiling rather than guaranteed
throughput, and monitor live leases, achieved RPM, latency, and failure rate.

## Failure and completion interpretation

- SLURM `COMPLETED` means the worker exited successfully; it does not prove
  that every scientific episode completed. Verify episode and cell seals.
- Retry HTTP 429/5xx, timeouts, and dropped connections within the bounded
  logical-request window through the shared coordinator.
- Keep strict semantic-response validation. Provider recovery must not coerce
  malformed ballots or identifiers.
- Preserve sealed episodes and validated checkpoints when recovering. Never
  wipe a study root merely to retry provider or scheduler failures.
