# ICLR false-arm / Potsdam session handoff

Session snapshot: 2026-09-18 10:16 CEST.

## What is running

This session launched only the false-control arm of the prepared ICLR
recommendation-only Task-003 study. The truth-control and no-control arms are
not queued.

- Study config:
  `configs/runs/relational_reasoning/blackboard_game/iclr_experiments/recomm_only_q12_chatoss_false_control/`
- Result root:
  `/work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/recomm_only_q12_chatoss_false_control`
- Shared paired-initialization bank:
  `/work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/recomm_only_q12_chatoss_initializations`
- Provider/model: DeepInfra `openai/gpt-oss-120b` with explicit low reasoning
  effort.
- Scientific design: Task-003, false target `ALLOCATION_2`, 24 agents, q=12,
  30 rounds, 60 matched repetitions, and a 3-by-3 grid over epistemic
  persistence `{0.70, 0.85, 1.00}` and intervention budget `{6, 12, 18}`.
  This is 9 physical cells and 540 episodes.

The false arm uses `paired_local_vote`. The completed 60-artifact bank fixes
one initial local-vote population per repetition and is reused across all
false-arm cells. Do not regenerate it or point the run at a different
initialization directory.

## Potsdam preparation performed in this session

The prepared config was adapted from a non-Potsdam path layout:

- results, initialization artifacts, runtime state, and Slurm logs were moved
  to the `/work/ojedamarin/Projects/LanguageGames/MA-CC/results` tree;
- the task dataset path was changed to the repository-relative calibrated task
  directory under `results/studies/musr_truthful_selective_task_calibration_01/tasks`;
- the selected Slurm partition is `all`, not the unavailable `main` partition;
- the cell-array plan uses 9 active single-cell shards, 8 CPUs and 12 GiB per
  shard, a 36-hour time limit, 60 episode/request slots per shard, and 540
  local request slots in total.

The MA-CC Conda environment now has a Redis server/client and the Python
Redis client installed. The generic Redis service publishes an authenticated,
mode-0600 endpoint file under the study runtime directory; its secret must
never be copied into configs, documentation, shell history, or messages.
Cross-node Redis `PING` was verified before the paid launch.

## Provider coordination and observed performance

The study selects `redis_adaptive`, not the legacy shared-filesystem JSON lock
coordinator. Redis gives each API request a short-lived *lease*: a temporary
permission to make one provider call. The global limit bounds how many calls
can be in flight at once; the lease is released when that call ends.

The persisted policy is:

| Setting | Value |
|---|---:|
| Initial / minimum / maximum leases | 200 / 25 / 400 |
| Target RPM | 1,000 |
| Lease / heartbeat | 180 s / 60 s |
| Provider retry and admission windows | 1,800 s / 1,800 s |
| Global failure sample / ratio | 12 / 0.25 |

DeepInfra's live account lookup before launch reported 400 concurrent requests
and 1,100,000 TPM. At a live session sample around 08:35 CEST, Redis reported
432 dispatches and 402 successful completions in the preceding 60 seconds,
with zero retryable failures. A later sample reported 431 dispatches/minute,
360 successful completions/minute, and 140 active leases. These are snapshots,
not a promise of sustained RPM. The code's coordinator event latency includes
shared-admission wait as well as transport time, so it is not a pure model
inference-latency metric.

Do not infer that adding CPUs per shard will make model responses faster. All
9 scientific shards are already live, and remote model latency plus the
sequential 30-round dynamics dominate collection time. Extra local CPUs are
useful later for aggregation.

## Initialization history

| Job | Outcome | Notes |
|---|---|---|
| `1882320` | failed after 1m29s | DeepInfra returned HTTP 429 after 4 artifacts. |
| `1882414` | cancelled after 4m | Replaced so retry protection could apply; completed artifacts remained valid. |
| `1882416` | completed after 15m27s | Produced and validated all 60 artifacts. |

The generic initialization launcher now supports bounded retry for HTTP 429,
HTTP 5xx, and connection errors when invoked with
`MAS_CC_INITIALIZATION_RETRY_ATTEMPTS`. A retry reruns the materializer against
the same bank: valid artifacts are read and skipped, while only missing
artifacts are generated. Non-retryable configuration, authentication, or
payment errors still fail closed.

## Submission and current scheduler state

| Job | Role | State at snapshot |
|---|---|---|
| `1882318` | Redis coordinator service | running; 48-hour limit |
| `1882418` | dependent submitter | completed; submitted the science array |
| `1882419` | false-arm science array `0-8%9` | 9 tasks running; 36-hour limit per task |
| `1882428` | partial-aggregation trigger | pending on completion of the whole science array |

`1882419` was submitted at 2026-09-18 00:28 CEST. At the 10:16 CEST snapshot,
all 540 trajectories existed and had retained 13,016 completed round records:
mean 24.10 completed rounds per trajectory, range 22--28. There were no
episode-completion markers and no cell seals yet. This is live progress, not a
completed scientific result.

The run's authoritative submission and execution plan are:

- `<study-root>/submission.json`
- `<study-root>/execution_plan.json`
- `<study-root>/execution_manifest.csv`

## Planned analysis

The false arm uses its committed `analysis.yaml`: established round-level
information estimators, bootstrap and null procedures, state-local outputs,
propensity-weighted causal response, blackboard calibration, epistemic phase
outputs, derived aggregates, and plots. No new estimator was written for this
session.

The dependent submitter queued job `1882428` with
`afterany:1882419`. Once every science-array task has ended--whether normally
or with failures--it invokes:

```bash
mas-cc study aggregate \
  --study-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/recomm_only_q12_chatoss_false_control \
  --backend slurm \
  --allow-incomplete
```

This deliberately matches the previous Potsdam ASTRA pattern: the trigger
submits generic prepare, information-group, and finalizer jobs. Because
`--allow-incomplete` is intentional here, any resulting package must be
treated as provisional. Its `analysis/validation.json`, reports, and manifest
will record incomplete/invalid status if episodes or cells are missing. Do not
present that package as a final pooled result. If the array later seals all
episodes and cells, rerun strict aggregation without `--allow-incomplete`.

## Monitoring and recovery

Use the MA-CC environment on Potsdam for all commands:

```bash
squeue -j 1882318,1882419,1882428
sacct -j 1882419 --format=JobID,State,Elapsed,ExitCode,MaxRSS
find /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/recomm_only_q12_chatoss_false_control \
  -type f -name episode_complete.json | wc -l
```

Inspect these first if a worker ends unexpectedly:

- `<study-root>/logs/slurm-1882419_<task>.err` and `.out`;
- `<study-root>/runtime/provider-control/job-1882419/settings.json`;
- durable trajectory, episode-completion, and cell-seal markers under the
  sharded run tree;
- the aggregation-trigger logs under
  `<study-root>/logs/partial-aggregation-trigger/` once the array ends.

Do not edit Redis state directly, raise the live concurrency limit manually,
or cancel the Redis service while `1882419` or its aggregation descendants are
running. Preserve incomplete trajectories and checkpoints; a later resume
must reuse their scientific identities and completed prefixes.

## Verification and caveats

- Credential-free preflight passed: 405,000 nominal, 474,120 expected, and
  1,992,600 conservative provider calls; expected cost 348.79 USD and
  conservative bound 1,466.59 USD. These are planning estimates, not observed
  spend.
- The resolved execution plan predicts 360 RPM under its conservative
  90-second latency assumption, despite the 1,000-RPM target. Live throughput
  should be read from Redis's rolling 60-second dispatch window.
- MA-CC imports for `mas_cc`, `pandas`, and `pyarrow` passed. Redis server,
  Python client, and cross-node authentication checks passed. The generic
  launcher scripts passed `bash -n`.
- A broader repository config test still fails because the separate, unrun
  three-arm preparation folder retains `/shared/home/cesar/...` references.
  That failure does not affect this isolated false-arm folder or its successful
  preflight.

No study-specific Slurm job file and no replacement CMI estimator were added.
