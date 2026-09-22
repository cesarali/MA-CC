# Preflight: 21-09-2026-full-vs-report-v1

Credential-free study preflight rerun on 2026-09-22: **PASS / permitted**.

## Workload

| Quantity | Value |
|---|---:|
| Configurations | 2 |
| Scientific cells | 28 |
| Episodes | 1,680 |
| Rounds per episode | 15 |
| Controller sensing | previous public board, sample size 12 |
| Controller budgets | 3, 12, 18 |
| Nominal provider calls | 626,400 |
| Expected provider calls | 733,920 |
| Conservative provider calls | 3,088,800 |

The controlled arm has 24 cells from two persistence values, two communication
profiles, two target semantics, and three budgets. The no-control arm has four
matched persistence/profile cells and is not duplicated across meaningless
controller budgets.

## Token and cost estimate

| Scenario | Input tokens | Output tokens | Estimated cost |
|---|---:|---:|---:|
| Lower / nominal | 679,773,600 | 626,400 | USD 25.26 |
| Representative expected | 800,196,000 | 3,006,136,320 | USD 540.65 |
| Conservative retry/context bound | 3,394,072,800 | 12,651,724,800 | USD 2,276.37 |

Costs use the repository's offline 2026-09-16 DeepInfra price snapshot for
`openai/gpt-oss-120b`. Expected values are planning estimates; conservative
values are configured retry and maximum-context bounds. Live provider limits
and pricing must be refreshed before a paid submission.

## Execution envelope

- repetitions / in-process parallelism: 60 / 5
- one scientific cell per shard
- 28 total cell shards; at most 25 active simultaneously
- four CPUs per shard; at most 100 requested CPUs and 125 episode slots
- configured coordinator initial / maximum concurrency: 100 / 100
- target RPM: 500
- planning latency: 15 seconds, informed by the v0 observed provider latency
- requested resources per shard: 4 CPUs, 12 GiB, 24 hours
- estimated dashboard-semantic storage: 2,113,816,320 bytes

## Remaining launch prerequisite

The paired initialization directory is
`/shared/home/cesar/work/results/studies/21-09-2026-full-vs-report-v1_initializations`.
The 60 compatible artifacts have not been materialized. Initialization and a
fresh Cygnus/provider preflight are required before any real launch.

No provider calls or scheduler submission were made during config preparation.
