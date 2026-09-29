# Preflight: 21-09-2026-full-vs-report-v1-b6-b9-b15

Credential-free study preflight on 2026-09-22: **PASS / permitted**.

## Workload

| Quantity | Value |
|---|---:|
| Configurations | 1 |
| Scientific cells | 24 |
| Episodes | 1,440 |
| Rounds per episode | 15 |
| Controller budgets | 6, 9, 15 |
| Nominal provider calls | 540,000 |
| Expected provider calls | 632,160 |
| Conservative provider calls | 2,656,800 |

The 24 cells are two persistence values x two communication profiles x two
target semantics x three budgets. No-control is deliberately not rerun; the
original V1 study retains its complete 240-episode baseline.

## Offline token and cost estimate

| Scenario | Estimated cost |
|---|---:|
| Expected | USD 465.58 |
| Conservative retry/context bound | USD 1,957.66 |

Costs use the repository's offline DeepInfra price snapshot for
`openai/gpt-oss-120b`. Live price and account limits must be refreshed before
submission.

## Cygnus execution plan

- execution mode: cell array, one scientific cell per shard;
- 24 shards, all 24 eligible to run simultaneously;
- 4 CPUs per shard, 96 requested CPUs total;
- 5 episode/request slots per shard, 120 episode slots total;
- study-wide adaptive provider maximum: 100 requests;
- target/planned RPM: 500 / 480;
- resources per shard: 12 GiB and 24 hours.

Exactly 60 initialization artifacts are present in the shared V1 initialization
directory. No provider call or Slurm submission was made during preparation.
