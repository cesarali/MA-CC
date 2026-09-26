# Preflight: b6/b9/b15 extension to 120 repetitions

Credential-free study preflight on 2026-09-24: **PASS / permitted**.

The extension compatibility dry run also passed:

- target cells: 24;
- final target episodes: 2,880;
- validated reusable episodes: 1,335;
- missing episodes to execute: 1,545.

The missing set contains 105 unfinished repetitions from the original run and
1,440 new repetitions needed to raise every cell from 60 to 120.

## Workload

| Quantity | Full 120-repetition target |
|---|---:|
| Configurations | 1 |
| Scientific cells | 24 |
| Episodes | 2,880 |
| Rounds per episode | 15 |
| Controller budgets | 6, 9, 15 |
| Nominal provider calls | 1,080,000 |
| Expected provider calls | 1,264,320 |
| Conservative provider calls | 5,313,600 |

## Offline token and cost estimate

| Scenario | Estimated cost |
|---|---:|
| Expected | USD 931.15 |
| Conservative retry/context bound | USD 3,915.32 |

These are full-target bounds from the versioned offline DeepInfra price
snapshot, not the incremental cost after reuse. Live provider limits and price
metadata must be checked again before a real submission.

## Cygnus execution plan

- cell-array mode with one scientific cell per shard;
- 24 shards and an array throttle of 24;
- 4 CPUs per shard, for 96 requested CPUs when all shards are active;
- 16 episode slots and 16 request slots per shard;
- study-wide adaptive provider maximum of 384 requests;
- target/planned throughput of 1,600 / 1,536 RPM;
- 12 GiB and 36 hours per shard.

No provider call or Slurm submission was made during preparation.
