# Preparation and preflight record

Prepared on 2026-09-23 for Cygnus. This is a preparation record only: no
initialization provider calls and no Slurm submission were made.

## Extension compatibility

- Existing lineage: `/shared/home/cesar/work/results/studies/21-09-2026-full-vs-report-v1`
- Target: 28 cells, 3,360 episodes (120 repetitions per cell)
- Validated completed episodes retained: 1,382
- Missing episodes to execute: 1,978
- Incompatible cells: 0
- Conflicted episodes: 0

The 1,978-episode delta consists of 298 unfinished episodes from repetitions
0–59 and 1,680 new episodes from repetitions 60–119.

## Paired initialization coverage

- Planned initialization artifacts: 120
- Existing artifacts: 60 (repetitions 0–59)
- Missing artifacts: 60 (repetitions 60–119)
- Unique planned seeds: 120

The missing initialization artifacts must be materialized with the provider-
backed initialization command before a real extension submission.

## Credential-free study preflight

Status: **PERMITTED**

- No-control: 4 cells, 480 target episodes, $150.15 expected / $637.42
  conservative
- Controlled: 24 cells, 2,880 target episodes, $931.15 expected / $3,915.32
  conservative
- Combined complete target: $1,081.30 expected / $4,552.74 conservative
- Nominal provider calls: 1,252,800
- Expected provider calls: 1,467,840
- Conservative provider calls: 6,177,600

These estimates cover the complete target, not only the incremental episode
plan. The actual extension plan will skip the 1,382 retained episodes.

## Cygnus execution plan

- Mode: `cell_array`
- Cell shards: 28
- Array throttle: 12
- CPUs per task: 4 (up to 48 reserved CPUs)
- Episode slots per shard: 16
- Total episode slots: 192
- Request concurrency per shard: 10
- Total request concurrency: 120
- Provider coordinator ceiling: 100
- Target / estimated RPM: 500 / 480
- Partition: `main`; QoS: `normal`
- Memory: 12G per task; time limit: 24 hours
