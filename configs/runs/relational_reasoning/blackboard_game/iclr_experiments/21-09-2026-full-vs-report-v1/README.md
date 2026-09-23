# 21-09-2026-full-vs-report-v1

Quick Task-003 comparison of `report_only` and `full_communication` with an
LLM-authored controller.

## Scientific design

- fixed `q=12` (`social_group_size=12`, board `sensor_sample_size=12`)
- controller sensing mode: previous completed public board (`sensing_mode: board`)
- 24 agents, 15 rounds, DeepInfra `openai/gpt-oss-120b`, prompt v5
- epistemic persistence `rho`: 0.75 and 1.0
- communication profiles: `report_only` and `full_communication`
- controller authoring: always `llm_authored` in controlled cells
- controller targets: truth (`correct`) and false (`ALLOCATION_2`)
- controller budgets: 3, 12, and 18
- matched no-control baseline for each rho/profile pair
- 60 paired repetitions per cell
- Cygnus execution: at most 25 of 28 cell shards at once, with 5 concurrent
  episode slots and 4 CPUs per shard (125 episode slots / 100 requested CPUs),
  while study-wide provider concurrency remains independently capped at 100

Ordinary agents are LLM-authored in every cell. No-control cells have no
controller, so `controller_authoring` is not applicable there. In controlled
cells, an inactive controller round emits zero posts and an active round emits
exactly `b` posts.

## Size

- controlled: 24 cells x 60 repetitions = 1,440 episodes
- no control: 4 cells x 60 repetitions = 240 episodes
- total: 28 cells, 1,680 episodes

The execution planner uses a 15-second latency assumption, informed by the v0
run's observed provider latency, and a 24-hour shard limit. Cost and call
estimates must be regenerated with study preflight before launch.

The study uses its own q=12 Task-003 paired initialization artifacts under
`/shared/home/cesar/work/results/studies/21-09-2026-full-vs-report-v1_initializations`
with execution seed `20260907`. All 60 episode seeds must match across every
cell; these artifacts have not been materialized by config preparation.

No paid provider run is authorized by these files. Run study preflight and
re-check live DeepInfra account limits before submission.
