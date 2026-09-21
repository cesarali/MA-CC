# 21-09-2026-full-vs-report-v0

Quick Task-003 comparison of `report_only` and `full_communication` with an
LLM-authored controller.

## Scientific design

- fixed `q=12` (`social_group_size=12`, board `sensor_sample_size=12`)
- controller sensing mode: previous completed public board (`sensing_mode: board`)
- 24 agents, 10 rounds, DeepInfra `openai/gpt-oss-120b`, prompt v5
- epistemic persistence `rho`: 0.75 and 1.0
- communication profiles: `report_only` and `full_communication`
- controller authoring: always `llm_authored` in controlled cells
- controller targets: truth (`correct`) and false (`ALLOCATION_2`)
- controller budgets: 3 and 12
- matched no-control baseline for each rho/profile pair
- 5 paired repetitions per cell
- Cygnus execution: all 20 cells at once, with 5 concurrent episode slots and
  5 CPUs per cell shard (100 episodes / 100 CPUs), while study-wide provider
  concurrency remains independently capped at 100

Ordinary agents are LLM-authored in every cell. No-control cells have no
controller, so `controller_authoring` is not applicable there. In controlled
cells, an inactive controller round emits zero posts and an active round emits
exactly `b` posts.

## Size

- controlled: 16 cells x 5 repetitions = 80 episodes
- no control: 4 cells x 5 repetitions = 20 episodes
- total: 20 cells, 100 episodes

The 2026-09-21 credential-free preflight passed. Using the repository's
2026-09-16 DeepInfra price snapshot, it estimates 29,100 provider calls and
USD 21.44 at representative token usage. The conservative retry/context bound
is 122,400 calls and USD 90.21; the nominal no-retry count is 24,800 calls.
See `PREFLIGHT.md` for the accounting details.

The study reuses the existing q=12 Task-003 paired initialization artifacts
with execution seed `20260907`, allowing the first five paired episode seeds to
match across every cell.

No paid provider run is authorized by these files. Run study preflight and
re-check live DeepInfra account limits before submission.
