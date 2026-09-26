# 21-09-2026-full-vs-report-v1-b6-b9-b15 extension to 120 repetitions

Extension configuration for the existing
`21-09-2026-full-vs-report-v1-b6-b9-b15` study lineage.

## Scientific design

- Preserves the submitted protocol: Task 003, q=12, 24 agents, 15 rounds,
  DeepInfra `openai/gpt-oss-120b`, prompt v5, rho 0.75/1.0, report-only and
  full-communication profiles, and truth/false LLM-authored controllers.
- Preserves the original recommendation-only controller message behavior; it
  does not enable the later mixed-message-types option.
- Preserves controller budgets 6, 9, and 15.
- Raises the target from 60 to 120 repetitions per scientific cell using the
  same seed contract and paired initialization directory.
- Does not add no-control cells. The no-control baseline remains in the
  complementary `21-09-2026-full-vs-report-v1` study.

## Size and reuse

- 24 controlled cells.
- Final target: 2,880 episodes.
- Nominal extension over a complete 60-repetition base: 1,440 episodes.
- The extension planner must also recover any unfinished base repetitions and
  reuse every validated completed episode from the existing result root.

The authoritative result root remains:

`/shared/home/cesar/work/results/studies/21-09-2026-full-vs-report-v1-b6-b9-b15`

No paid provider run or Slurm submission is authorized by these files.
