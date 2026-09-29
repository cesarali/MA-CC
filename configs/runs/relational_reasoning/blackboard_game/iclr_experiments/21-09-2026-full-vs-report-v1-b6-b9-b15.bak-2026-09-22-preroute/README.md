# 21-09-2026-full-vs-report-v1-b6-b9-b15

Standalone complementary study for the existing
`21-09-2026-full-vs-report-v1` results.

## Scientific design

- Same submitted V1 protocol: Task 003, q=12, 24 agents, 15 rounds,
  DeepInfra `openai/gpt-oss-120b`, prompt v5, rho 0.75/1.0, report-only and
  full-communication profiles, and truth/false LLM-authored controllers.
- New controller budgets only: 6, 9, and 15.
- 60 repetitions per cell using execution seed `20260907`.
- Reuses the exact V1 paired initialization directory:
  `/shared/home/cesar/work/results/studies/21-09-2026-full-vs-report-v1_initializations`.
- Does not rerun no-control cells. The complete 240-episode V1 no-control
  baseline remains in the original study.
- Deliberately does not enable the later, unsubmitted
  `controller_allow_mixed_message_types` option, because this study must remain
  scientifically compatible with the submitted V1 observations.

## Size

- 24 controlled cells: two rho values x two communication profiles x two
  target semantics x three budgets.
- 1,440 episodes total.

This study writes to its own result root:

`/shared/home/cesar/work/results/studies/21-09-2026-full-vs-report-v1-b6-b9-b15`

It does not modify, retry, or overwrite the 1,382 completed episodes retained
in the original V1 partial aggregation. After this study is complete, the two
analysis packages can be compared or combined through the paired cross-study
analysis path. A strict `study merge` requires complete inputs, so the original
partial V1 package must remain explicitly provisional unless its missing
episodes are later completed.

No paid provider run is authorized by these files. Run fresh Cygnus and live
DeepInfra preflight checks before submission.
