# Prompt-v4 adaptive-communication q=3 / DeepInfra study

Exact structural counterpart of `blackboard_truthful_reports_q3_deepinfra`.
Both controlled arms use `adaptive_communication`, with controller REQUEST and
DIRECTIVE enabled; REPORT remains available. The binary control decision,
task, grids, seeds, load, retention, and analysis are unchanged.

- rho: 0.70, 0.775, 0.85, 0.925, 1.00
- b: 3, 6, 9, 12, 15, 18, 21 (controlled arms only)
- 75 cells / 750 episodes; q=3, N=24, 10 rounds, 10 repetitions
- DeepInfra ceiling: 100 request slots over five active two-cell bundles
