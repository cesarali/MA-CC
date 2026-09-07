# Prompt-v4 adaptive-directive q=1 / Potsdam study

Matched `task_001` counterpart of the prompt-v3 truthful-report q=1 Potsdam
study. The controlled arms use `adaptive_communication`: REPORT and DIRECTIVE
are available, while controller REQUEST is disabled. The no-control arm uses
the same prompt-v4 participant contract for a matched baseline.

- rho: 0.70, 0.775, 0.85, 0.925, 1.00
- b: 3, 6, 9, 12, 15, 18, 21 (controlled arms only)
- 75 cells and 750 episodes total
- q=1, N=24, 10 rounds, 10 matched repetitions
- Potsdam ceiling: 60 episodes/requests across three two-cell bundles

Before submission, materialize the prompt-v4 Potsdam initializations at the
absolute directory declared by all three configs, then run strict study
preflight. This study has its own result and initialization roots and must not
reuse the prompt-v3 or DeepInfra initialization artifacts.
