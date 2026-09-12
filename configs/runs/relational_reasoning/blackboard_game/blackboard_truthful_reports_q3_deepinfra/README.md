# Prompt-v3 truthful-report q=3 / DeepInfra study

Matched task_001 persistence study using `deepseek-ai/DeepSeek-V4-Flash` and
the production prompt-v3 blackboard contract. Controlled arms both use
`truthful_strategic_report`; the third arm is autonomous.

- rho: 0.70, 0.775, 0.85, 0.925, 1.00
- b: 3, 6, 9, 12, 15, 18, 21 (controlled arms only)
- 75 cells and 750 episodes total
- q=3, N=24, 10 rounds, 10 matched repetitions
- execution ceiling: 100 episodes/requests across five two-cell bundles

Before submission, materialize the prompt-v3 DeepInfra initializations at the
absolute directory declared by all three configs, then run strict study
preflight. Do not reuse prompt-v2 initialization artifacts.
