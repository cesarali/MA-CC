# Prompt-v3 truthful-report q=1 / Potsdam study

Matched task_001 persistence study using `gwdg/openai-gpt-oss-120b` and the
production prompt-v3 blackboard contract. Controlled arms both use
`truthful_strategic_report`; the third arm is autonomous.

- rho: 0.70, 0.775, 0.85, 0.925, 1.00
- b: 3, 6, 9, 12, 15, 18, 21 (controlled arms only)
- 75 cells and 750 episodes total
- q=1, N=24, 10 rounds, 10 matched repetitions
- execution ceiling: 60 episodes/requests across three two-cell bundles

Before submission, materialize the prompt-v3 Potsdam initializations at the
absolute directory declared by all three configs, then run strict study
preflight. Provider-specific initialization artifacts deliberately remain
separate from the DeepInfra study.
