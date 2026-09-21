# ICLR recommendation-only q=12 ChatOSS study

Matched Task-003 DeepInfra study using `openai/gpt-oss-120b` for every arm.
The truth- and false-control arms use recommendation-only adaptive controller
communication. The no-control arm provides the autonomous baseline.

- arms: no control, truth-aligned control, false-target control
- social sample size: q=12
- intervention budgets: 6, 12, 18 (controlled arms only)
- epistemic persistence: 0.70, 0.85, 1.00
- 30 rounds and 60 episodes per scientific cell
- 21 cells and 1,260 episodes total
- common seed and paired initialization artifacts across all arms
- GPT-OSS reasoning effort: explicitly `low`
- provider coordination: `redis_adaptive`

This folder is preparation only. No provider work is authorized or submitted
by these files.

The 60 paired initialization artifacts are not included. Repository
initialization is provider-backed, so materializing them is a separate paid
prerequisite that must be explicitly authorized before submission.
