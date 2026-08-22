# Repository Agent Instructions

## Experiments

For MA-CC study creation, submission, SLURM execution, aggregation, or
post-processing, use the `ma-cc-study-workflow` skill. Read its complete
instructions at `.codex/skills/ma-cc-study-workflow/SKILL.md` before acting.

Do not create study-specific SLURM job files unless the scheduler topology
genuinely cannot be represented by the generic study launchers.

Authoritative architecture:

- `docs/tdd/features/orchestrator/22082026_TDD_standardized_study_submission_and_aggregation.md`
- `docs/handoff/22082026_standardized_study_submission_and_aggregation_handoff.md`
