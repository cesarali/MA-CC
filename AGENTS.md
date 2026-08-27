# Repository Agent Instructions

## Potsdam dedicated Python environment

This section applies only when operating on the Potsdam system or submitting
to its SLURM cluster. Potsdam Python commands, tests, preflights, submissions,
workers, aggregation, and post-processing must use the dedicated Conda
environment named `MA-CC`. Its canonical Conda executable is:

```text
/home/ojedamarin/.local/share/miniforge3/bin/conda
```

Use commands of the form:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC python ...
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC mas-cc ...
```

Add `--live-stream` for long-running commands whose progress must remain
visible. On Potsdam, do not use `/usr/bin/python`, create an alternative
project environment, or install project dependencies into the system
interpreter. Do not assume that `conda activate` persists between agent tool
calls. Before submitting a real Potsdam job, verify that the resolved `MA-CC`
Python imports `mas_cc`, `pandas`, and `pyarrow` from the expected
environment/repository.

Outside Potsdam, use the local machine's existing project environment and
setup instructions. Local agents must not look for, require, or reproduce the
Potsdam-specific absolute Conda path.

## Experiments

For MA-CC study creation, submission, SLURM execution, aggregation, or
post-processing, use the `ma-cc-study-workflow` skill. Read its complete
instructions at `.codex/skills/ma-cc-study-workflow/SKILL.md` before acting.

Do not create study-specific SLURM job files unless the scheduler topology
genuinely cannot be represented by the generic study launchers.

Authoritative architecture:

- `docs/tdd/features/orchestrator/22082026_TDD_standardized_study_submission_and_aggregation.md`
- `docs/handoff/22082026_standardized_study_submission_and_aggregation_handoff.md`
