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

Potsdam SLURM submission working directories and scientific output roots are
separate concerns. Results and SLURM logs must remain under `/work`, but the
generic Potsdam launchers establish
`/home/ojedamarin/Projects/LanguageGames/MA-CC` as their runtime working
directory so repository-local provider configuration (including `.env`) is
resolved consistently. Do not rely on the directory from which `sbatch` was
called. This rule is Potsdam-specific and must not be copied into local or
other deployment instructions.

Outside Potsdam, use the local machine's existing project environment and
setup instructions. Local agents must not look for, require, or reproduce the
Potsdam-specific absolute Conda path.

## Cloud result uploads

For uploading existing result files or directories to the cloud bucket behind
`results/aggregation_results`, use the `upload-results` skill at
`.codex/skills/upload-results/SKILL.md`. All agents working in this repository
must read it before an upload. Transfer only in environments with a confirmed
existing bucket connection; do not assume every checkout is connected.
The known local mount is read-only, so the skill uses direct rclone uploads
and verification. Keep credentials in the environment's existing configuration;
never print or commit credentials or copy them between environments.
This routing instruction does not authorize automatic uploads.

## Experiments

For combining complementary completed studies before aggregation, use
`mas-cc study merge` and read
`docs/documentation/metrics/merging_complementary_studies.md`. Merge into a
new result root, then run the existing aggregator separately. Do not pool
precomputed estimates or silently count overlapping cells twice.

For MA-CC study creation, submission, SLURM execution, aggregation, or
post-processing, use the `ma-cc-study-workflow` skill. Read its complete
instructions at `.codex/skills/ma-cc-study-workflow/SKILL.md` before acting.

For those operations on the Cygnus cluster or its `slurm-login` host, also use
the `ma-cc-cygnus-study-workflow` skill at
`.codex/skills/ma-cc-cygnus-study-workflow/SKILL.md`. It requires agents to
read `docs/handoff/cygnus-end-to-end.md` in full and keeps Cygnus environment,
launcher, path, resource, and aggregation rules separate from Potsdam. This
routing requirement does not itself authorize a real submission or paid
provider call.

That skill routes compute-site and provider-specific operations separately:

- Potsdam execution and the university provider:
  `.codex/skills/ma-cc-study-workflow/references/potsdam.md`
- DeepInfra provider operations on any compute site:
  `.codex/skills/ma-cc-study-workflow/references/deepinfra.md`

DeepInfra is a provider, not a cluster. A DeepInfra study running on Potsdam
must follow both references. Do not merge credentials, limits, or provider
coordinator state between providers.

For read-only experiment monitoring and pace summaries, use the
`report-job-pace` skill at `.codex/skills/report-job-pace/SKILL.md`.

For requesting or inspecting a cooperative drain of a study submission or
extension, use the `ma-cc-study-drain` skill at
`.codex/skills/ma-cc-study-drain/SKILL.md`. A drain request is distinct from
Slurm cancellation and does not authorize a new submission.

Do not create study-specific SLURM job files unless the scheduler topology
genuinely cannot be represented by the generic study launchers.

Authoritative architecture:

- `docs/tdd/features/orchestrator/22082026_TDD_standardized_study_submission_and_aggregation.md`
- `docs/handoff/22082026_standardized_study_submission_and_aggregation_handoff.md`
