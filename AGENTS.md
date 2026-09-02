# Repository Agent Instructions

## Rutgers Amarel environment

Develop and inspect Amarel support locally. Reach Amarel only through the
operator-provided `amarel` command; never SSH directly, configure a VPN, or
operate the courier MacBook. Lead remote work with `amarel snapshot`, batch
commands aggressively, and tell the operator to approve the Duo push before
every call. Each courier call is high-latency and requires one phone approval.

Amarel production work uses ordinary SLURM batch jobs on account `general`,
partition `main`, and QoS `normal`. The hard walltime limit is 72 hours. Use the
generic launchers under `scripts/Amarel/SLURM/`; never create a study-specific
job. Build/use the named `MA-CC` Conda environment from `environment.yml`
because the base Python installations are unsuitable.

Put results, scheduler logs, Hugging Face/model caches, and Comet caches under
`/scratch/df630`, never in the source checkout or home directory. Check free
scratch space before staging because the filesystem is nearly full. Prepare or
submit studies with `--execution-site amarel`; the generated workers retain the
same scientific cells, episode seeds, resume behavior, and provider limits as
other sites.

## NERSC Perlmutter Python environment

This section applies when operating in this checkout on NERSC Perlmutter. The
repository is at `/pscratch/sd/d/dfarough/MA-CC`, and the `MA-CC` Conda
environment is physically stored at:

```text
/pscratch/sd/d/dfarough/conda_envs/MA-CC
```

The logical path `~/.conda/envs/MA-CC` resolves to that `/pscratch` location
through the `~/.conda/envs` symlink. Do not create a duplicate environment in
the home directory. Conda's writable package cache is also on `/pscratch` at:

```text
/pscratch/sd/d/dfarough/conda_pkgs
```

Load NERSC's Python module before each group of Conda commands because module
state does not persist between agent tool calls. Run Python, tests, and the
project command-line interface with:

```bash
module load python/3.11-24.1.0
conda run -n MA-CC python ...
conda run -n MA-CC python -m pytest ...
conda run -n MA-CC mas-cc ...
```

Do not install project dependencies into the system interpreter or recreate
the project environment under `$HOME`. To confirm the physical environment
location, use `readlink -f ~/.conda/envs/MA-CC`; it must resolve to the
`/pscratch` path above.

### NERSC Perlmutter scheduler policy

Lightweight editing, static preflight, and unit tests may run on a Perlmutter
login node. Production experiments, study workers, aggregation, and any other
compute-intensive work must run on Perlmutter CPU compute nodes obtained with
`salloc`. Always pass both of these options explicitly:

```bash
--qos=interactive --constraint=cpu
```

Never use `sbatch`, the `regular` QoS, or an omitted/default QoS for NERSC
MA-CC work. The interactive QoS permits at most four nodes and four hours;
each Perlmutter CPU node has 128 physical CPU cores. Use the policy-enforcing
generic launchers under `scripts/nersc/`. They reject a non-interactive
allocation and do not offer a QoS override.

Use the CPU project account, not its `_g` GPU-account form. Put production
results, worker logs, and scheduler metadata under
`/pscratch/sd/d/dfarough/MA-CC-results` (or an explicitly configured
`NERSC_RESULTS_ROOT` on `/pscratch`), never inside the source repository or
home directory. NERSC study execution must first use `mas-cc study prepare` to
run preflight and write manifests without submitting a Potsdam `sbatch` job;
then use `scripts/nersc/run_study.sh` to execute those manifests in one
interactive allocation. For a study that may cross the four-hour walltime,
use `scripts/nersc/start_study_supervisor.sh`; it detaches from the agent/SSH
session and resumes the same prepared study through successive interactive
allocations. Do not substitute a login-session foreground loop or another QoS.
The real `mas-cc study submit` path is disabled when
`NERSC_HOST=perlmutter`; do not bypass that guard.

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

Outside Potsdam and NERSC Perlmutter, use the local machine's existing project
environment and setup instructions. Local agents must not look for, require,
or reproduce either cluster's absolute paths.

## Experiments

For MA-CC study creation, submission, SLURM execution, aggregation, or
post-processing, use the `ma-cc-study-workflow` skill. Read its complete
instructions at `.codex/skills/ma-cc-study-workflow/SKILL.md` before acting.

Do not create study-specific SLURM job files unless the scheduler topology
genuinely cannot be represented by the generic study launchers.

Authoritative architecture:

- `docs/tdd/features/orchestrator/22082026_TDD_standardized_study_submission_and_aggregation.md`
- `docs/handoff/22082026_standardized_study_submission_and_aggregation_handoff.md`
