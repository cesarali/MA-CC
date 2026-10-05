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

Potsdam SLURM submission working directories and scientific output roots are
separate concerns. Results and SLURM logs must remain under `/work`, but the
generic Potsdam launchers establish
`/home/ojedamarin/Projects/LanguageGames/MA-CC` as their runtime working
directory so repository-local provider configuration (including `.env`) is
resolved consistently. Do not rely on the directory from which `sbatch` was
called. This rule is Potsdam-specific and must not be copied into local or
other deployment instructions.

## Cesar cluster

Cesar is the private SLURM-on-Kubernetes cluster reached over Tailscale. It has
no Conda, no module system, and no root: the environment is a uv virtualenv on
the shared NFS mount at `~/envs/MA-CC`, and the repository is an rsync copy at
`/shared/MA-CC` rather than a clone, so the cluster has no git and studies
submitted there record an empty `git_commit`. Push with `scripts/Cesar/sync.sh`,
which refuses to run while jobs are active because workers import from the
synced `src/` at runtime. Compute nodes ship no CA certificates, so
`SSL_CERT_FILE` must point at a bundle staged on `/shared`; the launchers under
`scripts/Cesar/SLURM/` handle this and fail loudly if it is missing. Results
belong under `/shared/MA-CC-results`.

Outside Potsdam, NERSC Perlmutter, and Cesar, use the local machine's existing
project environment and setup instructions. Local agents must not look for,
require, or reproduce any cluster's absolute paths.

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

### The next MuSR blackboard-control setups

Two setups, `task003-symmetric` and `task003-nosolution`, are
specified under
`analysis/preanalysis_task003_new_setup_and_coarsegraining/50_next_experiment/`.

- **To build or launch them**, read
  `50_next_experiment/designs/AGENT_RUNBOOK.md` in full first. It lists the
  blockers, the build steps, the configuration traps, and what not to do.
- **For what the experiments are**, `50_next_experiment/EXPERIMENTS.md`.
- The settings live in `50_next_experiment/designs/config_template.yaml`, with
  per-phase overrides in `variants.yaml`.

Two things that are easy to get wrong and are covered there:
`task003-symmetric` has **two** controller pools, one per target, so it needs
two task directories (`task003_symmetric_to_a0`, `task003_symmetric_to_a2`)
and a run must use the directory matching its target; and
`controller_fact_pool_mode` is a different option from
`controller_report_pool_mode`.

The two setups are frozen (5 October 2026); the run settings are preliminary.
Do not freeze a study configuration without confirming with the author.


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
