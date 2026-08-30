# MA-CC on NERSC Perlmutter

These are the generic NERSC launchers for MA-CC. They adapt the Potsdam cell
and config workers to NERSC's interactive allocation model. They do not create
study-specific job files and never call `sbatch`.

For the same reason, `mas-cc study submit` refuses to run when
`NERSC_HOST=perlmutter`. Use `study prepare` and these launchers instead.

## Non-negotiable scheduler rule

Every launcher fixes the scheduler request to:

```text
salloc --qos=interactive --constraint=cpu
```

There is no QoS option to override. Passing `--qos` is an error. The scripts
also enforce NERSC's current interactive limits: one to four nodes and at most
four hours. A Perlmutter CPU node has 128 physical cores and 256 hardware
threads as represented by SLURM's CPU count.

Use CPU account `m4539` for this checkout, or export it once:

```bash
export NERSC_CPU_ACCOUNT=m4539
```

Production results and all per-worker logs belong below
`/pscratch/sd/d/dfarough/MA-CC-results`, outside this source repository.

## One interactive shell

```bash
scripts/nersc/allocate_cpu.sh --account m4539 --nodes 1 --time 01:00:00
```

After the allocation is granted, use `srun` for compute work. Exit the shell
when finished so NERSC stops charging the allocation.

## One compute command

`run_command.sh` creates one CPU-node allocation and runs one command with
`srun`:

```bash
scripts/nersc/run_command.sh --account m4539 --time 00:10:00 -- \
  hostname
```

Use `--dry-run` to print the exact `salloc` and `srun` command without
requesting resources.

## A standardized study

NERSC execution has two stages. Preparation runs the existing credential-free
preflight and writes the same deterministic manifests used by the Potsdam
array worker, but it does not contact SLURM:

```bash
module load python/3.11-24.1.0
conda run -n MA-CC mas-cc study prepare \
  --config-dir <study-config-folder> \
  --results-dir /pscratch/sd/d/dfarough/MA-CC-results/studies/<study-root> \
  --require-results-under /pscratch/sd/d/dfarough/MA-CC-results \
  --execution-site nersc
```

Then allocate one to four CPU nodes and execute the prepared manifest:

```bash
scripts/nersc/run_study.sh \
  --account m4539 \
  --nodes 4 \
  --study-dir /pscratch/sd/d/dfarough/MA-CC-results/studies/<study-root>
```

For a new study, both stages can be invoked together:

```bash
scripts/nersc/run_study.sh \
  --account m4539 \
  --nodes 4 \
  --config-dir configs/runs/<family>/<study> \
  --results-dir /pscratch/sd/d/dfarough/MA-CC-results/studies/<study-root>
```

`study_plan.py` keeps the generated provider-safe throttle. It also caps local
worker processes by the plan's physical CPU and memory request, so allocating
four nodes does not silently turn into 512 simultaneous provider workers.
`run_study_rank.py` partitions manifest rows across nodes and calls the same
`mas_cc.studies.cell_worker` or `array_worker` used by the standardized
Potsdam architecture. Per-shard stdout and stderr are stored below
`<study-root>/logs/nersc-<job-id>/`.

For a study longer than the four-hour interactive limit, start the detached
rollover supervisor. This is the normal production entry point:

```bash
scripts/nersc/start_study_supervisor.sh \
  --account m4539 \
  --nodes 4 \
  --time 04:00:00 \
  --study-dir /pscratch/sd/d/dfarough/MA-CC-results/studies/<study-root> \
  --aggregate
```

The command returns only after `run_study_until_complete.sh` is running under
`nohup` plus a new session and owns the study lock. Closing Herdr, ending the
agent turn, or losing the originating SSH connection therefore does not stop
the supervisor. Its state is auditable at:

```text
<study-root>/runtime/nersc-rollover.pid
<study-root>/runtime/nersc-rollover.log
<study-root>/runtime/nersc-rollover.lock
```

Use `--ensure` to make repeated starts idempotent. Add
`--wait-for-job <current-interactive-job-id>` when adopting an allocation that
was launched separately; the supervisor owns its lock while it waits.

After each allocation exits, including a walltime timeout, the supervisor
rechecks cell seals and episode resume manifests. If the study is clean but
incomplete, it requests another fresh interactive allocation and passes the same
prepared manifest, output root, cell identities, and seeds to the unchanged
worker. Completed episodes are validated and skipped by the existing resume
layer. Scientific failures stop the loop; scheduler timeout alone does not.
Once every cell has a completed seal, `--aggregate` runs strict aggregation in
its own interactive CPU allocation. If that allocation is unavailable or is
interrupted before the final manifest and ZIP exist, aggregation resumes in a
fresh allocation. A recorded strict-validation failure stops the loop.

`--time` may be set below the prepared shard time when the project balance or
an intentional rollover drill requires shorter allocations. The override is
enabled automatically through the resumable supervisor path: it keeps the same
episode checkpoints and uses the selected interactive walltime for both study
allocations and the later one-node aggregation allocation. Direct
`run_study.sh` calls still reject a walltime shorter than the prepared plan
unless their caller explicitly opts into resumable behavior.

The supervisor contains no fallback to regular QoS and rejects any QoS
override. Its study lock prevents two rollover loops from allocating for the
same result root. `run_study_until_complete.sh` remains available as the
foreground/debugging form, but should not be the normal long-study command.

This detached process survives user-session inactivity, but no login-node
process can be guaranteed across a login-node reboot or NERSC maintenance. A
fully scheduler-hosted control process requires access to NERSC's long-lived
`workflow` QoS; project `m4539` does not currently have that QoS. Do not work
around this with `regular`. After a platform interruption, rerun the detached
command with `--ensure`; the same checkpoints make restart safe.

When preparation starts through `run_study.sh`, the wrapper records the
approved NERSC result boundary as
`/pscratch/sd/d/dfarough/MA-CC-results` (or `NERSC_RESULTS_ROOT`). This safely
replaces a Potsdam `/work/.../results` boundary for the prepared artifacts, so
the existing scientific study folder does not need to be copied or edited.
It does not alter provider limits, experiment settings, cell identities, or
seeds.

Prepared artifacts are stamped with `execution_site: nersc`. The NERSC planner
rejects Potsdam or unstamped preparations and validates that every worker
output remains beneath the prepared `/pscratch` study root. Conversely, the
two generic Potsdam jobs stamp their workers as `potsdam`; the shared worker
entry points reject a mismatched preparation before configuring a provider or
running an episode. Checked-in study manifests therefore keep their Potsdam
`/work` result guard and scheduler defaults. NERSC rebinding happens only in
the external prepared result tree, never by editing the scientific config
folder.

## Credential-free scheduler smoke

This requests one interactive CPU node and runs a one-episode mock-provider
experiment:

```bash
scripts/nersc/run_command.sh --account m4539 --time 00:05:00 -- \
  scripts/nersc/smoke_compute.sh \
  /pscratch/sd/d/dfarough/MA-CC-results/smoke/nersc-$(date -u +%Y%m%dT%H%M%SZ)
```

The smoke payload refuses a non-interactive or non-CPU allocation, confirms
the 128 physical cores, imports MA-CC plus its aggregation dependencies from
the `/pscratch` environment, preflights the config, and completes the mock
experiment without sending any provider request.
