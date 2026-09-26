# Graceful study drain

The generic Potsdam and Cygnus study launchers can request a cooperative drain
before SLURM's hard time limit. Configure it in `study.yaml`:

```yaml
execution:
  graceful_drain:
    enabled: true
    signal: USR1
    lead_time: "04:00:00"
```

The lead time must be shorter than `execution.time_limit`. The generated
execution plan records the resolved setting, and submission passes a generic
`sbatch --signal=B:USR1@<seconds>` option. The shell launcher keeps its Python
worker alive while it drains. On Cygnus, `study submit` and `study extend` select the
matching generic Cygnus launcher from the active Slurm cluster name; an
explicit `--job-script` remains an override. The submission attempt is read from
`submission.json` for a base job or the extension attempt record for an extension,
so no `sbatch --export` option is used. Automatic drain is
opt-in until a mock scheduler gate and an explicitly authorized tiny scheduler
smoke have passed on the target site.

Request a drain manually with:

```bash
mas-cc study drain --study-dir <study-root> --job-id <job-id> --reason manual
mas-cc study status --study-dir <study-root>
```

For an extension, pass its new array job ID to the same drain command. Use
`study status --study-dir <study-root> --job-id <extension-job-id>` to inspect
that attempt. No YAML switch is needed for a manual drain; the opt-in YAML
setting only schedules an automatic pre-walltime signal. A job already running
with older worker/launcher code cannot gain drain support retroactively.

`study drain --wait` waits for every planned shard to acknowledge either
`drained` or `scientifically_complete`; `--timeout` sets the maximum wait in
seconds. The command verifies the job against the current submission and
SLURM queue. It does not cancel or resubmit jobs. Only consider `scancel` after
all shards have acknowledged, or if accepting an ordinary abrupt stop.

A drain stops queued episodes. Generic games finish active episodes. Relational
imitation episodes persist a validated decision ledger after a complete
population round and replay those decisions locally on resumption. Checkpoint
parent bundles finish and seal an active continuation branch, then stop before
the next branch. Failed provider calls remain failures. The next authorized
`mas-cc study submit --config-dir <same-folder>` creates a new job namespace
and reuses valid episode, parent, and branch seals in the unchanged result
root. If the scheduler hard timeout arrives first, the existing abrupt
recovery path remains available.

For `study extend`, resume through the same target config folder and study
root after reconciling any stale `SUBMITTED` extension state. The retry gets a
new submission attempt and job-scoped drain namespace. Do not launch two
extension targets against one result root concurrently.

Drain state lives under `<study-root>/runtime/drain/job-<job-id>/`. It is
operational metadata, excluded from scientific aggregation. `drained` is an
incomplete outcome even when the SLURM task exits successfully; strict
aggregation still requires all scientific seals.
