---
name: ma-cc-study-drain
description: Request or inspect a graceful, resumable drain of a running MA-CC Slurm study or study extension. Use for drain progress, safe-point acknowledgements, and drain-aware stopping; not for ordinary job pace reports.
---

# MA-CC study drain

Use the repository's `ma-cc-study-workflow` skill, plus
`ma-cc-cygnus-study-workflow` and its runbook on Cygnus. A drain is an
operational request to stop at durable scientific boundaries; it does not
cancel a Slurm job, complete the study, or authorize resubmission.

## Identify the exact job

Resolve the **study result root**, not the scientific config folder. Identify
the current array job ID from `submission.json` for `study submit`, or from
`extensions/extension-*/submissions/attempt-*.json` for `study extend`. Confirm
it against `squeue` before requesting a drain. Do not infer the active attempt
from a stale extension `state.json`, and do not target a previous job ID. The
CLI verifies the job and study manifest before writing a job-scoped request.
Jobs launched before drain-capable workers and generic launchers were installed
cannot be retrofitted while running.

## Request and inspect

On Cygnus, use the Python environment specified by
`docs/handoff/cygnus-end-to-end.md` (currently
`/shared/home/cesar/.local/share/mamba/envs/MA-CC/bin/python`). Elsewhere use
that site's workflow environment. The commands below use `mas-cc` for brevity;
`python -m mas_cc.cli.main` is equivalent in the correct environment.

```bash
mas-cc study drain --study-dir <study-result-root> --job-id <array-job-id> --reason manual
mas-cc study status --study-dir <study-result-root> --job-id <array-job-id>
```

Repeat `study status` to observe `phase`, `expected_shards`, shard counts,
`active_episodes`, `active_branches`, `last_durable_safe_point_time`, and
Slurm state/time remaining. `study drain ... --wait` also prints per-shard
acknowledgements, but waits for **every planned shard**; queued Slurm tasks can
make it wait longer than the active episodes. The manual request needs no
`execution.graceful_drain` YAML setting. That setting only asks Slurm to send
an automatic pre-walltime signal on a future submission.

Read `runtime/drain/job-<job-id>/request.json`, `state.json`, and
`shards/<array-index>.json` when the summary needs explanation. These files
are operational metadata, not scientific completion seals. Cross-check
`squeue`/`sacct` and task logs when acknowledgements stall or the job is gone.
Never treat a missing acknowledgement as proof that a checkpoint was written.

## Interpret the safe point

The request is noticed quickly, but draining is **not instantaneous**.
Queued episodes do not start. Generic games finish active episodes;
relational/blackboard episodes finish the current population round and write
the round-replay recovery checkpoint; checkpoint ensembles finish and seal an
active continuation branch. Provider latency or retries can extend that wait,
so do not promise a minutes-level deadline or an ETA from status alone.
`drained_incomplete` means the job stopped cooperatively, not that strict
aggregation may run.

Do not `scancel` on the strength of the request alone. If cancellation is
separately authorized, confirm each **live** shard has acknowledged `drained`
or `scientifically_complete` first; otherwise state clearly that cancellation
would be an abrupt stop. Do not resubmit or reconcile extension state without
separate authorization. An authorized resume uses the same result root and
target config via the normal `study submit` or `study extend` path; extension
retries get a new attempt and job-scoped drain namespace. Never run two
extension targets against the same study root concurrently.
