---
name: report-job-pace
description: Report the live pace and status of MA-CC experiments and scheduler jobs in a compact table. Use when the user asks how an experiment, study, deployment, extension, provider workload, or Slurm job is progressing, including episodes, rounds, concurrency, RPM, or latency.
---

# Report Job Pace

Inspect current state read-only and lead with a compact two-column table.
Prefer authoritative manifests, scheduler state, completion seals, live
trajectory records, and provider-control state over filenames or recollection.
Read `../ma-cc-study-workflow/SKILL.md` and its relevant provider/site
references before inspecting an MA-CC run.

## Required table

Include these rows whenever data exists. Use `—` with a short reason when a
metric is unavailable.

| Item | Current value |
|---|---:|
| Observed at | timestamp and timezone |
| Configuration | name/path |
| Provider/model | provider and model |
| Scheduler job | job ID and state |
| Episodes | per-cell and total target |
| Reused episodes | count |
| Newly completed | completed / extension delta |
| Overall completed | completed / final target and percent |
| Durably sealed | sealed / final target and percent |
| Currently running | count |
| Not started | count |
| Average completed round among running episodes | mean / configured horizon |
| Running-round range | minimum–maximum |
| Scheduler tasks | completed, running, pending, failed |
| Concurrency | configured ceiling, adaptive limit, and active leases |
| RPM | configured target and observed latest-60-second dispatches |
| Provider latency | mean completed-request latency in the same window |

Add one short health note with recent request failures/retries, stale-state
warnings, and failed scheduler tasks.

## Counting rules

- A durable cell/run completion marker is authoritative for sealed output.
- An episode-level completion marker means fully simulated. Keep completed but
  not-yet-sealed episodes separate from parent-cell seals.
- For extensions, overall completed equals reusable base episodes plus newly
  completed episodes. Do not double-count linked or copied artifacts.
- Currently running means started, uncompleted trajectories belonging to live
  scheduler tasks. Not started equals extension delta minus completed minus
  running after investigating inconsistencies.
- Derive round progress from durable trajectories and label it as completed
  rounds, not partial in-flight work.
- Calculate observed RPM from dispatch timestamps in the latest 60 seconds.
  Calculate latency from completed provider events in the same window and
  report the sample count when sparse.
- Distinguish experiment parallelism, per-shard request concurrency, scheduler
  throttle, adaptive global limit, and instantaneous active leases.
- Resolve ambiguous wording using conversation context, active jobs, recent
  submissions, result timestamps, and study lineage. Summarize multiple
  plausible jobs separately rather than combining them.

For an extension, inspect `study_lineage.json`, the latest extension's target
and compatibility manifests, execution plan/manifest, submission attempt,
`squeue`/`sacct`, completion markers, uncompleted round trajectories, and
`runtime/provider-control/job-<job-id>/state.json`. Flag stale inherited
metadata instead of letting it override the target manifest.

Do not mutate, resubmit, cancel, aggregate, or repair a run unless the user
separately requests that action.
