# Slurm cluster incident diagnostic — 2026-09-17

## Summary

The Slurm controller is responsive, but the compute-node `slurmd` services are unstable. Four of six configured nodes are currently `NOT_RESPONDING`. The two remaining nodes restarted `slurmd` within 31 seconds of each other around 21:00 UTC. Existing jobs were automatically requeued, and two initialization allocations temporarily remained in `RUNNING` state even though no batch process existed on the assigned node.

This is not specific to the DeepInfra workload. The failure occurs at the Slurm worker/node-service layer, before or independently of application activity.

Observed cluster version: Slurm 25.05.8. Cluster name: `cesar`.

## Current node state

Snapshot taken at 2026-09-17 21:09:05 UTC:

| Node | State | CPUs | `SlurmdStartTime` | Controller reason |
|---|---:|---:|---|---|
| `slurmd-0` | `IDLE+NOT_RESPONDING` | 16 | 18:15:15 | Not responding at 21:04:42 |
| `slurmd-1` | `IDLE+NOT_RESPONDING` | 16 | 18:15:03 | Not responding at 21:04:42 |
| `slurmd-2` | `IDLE+NOT_RESPONDING` | 16 | 18:21:08 | Not responding at 21:04:42 |
| `slurmd-3` | `IDLE+NOT_RESPONDING` | 16 | 18:21:08 | Not responding at 21:04:42 |
| `slurmd-4` | `MIXED` | 16 | **21:00:30** | none |
| `slurmd-big-0` | `MIXED` | 40 | **20:59:59** | none |

`scontrol ping` reports the primary controller `slurmctld-0` as `UP`. `sdiag` reports no pending RPCs and low scheduler-cycle latency, so the controller itself does not presently appear overloaded.

## Failure timeline and evidence

All times are UTC on 2026-09-17.

1. Initialization job 514 ran on `slurmd-big-0`. Slurm reported it as `RUNNING`, with three requeues in total, but inspection of the node showed no `514.batch` process and no job stdout/stderr files. Its allocation began at 20:38:57 during an earlier `slurmd-big-0` service restart at approximately 20:38:58. It made no application progress while in this state.
2. Job 514 was manually requeued. Slurm again reported it as running, but no batch process appeared.
3. A replacement initialization job 518 was assigned to `slurmd-4` at 21:00:27. The node's current `SlurmdStartTime` is 21:00:30. After that restart, Slurm reported job 518 as `RUNNING`, but `scontrol listpids` reported that the job did not exist on the node, no batch process was visible, and no stdout/stderr files were created.
4. An explicit `srun --jobid=518 --overlap ...` step worked after the node service returned. It completed the remaining initialization work, demonstrating that the compute host, shared output path, Python environment, and provider were functional once a real step was started.
5. Existing pilot array job 494 was also interrupted and requeued. Representative tasks currently report:
   - `494_5`: `Restarts=1`, current start time 21:02:47
   - `494_7`: `Restarts=4`, current start time 21:02:47
   - `494_10`: `Restarts=4`, current start time 21:02:47
6. New Redis job 515 and launcher job 519 started successfully after the node services stabilized. Launcher 519 completed normally. New study array job 520 currently has active workers with `Restarts=0`.

The first launcher attempt, job 516, failed for an unrelated application-wrapper issue: Slurm used `/bin/sh` for an `sbatch --wrap` script containing the Bash-only `set -o pipefail`. This was corrected in launcher 519 and is not part of the node-service incident.

## User-visible impact

- Jobs can be shown as `RUNNING` even when their batch process is absent.
- Automatic requeue preserves completed output, but work performed inside an unfinished episode can be repeated.
- API calls made since the application's latest durable checkpoint may also be repeated, increasing runtime and provider cost.
- Four unavailable nodes reduce usable capacity from 120 configured CPUs to 56 CPUs.
- Slurm accounting storage is disabled, so `sacct` cannot provide a durable event history. This makes incident reconstruction and automated recovery validation harder.

## Likely fault domain

The confirmed fault domain is the communication/lifecycle boundary between `slurmctld` and the compute-node `slurmd` services. The available evidence does not identify the root cause.

The `.svc.cluster.local` node addresses suggest that the Slurm daemons run in Kubernetes or a similar service-managed environment. If so, simultaneous pod restart, eviction, failed liveness probe, OOM kill, node pressure, or storage/network interruption are plausible causes. These are hypotheses and need administrator logs for confirmation.

## Requested administrator checks

Please inspect the following around 20:38–21:05 UTC on 2026-09-17:

1. Kubernetes pod status and events for `slurmd-0` through `slurmd-4`, `slurmd-big-0`, and `slurmctld-0`:
   - restart counts and previous container termination reason
   - `OOMKilled`, eviction, node pressure, failed probes, rollout, or rescheduling events
   - pod UID changes and container start times
2. Current and previous-container `slurmd` logs, especially for `slurmd-4` and `slurmd-big-0`:
   - loss of controller connection
   - failure to recover batch steps after daemon restart
   - credential/Munge errors
   - spool-directory loss or recreation
   - filesystem I/O, bus, or mount errors
3. `slurmctld` log `/var/log/slurm/slurmctld.log` for:
   - node registration loss and return
   - `Not responding` transitions
   - job 514 and 518 launch/requeue messages
   - why allocations remained `RUNNING` without a batch step
4. Node-local `slurmd` spool persistence. The reported `scontrol listpids` error included:

   ```text
   Domain socket directory /var/spool/slurmd: No such file or directory
   JobId=518 does not exist on node slurmd-4.
   ```

   Confirm `/var/spool/slurmd` exists before daemon startup, has correct ownership and permissions, and survives/reinitializes correctly across container restarts.
5. Shared filesystem and network events. Earlier workloads also experienced filesystem/node interruptions and a bus error. Check the shared mount used by `/shared/home` and `/shared/statesave` for disconnects, stale handles, I/O errors, or remounts.
6. Why `slurmd-0` through `slurmd-3` remain `NOT_RESPONDING`, with controller reason timestamp 21:04:42.
7. Consider enabling Slurm accounting (`slurmdbd`/accounting storage) so requeues, failures, exit states, and historical allocations remain queryable through `sacct`.

## Relevant Slurm configuration

```text
JobRequeue              = 1
ReturnToService         = 2
SlurmctldTimeout        = 120 sec
SlurmdTimeout           = 300 sec
KillWait                = 30 sec
UnkillableStepTimeout   = 60 sec
StateSaveLocation       = /shared/statesave
SlurmctldLogFile        = /var/log/slurm/slurmctld.log
SlurmdLogFile           = /var/log/slurm/slurmd.log
```

## Current workload status at diagnosis time

- Initialization completed successfully after running an explicit step: 60/60 artifacts plus manifest.
- Redis coordinator job 515 is running on `slurmd-4` with no restart.
- Study array 520 has four active workers on `slurmd-big-0`, currently with no restart; five array tasks are pending for resources.
- Pilot job 494 remains active but its tasks have experienced one to four requeues.

