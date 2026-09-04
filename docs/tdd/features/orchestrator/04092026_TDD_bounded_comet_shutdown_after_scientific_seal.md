# TDD: Bounded Comet Shutdown After Scientific Seal

**Date:** 2026-09-04  
**Status:** implementation plan  
**Scope:** generic observability and standardized SLURM study execution

## 1. Observed failure

Job `1863582`, array task `29`, completed its ten episodes and wrote a valid
`cell_complete.json`, but remained in `RUNNING` for more than six hours. Its
last output was Comet shutdown activity:

```text
Please wait for metadata to finish uploading (timeout is 3600 seconds)
Please wait for assets to finish uploading (timeout is 10800 seconds)
All assets have been sent, waiting for delivery confirmation
```

The shared provider coordinator had zero active requests. All 297 extension
episodes and all 33 extension cells were complete. The SLURM task was therefore
retaining a node solely for optional telemetry delivery confirmation.

## 2. Goal and invariants

Optional remote observability must never keep a scientifically complete shard
alive indefinitely. Preserve these invariants:

- canonical local results and the validated cell seal are authoritative;
- Comet remains aggregate-only and optional;
- scientific execution, seeds, estimators, retention, and provider behavior do
  not change;
- telemetry failure or timeout is reported, but does not invalidate a sealed
  cell;
- unsealed workers are never declared successful or killed by the telemetry
  safeguard;
- the solution is generic, not specific to the blackboard study or one job.

## 3. Design

### 3.1 Bounded sink close

Add an explicit, configurable Comet shutdown deadline with a conservative
default of 120 seconds for cluster studies. `CometMetricSink.close()` must
return a compact status containing the requested deadline, elapsed time, and
one of `completed`, `timed_out`, or `error`. Never persist exception text that
could contain credentials.

Because a Python thread cannot safely interrupt a blocking SDK `end()` call,
do not implement the deadline as an abandoned background thread in the worker.
Prefer a process-isolated Comet sink/helper so the telemetry process can be
terminated without affecting canonical writes. If the installed Comet SDK has
a documented hard cleanup timeout that demonstrably bounds every internal
thread, it may be used only after an integration test proves the worker exits
within the configured deadline.

### 3.2 Seal-aware launcher backstop

Add a generic backstop to the cell-array launcher:

1. run the scientific worker as a supervised child;
2. normally propagate its exit status;
3. if the expected cell seal exists and passes the ordinary identity and
   completeness validation, allow a short telemetry grace period;
4. after that grace period, terminate a still-running child, record
   `telemetry_shutdown_timeout`, and return success for the already sealed
   shard;
5. never use this path when the seal is missing, incomplete, or invalid.

The backstop protects allocations even if an SDK regression bypasses the
in-process deadline. SLURM stdout/stderr and the submission manifest must make
the intervention visible.

### 3.3 Configuration

Expose the deadline through the existing observability/execution configuration
with one standard behavior only; do not create study-specific job files or a
legacy unbounded mode. Preflight must report the deadline. The generic Potsdam
launcher must continue to use the `MA-CC` Conda environment.

### 3.4 Ordering

The worker must complete and atomically publish all canonical scientific files
and the cell seal before beginning optional sink shutdown. No Comet reference,
upload acknowledgement, or remote delivery receipt is part of the scientific
seal contract.

## 4. Tests

1. A normal Comet close finishes and reports `completed`.
2. A fake SDK whose `end()` blocks beyond the deadline cannot hold the worker
   or launcher beyond the configured grace period.
3. A sealed shard exits successfully after a telemetry timeout and records the
   timeout in operational provenance.
4. An unsealed blocked/failed shard is not converted into success.
5. Invalid or mismatched seals are rejected by the launcher backstop.
6. Exceptions during Comet close do not alter canonical data or cell status.
7. Existing aggregate-only privacy tests continue to pass.
8. A multi-process SLURM smoke leaves no task running after all cell seals are
   present.
9. Standard study aggregation accepts the sealed cell and remains numerically
   unchanged.

## 5. Acceptance criteria

- No completed cell holds a SLURM allocation for more than the configured
  telemetry grace period.
- Telemetry timeout/failure is visible and distinguishable from scientific or
  provider failure.
- A successful local scientific seal never depends on Comet availability.
- No prompt, raw response, credential, or execution tree is added to Comet.
- No scientific configuration, game/controller behavior, estimator, or
  retained observation changes.
- The generic study launcher remains the only production launcher.

## 6. Recovery for job 1863582

Verify that all 33 extension cells and 297 new episodes are sealed, cancel only
the lingering array task, and aggregate the complete extension together with
its 93 reused episodes. Do not rerun any episode.
