# TDD: Reliable Cross-Node Provider Load Coordination

**Date:** 2026-08-30  
**Status:** implementation plan  
**Scope:** generic remote-provider runtime and standardized study execution

## 1. Motivation and observed failure

Study 09f exercised the shared adaptive provider coordinator from two Potsdam
compute nodes. Provider traffic was configured conservatively (global
concurrency 20 and target 500 RPM), but workers repeatedly logged:

```text
provider load-control release failed; the lease will expire: RuntimeError
```

The coordinator consequently accumulated stale leases. Its default lease is
600 seconds, while the configured logical-request retry window is 300 seconds.
Once stale leases occupied the concurrency limit, useful dispatch nearly
stopped and otherwise recoverable calls exhausted their episode-level request
window. Study 09f was stopped; it must not be resumed unchanged.

This is an execution-coordination failure, not evidence that the university
provider rejected 500 RPM or concurrency 20. Study 09g, whose two concurrent
workers were placed on one node, remained healthy under the same global target.
That comparison is diagnostic evidence, not yet proof of the precise shared
filesystem exception.

## 2. Goal

Make the existing shared adaptive provider coordinator reliable across SLURM
nodes so that:

- every provider attempt owns one globally visible renewable lease;
- successful responses are never discarded because telemetry/release failed;
- a failed release cannot suppress capacity for ten minutes;
- temporary coordinator I/O failures pause and recover safely rather than
  silently bypassing global limits;
- retryable provider failures remain within the same in-memory episode for at
  most the configured five-minute logical-request budget;
- AIMD concurrency, per-node pauses, global pauses, and the rolling RPM gate
  remain generic and independent of games/studies;
- scientific identities, episode seeds, controller semantics, and outputs are
  unchanged.

Do not add study-specific throttling or a Study-09f job file.

## 3. Existing implementation map

- `src/mas_cc/llm_runtime/providers/load_control.py`
  owns shared state, file locking, leases, RPM dispatches, pauses, and AIMD.
- `src/mas_cc/llm_runtime/providers/adapters/_openai_compatible.py`
  acquires/releases a lease around every HTTP attempt and applies bounded
  provider retries.
- `src/mas_cc/studies/runtime.py` injects the study-wide coordinator policy.
- `src/mas_cc/studies/execution.py` resolves the submitted execution plan.
- `tests/mas_cc/test_provider_load_control.py` covers coordinator behavior.
- `tests/mas_cc/test_llm_providers.py` covers adapter/coordinator integration.

## 4. Required design

### 4.1 Preserve the first real exception

The current release warning reports only the exception class. Log the
operation, coordinator root, node/worker identity, lease token prefix,
exception message, attempt count, and traceback. Never log credentials,
requests, prompts, or response bodies.

Expose compact coordinator health counters in the shared snapshot and normal
study monitoring:

- transaction successes/failures by operation;
- acquire/release/renew retry counts;
- active and expired/reaped leases;
- oldest lease age;
- dispatches and provider events in the current window;
- current limit and pause deadlines;
- last coordinator error type/time/node.

### 4.2 Retry shared-state transactions

Make acquire, release, renewal, and snapshot transactions resilient to
transient shared-filesystem `OSError`, incomplete/stale reads, and atomic
replace visibility races. Use bounded exponential backoff with jitter and
re-read the authoritative state on every retry.

Release must be idempotent. Releasing an already-removed token is successful
and must not duplicate its provider outcome event. Give each outcome a stable
attempt/event identity so a transaction whose commit status is uncertain can
be retried safely.

Do not treat arbitrary programming/schema errors as transient. Corrupt or
unsupported state must fail with an actionable error and preserve the bad
state for diagnosis.

### 4.3 Replace long passive leases with renewable leases

Use a short configurable lease TTL plus heartbeat renewal while an HTTP
attempt is in flight. The renewal period must be comfortably below the TTL
(for example, no more than one third). A healthy slow provider response must
retain its slot; a crashed worker must surrender capacity shortly after
heartbeats stop.

The adapter must stop the heartbeat in a `finally` path and then perform an
idempotent release. Heartbeat/release failures must not replace a valid
provider response, but they must enter bounded coordinator recovery and be
observable.

Validate configuration invariants at preflight:

- heartbeat interval < lease TTL;
- lease TTL leaves adequate scheduling/I/O margin over the heartbeat interval;
- stale-capacity recovery is materially shorter than the logical provider
  retry window;
- the overall logical request remains bounded by
  `retry_max_elapsed_seconds` (300 seconds for these studies).

Remove the current automatic rule that stretches every lease to provider
timeout × attempts + 60 seconds once heartbeats safely cover live requests.

### 4.4 Fail closed without sacrificing the episode

If shared coordination is temporarily unavailable, do not dispatch an
uncoordinated provider request. Pause only the affected worker/node, retry the
coordinator within the remaining logical-request budget, and continue the same
episode after recovery.

Coordinator wait and HTTP retry time must share one explicit five-minute
deadline. On deadline exhaustion, raise a distinct normalized error such as
`provider_coordination_unavailable`, separate from HTTP 429/5xx and malformed
model output. This makes operational failures countable and retryable through
the existing safe episode/cell recovery path.

### 4.5 Shared-state correctness

Keep the execution-only coordinator under the study result root, but make the
storage protocol explicit and tested for the Potsdam shared filesystem.
Continue using a stable lock file if verified reliable. Writes must remain
atomic, durable enough for this coordination purpose, schema-versioned, and
recoverable from a process dying before or after replace.

If a multi-node probe demonstrates that the filesystem cannot provide the
required lock/visibility semantics, replace the JSON transaction backend with
a repository-supported cluster-visible coordinator backend. Do not silently
fall back to independent per-node concurrency because that loses the global
RPM/concurrency guarantee.

## 5. TDD sequence

Implement in this order.

1. Add failure-injection tests that reproduce release/read/write/replace
   failures and uncertain commits; confirm the current implementation leaks
   capacity.
2. Add detailed safe diagnostics and coordinator health counters.
3. Add bounded transaction retries and idempotent outcome commits.
4. Add short renewable leases and adapter heartbeat lifecycle handling.
5. Bind coordinator acquisition and provider retries to one logical deadline.
6. Add preflight/config validation for the new timing invariants.
7. Run a local multi-process contention test.
8. Run a small Potsdam two-node mock-provider/shared-filesystem smoke test.
9. Run a credentialed, low-cost two-node provider probe only after the mock
   smoke passes.
10. Resume Study 09f from its existing canonical outputs using the ordinary
    safe-resume cell-array workflow; do not rerun sealed episodes.

## 6. Required tests

### Coordinator unit tests

- two coordinators share one concurrency ceiling;
- release is idempotent and records one outcome event;
- transient lock/read/write/replace failures recover within the retry budget;
- an uncertain successful commit does not duplicate leases/events;
- active leases renew and cannot be reaped;
- a crashed/non-renewing worker's lease is reaped within the short TTL;
- a slow valid HTTP attempt keeps its lease via heartbeat;
- RPM counts every dispatched HTTP attempt, including retries;
- AIMD decrease/recovery and node/global pauses retain current behavior;
- invalid/corrupt state fails explicitly rather than resetting silently;
- health counters accurately describe injected failures and recovery.

### Adapter tests

- every HTTP attempt acquires exactly one lease and releases it;
- retry attempts receive distinct leases and dispatch records;
- heartbeat starts after acquire and stops on success, HTTP error, parse error,
  cancellation, and transport exception;
- transient release failure is retried without losing a valid response;
- coordinator outage sends no uncoordinated request;
- coordinator recovery continues the same logical request/episode;
- the five-minute deadline terminates persistent coordinator/provider outage
  with the distinct normalized error;
- existing 429, 5xx, connection, malformed-envelope, and reasoning-exhausted
  behavior remains unchanged.

### Process/integration tests

- concurrent processes cannot exceed the shared limit;
- killing a lease-owning process restores capacity within the TTL;
- repeated atomic state replacement remains readable under contention;
- a two-node Potsdam smoke sustains acquire/renew/release cycles with zero
  leaked leases and reports the expected dispatch count;
- a small real-provider probe demonstrates upward/downward AIMD movement and
  no episode loss caused by coordinator release failure.

## 7. Acceptance criteria

The change is complete only when:

1. All existing provider, study, persistence, and resume tests pass.
2. New failure-injection and multi-process tests pass repeatedly.
3. The Potsdam two-node smoke completes with no unreadable state, leaked
   leases, duplicate outcome events, or concurrency overshoot.
4. The monitor exposes enough detail to distinguish provider HTTP failures,
   invalid model output, and coordinator failures.
5. A valid provider response survives a transient release failure.
6. A dead worker restores capacity well before the 300-second request limit.
7. No provider request is issued while global coordination is unavailable.
8. Study execution still uses the generic cell-array launcher and the Potsdam
   `MA-CC` Conda environment.
9. Scientific configs, episode seeds, retained observations, estimators, and
   aggregation results are unchanged by execution topology.

## 8. Operational rollout

Keep Study 09f stopped until the acceptance smoke passes. Preserve any valid
09f episode outputs and provenance. Then resume it with the same scientific
configuration, global concurrency 20, target 500 RPM, array throttle 2, and
existing results root; safe resume must skip already sealed episodes.

Monitor the first two cells for at least:

- coordinator transaction errors and retry recovery;
- live/oldest lease counts;
- achieved RPM and request latency;
- HTTP/provider vs coordination failure rates;
- completed and failed episodes.

Stop and diagnose if coordination errors leak capacity, if failure frequency
is substantial, or if the global ceiling is exceeded. Do not weaken response
validation or scientific game semantics to improve completion rate.

## 9. Non-goals

- changing provider/model;
- changing Study 09f/09g scientific design;
- weakening `shared_fact_id` validation;
- changing game, controller, persistence, MI/CMI, bootstrap, null, or support
  estimator semantics;
- adding study-specific scheduling code;
- maximizing request volume before cross-node correctness is demonstrated.

