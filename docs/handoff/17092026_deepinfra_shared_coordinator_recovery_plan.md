# DeepInfra Shared-Coordinator Repair and Pilot Recovery Plan

Implementation-planning snapshot: 2026-09-17.

This document is an implementation handoff for repairing the shared adaptive
provider coordinator and safely resuming the interrupted Task-003 false-control
pilot. It is a plan, not evidence that the repair or resume has been performed.
Do not submit paid provider work merely by following this document; obtain
explicit authorization after the implementation, tests, preflight, and resume
manifest are ready for review.

## 1. Objective

Make shared provider admission behave as backpressure rather than an episode
failure when many SLURM workers are waiting behind a reduced DeepInfra
concurrency limit. Then resume the existing 30 checkpointed episodes without
changing their scientific identities, initializations, seeds, completed
decisions, or retained round records.

The immediate target is:

```text
study:
  astra_task003_false_control_q12_deepinfra_gptoss120b_rho3_pilot_r2

config:
  configs/runs/relational_reasoning/blackboard_game/
    astra_task003_false_control_q12_deepinfra_gptoss120b_rho3_pilot_r2/

results:
  /shared/home/cesar/work/results/studies/
    astra_task003_false_control_q12_deepinfra_gptoss120b_rho3_pilot_r2

initializations:
  /shared/home/cesar/work/results/studies/
    astra_task003_q12_deepinfra_gptoss120b_rho3_pilot_r2_initializations
```

Only the false arm is in scope for the recovery. Do not submit the prepared
truth arm or aggregate incomplete output as a final result.

## 2. Required workflow and environment

Read and follow before changing or submitting anything:

- `.codex/skills/ma-cc-study-workflow/SKILL.md`
- `.codex/skills/ma-cc-study-workflow/references/deepinfra.md`
- `docs/tdd/features/orchestrator/22082026_TDD_standardized_study_submission_and_aggregation.md`
- `docs/handoff/22082026_standardized_study_submission_and_aggregation_handoff.md`

This checkout is currently on the shared Cerebrals-style cluster, not the
Potsdam system. Use the existing project interpreter:

```bash
/shared/home/cesar/.local/share/mamba/envs/MA-CC/bin/python
```

Use compute nodes for provider calls, load tests, and real submissions. Keep
results and SLURM logs beneath `/shared/home/cesar/work/results`. Do not copy
Potsdam-specific paths or Conda commands into this workflow.

The working tree already contains unrelated staged and unstaged work. Inspect
`git status --short` first and do not reset, overwrite, or fold unrelated
changes into this repair.

## 3. Incident evidence

The failed run was SLURM array job `442`.

Observed outcome:

| Item | Value |
|---|---:|
| Scientific cells | 15 |
| Repetitions per cell | 2 |
| Episode target | 30 |
| Completed and sealed episodes | 0 |
| Checkpointed incomplete episodes | 30 |
| Retained completed rounds | 115 / 900 |
| Mean retained rounds per episode | 3.83 |
| Retained-round range | 2-19 |
| Committed provider requests | 3,269 |
| Recorded input tokens | 7,206,081 |
| Recorded output tokens | 3,979,351 |
| Recorded cost | 0.943114667 USD |

All 30 `.resume/<episode-id>/failure_checkpoint.json` files record
`interruption_type: provider_error` and retain the failed-call identity plus
the episode runtime state. Worker logs classify 29 episode failures at
`_openai_compatible.py:565` (`provider_coordination_unavailable`) and one at
line 571 (normalized transport error). The final coordinator event had no HTTP
status, was retryable, and lasted about 209.6 seconds. There is no evidence of
HTTP 402, a budget stop, or invalid credentials.

The coordinator state ended at concurrency 2. Its health record contains one
explicit shared-filesystem lock timeout during lease renewal. This proves that
filesystem metadata contention occurred, but it does not explain all 29
coordination failures by itself.

## 4. Root cause to preserve in the implementation notes

There were three interacting causes.

First, real provider calls experienced retryable transport failures or long
timeouts. The adaptive controller correctly interpreted those as a reason to
reduce concurrency.

Second, the coordinator uses the same 300-second window for provider recovery
and waiting for shared admission. Once the adaptive limit fell to 2, the 15
worker processes had many admitted local requests waiting for two global
leases. A request that waited longer than that shared deadline raised
`ProviderCoordinationUnavailable`; the adapter then converted ordinary
capacity backpressure into a fatal episode-level `ProviderError`.

The relevant paths are:

- `src/mas_cc/llm_runtime/providers/load_control.py`, especially
  `SharedProviderCoordinator.acquire()`;
- `src/mas_cc/llm_runtime/providers/adapters/_openai_compatible.py`, especially
  the deadline construction around the shared acquire and the conversion to
  `provider_coordination_unavailable`;
- `src/mas_cc/studies/runtime.py`, which gives every worker the resolved
  coordinator policy from `execution_plan.json`.

Third, an operator live override raised persisted coordinator state to 100
after the workers had already loaded the original maximum-10/minimum-2 policy.
The override also moved `last_increase_at` into the future to prevent the old
workers from immediately clamping the state back to 10. Provider failures
could still reduce the limit, but normal additive recovery was suppressed.
This was an invalid mixed-policy state and must not be repeated.

The cluster shared filesystem amplified the problem because every acquire,
release, and heartbeat uses a metadata-heavy file-backed FIFO lock. A robust
coordinator must tolerate that environment, but the evidence does not support
blaming a particular compute node.

## 5. Non-negotiable invariants

1. Preserve the false-arm scientific YAML, model, temperature, game settings,
   grid coordinates, seeds, repetitions, initialization artifacts, and episode
   IDs.
2. Do not delete or rewrite the existing failure checkpoints or retained round
   trajectories.
3. Do not treat a checkpointed episode as completed. Only normal episode and
   cell seals count as durable completion.
4. Do not create a study-specific SLURM job. Use the generic study launcher.
5. Keep one coherent load-control policy from planning through every worker.
   Do not edit live `state.json` directly.
6. Provider-capacity waiting must not consume the provider transport-retry
   budget.
7. Cancellation, SLURM termination, and real provider failures must still have
   bounded and observable behavior; do not replace the current deadline with
   an uninterruptible infinite wait.
8. No paid smoke test or resume until the user explicitly authorizes it after
   reviewing the implementation, tests, preflight, and expected remaining
   cost.

### Storage and memory guardrails

The current partial pilot occupies 44 MB across 411 files. Its 30 failure
checkpoints occupy 6.15 MiB in total, and the two initialization artifacts
occupy 144 KB. These are small enough to preserve in place. Do not make a full
copy of the run tree merely to calculate hashes or prove compatibility.

The repair must also satisfy these resource constraints:

- hash and inspect the existing run tree in place;
- use `pytest` temporary directories for mock runs and allow normal test
  cleanup to remove them;
- do not write stress-test artifacts beneath the scientific result root;
- do not persist one coordinator file, request body, response body, traceback,
  or debug log per provider attempt;
- keep coordinator state bounded to the active leases and rolling event/RPM
  windows already required for control decisions;
- keep ordinary worker logs concise and rotate or bound any new diagnostic
  stream;
- use small fake payloads for concurrency tests and cap a compute-node stress
  smoke at 2 GiB unless measurement demonstrates a real need for more;
- reuse the existing result root and checkpoints for the eventual resume; do
  not create a second 30-episode scientific tree as a workaround;
- retain only the canonical scientific records required by the configured
  estimators and the existing dashboard artifact profile; do not add permanent
  retry caches or coordination traces;
- estimate the completed pilot's disk footprint before launch and pause for
  review if it is likely to exceed 1 GiB;
- monitor disk usage during the resume and stop for investigation if the pilot
  root crosses 1 GiB or file count grows unexpectedly through per-request
  artifacts.

The 1 GiB value is a review threshold, not a target. Scaling the current
44 MB at 115 retained rounds to the 900-round target suggests a final footprint
of a few hundred megabytes, although record size varies with later-round board
state. Aggregation may create its documented compact analysis package, but it
must not package or duplicate the source run tree.

## 6. Implementation plan

### Phase A: Reproduce the failure without provider calls

Add deterministic tests around the existing fake transport and shared
coordinator. Reproduce this sequence:

1. Start substantially more logical callers than the global coordinator limit.
2. Hold leases long enough that at least one caller waits beyond the old
   300-second-equivalent test deadline.
3. Verify the current behavior raises `provider_coordination_unavailable` even
   though the fake provider itself remains healthy.
4. Add a reduced-time test for an adaptive transition such as 30 -> 15 -> 10
   while 30 logical callers remain queued.

Keep the test fast by using a fake clock or short policy intervals. Do not make
tests sleep for production-scale minutes.

### Phase B: Separate admission waiting from provider recovery

Introduce a distinct coordinator admission policy rather than reusing
`retry_max_elapsed_seconds`. A reasonable design is a new configuration field
such as:

```yaml
admission_max_elapsed_seconds: 1800
```

Required semantics:

- local semaphore waiting remains outside all provider-attempt deadlines;
- waiting for the first shared lease uses the admission deadline;
- the provider retry window begins only when the first transport attempt is
  ready to start;
- releasing a failed attempt and reacquiring for a retry must not allow an
  unbounded provider outage loop;
- task cancellation must interrupt admission promptly;
- an admission timeout must use a distinct diagnostic code and preserve a
  resumable failure checkpoint;
- ordinary admission waiting must never be reported as an HTTP/provider
  failure or counted in the adaptive provider failure ratio.

Prefer a bounded long wait over an unconditional infinite wait. The SLURM time
limit remains the outer execution bound.

### Phase C: Prevent shared-filesystem acquisition storms

Reduce write-lock pressure without weakening the global concurrency or RPM
guarantees.

Evaluate and implement the smallest safe combination of:

1. Read `state.json` optimistically before entering the exclusive lock. When
   capacity is plainly full or a pause/RPM gate is active, sleep without
   creating a lock ticket. Recheck every condition under the lock before
   granting a lease.
2. Preserve the per-process `_acquire_gate`, and ensure its polling backoff does
   not reset on every unsuccessful capacity check.
3. Prioritize lease release and renewal sufficiently that acquisition polling
   cannot starve them. Any priority mechanism must retain bounded fairness.
4. Use longer leases with less frequent renewal for this long-latency model.
   Validate the existing constraints between lease duration, heartbeat, stale
   recovery, and retry/admission windows.

Do not replace the coordinator with SQLite on a network filesystem without a
separate correctness and locking analysis. A service-backed coordinator may be
a future improvement, but it is not required for this recovery.

### Phase D: Make policy changes coherent and auditable

Choose one of these approaches and document it:

- reject runtime policy changes while workers are active; or
- implement a versioned settings update that every worker reloads and validates
  before using the new limits.

For the immediate repair, rejecting live mutation is sufficient. Add a guard
or operational command that makes it difficult to reproduce the mixed state
created in job 442. Never manipulate `state.json` timestamps to suppress
adaptive behavior.

The execution plan, persisted settings, and every worker must agree on initial,
minimum, and maximum concurrency. Record a policy hash or version in state and
fail clearly on mismatches instead of continuing with mixed policies.

### Phase E: Use a conservative pilot policy

The pilot has 15 shards, two episode slots per shard, and local
`request_concurrency: 2`. Its natural effective request ceiling is therefore
30. A shared ceiling of 100 adds no throughput unless local concurrency is also
changed, and changing local concurrency is unnecessary for this recovery.

Prepare a coherent study-level policy near the following starting point, then
validate it with the mock stress test:

```yaml
provider_load_control:
  mode: shared_adaptive
  initial_concurrency: 30
  minimum_concurrency: 10
  maximum_concurrency: 30
  target_rpm: 1000
  lease_seconds: 180
  heartbeat_seconds: 60
  retry_max_elapsed_seconds: 1800
  admission_max_elapsed_seconds: 1800
```

Do not copy these numbers blindly if code validation or the mock stress results
show a better combination. In particular, tune the global failure sample count,
failure ratio, decrease factor, cooldown, and recovery step so a short burst of
timeouts reduces pressure without collapsing 30 logical callers to two leases.
Record the rationale and test evidence for the final values.

Keep all 15 cell shards active so all 30 episodes remain scheduled. This is
separate from the 30-request provider ceiling.

### Phase F: Prove checkpoint compatibility before submission

Before any paid call:

1. Inventory all 30 failure checkpoints and record their hashes.
2. Inventory both initialization artifacts and record their hashes.
3. Resolve the proposed study and compare scientific identity, grid cell IDs,
   repetition seeds, initialization compatibility keys, and output paths with
   the existing manifests.
4. Verify that study-level coordinator changes do not alter the episode's
   scientific resolved-config hash or checkpoint identity.
5. Exercise resume against a copied miniature mock-provider run. Prove that it
   resumes the failed call, preserves preceding decisions/rounds, skips sealed
   episodes, and produces exactly one final seal.
6. Determine the supported generic submission/resume entry point for this
   existing result root. Do not construct an ad hoc `sbatch` command if
   `mas-cc study submit` or the extension workflow already represents it.

If the current checkpoint identity rejects a safe execution-only coordinator
change, fix the identity classification rather than editing checkpoint hashes.
Execution policy is not a scientific coordinate, but the code must explicitly
enforce that distinction.

### Phase G: Preflight and present the resumable launch

Run credential-free preflight and produce a reviewable launch summary with:

- one config and 15 cells;
- two repetitions and 30 target episodes;
- zero sealed episodes and 30 checkpointed episodes;
- retained progress of 115 completed rounds;
- expected remaining calls, tokens, cost, and wall time, clearly separated
  from the already recorded 3,269 calls and 0.943114667 USD;
- 15 active shards, 2 CPUs per shard, 30 episode slots, 30 request slots, and
  the final adaptive policy;
- absolute result and log roots;
- exact generic launcher and resume command;
- confirmation that the truth arm and aggregation are not scheduled.

Ask for explicit paid-run authorization only after this concrete summary is
ready. Submission is the final step after approval.

### Phase H: Monitor the authorized resume

On an authorized resume, use the `report-job-pace` skill and verify within the
first few minutes:

- all 15 shards are active;
- all 30 episodes were recognized as resumptions;
- retained round counts did not go backward;
- the coordinator policy hash matches across workers;
- active leases never exceed 30;
- waiting callers remain queued beyond the old short deadline without episode
  failure;
- renewals/releases are not starved;
- provider failures reduce concurrency gradually and successful traffic
  restores it;
- no HTTP 402 or budget stop occurs.

Stop and preserve checkpoints if scientific identity mismatches, retained
rounds disappear, the coordinator policy differs across workers, or repeated
coordination failures return. Do not repeatedly resubmit a structurally broken
run.

Aggregate strictly only after all 30 episodes and all 15 cells are durably
sealed.

## 7. Required tests

At minimum, add or extend tests for:

1. shared admission waiting longer than the provider retry window;
2. provider retry timing beginning at the transport attempt rather than queue
   admission;
3. 30 logical callers completing through a simulated limit reduction to 10;
4. adaptive reduction and subsequent recovery under a coherent policy;
5. policy-version mismatch rejection;
6. optimistic unlocked reads followed by authoritative locked rechecks;
7. renewal/release progress under heavy acquisition polling;
8. cancellation while waiting for admission;
9. lease cleanup after transport cancellation and exceptions;
10. exact checkpoint resume without replaying completed decisions;
11. existing DeepInfra adapter behavior for 401/402/403/429/5xx, malformed
    responses, connection failures, and successful responses;
12. the complete existing provider-load-control and study workflow test suites.

Use deterministic fakes for ordinary CI. If a multi-process shared-filesystem
smoke is useful, run it through `srun` on compute nodes and make it credential
free, write it to a temporary directory outside the study root, and remove the
temporary artifacts after recording the compact pass/fail summary. A real
DeepInfra smoke requires separate authorization.

## 8. Acceptance criteria

The repair is ready for launch review when all of the following are true:

- the reproduced pre-fix admission failure passes after the fix;
- no healthy fake-provider episode fails merely because it waited behind the
  shared limit;
- provider retry windows remain bounded;
- adaptive reduction and recovery both work after a limit change;
- file-lock contention does not starve renewals or releases in the 30-caller
  stress test;
- mixed coordinator policies are rejected or safely version-reloaded;
- all relevant unit and integration tests pass;
- the existing 30 checkpoint identities and two initialization hashes remain
  unchanged;
- the resume dry run demonstrates preserved decisions and rounds;
- mock and stress verification leaves no persistent per-request artifacts and
  stays within the declared memory cap;
- projected pilot storage remains below the 1 GiB review threshold;
- preflight permits the remaining workload and reports cost in USD;
- no study-specific SLURM file or duplicate estimator implementation was
  introduced;
- the user has a concrete launch summary and has explicitly authorized the
  paid resume.

## 9. Expected deliverables

The implementing agent should leave:

1. the coordinator and adapter code changes;
2. focused regression and stress tests;
3. any schema/config documentation for the new admission policy;
4. the coherent pilot `study.yaml` execution policy;
5. a checkpoint compatibility report with hashes and identity comparisons;
6. a fresh preflight report and remaining-cost estimate;
7. a short implementation handoff stating what was tested and any residual
   risks;
8. after separate authorization, the new SLURM job ID and an initial live pace
   report.

Do not claim recovery success until normal completion markers exist for all 30
episodes and strict study validation passes.
