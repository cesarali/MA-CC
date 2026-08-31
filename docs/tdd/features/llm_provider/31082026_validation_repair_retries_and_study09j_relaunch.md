# TDD: Validation-Aware Response Repair and Study 09j Relaunch

**Date:** 2026-08-31  
**Status:** implementation plan; no code or scheduler changes made by this document  
**Scope:** generic validated-decision runtime, bounded diagnostics, and a clean Study 09j relaunch

## 1. Motivation and evidence

Study 09j job `1862075` uses `gwdg/openai-gpt-oss-120b`, `N=24`, 30 rounds,
20 episodes per cell, and strict relational-ballot validation. During its first
four active cells, 11 of the first 80 episodes logged a terminal validation
failure (about 13.75% at that early checkpoint):

```text
response.shared_fact_id: must be a bare fact identifier such as f1
```

This particular error is not an output-token exhaustion signal. It is emitted
only after the runtime has extracted a JSON object, accepted the vote and
reason, found `shared_fact_id` to be a string, and then rejected that string as
not being a bare identifier. Likely shapes include `"Fact f2"`, `"f2."`, or a
fact identifier plus its text, but the current `results_only` profile does not
retain the rejected value, so those examples are hypotheses rather than
observed values.

The configured output allowance is 4096 tokens and valid responses are short.
Increasing the token allowance is therefore not the proposed remedy.

The runtime currently permits one invalid-response retry: two invalid model
responses terminate the entire episode. For an `N=24` trajectory, the larger
number of decisions per episode makes even a small per-decision formatting
error rate accumulate into an unacceptable episode-loss rate.

At the same checkpoint, the new cross-node provider coordinator was healthy:
no transaction failures, expired leases, global pauses, or node pauses were
reported. This TDD must not modify or bypass provider load coordination.

## 2. Goals

1. Keep validation strict and make an invalid model response repairable inside
   the same in-memory episode.
2. Tell the model what failed and what exact output values are permitted on
   the next validation attempt.
3. Permit three correction attempts after the initial request for Study 09j.
4. Preserve identical behavior and request content for responses that validate
   on the first attempt.
5. Retain a small, bounded operational diagnostic that reveals the malformed
   field without adding prompt histories or response archives to the
   scientific package.
6. Relaunch Study 09j with one uniform response-repair protocol and without
   mixing the current partial attempt into the final scientific result.

## 3. Non-goals and invariants

Do not:

- silently coerce `"Fact f2"`, `"f2."`, or other invalid strings to `"f2"`;
- weaken the fact-ownership check or permit an agent to cite an inactive or
  unknown fact;
- invent a default vote, fact, or no-op after validation failure;
- change the game, controller, persistence, sensing, initialization, seeds,
  task, model, temperature, rounds, estimator definitions, or analysis;
- merge partial Study 09j episodes produced under two retry protocols;
- confuse validation repair retries with HTTP/provider retries;
- add a Study-09j-specific SLURM script or game-specific execution loop;
- place malformed-response diagnostics in the final analysis ZIP.

The production semantics remain authoritative: a response becomes an action
only after the existing response contract and game action validation both
pass.

## 4. Existing implementation map

- `src/mas_cc/runtime/loop_runtime.py`
  runs the shared ask/validate/retry loop.
- `src/mas_cc/games/relational_reasoning/imitation_round_feedback/prompts.py`
  owns `RelationalBallotContract`, its allowed fact identifiers, and strict
  validation.
- `src/mas_cc/games/relational_reasoning/imitation_round_feedback/runtime.py`
  derives deterministic per-attempt seeds and records attempts.
- `src/mas_cc/observability/recorder.py`
  receives validation-attempt records and applies retention policy.
- `src/mas_cc/experiments/orchestrator.py`
  owns episode/cell execution and safe episode-boundary resume.
- `configs/runs/relational_reasoning/population_study_09j/`
  contains the Study 09j scientific and launch configuration.

## 5. Required design

### 5.1 Contract-authored repair guidance

Add a small response-contract interface that can render correction guidance
from a structured validation failure. The generic loop should not know what a
relational fact identifier is.

For the relational ballot, a correction message should be equivalent to:

```text
Your previous response was invalid:
shared_fact_id must be a bare fact identifier.

Return the complete JSON object again. shared_fact_id must be exactly one of:
"f2", "none"

Do not include a label, fact text, punctuation, or explanation in that field.
```

The allowed values must come from the active contract for that request:
`contract.fact_ids + ("none",)`. These are facts already exposed to the agent
in its prompt. Never reveal a task fact, controller fact, truth label, or
hidden state that was absent from the original request.

Other validation errors should receive similarly minimal contract-authored
guidance where available. If a contract has no specialized repair renderer,
the generic fallback may state the validation error and ask for a complete
response matching the original schema. Do not embed Python tracebacks.

### 5.2 Retry the same logical decision

Attempt 1 must remain byte-for-byte equivalent in message content to the
current implementation. After an invalid response:

1. retain the original compiled messages;
2. append one user-role correction message for the next provider request;
3. request a complete replacement JSON object, not a patch fragment;
4. validate the replacement through the same authoritative contract and game
   validation;
5. continue the same episode immediately when a response validates.

For Study 09j set:

```yaml
game:
  options:
    invalid_response_retries: 3
```

This means one initial request plus at most three correction requests for a
logical decision. Provider HTTP retry limits remain independent and unchanged.

Do not restart the episode merely because one decision required correction.
Only exhaust the episode after all configured validation corrections fail or
after an independent unrecoverable provider/coordination error.

### 5.3 Determinism and provenance

Keep the existing deterministic seed derivation by validation-attempt index.
Repetition seeds and scientific identities must not change.

Because repaired attempts send an additional message, their recorded request
metadata must identify the effective request accurately. Add compact fields
equivalent to:

- `validation_attempt`;
- `validation_repair: true|false`;
- `repair_schema_version`;
- validation issue path/category;
- a hash of the correction message or effective transmitted messages.

Do not claim that a repaired request has the same prompt-instance identity as
the original transmitted message sequence unless the hashing contract
explicitly defines that behavior. The first-attempt prompt definition and
instance hashes must remain unchanged.

### 5.4 Bounded malformed-response diagnostics

The current `results_only` behavior does not preserve enough evidence to tell
whether the model emitted `"Fact f2"`, `"f2."`, or another invalid value.
Add a bounded operational diagnostic under the execution/runtime area, not the
canonical scientific tables and not the final analysis package.

Prefer a compact row containing:

- study/job/run/cell/episode identifiers;
- round, agent, decision stage, and validation attempt;
- model/provider and finish reason;
- validation issue path and message;
- received value and received type for the offending field, when safely
  extractable;
- input/output token counts;
- timestamp and repair outcome.

For relational ballots, retaining `raw_shared_fact_id` is sufficient for this
failure and avoids retaining the private free-text reason. Do not retain full
prompts by default. Cap rows per job/study, write atomically or append with the
existing safe recorder mechanism, and document truncation when the cap is
reached.

The standard analysis packager must explicitly exclude this diagnostic along
with provider logs, prompt histories, and other execution artifacts.

### 5.5 Structured output is a later optional enhancement

Do not make provider JSON-schema support a prerequisite for this repair. The
university endpoint/model must first be probed for reliable schema-constrained
output. A future provider capability may encode `shared_fact_id` as a dynamic
enum, but the validation-aware correction loop is still required as the
portable implementation.

## 6. TDD implementation sequence

1. Add failing unit tests that show the current second identical invalid
   response exhausts a decision and does not receive corrective guidance.
2. Make validation failures available to the response contract in structured
   form without changing successful validation behavior.
3. Add the contract repair-guidance interface and relational implementation.
4. Append repair guidance only on attempts after an invalid model response.
5. Preserve deterministic per-attempt seeds and add correct repair provenance.
6. Add bounded malformed-field diagnostics compatible with `results_only`.
7. Set Study 09j `invalid_response_retries: 3`; make no other scientific
   configuration changes.
8. Run focused unit, integration, retention, packaging, and persistence tests.
9. Run a credential-free mock episode in which attempt 1 returns a malformed
   ID and attempt 2 corrects it.
10. Stop/preserve the old Study 09j attempt if still active, preflight the new
    attempt, and submit through the generic study launcher.

## 7. Required tests

### Validation and runtime tests

- A valid first response sends exactly the original messages, makes one call,
  and produces the same action as before.
- `"shared_fact_id": "Fact f2"` is rejected, not coerced.
- `"shared_fact_id": "f2."` is rejected, not coerced.
- A retry sees the error plus exactly the currently allowed fact IDs and
  `none`.
- A corrected second response continues the same logical decision and episode.
- Three correction failures exhaust after four total validation attempts when
  `invalid_response_retries: 3`.
- A repaired citation of a syntactically valid but unknown/inactive fact still
  fails the existing ownership/action validation.
- Repair guidance never includes facts outside the requesting agent's active
  contract.
- Attempt seeds remain deterministic, distinct, and compatible with the
  existing derivation convention.
- Provider 429/5xx/timeout retries and coordinator deadlines remain unchanged.
- Cancellation still exits promptly and does not create an extra request.

### Diagnostics and retention tests

- The compact diagnostic records the actual rejected field value and error.
- The row cap is enforced and cap exhaustion is visible.
- Valid responses do not create malformed-response rows.
- `results_only` still omits full prompt/response histories.
- The standardized analysis ZIP excludes the diagnostic path and raw malformed
  responses.
- No new persistent analysis cache or duplicated scientific table is created.

### Regression tests

- Relational persistence, controller, and scientific-observable tests remain
  numerically unchanged for fixtures whose responses validate initially.
- Existing response-contract tests retain strict behavior.
- Study preflight still resolves 18 cells and 360 episodes for Study 09j.
- Cross-node load-control tests remain green; no coordinator code is changed
  unless a test exposes a direct integration defect.

## 8. Stop and clean relaunch procedure

The implementing agent must inspect scheduler state first:

```bash
squeue -j 1862075
sacct -j 1862075 --format=JobID,State,ExitCode,Elapsed,Start,End
```

If job `1862075` is active, cancel it and wait until no array task remains
running. Do not delete its result tree. Record its final scheduler state and
the observed completed/failed/in-progress counts as operational provenance.

Do not resume partially executed Study 09j cells under the new protocol. No
cell was sealed at the time this plan was written, and mixing episodes across
validation protocols would make the run harder to audit. Use a new explicit
attempt result root (for example a repair-v1 suffix) or preserve/rename the old
attempt before recreating the canonical root. Resolve the exact destination
with read-only checks before any move; never overwrite or recursively delete
the old attempt.

Reuse unchanged:

- the Study 09j configs except `invalid_response_retries: 3`;
- the 20 shared initialization artifacts;
- all episode/repetition seeds;
- model/provider, N=24, task, rounds, rho/b grid, controller, persistence, and
  analysis;
- the generic cell-array launcher;
- the shared adaptive provider coordinator and approved concurrency/RPM plan;
- the Potsdam `MA-CC` Conda environment.

Run strict preflight and report the result root, 18 cells, 360 episodes, call
budget including the possible correction calls, concurrency, RPM, memory, and
wall time. Submit only after tests and preflight pass.

## 9. Acceptance criteria

1. First-attempt-valid decisions are behaviorally unchanged.
2. Invalid output is never silently normalized into a scientific action.
3. A corrected response continues the same episode.
4. Study 09j permits three correction attempts after the initial request.
5. Diagnostics show the exact malformed field value without retaining an
   unbounded response archive.
6. Scientific packages remain free of prompts, raw invalid responses, and
   execution diagnostics.
7. All focused and existing relevant tests pass.
8. Strict Study 09j preflight resolves the unchanged 18-cell/360-episode
   scientific design.
9. The old job/result attempt remains preserved and excluded from the new
   scientific aggregation.
10. The relaunched job initially demonstrates materially lower terminal
    validation-failure incidence; if it does not, stop and diagnose rather
    than repeatedly consuming the full study budget.

## 10. Implementer handoff report

The implementing agent should report:

- files changed;
- exact repair-message and retry semantics;
- how effective request hashes/provenance are recorded;
- diagnostic location, cap, and package exclusion;
- tests and results;
- old job final status and preserved result path;
- new preflight totals and correction-call budget;
- new submission job ID/result root;
- validation-failure rate after an initial monitored sample;
- confirmation that scientific/game/controller/persistence semantics were not
  changed.
