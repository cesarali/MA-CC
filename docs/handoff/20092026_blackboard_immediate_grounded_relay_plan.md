# Blackboard Immediate Grounded-Report Relay Plan

Implementation-planning snapshot: 2026-09-20.

This document is an implementation handoff. It does not claim that the change
has been implemented, tested, or used in a paid run. Do not resume or replace
the failed report-only study until the implementation, provider-free regression
tests, downscaled smoke, and preflight evidence described here are complete.

## 1. Objective

Make truthful information observed on the blackboard immediately available for
truthful relay by the focal participant in the same microscopic update.

The new default semantics for new blackboard prompt/config versions must be:

1. A participant may cite a fact that is active in its memory.
2. A participant may also cite a grounded `REPORT` that was actually included
   in its sampled blackboard view for the current update.
3. A participant may not cite an unseen, expired, unsampled, or ungrounded fact.
4. A controller or participant `REQUEST` or `DIRECTIVE` does not carry evidence
   and therefore does not add a citable fact.
5. If the participant neither remembers nor currently sees any citable fact,
   its public communication action is deterministically `NONE`.
6. The participant still makes its private vote/reasoning decision when its
   public communication action is `NONE`.
7. The previous active-memory-only citation behavior remains available as an
   explicit legacy policy and remains the implicit behavior of historical
   prompt/config versions.

The feature must work independently of the controller policy. In particular,
it must behave correctly when adaptive communication chooses `REPORT`,
`REQUEST`, or `DIRECTIVE`, as well as with recommendation-only, silent, and
other existing controller actuation modes.

Preserve the existing public controller pseudonym: participants must see the
controller only as `Agent N+1`, rendered through the same board-message format
as every other author. The internal `author_kind: controller` and
`source_type: control` fields are required for audit and causal analysis but
must never be rendered into a participant prompt or otherwise reveal the
special role.

## 2. Incident that exposed the problem

The diagnostic run was:

```text
config:
  configs/runs/relational_reasoning/blackboard_game/iclr_experiments/
    report_only_authored_q12_chatoss_false_control/false_control.yaml

results:
  /shared/home/cesar/work/results/studies/
    report_only_authored_q12_chatoss_false_control

Slurm array job:
  67
```

Its final scientific outcome was:

| Epistemic persistence | Completed episodes | Failed episodes |
|---:|---:|---:|
| 0.70 | 0 / 180 | 180 |
| 0.85 | 22 / 180 | 158 |
| 1.00 | 180 / 180 | 0 |
| **Total** | **202 / 540** | **338** |

The nine shards recorded 165,548 committed provider requests, 495,725,986
input tokens, 54,598,046 output tokens, and 27.623529302 USD in local budget
accounting. Provider transport was healthy. Failed episodes ended in
`RelationalDecisionFailed` at the `focal_update` stage after the configured
validation attempts were exhausted.

The persistence pattern is the key diagnostic. At `rho = 1.0`, active memory
never loses known facts and every episode completed. At lower persistence,
participants can enter an update with no active private fact while still being
shown grounded reports on the board. The prompt makes those reports visible,
but the current response contract accepts citations only from the participant's
pre-update active-memory set.

The failure checkpoints do not retain the exact rejected response and
validation issue for every exhausted decision. The root-cause statement above
therefore combines the observed persistence-specific failure pattern with the
current prompt, validator, and state-transition code. The implementation must
add bounded validation diagnostics so a future incident does not require this
inference.

## 3. Current semantics and precise design gap

The existing implementation already has most of the required information-flow
machinery:

- `RelationalAgentState.known_fact_ids` is the historical knowledge set
  `H_i(t)`.
- `RelationalAgentState.active_fact_ids` is the currently usable set `K_i(t)`.
- `fact_provenance` records how a participant acquired a fact.
- `RelationalImitationRoundFeedbackGame._exposures()` extracts grounded facts
  from the exact social sources shown in the prompt.
- `apply_round_event_transition()` adds those exposures to historical and
  active knowledge and records their provenance.

The ordering is wrong for immediate relay:

```text
1. Build prompt from active facts and sampled board messages.
2. Validate shared_fact_id against active facts only.
3. Apply the transition.
4. Add facts exposed by the sampled board messages to knowledge.
```

Consequently, a participant can learn a displayed fact at step 4 and cite it
on a later update, but it cannot cite that fact in the response to the update
where it first saw it. The model sees the report and naturally attempts to use
it, while the validator treats it as unavailable.

The corrected ordering is conceptually:

```text
1. Build the sampled social view.
2. Derive grounded facts actually exposed in that view.
3. Construct this update's citable set from active memory plus those exposures.
4. Render and validate against that exact citable set.
5. Atomically commit the vote, knowledge acquisition/reactivation, provenance,
   and optional public report.
```

This is an immediate-reception rule. It must not expose the entire board or the
global task fact set to the focal participant.

## 4. Scientific semantics

### 4.1 Three distinct fact sets

Keep these concepts separate in code, prompts, events, and analysis:

```text
historically known facts H_i(t)
    facts the participant has legitimately received at any prior time

active memory K_i(t)
    historically known facts currently available after persistence dynamics

current observed facts O_i(t)
    grounded REPORT facts in the exact sampled social view for this update
```

For the new policy, the legal citation set is:

```text
C_i(t) = K_i(t) union O_i(t)
```

Use task fact order for deterministic ordering and deduplicate repeated
exposures. `O_i(t)` is update-local. The normal transition adds or reactivates
its facts in `H_i` and `K_i` after the action is accepted.

### 4.2 What counts as an observed grounded fact

A fact belongs to `O_i(t)` only when all of the following hold:

- its source record was actually rendered to the focal participant;
- the source has `message_type: REPORT` when it is a board message;
- it carries a non-null `shared_fact_id`;
- that identifier exists in the task's immutable fact registry;
- the message is live, sampled, and permitted by self-exclusion rules;
- its stored fact identifier and canonical fact text agree.

Do not include facts from:

- board messages that were not sampled for this focal update;
- expired messages;
- messages excluded because the focal participant authored them;
- `REQUEST` or `DIRECTIVE` messages;
- transient direct recommendations without a grounded fact identifier;
- free-form public text that merely resembles a fact;
- controller-private eligible-fact pools not actually posted or exposed.

### 4.3 Immediate relay and provenance

An immediately relayed fact is acquired and reported in one atomic transition.
The emitted report must identify whether its citation came from:

- `active_memory`; or
- `current_observation`.

For `current_observation`, retain the source message ID, author kind, source
agent/controller identifier, and observation round/micro-step. If several
visible messages expose the same fact, preserve the existing deterministic
slot-order tie-break for acquisition provenance.

Do not claim that an observed fact was privately verified by the relaying
agent. It is a globally grounded fact received through a verified report. The
prompt should use language such as "grounded facts you remember or can
currently see," rather than calling every citable fact private evidence.

### 4.4 No citable fact means `NONE`

When `C_i(t)` is empty, public communication has a singleton legal action:

```json
{
  "type": "NONE",
  "text": null,
  "shared_fact_id": null,
  "reply_to": null
}
```

Implement this as an explicit action mask or contract specialization, not as a
fabricated fact identifier. The LLM must still supply its vote and private
reasoning. Record that the communication action was constrained because there
was no citable fact.

The default no-fact policy is `force_none`. Preserve an explicit legacy option
that leaves message choice to the old contract. Do not silently turn an invalid
identifier into a different valid identifier.

When participant `REQUEST` messages are enabled, keep their general semantics,
but the new default `force_none` rule still applies to a no-fact update. A
separate explicit policy may allow `REQUEST` as a no-fact action in experiments
that scientifically require it; it must not become the default through the
controller's ability to emit requests.

## 5. Configuration and compatibility contract

Add explicit, canonical board policies. Suggested names are:

```yaml
game:
  options:
    prompt_version: 5
    board:
      report_citation_scope: active_or_observed
      no_citable_fact_action: none
```

Supported citation scopes:

```text
active_or_observed   new default for prompt/config version 5 and later
active_only          exact legacy citation behavior
```

Supported no-fact actions:

```text
none                 deterministic NONE public action; new default
model_select         legacy response-contract behavior
request_or_none      explicit opt-in only when participant requests are enabled
```

Compatibility rules:

1. Prompt/config version 4 and earlier must retain `active_only` plus
   `model_select` when the fields are absent.
2. Prompt/config version 5 and later must resolve absent fields to
   `active_or_observed` plus `none`.
3. Explicit `active_only` must remain supported under the new version so new
   studies can deliberately reproduce the old scientific mechanism.
4. Resolved configs, canonical study manifests, config hashes, checkpoints,
   and analysis provenance must record both policies.
5. A checkpoint created under one policy must not resume under another policy.
6. Existing frozen bundles and historical result readers must remain readable.
7. Do not rewrite existing YAML or result manifests in place.

This version gate makes the corrected mechanism the default for new work while
preserving byte-reproducible access to the previous behavior.

## 6. Implementation design

### Phase A: Introduce one authoritative citable-set derivation

Add a small immutable value object or pure helper representing the citation
context for one focal update. It should contain at least:

```text
active_fact_ids
observed_fact_ids
citable_fact_ids
source metadata for each observed fact
policy/version
```

Build it once from the focal state, `social_sources`, and resolved rules. Pass
that same object to prompt rendering, response validation, action validation,
transition application, checkpoint diagnostics, and event emission. Do not
recompute the set independently in several layers.

Likely code locations:

- `src/mas_cc/games/relational_reasoning/imitation_round_feedback/game.py`
- `src/mas_cc/games/relational_reasoning/imitation_round_feedback/state.py`
- `src/mas_cc/games/relational_reasoning/imitation_round_feedback/runtime.py`

Reuse `_exposures()` or extract its logic into the shared helper. Preserve its
rule that requests and directives are not evidence carriers.

### Phase B: Add prompt/contract version 5

Create a versioned blackboard prompt contract rather than changing version 4
in place.

Version 5 must:

- show private active facts separately from currently observed grounded facts;
- label observed facts with their visible message/source provenance;
- enumerate exactly the fact IDs accepted for this response;
- say that a report may faithfully relay either category;
- prohibit treating request/directive text as verified evidence;
- require `NONE` when the citable set is empty under `no_citable_fact_action:
  none`;
- continue requiring truthful text paired with the selected fact identifier;
- retain current limits on reason and public-message length;
- retain current reply visibility validation.

When the legal public action is deterministically `NONE`, prefer a specialized
response contract that asks the model for the vote and private reason while
making the public action fixed. If the full JSON shape is retained, the parser
must still record that `NONE` was action-masked rather than freely selected.

Likely code location:

- `src/mas_cc/games/relational_reasoning/imitation_round_feedback/prompts.py`

### Phase C: Align both validation layers

The prompt response contract and `game.validate_action()` must validate against
the same `C_i(t)` object.

Required cases:

- an active-memory fact is accepted;
- a grounded fact in the sampled current view is accepted;
- a fact on the live board but absent from the focal sample is rejected;
- an expired or self-excluded fact is rejected;
- a fact appearing only in free-form request/directive text is rejected;
- an unknown fact identifier is rejected;
- `REPORT` with null evidence is rejected when grounded reports are required;
- `NONE` uses null text, fact ID, and reply target;
- the empty-citable-set action mask cannot exhaust retries over the message
  portion of an otherwise valid ballot.

Replace the transition's current belt-and-braces check against
`active_before` with a check against the authoritative citable set. Keep the
check strict; do not remove it.

### Phase D: Commit acquisition, relay, and provenance atomically

Update `apply_round_event_transition()` so an observed fact can be acquired or
reactivated and relayed in the same accepted action.

The transition must:

1. derive `known_before` and `active_before`;
2. use the already-bound current exposures;
3. calculate `known_after` and `active_after`;
4. validate the report against `active_before union observed_now`;
5. append the blackboard message;
6. record acquisition and relay provenance;
7. commit all state together.

If transition validation fails, none of these state changes may be partially
applied.

Extend retained scientific event fields with compact audit columns such as:

```text
focal_citable_fact_ids_before_action
focal_observed_fact_ids_this_update
new_message_citation_source
new_message_source_message_id
communication_action_masked
communication_action_mask_reason
```

Avoid copying full prompt text or full board state into every row. IDs and
compact source metadata are sufficient and keep result size bounded.

If `BlackboardMessage` requires new provenance fields, introduce a new message
schema version and keep readers for versions 1 and 2. Prefer event-level fields
when they provide complete auditability without expanding every persisted
message.

### Phase E: Cover every controller communication mode

The participant citation rule must be independent of how the controller chose
its action.

| Controller output | Participant effect |
|---|---|
| Grounded `REPORT` | Fact enters `O_i(t)` only when that report is sampled and shown; it is immediately relayable. |
| `REQUEST` | May affect reasoning or reply behavior, but contributes no fact to `O_i(t)`. |
| `DIRECTIVE` | May recommend an action, but contributes no fact to `O_i(t)`. |
| Direct/transient recommendation | Contributes no fact unless the source explicitly carries a valid grounded report fact under an existing evidence-bearing mode. |
| Silent/no controller action | No controller exposure. Peer reports continue to work normally. |

Exercise at least these controller configurations:

- `controller_actuation_mode: adaptive_communication` with each of `REPORT`,
  `REQUEST`, and `DIRECTIVE` forced deterministically in tests;
- the current `llm_authored_report_only_v1` report policy;
- contextual/canonical report fallback;
- recommendation-only/direct recommendation;
- controller-disabled or silent control;
- participant requests enabled and disabled.

Controller `REQUEST` and `DIRECTIVE` messages must never acquire a
`shared_fact_id` merely to satisfy the participant relay feature.

#### Preserve the existing Agent N+1 public identity

The current renderer already supplies the intended public presentation:

- `control_label(population_size)` returns `Agent N+1`;
- `_message_source()` retains internal controller provenance but assigns that
  ordinary-looking label;
- `render_board_message()` renders the supplied label and does not render
  `author_kind` or `source_type`.

Do not replace this with a visible `Controller`, `Control source`, or similar
label while implementing citation context. Add regression tests asserting that
participant prompts contain `Agent N+1` and do not contain role-revealing
controller metadata. Keep the internal identity distinct from the N voting
participants so controller messages do not change population size, vote
denominators, focal-agent scheduling, or ordinary-agent metrics. In other
words, it is the `(N+1)`th public communication persona, not an extra voting
member of the simulated population.

The controller's authored `REPORT` must use the same visible message fields as
an ordinary report: agent label, message type, current vote, public prose, and
grounded fact. Its larger truthful fact pool, sensor access, target, timing,
and controller role remain private runtime state.

### Phase F: Serialization, checkpoint, and analysis compatibility

Update all relevant state/checkpoint/canonical paths so that:

- policy fields survive serialization and resume;
- provenance survives resume;
- a mid-episode checkpoint cannot switch citation semantics;
- old checkpoints restore without invented observed facts;
- old scientific event schemas remain readable;
- new audit fields are optional for historical data;
- aggregation groups or reports identify the citation policy used;
- studies with different citation policies cannot be silently pooled as the
  same treatment.

Likely code areas include:

- `checkpoint.py`;
- study canonicalization and compatibility manifests;
- relational pilot artifact/event schemas;
- blackboard documentation and preflight summaries.

Do not resume the failed job-67 episodes under the corrected policy. Their
scientific mechanism differs. A corrected study needs a new config identity and
result root, while the failed run remains diagnostic evidence.

### Phase G: Fail-fast safety for future paid runs

The incident also exposed a workflow safeguard gap. `fail_fast: false` allowed
hundreds of homogeneous semantic failures while Slurm tasks still exited zero.

Add a study-worker circuit breaker that is distinct from provider retry logic.
Suggested configurable conditions are:

```yaml
execution:
  semantic_failure_guard:
    minimum_finished_episodes: 2
    maximum_failure_fraction: 0.25
    minimum_failures: 2
```

When tripped, the shard should:

- stop starting new episodes;
- preserve all existing success records and failure checkpoints;
- exit nonzero with a distinct semantic-failure status;
- leave the cell unsealed;
- print a compact failure-stage/count summary;
- never classify the condition as a provider outage.

Keep this guard separately reviewable if it would make the immediate-relay
patch too broad, but require it before another 540-episode paid launch.

## 7. Test plan

### 7.1 Pure citable-set tests

Add deterministic unit tests for:

1. active private fact only;
2. observed peer report only;
3. observed controller report only;
4. active plus observed union in task fact order;
5. duplicate reports of one fact;
6. request and directive sources excluded;
7. unsampled live message excluded;
8. expired message excluded;
9. self-authored message excluded when configured;
10. unknown fact ID rejected.
11. controller report rendered as `Agent N+1` with no public controller-role
    marker.

### 7.2 Prompt and validation tests

Verify the exact prompt/contract behavior for:

- empty memory plus one visible report: that fact is citable immediately;
- empty memory and no visible grounded report: public action is `NONE`;
- empty memory plus only a request/directive: public action is `NONE` under the
  default policy;
- a valid vote remains valid when communication is action-masked to `NONE`;
- attempted unseen-fact citation receives precise repair guidance;
- observed-fact relay retains faithful fact text and source metadata;
- version 4 retains its exact old allowed-fact behavior;
- explicit `active_only` under version 5 reproduces the old behavior.

### 7.3 Transition and propagation tests

Construct a deterministic three-participant chain:

```text
A owns f1
A reports f1
B samples A's report and immediately relays f1
C samples B's report and acquires f1
```

Assert the knowledge sets, active sets, board messages, source message IDs, and
provenance after every update. Repeat with persistence deactivating `f1`, then
verify that a new visible report reactivates and permits immediate relay.

### 7.4 Controller-mode matrix

Use fake providers and deterministic controller policies. For every supported
controller actuation mode, assert that report facts are relayable and that
requests/directives do not masquerade as facts. Cover participant request
permissions independently from controller message permissions.

### 7.5 Legacy regression gate

Run the repository's byte-identity regression fixture with:

```yaml
report_citation_scope: active_only
no_citable_fact_action: model_select
prompt_version: 4
```

The old event stream, state transitions, prompt hashes, and retained tables
must remain byte-identical for the fixed fixture. If prompt hashes differ under
the legacy policy, treat that as a regression.

### 7.6 Checkpoint tests

- resume new-policy state without changing citable facts or provenance;
- restore historical checkpoints with legacy defaults;
- reject resume when citation policy or prompt version changes;
- ensure a same-update relay checkpoint cannot double-acquire a fact;
- ensure `NONE` action masking is stable across retry/resume.

### 7.7 Provider-free end-to-end smoke

Run a small fake-provider grid spanning persistence `0.70`, `0.85`, and `1.0`
with the report-only participant policy. Use at most two repetitions per cell.
Acceptance requires:

- every episode completes and seals;
- no `RelationalDecisionFailed` caused by an empty or newly observed fact set;
- immediate relay events occur at finite persistence;
- no request/directive is counted as evidence;
- strict aggregation accepts the complete smoke;
- retained artifacts remain bounded.

## 8. Paid-run acceptance sequence

After implementation and provider-free tests pass:

1. Create a new corrected false-arm config identity and result root.
2. Keep the failed job-67 tree unchanged as diagnostic evidence.
3. Preflight the corrected config and report cells, episodes, expected requests,
   expected/conservative cost, concurrency, RPM, memory, and wall time.
4. Run one paid episode at persistence `0.70` with low request concurrency.
5. Inspect its prompt, observed/citable sets, relay provenance, completion seal,
   and budget state.
6. Run a two-repetition, three-persistence pilot.
7. Require zero semantic-validation episode failures before proposing the full
   study.
8. Obtain explicit authorization before submitting the full paid run.

Do not infer readiness from Slurm `COMPLETED` alone. Require episode seals, cell
seals, and strict scientific completeness.

## 9. Documentation changes

Update the blackboard game documentation to explain:

- historical knowledge, active memory, current observation, and citable set;
- immediate relay semantics;
- the distinction between globally grounded truth and personally sourced
  evidence;
- provenance retained for relayed reports;
- why requests and directives do not transmit facts;
- deterministic `NONE` when no fact is remembered or seen;
- versioned legacy behavior and how to select it explicitly.

Also document that message lifetime controls visibility, while knowledge
acquisition controls memory. Expiring a board message must not erase a fact
that was legitimately acquired; persistence dynamics govern whether the fact
remains active.

## 10. Resource and artifact constraints

This feature does not require per-request files or copies of the board state.
Keep diagnostics compact:

- retain bounded validation summaries in failure checkpoints;
- store identifiers and provenance fields rather than duplicate fact text;
- do not persist every repair prompt/response by default;
- use temporary directories for tests;
- do not copy the failed run tree;
- do not create a paid smoke result until authorized;
- do not aggregate the incomplete diagnostic run as a final study.

The implementation should add no study-specific Slurm job and no new analysis
estimator.

## 11. Definition of done

The work is complete only when all of the following hold:

- new blackboard studies default to `active_or_observed` citation semantics;
- historical/version-4 and explicit `active_only` behavior remains available;
- a grounded report sampled in the current update is immediately relayable;
- an unseen or non-evidence message cannot authorize a citation;
- empty remembered-and-observed evidence produces deterministic `NONE` without
  sacrificing the private vote;
- all controller communication modes pass the behavior matrix;
- controller messages remain publicly attributable only to `Agent N+1`, while
  internal controller provenance remains available for analysis;
- acquisition and relay provenance are auditable and survive checkpoints;
- legacy byte-identity, unit, transition, checkpoint, and provider-free smoke
  tests pass;
- the semantic-failure guard prevents another large homogeneous failure run;
- a paid one-episode persistence-0.70 smoke completes before any full rerun;
- the corrected full study is submitted only with separate explicit user
  authorization.
