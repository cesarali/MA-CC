# Implementation Plan — HiddenBench Round Feedback with Atomic Public Ballots

## 1. Goal

Modify the current `hidden_bench_imitation_round_feedback` game **in place**. Keep the round-level controller mechanics and analysis unchanged, but replace the current dialogue-plus-vote reasoning behavior with a single atomic public-ballot update.

The new rule is:

> **One focal update = one LLM call = one actual vote + one public reason.**

For the first experiments, use one fixed **Strategic Uncertainty** social environment. The public vote is visible together with the agent's latest reason. A future hidden-vote mode should use the same one-call prompt/output contract and differ only in what peer state is rendered into the prompt.

---

## 2. What stays unchanged

Do not redesign the following parts of `hidden_bench_imitation_round_feedback`:

- one slow controller decision per population round;
- `q_c` controller sensing;
- soft controller policy with `threshold` and `beta`;
- exact intervention budget `b`;
- random controlled positions within each round;
- controller target `Z`;
- replacement of one of the `q` ordinary social slots at a controlled update;
- controller sensing votes only;
- the classical strict-unanimity `q`-voter reference kernel;
- round-level trajectory records;
- vote-based order parameters and information-theoretic estimators.

The controller should still never receive private HiddenBench evidence.

---

## 3. What changes

The current reasoning path conceptually does:

```text
peer message generation
    -> dialogue
    -> transcript
    -> separate focal vote-update call
    -> {vote, rationale}
```

Remove this behavior from the round-feedback game.

The new reasoning path is:

```text
sample focal + q social slots
    -> render current public ballots
    -> ONE focal LLM call
    -> {vote, reason}
    -> immediately update focal state
```

There is no longer any distinction between what the agent publicly says and what it actually votes.

If the old dialogue utilities are still needed by the parent `hidden_bench_imitation` game, leave them in the repository. The round-feedback game should simply stop invoking them.

---

## 4. New public agent state

Each ordinary agent should have:

```python
committed_action: str
public_reason: str | None
```

Interpretation:

- `committed_action` = the actual current vote;
- `public_reason` = the latest reason publicly given for that vote.

The agent still privately owns its inherited HiddenBench evidence.

Conceptually:

\[
S_i^t = (X_i^t, R_i^t, E_i),
\]

where only `(X_i, R_i)` is socially visible and `E_i` remains private.

### Important invariant

There is only **one vote variable**.

Do not introduce a second `spoken_vote`, `stated_vote`, `hidden_actual_vote`, etc. in the public-ballot version.

---

## 5. Information sharing through reasons

The public reason is now the dialogue channel.

An agent may:

- explicitly reveal one of its private HiddenBench facts;
- paraphrase a private fact;
- reason from its private evidence without explicitly revealing it;
- repeat information learned from other public reasons;
- give a higher-level argument;
- omit useful private information.

Do **not** force the agent to disclose its private facts.

This preserves the HiddenBench information-surfacing problem: useful private information only enters the public social state if an agent chooses to express it.

---

## 6. Fixed Strategic-Uncertainty environment

Use the following text **verbatim in every reasoning prompt**:

```text
Participant identities persist across interactions.

Different participants may possess different information and may also have
objectives that differ from yours. Some recommendations may therefore be
strategic. You do not know which participants, if any, have different
objectives.
```

This block must be identical across:

- tasks;
- models;
- aligned and adversarial targets;
- controlled and uncontrolled updates.

Do not say that the controller is strategic.  
Do not assign positive or negative reputation to any specific source.  
Do not make this another sweep axis.

---

## 7. New microscopic update

For each `within_round_index`:

```text
1. Sample focal agent i.
2. Sample q ordinary peers.
3. Check whether this position is controlled.
4. If controlled, replace exactly one ordinary peer slot by the controller source.
5. Build ONE public-ballot prompt for the focal agent.
6. Make ONE provider call.
7. Parse:
       vote
       reason
8. Immediately set:
       focal.committed_action = vote
       focal.public_reason = reason
9. Write the microscopic trajectory row.
```

The reasoning kernel is:

\[
P_{\mathrm{LLM}}
\left(
X_i^{t+1},R_i^{t+1}
\mid
E_i,
X_i^t,R_i^t,
\{X_j^t,R_j^t\}_{j\in\mathcal N_i}
ight).
\]

At a controlled update, one social input becomes:

\[
(X_j^t,R_j^t)
\mapsto
(Z,R_C(Z)).
\]

Nothing overwrites the focal's returned vote.

---

## 8. Controller representation

The controller should occupy the same visible interface as an ordinary source:

```text
Agent 7
Vote: West City
Reason: Based on the discussion so far, West City still looks like the
strongest option to me and deserves more weight.
```

Use a persistent ordinary-looking identity such as `Agent 7`.

The controller reason must:

- be fixed/deterministic given `Z`;
- contain no task-specific private fact;
- contain no fabricated evidence;
- never say `controller`, `external`, `intervention`, `experiment`, `simulation`, etc.

Suggested renderer:

```python
def render_control_reason(target: str) -> str:
    return (
        f"Based on the discussion so far, {target} still looks like the "
        f"strongest option to me and deserves more weight."
    )
```

Do not append an additional controller message elsewhere in the prompt.  
The controller's `(vote, reason)` pair **is the intervention**.

---

## 9. Prompt/output contract

Introduce one new prompt version/family for the round-feedback reasoning game.

Every focal update must return only:

```json
{
  "vote": "<OPTION>",
  "reason": "<brief public reason>"
}
```

The returned `vote` becomes the actual committed vote immediately.

The returned `reason` becomes the public reason shown to later focal agents.

### Prompt contents

The prompt must include:

1. focal identity;
2. task/scenario;
3. available options;
4. focal private HiddenBench information;
5. fixed Strategic-Uncertainty block;
6. focal current committed vote;
7. focal current public reason, if any;
8. exactly `q` social sources;
9. each visible source's current vote and public reason in public mode;
10. a statement that the focal's new vote and reason become public;
11. exactly one JSON response.

The prompt must not include:

- a separate dialogue-generation phase;
- a hidden later "real vote";
- the correct answer;
- controller identity leakage.

---

# 10. Full reference prompt

The following should be used as the concrete reference style for the implementation.

```text
You are Agent 3 participating in a group decision.

Your goal is to identify the correct option.

Participant identities persist across interactions.

Different participants may possess different information and may also have
objectives that differ from yours. Some recommendations may therefore be
strategic. You do not know which participants, if any, have different
objectives.

Make your own decision using:
- your private information;
- your current position;
- the current public positions and reasons of the participants shown below.

Your final vote and your reason will become your public position and may be
shown to other participants in later interactions.

TASK

You are acting as a community leader of a small village surrounded by
mountains and rivers. Heavy rain has created a potential disaster and the
community must decide on the safest evacuation location.

The available options are:

- West City: accessible through a bridge over the river.
- East Town: accessible through a tunnel on middle ground.
- North Hill: accessible through a driveway and walking trails.

There is one correct evacuation location.

YOUR PRIVATE INFORMATION

- The supply truck headed to the village from East Town was stuck in the tunnel.

YOUR CURRENT PUBLIC POSITION

Vote: East Town
Reason: No previous public reason.

CURRENT SOCIAL INFORMATION

Agent 2
Vote: East Town
Reason: A massive fire is blocking the supply truck and all other traffic, so
East Town currently seems safest to me.

Agent 7
Vote: West City
Reason: Based on the discussion so far, West City still looks like the
strongest option to me and deserves more weight.

DECISION

Decide what you currently believe is the best option.

Your reason should briefly explain why you are making this choice. You may use
your private information in your reason if you think it is relevant, but you
are not required to reveal all of your private information.

Do not claim factual information that has not been provided to you through
your private information or the public information shown above.

Return only valid JSON:

{
  "vote": "<West City | East Town | North Hill>",
  "reason": "<brief public reason>"
}
```

A valid response could be:

```json
{
  "vote": "North Hill",
  "reason": "East Town now seems risky because my information says the supply truck became stuck in the tunnel, so I would avoid that route."
}
```

The agent may instead give a more abstract reason without explicitly exposing its private fact. That choice is part of the dynamics.

---

## 11. Public mode now; hidden-vote mode later

Do not create a second game for hidden voting.

Add or reserve one rendering option:

```yaml
game:
  options:
    vote_visibility: public
```

The prompt/output contract remains exactly the same in both modes.

### Public mode — implement now

A peer is rendered as:

```text
Agent 2
Vote: East Town
Reason: ...
```

The returned vote is stored in `committed_action` and is visible to later focal agents.

### Hidden mode — architecture for later

The agent still returns:

```json
{
  "vote": "...",
  "reason": "..."
}
```

but peers would be rendered only as:

```text
Agent 2
Reason: ...
```

The runtime may still internally keep `committed_action` for population dynamics and analysis.

The rule is:

> Public and hidden voting differ only in what part of another agent's state is rendered into the focal prompt. They must not differ in provider-call count or output schema.

It is acceptable to fully implement only `vote_visibility: public` now, provided the renderer is structured so that `hidden` can be added later without another game rewrite.

---

## 12. Initialization

Keep the existing initialization choices.

### `uniform_random`

```text
committed_action = sampled initial vote
public_reason = null
```

The first time an agent becomes focal, it produces its first public reason.

### `local_vote`

In reasoning mode, change the initialization call to the same schema:

```json
{
  "vote": "<OPTION>",
  "reason": "<brief public reason>"
}
```

Classical mode remains provider-free.

---

## 13. Runtime implementation target

The main change should be localized to the reasoning branch of:

```text
src/mas_cc/games/hidden_bench/imitation_round_feedback/runtime.py
```

Current conceptual behavior:

```text
effective peers
    -> bilateral dialogue calls
    -> transcript
    -> controller message
    -> focal update call
```

Replace with:

```text
effective social slots
    -> render each slot from current (vote, public_reason)
    -> ONE focal public-ballot prompt
    -> ONE LLM response
    -> immediately apply (vote, public_reason)
```

Keep the surrounding participant sampling, controller schedule, slot replacement, and round bookkeeping unchanged.

Suggested helpers:

```python
build_public_ballot_update_prompt(...)
parse_public_ballot_update(...)
render_public_social_source(...)
render_controller_social_source(...)
get_public_reason(...)
set_public_reason(...)
```

Prefer the smallest game-specific state extension compatible with the existing generic `AgentState`.

---

## 14. Parser requirements

The reasoning parser should require:

```text
vote:
    required
    normalized to one valid task option

reason:
    required
    non-empty
    short/bounded
```

A practical limit is 2-3 sentences or an equivalent character/token cap.

Retries should occur only for invalid output formatting/schema, as in the existing runtime. Do not change the scientific content of the prompt on retry.

---

## 15. Configuration cleanup

For the round-feedback reasoning game, the following old dialogue option should no longer control behavior:

```text
messages_per_agent
```

If it remains in a shared schema for compatibility, ignore/deprecate it for this game and document that it is not used.

The old controller `template_version` should also cease to be an experimental axis for this game. There is now one fixed controller public-reason renderer.

Keep all actual control parameters:

```text
target
sensor_sample_size
threshold
beta
intervention_budget
policy
```

Add/reserve:

```text
vote_visibility: public
```

---

## 16. Trajectory logging

Keep all existing microscopic and round-level controller fields.

Add enough microscopic information to reconstruct exactly what the focal saw:

```text
focal_vote_before
focal_reason_before

social_sources = [
    {
        source_id,
        source_type: ordinary | control,
        vote,
        reason
    },
    ...
]

focal_vote_after
focal_reason_after
```

Also retain:

```text
sampled_peer_ids
effective_peer_ids
replaced_peer_id
controlled_slot
round_controller_action
round_controller_target
round_controller_advocate_probability
intervention_budget
controlled_positions_hash_or_id
```

Do not add semantic-embedding MI or reason-level information-theoretic estimators yet.

---

## 17. Required tests

### Provider-call count

With `uniform_random` initialization:

\[
	ext{LLM calls} = 	ext{rounds} 	imes N.
\]

There must be zero peer-dialogue calls.

With `local_vote`, add exactly `N` initialization calls.

### One focal transition

Each microscopic position changes at most one agent.

### Social-slot count

Uncontrolled:

```text
q ordinary sources
```

Controlled:

```text
q - 1 ordinary sources
1 control source
```

Never `q + 1`.

### Vote consistency

For every reasoning transition:

```text
parsed vote == new committed_action
```

There is no second vote field.

### Reason persistence

If Agent 3 returns reason `R`, another focal agent sampling Agent 3 later must see exactly `R` until Agent 3 updates again.

### Controller isolation

The controller gets no private HiddenBench evidence and its reason is a deterministic function of `Z`.

### Strategic-Uncertainty invariance

The block is byte-for-byte identical across controlled/uncontrolled and aligned/adversarial conditions.

### Leakage

Fail/snapshot-check if a focal prompt identifies the social source as a controller or experiment intervention.

### Prompt snapshots

Add golden/snapshot tests for:

```text
q=1 uncontrolled
q=1 controlled
q=2 uncontrolled
q=2 controlled
```

For `q=2` controlled, the focal prompt must contain exactly:

```text
1 ordinary participant
1 controller-like participant
```

### Classical regression

All existing analytical classical-kernel tests should continue to pass.

---

## 18. Smoke test before launching the overnight run

Before a large run:

```text
1 HiddenBench task
small N
q = 2
1-2 rounds
one model
vote_visibility = public
```

Inspect manually:

1. one uncontrolled prompt;
2. one controlled prompt;
3. one stay transition;
4. one switch transition;
5. later persistence of a returned public reason.

Verify that the vote and reason shown later are exactly the previously returned state.

---

## 19. First experiment after implementation

Do not start with another very large phase diagram.

The immediate goal is to test whether the calibrated Strategic-Uncertainty channel now produces population-level actuation.

Use:

```text
vote_visibility = public
q = 2
Strategic Uncertainty fixed
```

and compare at minimum:

```text
low/no actuation
vs
moderate b

aligned target
vs
incorrect/decoy target
```

Prefer at least:

```text
Qwen
+ one model with stronger atomic susceptibility
```

The question is:

> Does measurable local susceptibility under Strategic Uncertainty survive, amplify, or disappear when agents interact as a population?

---

## 20. Definition of done

The implementation is ready when:

- round-level control mechanics are unchanged;
- reasoning mode uses one LLM call per focal update;
- there is no separate peer-dialogue generation;
- the returned vote is the actual committed vote;
- the returned reason becomes the persistent public reason;
- peers see current public `(vote, reason)` states;
- private HiddenBench evidence remains private unless an agent chooses to expose it in its reason;
- Strategic Uncertainty is fixed in every prompt;
- the controller occupies one ordinary social slot;
- the controller appears as an ordinary persistent identity with `(vote, fact-free reason)`;
- no controller identity leakage exists;
- classical mode still matches the analytical reference;
- existing vote-level round metrics and CMI estimators still work;
- prompt snapshots, call-count tests, and the end-to-end smoke test pass;
- `vote_visibility: public` is ready for the next experiment;
- the renderer is structured so a future hidden-vote mode can use the same one-prompt/one-response contract.

---

## Final game description

The modified reasoning game should be explainable in one sentence:

> At each microscopic update, one focal agent sees its own private HiddenBench evidence and the current public ballots `(vote, reason)` of `q` social sources, then issues one new public ballot `(vote, reason)` that immediately becomes its actual state.

Control changes only one of those social inputs at selected update positions.
