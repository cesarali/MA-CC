# Implementation Handoff: Finite-Memory q-Message Blackboard Extension

## Game

Existing game:

```text
src/mas_cc/games/relational_reasoning/imitation_round_feedback/
```

This is a **separate additive extension of the existing game**, not a new game and not a rewrite.

The MuSR-style Team Allocation data preparation/validation is handled separately and is assumed ready. This handoff concerns only the **social interaction / communication layer and controller delivery mechanism**.

---

# 1. Scientific objective

Add a new social mode in which the population interacts through a **finite-memory public blackboard**.

The central design is a direct generalization of the existing q-voter-like interaction:

```text
legacy peer mode:
    private evidence + q sampled current peers
        -> focal LLM update
        -> new vote

new board mode:
    private evidence + q sampled live board messages
        -> focal LLM update
        -> new vote + optional new public message
```

There must be **one social dynamics only** in board mode.

Do **not** give a focal agent both:

```text
q peers + q board messages
```

The board replaces the peer social channel.

The macroscopic population variable remains the current vote configuration:

\[
X_i(t) \in \{1,\ldots,K\}, \qquad
N_k(t)=\sum_i \mathbf{1}[X_i(t)=k].
\]

The board is the microscopic communication substrate that generates those vote transitions.

---

# 2. Why this extension

The current game samples a small number `q` of contemporaneous peer states. The new design should retain the same limited-information idea, but make the sampled object a **persistent natural-language message** rather than a current peer snapshot.

This permits:

- asynchronous indirect influence;
- information requests;
- replies;
- corrections;
- reporting intermediate conclusions;
- temporary coordination;
- second-order propagation of controller interventions.

At the same time, finite message lifetime prevents the board from growing without bound and keeps random message sampling scientifically interpretable.

The intended controlled analogy is:

```text
q-voter:
    q sampled current social states

q-message:
    q sampled live messages from a finite-memory board
```

---

# 3. Hard compatibility constraints

Preserve as much of the existing implementation as possible.

The current game already has:

- current semantic vote per agent;
- exact known evidence/fact IDs;
- one focal update at a time;
- `q` social-source sampling;
- round clock and microscopic update clock;
- controller sensing `q_c`;
- `NO_OP / ADVOCATE_Z`;
- exact actuation budget `b`;
- semantic option shuffling;
- provider call/retry/validation machinery;
- round and microscopic records;
- control / information analysis.

The current implementation map places the relevant responsibilities in:

```text
state.py
prompts.py
game.py
controller.py
runtime.py
metrics.py
analysis.py
```

The blackboard should be implemented by extending these components rather than creating a parallel runtime.

**Regression requirement:**

```yaml
social_mode: peer
```

must preserve the existing behavior.

No existing peer-mode study should silently change.

---

# 4. New social mode

Introduce a first-class categorical mode, conceptually:

```yaml
game:
  options:
    social_mode: peer   # legacy
```

or:

```yaml
game:
  options:
    social_mode: board  # new
```

Use the repository's native config style after inspecting the current models.

The meaning of `q` remains:

```text
q = maximum number of social observations available to one focal update
```

but the observation type depends on `social_mode`.

### Peer mode

```text
sample q peer agents
```

### Board mode

```text
sample q live board messages
```

Do not create an independent `q_board` parameter in the first implementation.

---

# 5. Blackboard state

Add a typed append-only message object plus a board container.

Conceptually:

```python
@dataclass(frozen=True, slots=True)
class BlackboardMessage:
    message_id: str
    author_id: int | str
    message_type: str
    text: str
    vote: str
    shared_fact_id: str | None
    reply_to: str | None
    round_created: int
    micro_step_created: int
    expires_after_round: int

    # provenance only; not necessarily rendered
    author_kind: str = "agent"  # agent | controller
```

Prefer reusing the existing `shared_fact_id` field internally for compatibility rather than renaming the entire epistemic machinery. For MuSR tasks, that ID can refer to an evidence card through the task adapter.

Board state conceptually:

```python
@dataclass
class BlackboardState:
    messages: list[BlackboardMessage]
```

Required operations:

```text
append(message)
live_messages(round_idx)
sample_live(...)
expire(...)
find(message_id)
```

Do not introduce embeddings, vector search, ranking, or semantic retrieval in this change.

---

# 6. Message types: communication roles

Ordinary agents do **not** receive fixed permanent roles.

Instead, coordination roles are expressed through the type of message they choose to post. This allows roles such as requester, responder, and corrector to emerge dynamically.

Use a small controlled vocabulary:

```text
CLAIM
QUESTION
REQUEST
RESULT
REPLY
CORRECTION
```

Semantics:

### `CLAIM`

A current hypothesis or inferred conclusion.

### `QUESTION`

Asks for missing information.

### `REQUEST`

Asks others to check, compare, or report something.

### `RESULT`

Reports the outcome of reasoning from available evidence.

### `REPLY`

Direct response to an earlier message. `reply_to` is required.

### `CORRECTION`

Explicitly disputes or updates an earlier message. `reply_to` is required.

Do not create a large taxonomy in the first version.

---

# 7. Private reason versus public message

Preserve the current private reasoning field.

A ballot in board mode should conceptually become:

```json
{
  "vote": "B",
  "reason": "private reasoning recorded for analysis",
  "shared_fact_id": "e17",
  "public_message": {
    "type": "RESULT",
    "text": "The evidence I have suggests Ana has strong technical experience.",
    "reply_to": null
  }
}
```

`reason` remains private.

`public_message.text` is deliberately public and may be read by later agents.

Allow:

```json
"public_message": null
```

so an agent can choose not to post.

One focal update may create **at most one** new board message.

If `public_message` is null in board mode, `shared_fact_id` must also be null because there is no public carrier for the evidence.

---

# 8. Important epistemic-semantics change

The old relational game deliberately enforced:

```text
shared_fact_id = the only inter-agent information channel
```

That exact interpretation **cannot remain true in board mode**, because public natural-language messages intentionally communicate derived claims, questions, summaries, and corrections.

Therefore preserve the exact evidence-ID state:

\[
K_i(t)=\text{set of exact evidence/fact IDs explicitly acquired by agent }i
\]

but reinterpret it carefully:

> `K_i(t)` is an exact record of source-evidence acquisition, not a complete representation of everything the agent may have learned semantically from board prose.

Existing knowledge metrics based on `K_i` remain useful as **evidence coverage / evidence propagation** observables, but must not be described as exhaustive epistemic state under board mode.

---

# 9. Evidence honesty and provenance

A public message may contain free-form reasoning, but structured evidence references remain auditable.

If an ordinary agent posts:

```text
shared_fact_id = e17
```

require:

```text
e17 in K_i(t)
```

exactly as in the current game.

The rendered board message should include the authoritative frozen text for that evidence ID, not an invented paraphrase.

When another focal agent actually reads that message:

```text
K_focal <- K_focal union {e17}
```

and record acquisition provenance.

A free-form `CLAIM` or `RESULT` without a `shared_fact_id` does **not** create a new exact evidence ID.

Do not attempt to parse new symbolic facts from prose.

---

# 10. Message lifetime / board memory

Introduce an explicit finite memory parameter:

```yaml
board:
  message_lifetime_rounds: 1
```

Default:

```text
tau_B = 1 round
```

Interpret one existing population round as one communication "day."

For a message created during round `r` and `tau_B = 1`:

```text
readable for the remainder of round r
expired before round r+1 begins
```

Thus the live board grows during a round and is cleared at the round boundary for `tau_B = 1`.

Implement lifetime generically enough that later experiments can use:

```text
tau_B = 2, 3, ...
```

without redesigning the state object.

Do not implement probabilistic decay yet.

---

# 11. Empty / small board at the beginning of a round

Do not reintroduce peer sampling as a fallback.

Let:

```text
eligible = live board messages not authored by the focal agent
q_eff = min(q, len(eligible))
```

Then sample exactly `q_eff` messages.

If:

```text
q_eff = 0
```

the focal update proceeds from private state only.

Record both:

```text
q_requested
q_effective
```

This early-round sparsity is part of the finite-memory board dynamics and should be observable rather than hidden by another channel.

---

# 12. Board sampling rule

Initial board retrieval must be deliberately simple:

```yaml
board:
  sampling: uniform
  exclude_self_authored: true
```

Sample uniformly without replacement from currently live eligible messages.

Important:

- sample **messages**, not authors;
- multiple sampled messages may come from the same author;
- the same exact message cannot occupy two slots in one focal update;
- self-authored messages are excluded by default;
- no relevance ranking;
- no recency weighting;
- no embeddings.

This is the cleanest q-message analogue for the first experiment.

---

# 13. One microscopic board update

For `social_mode: board`, the ordinary microscopic update is:

```text
1. sample focal agent i
2. determine currently live eligible board messages
3. uniformly sample up to q messages
4. render:
      - task
      - focal private evidence
      - focal current vote / own state
      - sampled board messages
5. make one normal LLM decision call
6. validate vote
7. validate optional public message
8. validate optional shared_fact_id honesty
9. apply focal vote immediately
10. acquire any structured evidence actually exposed in sampled messages
11. append focal public message if one was produced
12. record the complete micro transition
```

There is no peer sample in this path.

The provider-call count remains one call per microscopic focal update.

---

# 14. Controller: preserve the existing sensing/action policy

Do **not** redesign the controller's decision policy in this extension.

Preserve the existing round-level controller:

```text
sample q_c agent votes
        ↓
sensor Y
        ↓
soft policy
        ↓
U ∈ {NO_OP, ADVOCATE_Z}
```

Keep:

```text
q_c
beta
theta
target Z
b
controlled-position scheduling
```

as they currently work.

The new experiment should isolate a change in **how ADVOCATE_Z is delivered**, not simultaneously change the sensor.

The controller still does not inspect hidden agent evidence.

---

# 15. Two controller actuation modes

Introduce one explicit controller-delivery coordinate for board experiments.

Conceptually:

```yaml
controller_actuation_mode: direct_recommendation
```

or:

```yaml
controller_actuation_mode: coordination_request
```

Do not overload the existing `message_mode` if that field already means `recommendation_only` versus `recommendation_plus_fact`.

The new coordinate is about **delivery mechanism**.

---

# 16. Mode A: `direct_recommendation`

This is the closest board-mode analogue of the current controller and serves as the control reference.

On an `ADVOCATE_Z` round:

- sample exactly `b` distinct microscopic positions using the current schedule machinery;
- at each controlled position, construct the focal's normal board sample;
- replace one social slot with a transient controller recommendation;
- the controller recommendation is **not appended to the board**;
- it disappears after that focal update.

Conceptually for `q = 1`:

```text
ordinary position:
    focal reads one sampled live board message

controlled position:
    focal reads one transient controller recommendation instead
```

The agent is free to reject the recommendation.

The controller should remain rendered as an ordinary participant, consistent with the existing game; do not label it "controller", "expert", or "authority."

This mode preserves the meaning:

```text
b = number of directly controlled social exposures
```

as closely as possible.

---

# 17. Mode B: `coordination_request`

This is the new METR-inspired controller.

The controller does **not** directly force a recommendation into the focal's social sample.

Instead, on an `ADVOCATE_Z` round:

- use the same existing set of `b` scheduled microscopic positions;
- at each scheduled position, append one controller-authored `REQUEST` message to the board;
- append it **before** the ordinary board sample for that position is drawn;
- the focal then samples normally from the board;
- therefore the newly posted controller message may or may not be sampled immediately;
- it persists according to `tau_B` and may be read by later agents;
- it can receive replies/corrections/results from ordinary agents.

Thus:

```text
b = number of coordination messages injected into the shared environment
```

not the number of guaranteed direct exposures.

This difference is intentional and must be logged clearly.

A single controller post can potentially generate multiple later exposures and second-order responses.

---

# 18. Controller coordination-message content

Do not use another LLM for the controller in the first implementation.

Keep controller messages deterministic and auditable.

The controller knows:

- target option `Z`;
- the `q_c` sampled votes used by its sensor;
- therefore the strongest sampled rival option can be determined.

A deterministic template can be:

```text
Message type: REQUEST
Current vote: <Z>

Please share evidence that helps distinguish option <Z> from option <RIVAL>.
If you have evidence supporting or contradicting either allocation, report it.
```

If no unique rival is available:

```text
Please share evidence that bears on whether option <Z> is the best allocation.
Report evidence that supports or contradicts it.
```

Do not fabricate task evidence.

For the first implementation:

```text
controller shared_fact_id = null
```

under `coordination_request`.

This is coordination through information solicitation, not evidence injection.

The controller's hidden provenance should be recorded, but the rendered identity should remain an ordinary participant identity consistent with the current game.

---

# 19. Meaning of the two controller modes

The scientific contrast is:

### Direct recommendation

```text
controller
    -> direct transient exposure
    -> focal agent
    -> possible vote change
```

### Coordination request

```text
controller
    -> persistent request on board
    -> stochastic readers
    -> replies / results / corrections
    -> further readers
    -> eventual population response
```

The second mode allows control through **organization rather than coercion**.

Agents retain autonomy:

- they may never sample the controller message;
- they may read and ignore it;
- they may disagree;
- they may answer it;
- they may correct it;
- they may change vote;
- they may propagate relevant evidence to others.

---

# 20. Board/controller logging

Extend microscopic and round records rather than creating a separate data pipeline.

At minimum record per microscopic update:

```text
social_mode
q_requested
q_effective
sampled_message_ids
sampled_message_authors
sampled_message_types
sampled_message_ages
sampled_controller_message_ids

focal_posted_message
new_message_id
new_message_type
new_message_reply_to
new_message_shared_fact_id

board_size_before
board_size_after

controller_actuation_mode
controlled_position
controller_message_posted
controller_message_id
controller_message_directly_exposed
```

At round level record:

```text
board_messages_created
board_messages_expired
board_peak_size
board_mean_size

message_type_counts
reply_count
correction_count
request_count
result_count

controller_posts
controller_message_exposures
controller_unique_readers
controller_direct_replies
controller_reply_descendants

peer_evidence_exposures
new_evidence_acquisitions
```

Preserve current vote-count and controller fields unchanged where possible.

---

# 21. Reply graph / coordination provenance

The message board naturally creates a directed graph through:

```text
reply_to
```

Record it exactly.

Do not implement complex graph analysis inside the runtime.

The records should be sufficient to reconstruct:

```text
REQUEST -> REPLY
REQUEST -> RESULT
CLAIM -> CORRECTION
controller request -> agent reply -> later reply
```

Offline analysis can later compute:

- reply depth;
- branching;
- response latency;
- controller descendant count;
- fraction of requests answered;
- question-to-result transitions.

The first implementation only needs to preserve the graph faithfully.

---

# 22. Basic new observables

Add lightweight diagnostics; do not turn this extension into a large new analysis project.

Useful quantities include:

- communication activity: ordinary messages posted per round;
- controller exposures: number of times controller board messages are actually sampled;
- unique controller reach: distinct ordinary agents that read a controller message;
- direct response count: messages replying directly to controller messages;
- descendant activity: ordinary messages descending from controller messages;
- evidence response: exact evidence IDs shared in direct/indirect response to controller requests.

Do not define a new final "coordination efficiency" yet. Log the ingredients first.

---

# 23. Message lifetime observables

Because persistence is now an experimental property, retain enough fields to study it.

For every message exposure record:

```text
message_age_micro_steps
message_age_rounds
```

For each round record:

```text
expired_message_count
surviving_message_count
```

For the initial study use:

```text
tau_B = 1
```

but make the implementation generic.

---

# 24. Interaction with MuSR evidence

The task adapter should continue exposing stable evidence IDs.

An ordinary agent's private state remains conceptually:

```text
X_i(t) = current allocation vote
K_i(t) = exact evidence IDs acquired
```

A board message may:

1. contain a derived natural-language statement;
2. optionally expose one exact evidence item from `K_i`.

This is particularly useful for the MuSR task because an agent can communicate both raw evidence and a derived inference.

Only the raw evidence ID enters another agent's exact `K_i`.

The derived inference remains a semantic message.

---

# 25. Prompt design

Board-mode prompts must tell agents the real communication contract.

They should understand:

- different agents hold different evidence;
- they see only a small sample of currently live board messages;
- messages are temporary;
- their private reasoning is not public;
- they may post at most one public message;
- public messages may be read by later agents;
- they may ask questions, report results, reply, or correct;
- other participants may have different objectives if the configured epistemic prompt class already says so;
- they should evaluate messages rather than automatically follow them.

Do not identify the controller as an authority.

Do not silently change existing epistemic prompt classes. Integrate board instructions orthogonally.

---

# 26. Public-message validation

Validate the response structurally.

Rules:

- message type must be in the allowed enum;
- text must be non-empty if a message exists;
- `REPLY` and `CORRECTION` require a valid `reply_to`;
- `reply_to` must reference a message actually visible to the focal agent in that update, unless the existing prompt explicitly gives broader board IDs;
- shared evidence must already be known by the author;
- at most one exact evidence item is exposed initially;
- no hidden task metadata may enter the rendered prompt;
- no extra provider call is made to classify message type.

Do not attempt semantic truth validation of free-form prose at runtime.

---

# 27. Board clearing

For `tau_B = 1`, perform expiration at a single well-defined clock boundary.

Preferred:

```text
end round r
    -> emit final round record
    -> expire all messages whose lifetime ends at r
    -> begin round r+1 with an empty board
```

or an equivalent deterministic ordering if the current recorder requires expiration before the record.

Choose one convention, document it, and test it.

Do not allow off-by-one lifetime ambiguity.

---

# 28. Theory / analysis warning

The existing q-voter reference is an exact/matched reference for the current peer-interaction protocol under its assumptions.

A finite-memory q-message board changes the microscopic social process.

Therefore:

- keep existing empirical vote/control observables;
- keep CMI/sensing/current estimators where their definitions remain valid;
- do **not** silently claim that the existing q-voter theory is an exact theory of board mode;
- mark board-mode theory status explicitly as `reference_only`, `unsupported`, or another repository-consistent label until a q-message theory is derived;
- do not feed board-mode runs into theory code that assumes contemporaneous q-peer sampling without an explicit guard.

The classical q-voter remains a useful baseline/reference, but not automatically the exact microscopic model of the board.

Likewise, evidence-coverage metrics from `K_i` remain exact evidence metrics but are no longer exhaustive measures of semantic knowledge because prose is now a deliberate information channel.

---

# 29. Suggested configuration

Conceptually:

```yaml
game:
  type: relational_imitation_round_feedback
  options:
    task_family: musr_team_allocation
    social_mode: board
    q: 1

    board:
      sampling: uniform
      message_lifetime_rounds: 1
      exclude_self_authored: true
      allow_no_post: true

control:
  mechanism: relational_round_budgeted
  q_c: 12
  b: 6
  beta: 4.0
  theta: 0.5
  controller_actuation_mode: coordination_request
```

Reference arm:

```yaml
controller_actuation_mode: direct_recommendation
```

Adapt exact field placement to the repository's current config models.

---

# 30. Implementation staging

This should be feasible as one implementation task, but make the code changes in a disciplined sequence.

## Stage 1 — state/schema only

Add:

- board state;
- message schema;
- lifetime logic;
- serialization;
- config parsing.

No behavior change in peer mode.

## Stage 2 — ordinary board dynamics

Implement:

```text
sample q live messages
render them
one focal LLM call
append optional public message
expire by lifetime
```

Add micro/round logging.

Run mock-provider tests.

## Stage 3 — direct recommendation controller

Adapt current slot replacement so it works with the board social source.

Verify exact `b` direct exposures on advocacy rounds.

## Stage 4 — coordination-request controller

Use the same round policy and `b` schedule, but append persistent deterministic `REQUEST` messages rather than force exposure.

Record actual downstream read/reply behavior.

## Stage 5 — metrics/reporting

Add only the basic communication diagnostics listed above.

Do not redesign the existing analysis stack.

---

# 31. Tests: backward compatibility

### Peer-mode regression

With fixed seeds and mock provider:

```yaml
social_mode: peer
```

must preserve:

- focal sampling;
- peer sampling;
- prompt content where possible;
- controller sensing;
- controller action;
- controlled-position schedule;
- vote transitions;
- evidence propagation;
- provider call count;
- round records;
- termination.

The blackboard code must be dormant.

---

# 32. Tests: ordinary board mode

Test:

1. no peer agents are sampled in board mode;
2. q messages are sampled when enough live messages exist;
3. `q_effective < q` when the board is too small;
4. empty board gives zero social messages without a peer fallback;
5. self-authored messages are excluded;
6. same author may contribute multiple different sampled messages;
7. one exact message is never sampled twice in one update;
8. optional `NO_POST` produces no appended message;
9. at most one new message per update;
10. public message is visible to later agents;
11. private reason is never rendered socially;
12. shared evidence ID must already be known by author;
13. reading a message containing structured evidence updates exact evidence provenance;
14. prose without evidence ID does not create exact evidence state.

---

# 33. Tests: lifetime

For `tau_B = 1`:

- a message created early in round `r` can be read later in round `r`;
- it cannot be read in round `r+1`;
- all expiration counts are deterministic.

Also add at least one unit test for:

```text
tau_B = 2
```

to ensure the data model genuinely supports longer persistence.

---

# 34. Tests: message types and replies

Test:

- all allowed types parse;
- unknown type fails;
- `REPLY` without `reply_to` fails;
- `CORRECTION` without `reply_to` fails;
- invalid/non-visible reply target fails;
- reply graph survives serialization;
- message type counts are correct.

---

# 35. Tests: `direct_recommendation`

On an advocacy round:

- exactly `b` controlled positions;
- each controlled position contains one transient controller social source;
- controller message is not appended to the board;
- ordinary board messages occupy the remaining social slots if `q > 1`;
- controller is rendered without authority labels;
- `NO_OP` produces zero controller exposures.

For `q = 1`, verify:

```text
ordinary position -> one sampled board message
controlled position -> one transient controller recommendation
```

---

# 36. Tests: `coordination_request`

On an advocacy round:

- exactly `b` controller messages are appended;
- each has type `REQUEST`;
- each has distinct message ID;
- controller posts use the scheduled positions;
- posts occur before sampling at that position;
- no controller post is guaranteed to be read;
- a controller post can be read multiple times by different later agents;
- controller messages obey the configured lifetime;
- replies to controller messages are valid;
- `NO_OP` appends zero controller messages;
- controller injects no fabricated evidence;
- controller sensing remains the existing `q_c` vote sensor.

---

# 37. Provenance requirements

Canonical output must make it possible to determine without inspecting prompts:

```text
task_family
social_mode
q
board sampling policy
message_lifetime_rounds
message type
message author
message author kind
message creation time
message expiry time
message reply target
message shared evidence ID

controller_actuation_mode
controller action U
controller target
q_c
b
controlled schedule
controller post IDs
controller exposures
```

Prompt definition hashes should distinguish peer and board prompt families/modes.

---

# 38. Documentation

Update the game README with a new section explaining:

```text
peer mode
board mode
message types
finite lifetime
exact evidence state versus semantic prose
direct recommendation controller
coordination-request controller
```

Include one worked microscopic example for `q = 1`, `tau_B = 1`.

Example:

```text
Round starts: board empty.

Update 1:
Agent 4 sees no board message.
Agent 4 votes B and posts QUESTION m1.

Update 2:
Agent 9 samples m1.
Agent 9 votes C and posts REPLY m2 with evidence e17.

Update 3:
Agent 2 samples m2, acquires e17, and changes vote.

...

End of round:
m1, m2, ... expire.
```

Then show the coordination-controller variant where a controller `REQUEST` is posted and later answered.

---

# 39. Non-goals

Do not implement in this change:

- semantic retrieval;
- embeddings;
- message ranking;
- recency weighting;
- global board summaries;
- agent-specific inboxes;
- fixed permanent ordinary-agent roles;
- explicit task delegation queues;
- tool use;
- new LLM controller;
- probabilistic message decay;
- new thermodynamic theory;
- new final coordination-efficiency metric;
- cross-round long-term board memory beyond generic TTL support.

Keep the first version experimentally interpretable.

---

# 40. Acceptance criteria

The extension is complete when:

1. legacy `social_mode: peer` still works unchanged;
2. `social_mode: board` uses **only q board messages** as the social channel;
3. messages support the six controlled communication roles;
4. private reasons remain private;
5. optional public messages persist according to `tau_B`;
6. `tau_B = 1` clears the previous round's messages deterministically;
7. structured evidence sharing remains auditable;
8. exact evidence state is clearly distinguished from semantic knowledge;
9. both controller modes work:
   - `direct_recommendation`
   - `coordination_request`;
10. direct recommendation preserves exact-budget transient exposure semantics;
11. coordination request injects exactly `b` persistent requests on advocacy rounds;
12. agents are free to ignore, answer, correct, or act on controller requests;
13. controller messages are not rendered as authoritative/system messages;
14. reply/provenance graph is reconstructable from stored records;
15. board/controller diagnostics are emitted;
16. board runs are guarded from being silently interpreted as exact q-voter-theory runs;
17. mock-provider tests cover all critical behavior;
18. no extra provider call is introduced per ordinary microscopic update.

---

# 41. Scientific interpretation to preserve

The intended object is a **finite-memory q-message process**:

\[
B_t
\xrightarrow{\text{sample }q}
\text{focal LLM}
\xrightarrow{}
(X_i',m_{\mathrm{new}})
\xrightarrow{}
B_{t+1}.
\]

The controller has two experimentally distinct ways to intervene:

\[
\text{direct recommendation}
\]

versus

\[
\text{persistent coordination request}.
\]

The second mode is intentionally designed to test whether a resource-limited controller can steer a population by **organizing information production and propagation**, rather than by directly coercing votes.

The primary macroscopic state remains the population vote configuration. The board is the finite-memory communication mechanism through which that population evolves.
