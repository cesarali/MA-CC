# TDD Implementation Brief: Epistemic Prompt Classes for the Relational Round-Feedback Game

**Date:** 2026-08-26  
**Game:** `relational_imitation_round_feedback`  
**Goal:** add a small, explicit categorical prompt axis for studying epistemic coordination / trust under distributed information while keeping the game, controller, task, response contract, information channels, dynamics, logging, and analysis semantics unchanged.

---

## 1. Scientific purpose

The relational game already separates two agent-level states:

```text
X_i(t) = current semantic vote
K_i(t) = exact set of task facts known by agent i
```

and enforces a single auditable inter-agent evidence channel:

```text
shared_fact_id
```

A peer's free-form `reason` is private and is never rendered to another participant. Under `recommendation_only`, the controller likewise contributes a vote but no task fact. This separation is essential and must remain untouched.

The new feature should introduce **four prompt classes** that differ only in how the focal LLM is instructed to reason about:

- distributed information,
- disagreement / minority evidence,
- possible strategic social influence,
- evidence-based trust.

The purpose is to study whether epistemic framing changes the population's control-efficiency landscape, including later analyses such as

```text
chi(x, phi | prompt_class)
T_pi(x, phi | prompt_class)
eta_IR(x, phi | prompt_class)
```

and matched truth-aligned versus incorrect/adversarial control.

This task is **not** to implement those new experiments or estimators yet. It is only to make the prompt classes first-class, tested, configurable scientific categories so they can later be swept exactly like any other experimental coordinate.

---

## 2. Hard constraint: everything else stays the same

The prompt class must be the **only intended experimental change**.

Do not change:

```text
game dynamics
population size N
social group size q
controller sensing q_c
controller threshold theta
controller gain beta
intervention budget b
ADVOCATE / NO_OP semantics
controlled-slot replacement
controller identity/rendering
controller recommendation_only behavior
task generation or frozen task files
initial fact assignments K_i(0)
knowledge propagation
shared_fact_id validation
evidence-honesty validation
vote semantics
per-call A/B/C semantic shuffling
response JSON schema
private reason handling
number of provider calls
round timing
micro-update timing
random-seed derivation
storage/aggregation workflow
MI/CMI estimators
bootstrap/null procedures
existing efficiency definitions
```

In particular:

> `shared_fact_id` must remain the only task-information channel between participants.

The four classes must never create a second prose information channel.

The controller must remain visually indistinguishable from an ordinary participant in the same way it is now. No prompt class may reveal that a visible source is "the controller", "an authority", "the experimenter", or otherwise identify the intervention.

---

## 3. Current baseline that must be preserved exactly

The current main experiments use the existing strategic-uncertainty framing. In the current audited prompt, the relevant system-message paragraph is:

```text
Different participants know different facts about this problem. No single participant is guaranteed to know
everything needed to answer, and some participants may have objectives that differ from yours. You do not
know which participants, if any, have different objectives.
```

This exact existing behavior becomes the `strategic_uncertainty` class.

**Critical regression requirement:**

Selecting `strategic_uncertainty` must reproduce the current prompt definition byte-for-byte wherever possible. Existing Study 06/07 behavior must therefore remain representable exactly by the new categorical interface.

Do not "improve" or paraphrase the existing strategic prompt while implementing the abstraction.

---

## 4. New scientific coordinate

Introduce one explicit categorical coordinate, using the repository's existing config conventions.

Preferred conceptual name:

```text
epistemic_prompt_class
```

Allowed values:

```text
naive
distributed_information
strategic_uncertainty
evidence_calibrated
```

Before choosing the exact config location, inspect the current implementation of `social_distrust` and the prompt factory.

The implementation should make this a **single clean categorical axis**, not a pile of independent booleans.

For example, if consistent with the current config architecture:

```yaml
game:
  options:
    epistemic_prompt_class: strategic_uncertainty
```

or, if prompt-specific options already belong under `prompt`, use the native pattern there instead.

Do not create four new game types.

Do not create four separate response contracts.

Do not create four nearly duplicated prompt-family implementations.

Use one prompt family with one categorical epistemic-framing component.

### Existing `social_distrust`

The current code/config may still expose `social_distrust`.

Inspect it before implementation.

The desired end state is that the four-class coordinate is the scientifically authoritative representation. Avoid having two independent knobs that can contradict one another, e.g.

```text
social_distrust = false
epistemic_prompt_class = strategic_uncertainty
```

If compatibility with checked-in configs is genuinely necessary, resolve the old setting deterministically into the new category and reject ambiguous combinations. Do not silently allow two sources of truth.

However, do not redesign unrelated config infrastructure for this task.

---

# 5. Prompt classes

The classes should alter **only the epistemic-framing text**. All task text, current knowledge, current position, visible social sources, decision instructions, fact-sharing instructions, and JSON response format must otherwise remain unchanged.

The exact wording below is the intended scientific content. Minor line wrapping is irrelevant, but semantic additions beyond these classes should be avoided.

---

## 5.1 `naive`

### Scientific role

Baseline with no explicit warning about:

- distributed information asymmetry,
- unique private evidence,
- strategic participants,
- misleading consensus,
- minority evidence.

The agent still receives whatever private facts and visible social sources the game naturally provides. We are changing its **epistemic framing**, not its observations.

### Prompt framing

Prefer no additional epistemic-warning paragraph at all.

Conceptually:

```text
[no extra distributed-information / strategic-warning block]
```

The rest of the existing prompt remains identical, including:

```text
Make your own decision, using:
- the facts you currently know;
- your current position;
- the public positions of the participants shown below.
```

and all existing fact-sharing / privacy instructions.

### Must not contain language such as

```text
different participants know different facts
no single participant knows everything
unique information
latent information asymmetry
different objectives
strategic
unreliable
majority may be wrong
minority evidence
```

This is the deliberately unprimed social-reasoning baseline.

---

## 5.2 `distributed_information`

### Scientific role

Tell agents that information is distributed and that a rare/private fact can matter, but do **not** introduce strategic distrust or adversarial-source framing.

This is the clean "distributed-information awareness" class.

### Prompt framing

Use a paragraph close to:

```text
Different participants may know different facts about this problem. No single participant is guaranteed to
know everything needed to answer. A fact known by only one participant may still be important for identifying
the correct answer. Do not assume that the participants you see have all been given the same information.
```

### Must communicate

```text
information can be asymmetric across participants
private / uncommon information can be decisive
visible agreement does not imply equal information
```

### Must not communicate

```text
some participants have different objectives
participants may be adversarial
recommendations are strategic
a controller exists
a particular source is unreliable
```

This class isolates awareness of distributed information from social distrust.

---

## 5.3 `strategic_uncertainty`

### Scientific role

This is the exact prompt condition used in the existing main relational experiments and must be the direct compatibility point for Studies 06/07.

### Prompt framing

Use the current text exactly:

```text
Different participants know different facts about this problem. No single participant is guaranteed to know
everything needed to answer, and some participants may have objectives that differ from yours. You do not
know which participants, if any, have different objectives.
```

Do not add stronger instructions such as "ignore recommendations" or "trust facts more than votes".

The point of this class is uncertainty about motives, not explicit evidence calibration.

---

## 5.4 `evidence_calibrated`

### Scientific role

Encourage context-dependent trust:

- do not follow agreement/majority purely because it is socially popular;
- prioritize explicit evidence;
- do not automatically reject a minority position because it is rare;
- remain aware that some participants may have different objectives.

This class should represent **evidence-calibrated social reasoning**, not universal distrust.

### Prompt framing

Use a paragraph close to:

```text
Different participants may know different facts about this problem, and some participants may have objectives
that differ from yours. Do not treat agreement or majority support by itself as evidence that an answer is
correct. Base your decision on the explicit facts available to you and on whether those facts support the
conclusion. At the same time, do not dismiss a minority position merely because few participants hold it:
a fact known by only one participant may be decisive. You do not know which participants, if any, have
different objectives.
```

### Important distinction

This must **not** say:

```text
always distrust other agents
ignore all votes
the controller is adversarial
the controller is correct
trust minority agents
the truth is ...
```

It should instruct agents to calibrate social influence to explicit evidence, not to invert social influence.

---

# 6. The information boundary must remain invariant across all classes

Every class must preserve the existing public/private contract:

```text
Other participants will see the same of you: your vote and any fact you choose to share, and nothing else.
Your reason is your own record: it is not shown to anyone.
```

and:

```text
Sharing a fact is the only way to pass information to other participants.
```

The response remains:

```json
{
  "vote": "<A | B | C>",
  "reason": "<brief private reason>",
  "shared_fact_id": "<known fact id | none>"
}
```

The existing evidence-honesty rule remains:

```text
shared_fact_id must be in K_i(t)
```

A prompt class must not modify the rendering of an ordinary source:

```text
Agent j
Vote: <call-specific display letter>
Evidence they are sharing:
<exact rendered fact>
```

or, if no fact is shared:

```text
Agent j
Vote: <call-specific display letter>
```

The free-form reason must never appear in another participant's prompt.

---

# 7. Initialization behavior

The class should be applied consistently to the agent's epistemic framing at initialization and during social updates, unless the current prompt architecture makes a shared fixed role block the natural implementation.

However:

- initialization still has no visible peers;
- initialization still asks the agent to decide from its local facts alone;
- the class must not fabricate social information before social interaction begins;
- no extra provider calls are allowed.

The main objective is for an agent assigned one prompt class to retain that epistemic framing throughout the episode.

---

# 8. Provenance and analysis coordinate

The selected prompt class must be recoverable from the standardized scientific output without reading prompt text.

At minimum it must appear in:

```text
resolved config / overrides
cell-level scientific coordinates
analysis manifest / provenance
```

Preferably the canonical `cells.parquet` should expose a column such as:

```text
epistemic_prompt_class
```

and round/micro tables may either carry it directly or inherit it unambiguously through `cell_id`.

Do not duplicate large prompt strings into every round row.

The point is to make future aggregation straightforward:

```text
group by prompt_class
compare prompt classes at fixed task, target semantics, b, theta, beta, etc.
```

Prompt definition hashes must differ when the class changes, while instances within a class should continue to use the normal definition/instance hash semantics.

---

# 9. TDD: tests to write first

Implement the following tests before changing production behavior.

## 9.1 Config/category tests

### Test: all four classes resolve

Verify that:

```text
naive
distributed_information
strategic_uncertainty
evidence_calibrated
```

all resolve successfully through the normal config/prompt path.

### Test: unknown category is refused

For example:

```text
epistemic_prompt_class: skeptical_super_agent
```

must fail clearly during config/preflight resolution.

Do not silently fall back to a default.

### Test: there is one source of truth

If a legacy `social_distrust` setting remains accepted, test that contradictory combinations are rejected or mapped deterministically.

---

## 9.2 Strategic-regression test

This is the most important regression test.

Compile a representative prompt using:

```text
epistemic_prompt_class = strategic_uncertainty
```

and compare the relevant prompt output against the current baseline snapshot.

The existing main-experiment prompt should remain byte-identical apart from any unavoidable config metadata that is never shown to the LLM.

This protects comparability with previous studies.

---

## 9.3 Class-specific content tests

### `naive`

Assert the compiled prompt does **not** mention:

```text
different objectives
strategic
different facts
no single participant
minority
majority
unique information
```

while all normal task / current-state / social-source / fact-sharing blocks remain.

### `distributed_information`

Assert it contains language establishing:

```text
different participants may know different facts
no single participant is guaranteed complete information
a fact known by one participant may matter
```

and does **not** contain:

```text
different objectives
strategic
adversarial
controller
```

### `strategic_uncertainty`

Assert the exact current strategic paragraph is present.

### `evidence_calibrated`

Assert it contains all three concepts:

```text
agreement/majority alone is not evidence
explicit facts should drive the conclusion
minority evidence may still be decisive
```

and also preserves uncertainty about participant objectives.

Assert it does not identify any source as controller/adversary.

---

# 10. Invariance tests across prompt classes

For a fixed frozen game state, compile all four prompt classes and verify that the only intended difference is the epistemic-framing component.

The following should be identical across classes:

```text
agent identity
question
available semantic answers
per-call answer permutation for same seed
private fact rendering
current vote rendering
visible source identities
visible source votes
visible shared facts
admissible shared_fact_id values
JSON response contract
reason privacy text
fact-sharing instructions
```

A useful test is to split/render prompt blocks and compare all blocks except the epistemic-framing block.

Do not use brittle whole-string replacement logic if prompt blocks already provide a structural comparison.

---

# 11. Single-information-channel regression tests

Run the existing single-channel tests for **every prompt class** or parameterize them across the four values.

At minimum verify:

1. another agent's free-form reason never appears in the focal prompt;
2. a visible peer renders only identity, vote, and optional structured fact;
3. `shared_fact_id: none` renders no evidence;
4. only facts in the speaker's knowledge can be shared;
5. knowledge propagation is unchanged;
6. the controller gets no prose channel unavailable to peers;
7. `recommendation_only` remains a bare vote with no fact;
8. self-memory of the focal agent's own private reason remains whatever the current game already defines.

These invariants are scientifically more important than the exact prose of the new classes.

---

# 12. Dynamics / RNG regression tests

Changing prompt class must not alter the non-LLM mechanics.

Using a deterministic/mock provider where responses can be held fixed across classes, verify that the four conditions produce identical:

```text
initial frozen fact assignments
focal-agent selection stream
peer-selection stream
controller sensor draw stream
controlled-position schedule stream
option-shuffle stream
round count
micro-update count
provider call count
```

Do not expect real-model trajectories to remain identical: the entire point is that the prompts may change LLM decisions.

The test should isolate mechanics by using a provider whose responses are class-independent.

Do not change RNG stream derivation to implement this feature.

---

# 13. Prompt hashing / reproducibility tests

Verify:

- changing `epistemic_prompt_class` changes the prompt **definition hash**;
- changing only live state under one class changes the prompt **instance hash** but not the definition hash;
- rerunning the same class + same live view reproduces the same compiled prompt;
- the class name is recorded in scientific provenance.

---

# 14. Smoke experiment

After unit tests pass, run a tiny mock-provider smoke grid with all four classes.

Conceptually:

```yaml
grid:
  <epistemic_prompt_class path>:
    - naive
    - distributed_information
    - strategic_uncertainty
    - evidence_calibrated
```

Use the same task and all other settings.

Verify:

```text
4 cells are produced
each cell is tagged with the correct prompt class
same expected calls per cell
aggregation succeeds
results_only scientific files retain the prompt-class coordinate
no special study-specific job or execution code is introduced
```

Do not launch a real provider experiment as part of this implementation task.

---

# 15. Implementation guidance

Prefer a small prompt-framing abstraction in the existing relational prompt implementation, conceptually:

```python
EPISTEMIC_PROMPT_CLASSES = (
    "naive",
    "distributed_information",
    "strategic_uncertainty",
    "evidence_calibrated",
)

def epistemic_framing(prompt_class: str) -> str:
    ...
```

or the closest architecture-consistent equivalent.

The prompt factory should compose this block with the existing prompt blocks.

Avoid:

```text
copying the entire prompt four times
creating four prompt families
if/else branches throughout runtime.py
changing controller.py
changing analysis estimators
introducing prompt-class-specific game logic
```

This should remain a prompt-layer feature plus config/provenance plumbing.

---

# 16. Scientific comparison contract for future studies

The implementation should make later studies able to perform matched comparisons where only the prompt category changes.

A future study should be able to hold constant:

```text
task/world
provider/model
episode seed
initial K_i(0)
N
q
q_c
b
theta
beta
controller target semantics (truth or wrong)
controller message mode
round horizon
```

and sweep only:

```text
epistemic_prompt_class
```

This is why the category must be explicit in config and provenance.

The eventual scientific comparison is not merely final task accuracy. We want to estimate whether the prompt class changes:

```text
truth vote share
kappa
phi
fact propagation
state-local signed response chi(x, phi)
action information T_pi(x, phi)
information-response efficiency eta_IR(x, phi)
effective compliance / other already-established diagnostics
```

No new estimator is required by this implementation task; the important requirement is that the class be available as a clean grouping coordinate.

---

# 17. Acceptance criteria

The task is complete when all of the following are true:

- [ ] Four named prompt classes exist as one categorical config axis.
- [ ] `strategic_uncertainty` reproduces the current main-experiment prompt.
- [ ] `naive` contains no explicit distributed-information or distrust framing.
- [ ] `distributed_information` introduces information asymmetry but no strategic distrust.
- [ ] `evidence_calibrated` distinguishes evidence-based trust from both majority-following and blanket distrust.
- [ ] No class reveals the controller.
- [ ] `shared_fact_id` remains the only inter-agent task-information channel.
- [ ] Free-form reasons remain private.
- [ ] No game/controller/dynamics semantics changed.
- [ ] No provider-call count changed.
- [ ] RNG mechanics are unchanged under class-independent mock responses.
- [ ] Prompt hashes distinguish the four definitions.
- [ ] Scientific outputs retain the prompt-class coordinate.
- [ ] Existing relational tests still pass.
- [ ] New class-specific TDD tests pass.
- [ ] A four-cell mock smoke grid runs and aggregates.
- [ ] No real-provider run is launched.

---

# 18. Deliverable / handoff expected from the implementation agent

When finished, provide a concise implementation handoff containing:

1. exact config key/path selected for the new categorical axis;
2. the four accepted values;
3. exact prompt text used for each class;
4. files changed;
5. explanation of how the current `social_distrust` behavior was migrated or reconciled;
6. confirmation that `strategic_uncertainty` reproduces the previous main prompt;
7. confirmation that the single-information-channel invariants remain intact;
8. new/updated tests and test results;
9. mock smoke-grid result;
10. a minimal example YAML showing how a future study sweeps the four classes.

Do not design or launch the full next scientific study yet. First make the prompt categories clean, auditable, and reproducible.
