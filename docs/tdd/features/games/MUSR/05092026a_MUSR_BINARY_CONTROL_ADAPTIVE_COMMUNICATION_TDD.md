# TDD: Binary-Control Adaptive Communication for the MuSR Blackboard

**Date:** 2026-09-05  
**Status:** implementation handoff  
**Scope:** MuSR blackboard controller + configurable communication policy  
**Primary requirement:** preserve the existing **binary controller variable** and backward compatibility with prior controller modes

---

# 1. Goal

Extend the MuSR blackboard game so that, **conditional on the existing binary controller action being ON**, the controller may choose how to communicate.

The controller should be able to choose among:

```text
REPORT
REQUEST
DIRECTIVE
```

subject to experiment-level feature switches.

At the same time, preserve the main theoretical control variable exactly as:

\[
U_k \in \{0,1\},
\]

with:

```text
U_k = 0  -> controller does nothing
U_k = 1  -> controller acts
```

The communication form chosen when `U_k=1` is a **secondary internal actuation mechanism**, not a replacement for `U`.

This is required so the existing binary-control information quantities and theoretical information bound remain the primary objects.

---

# 2. Scientific architecture

Preserve the causal structure:

\[
n_k
\rightarrow
Y_k
\rightarrow
P(U_k\mid Y_k)
\rightarrow
U_k
\rightarrow
\text{communication realization}
\rightarrow
n_{k+1}.
\]

Do not replace the existing sensor or binary controller policy.

The existing controller must still first decide:

```text
NO_OP
or
ACT
```

using the existing binary stochastic policy.

Only after:

```text
U_k = ACT
```

should the new adaptive communication layer choose a message strategy.

Define a secondary diagnostic variable:

\[
M_k \in
\{
\text{REPORT},
\text{REQUEST},
\text{DIRECTIVE}
\}.
\]

`M_k` exists only when `U_k=1`.

The main theoretical intervention variable remains binary.

---

# 3. Preserve the existing information-theoretic analysis

The primary transfer-information quantity remains:

\[
T_\pi = I(U_k;n_{k+1}\mid n_k)
\]

with binary `U`.

Do not redefine this quantity to use the categorical communication mode.

Likewise, preserve all existing analysis paths whose theoretical interpretation depends on binary intervention, including where applicable:

```text
chi
eta_IF
eta_IR
binary-control information bound
existing state-local estimators
existing U-conditioned response statistics
```

The chosen message mode may be logged and analyzed secondarily, for example:

\[
P(M_k=\text{REPORT}\mid U_k=1),
\]

\[
P(M_k=\text{REQUEST}\mid U_k=1),
\]

\[
P(M_k=\text{DIRECTIVE}\mid U_k=1).
\]

Optional secondary diagnostics may later include:

\[
I(M_k;n_{k+1}\mid n_k,U_k=1),
\]

but this is **not** part of the primary theory in this implementation.

---

# 4. Theoretical interpretation

Treat the adaptive communication layer as part of the controlled kernel.

Conceptually:

\[
Q_0 = \text{uncontrolled dynamics},
\]

while:

\[
Q_1 = \text{effective controlled dynamics when }U=1.
\]

Internally, `Q1` may be realized through different communication modes:

\[
Q_1
=
\sum_m
P(M=m\mid U=1,\text{context})Q_{1,m},
\]

but the main theory should continue to use the effective binary pair:

\[
Q_0,\;Q_1.
\]

Do not expose the categorical decomposition as a required theoretical variable unless a later theory extension explicitly asks for it.

---

# 5. Experiment-level feature switches

Add explicit configuration switches.

Recommended:

```yaml
allow_participant_requests: true
allow_controller_requests: true
allow_controller_directives: true
```

Keep REPORT available to the controller in adaptive mode.

The switches define the **allowed communication vocabulary when U=1**, not the value of `U`.

---

# 6. Participant REQUEST on/off switch

Ordinary agents currently have:

```text
REQUEST
REPORT
NONE
```

Add:

```yaml
allow_participant_requests: true | false
```

## If true

Ordinary agents may choose:

```text
REQUEST
REPORT
NONE
```

Use the stronger prompt guidance encouraging specific information-seeking when evidence is missing or ambiguous.

## If false

Ordinary agents may choose only:

```text
REPORT
NONE
```

The prompt and response contract must reflect the allowed types exactly.

Do not leave REQUEST described in the prompt when it is disabled.

---

# 7. Controller actuation modes

Preserve explicit controller modes for backward compatibility.

Recommended:

```yaml
controller_actuation_mode:
  legacy_coordination_request
  truthful_strategic_report
  adaptive_communication
```

If the repository already uses different mode names, preserve those names and add only the new mode.

---

# 8. Legacy mode: retrospective behavior

`legacy_coordination_request` must reproduce previous controller behavior as closely as possible.

When selected:

- use the previous controller actuation path;
- use the previous message semantics;
- do not invoke the adaptive communication chooser;
- preserve the old `b` interpretation for that mode;
- preserve old prompt/version routing where required.

Historical studies must remain reproducible.

---

# 9. Truthful report mode: retrospective behavior

`truthful_strategic_report` must preserve the existing report-only controller.

When:

```text
U=0
```

post nothing.

When:

```text
U=1
```

execute the existing truthful strategic REPORT behavior.

Do not invoke the adaptive communication chooser in this mode.

---

# 10. New adaptive mode

Add:

```yaml
controller_actuation_mode: adaptive_communication
```

In this mode:

1. sense population state exactly as before;
2. sample binary `U` exactly as before;
3. if `U=0`, controller is silent;
4. if `U=1`, choose one permitted communication strategy from the current context;
5. execute that strategy through the normal board interfaces.

The communication strategy must be chosen dynamically, not frozen for the entire experiment.

---

# 11. Controller-visible context

The adaptive chooser may use only controller-authorized information, such as:

```text
configured target
sensed votes Y_k
current round index
current live board summary available to the controller
its own previous communication history if already part of controller state
available truthful controller-reportable fact pool
feature switches indicating which message modes are allowed
```

It must not see:

```text
private agent reasons
hidden exact world state beyond approved controller knowledge
unexposed private evidence
future outcomes
analysis-only metrics
gold answer unless explicitly part of the controller definition
```

Preserve the truthful-controller epistemic restrictions.

---

# 12. Adaptive communication decision

When `U=1`, the controller should decide which allowed communication strategy is most useful:

```text
REQUEST   -> ask for missing information
REPORT    -> share truthful strategically selected evidence
DIRECTIVE -> coordinate attention/work without fabricating evidence
```

Do not hard-code a fixed strategy by round or budget.

Examples:

```text
allow_controller_requests = true
allow_controller_directives = false

allowed:
REQUEST
REPORT
```

```text
allow_controller_requests = false
allow_controller_directives = true

allowed:
REPORT
DIRECTIVE
```

```text
allow_controller_requests = false
allow_controller_directives = false

allowed:
REPORT
```

---

# 13. Truthfulness contract

Adaptive communication must not weaken the truthful strategic controller contract.

## REPORT

- every fact ID must exist in the frozen task;
- every fact must be true;
- canonical semantics must be preserved;
- no fabricated implication;
- no false certainty;
- no hidden-answer or hidden-score leakage.

## REQUEST

- may ask for evidence;
- may not fabricate facts;
- should ask for specific missing information;
- cannot attach exact evidence.

## DIRECTIVE

- coordinates attention or work;
- must not fabricate facts;
- must not impersonate system authority;
- must not claim hidden knowledge;
- must not present an unsupported conclusion as established fact.

---

# 14. Controller REQUEST semantics

Controller REQUEST should use the ordinary public message type where possible.

Good examples:

```text
Does anyone have evidence about Bruno's ability to analyze field measurements?
```

```text
Can someone report evidence about how Alice and Chandra work together?
```

Avoid generic questions such as:

```text
Any thoughts?
```

REQUEST must not carry `shared_fact_id`.

---

# 15. Controller REPORT semantics

Reuse the truthful strategic report mechanism.

A REPORT should:

- use the ordinary REPORT schema;
- use an ordinary stable participant identity;
- attach only valid canonical evidence;
- strategically select truthful target-compatible facts;
- avoid option-letter contamination in free text;
- preserve exact provenance.

The adaptive chooser decides whether REPORT is appropriate.

The existing truthful fact selector decides which reportable fact(s) to use.

Keep these responsibilities separate.

---

# 16. Controller DIRECTIVE semantics

DIRECTIVE is a coordination act, not evidence.

Examples:

```text
Please compare the evidence about Alice and Bruno's analytical ability before deciding.
```

```text
Focus on whether the proposed workshop pair has direct cooperation evidence.
```

DIRECTIVE must not:

- fabricate a fact;
- claim a conclusion is proven without evidence;
- attach forbidden evidence;
- use privileged/system/controller language in participant-visible text.

---

# 17. Budget b

Do not redefine `b` globally.

Its interpretation may remain mode-dependent where backward compatibility requires it.

For the new adaptive mode, document the meaning explicitly.

Recommended:

```text
b = maximum number of controller-authored public messages admitted on an ACT round
```

or, if the runtime already requires exact posts per active round:

```text
b = exact number of controller communication slots on an ACT round
```

Do not force redundant content merely to fill `b`.

For truthful REPORT information dose, preserve the currently recommended grid:

```text
b in {3, 6, 9, 12}
```

Treat `b=24` only as an optional saturation diagnostic unless explicitly requested.

Always log:

```text
requested_b
actual_posts
message_mode
```

---

# 18. Backward compatibility requirement

The implementation must continue to support:

```text
old directive-style controller
truthful report-only controller
adaptive controller
participant REQUEST enabled
participant REQUEST disabled
controller REQUEST enabled
controller REQUEST disabled
controller DIRECTIVE enabled
controller DIRECTIVE disabled
```

Where technically possible:

\[
\text{new features OFF}
\Rightarrow
\text{same seeded behavior as the legacy path}.
\]

At minimum, historical semantics and historical prompt versions must remain unchanged.

Use new prompt hashes / protocol fingerprints for new behavior.

---

# 19. Preserve the binary sensor-policy path

The adaptive chooser runs **after** the existing binary control decision.

Do not change:

```text
q_c
beta
theta
sensor sampling
NO_OP vs ACT policy
binary U logging
```

The causal sequence remains:

```text
n_k
-> Y_k
-> binary action probability
-> sample U_k
-> if U_k=1 choose communication mode
-> execute communication
-> ordinary population update
-> n_{k+1}
```

---

# 20. Required logging

For every controller round record:

```text
U_k
controller_action_probability
sensor summary / existing Y representation
allowed_message_modes
chosen_message_mode
requested_b
actual_controller_posts
controller_post_ids
selected_fact_ids if REPORT
request topic if REQUEST
directive topic if DIRECTIVE
```

The primary analysis still conditions on binary `U`.

---

# 21. Secondary communication diagnostics

Add optional diagnostics:

```text
fraction of ACT rounds using REPORT
fraction of ACT rounds using REQUEST
fraction of ACT rounds using DIRECTIVE

posts by mode
reads by mode
unique readers by mode
fact acquisitions from REPORT
reactivations from REPORT
reply rate to REQUEST
reply rate to DIRECTIVE
target adoption after exposure by mode
```

These do not replace the primary theory metrics.

---

# 22. Analysis compatibility

Existing analysis must still compute:

\[
I(U;n_{k+1}\mid n_k)
\]

with binary `U`.

Do not change estimator schemas unnecessarily.

If the analysis currently infers `U` from controller event type, make explicit stored binary `U` authoritative.

Add `message_mode` only as optional metadata.

---

# 23. Experimental 2 x 2 design

Prepare matched configuration support for:

\[
Q\in\{0,1\},
\qquad
D\in\{0,1\},
\]

where:

```text
Q = question capability
D = controller directive capability
```

Recommended conditions:

```text
Q0 D0
Q1 D0
Q0 D1
Q1 D1
```

All conditions should match:

```text
task
seed structure
N
q
q_c
rho
rounds
beta
theta
binary controller policy
budget grid
model
```

unless intentionally varied.

---

# 24. Prompt behavior

Reuse the updated blackboard prompt patch.

Preserve:

- semantic vote storage;
- recipient-specific option-letter rendering;
- public prose should not rely on A/B/C labels;
- neutral evaluation guidance;
- verified evidence vs REPORT interpretation distinction;
- stronger REQUEST guidance;
- previous-vote wording;
- no unnecessary duplicate shared-fact rendering.

When REQUEST is disabled, remove REQUEST from:

```text
instructions
allowed enum
examples
response contract
```

When enabled, include the stronger information-seeking guidance.

---

# 25. Controller chooser implementation

Implement the adaptive chooser as a distinct component.

Suggested interface:

```python
choose_communication_mode(
    controller_context,
    allowed_modes,
    rng,
) -> CommunicationMode
```

where:

```text
CommunicationMode = REPORT | REQUEST | DIRECTIVE
```

Then dispatch to the existing specialized implementation.

Suggested architecture:

```text
binary controller policy
    |
    +-- U=0 -> NO_OP
    |
    +-- U=1
          |
          +-- adaptive chooser
                 |
                 +-- REPORT handler
                 +-- REQUEST handler
                 +-- DIRECTIVE handler
```

Do not duplicate the full controller runtime for each communication type.

---

# 26. Determinism and reproducibility

Given:

```text
same task
same seed
same configuration
same provider outputs
```

the communication-mode path should be reproducible to the extent supported by the provider architecture.

If the chooser uses an LLM, archive:

```text
chooser prompt version
chooser model
allowed modes
raw structured choice
resolved mode
```

If rule-based, archive rule/version.

Changing chooser behavior must change the protocol fingerprint.

---

# 27. Recommended initial chooser contract

Prefer a structured choice rather than unconstrained free text.

Example:

```json
{
  "mode": "REPORT",
  "reason": "..."
}
```

with `mode` constrained to the currently allowed set.

Then execute the existing handler for the selected mode.

Do not let one unconstrained response simultaneously invent:

```text
message type
fact ID
evidence text
directive
request
```

without validation.

---

# 28. Required tests

## Binary-control preservation

1. `U` remains strictly binary.
2. sensor/policy sampling occurs before communication-mode choice.
3. `U=0` always produces zero controller communication.
4. `U=1` invokes exactly one allowed communication strategy.
5. main analysis still receives binary `U`.

## Capability switches

6. participant REQUEST disabled -> prompt/schema exclude REQUEST.
7. participant REQUEST enabled -> REQUEST is available.
8. controller directives disabled -> chooser cannot select DIRECTIVE.
9. controller requests disabled -> chooser cannot select REQUEST.
10. REPORT remains available in adaptive mode.

## Backward compatibility

11. legacy controller bypasses adaptive chooser.
12. truthful-report controller bypasses adaptive chooser.
13. old prompt versions remain reproducible.
14. historical configs still load.
15. feature-off configuration reproduces prior semantics.

## Truthfulness

16. REPORT uses only valid true fact IDs.
17. REQUEST cannot attach evidence.
18. DIRECTIVE cannot attach forbidden evidence or fabricate facts.
19. controller-only metadata does not leak into participant prompts.

## Rendering

20. recipient-specific vote mapping remains correct.
21. no stale A/B/C semantics in structured rendering.
22. enabled message types exactly match the response schema.

## Logging

23. every ACT round records chosen communication mode.
24. every NO_OP round records null/no communication mode.
25. binary `U` can be reconstructed independently of message type.

---

# 29. Fake-provider integration test

Create a deterministic short episode:

```text
round 1: U=0
round 2: U=1 -> REPORT
round 3: U=1 -> REQUEST
round 4: U=1 -> DIRECTIVE
```

with all communication modes enabled.

Verify:

```text
U remains binary
message types match chosen modes
board handling is unchanged
analysis receives n_k -> U_k -> n_{k+1}
population observables still exclude controller vote if that remains current protocol
```

Then repeat with:

```text
allow_controller_directives: false
```

and confirm DIRECTIVE cannot occur.

---

# 30. Small real-model pilot

After provider-free tests pass, run only a small pilot.

Suggested:

```text
1 frozen task
few repetitions
b in {3, 6}
questions/directives enabled
```

Inspect:

```text
frequency of chosen controller modes
specificity of REQUESTs
truthfulness of REPORTs
quality of DIRECTIVEs
whether repetitive spam disappears
whether agents reply to questions
parse rate
```

Do not launch the full production study from this TDD.

---

# 31. Acceptance criteria

PASS if:

- binary `U` is unchanged;
- existing information-theoretic estimators still use binary `U`;
- legacy and truthful-report modes remain available;
- participant questions can be switched ON/OFF;
- controller REQUEST can be switched ON/OFF;
- controller DIRECTIVE can be switched ON/OFF;
- adaptive controller dynamically chooses among allowed REQUEST/REPORT/DIRECTIVE modes only when `U=1`;
- `U=0` is always silent;
- truthfulness constraints are preserved;
- prompts reflect enabled capabilities correctly;
- historical studies are unchanged;
- regression tests pass;
- fake-provider tests demonstrate all intended branches.

---

# 32. Explicit non-goals

Do not in this implementation:

- redefine `U` as categorical;
- rewrite the information bound;
- replace the existing `T_pi` estimator;
- redesign the blackboard runtime;
- change persistence;
- change `q` or `q_c`;
- change the binary sensor policy;
- modify historical results;
- merge mode diagnostics into the main thermodynamic theory;
- run a full population study.

---

# 33. Final implementation report

At completion provide:

```text
files changed
new config fields
controller mode enum
adaptive chooser implementation
prompt/version changes
tests added
test results

confirmation that:
- U remains binary
- T_pi still uses binary U
- legacy behavior remains available
- truthful-report behavior remains available
- participant REQUEST can be toggled
- controller REQUEST can be toggled
- controller DIRECTIVE can be toggled
- adaptive controller chooses among allowed modes only after U=1

one example round for:
- NO_OP
- REPORT
- REQUEST
- DIRECTIVE

results path for fake-provider smoke test
recommendation for the next small real-model pilot
```

---

# 34. Core scientific principle

The implementation must preserve the distinction:

\[
\boxed{
U_k
=
\text{whether the controller intervenes}
}
\]

versus:

\[
\boxed{
M_k
=
\text{how the controller realizes that intervention}
}
\]

The main theory remains binary in `U`.

The richer communication behavior belongs inside the effective controlled dynamics `Q1`.

This is the central compatibility requirement.
