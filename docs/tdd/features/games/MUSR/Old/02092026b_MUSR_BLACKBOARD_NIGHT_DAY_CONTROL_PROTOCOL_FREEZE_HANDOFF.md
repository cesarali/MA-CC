# Handoff: Freeze the MuSR Blackboard Control Protocol — Night/Day Separation + Dawn Actuation

## Purpose

Refactor the current MuSR blackboard game so that the **control protocol has a clean temporal interpretation** while preserving the blackboard coordination behavior already demonstrated in the first `task_001` pilot.

This is a **protocol-freeze implementation**, not a new scientific sweep.

The first pilot already established that:

- ordinary agents can use `REQUEST` / `REPORT`;
- the coordinating participant can use `DIRECTIVE`;
- exact evidence can move through `REPORT.shared_fact_id`;
- `K_hist` and `K_active` are distinct;
- persistence and refresh events work;
- the board can support endogenous coordination.

Now make the controller timing and public-message semantics precise enough that the later control-efficiency quantities

\[
\chi(n,b), \qquad I(U_k;n_{k+1}\mid n_k),
\]

have an unambiguous one-round interpretation.

After implementing and testing the changes, run **one short 5-round `task_001` engineering pilot only** and regenerate the dashboard/report. Do not launch the overnight parameter sweep.

---

# 1. Frozen conceptual protocol

Treat the following as hard design invariants.

## 1.1 Night/day separation

One macro-round is:

```text
END OF PREVIOUS DAY
    |
    |  population votes define n_k
    v
NIGHT
    |
    |-- sample q_c ordinary-agent votes -> Y_k
    |-- compute existing P(U_k = 1 | Y_k)
    |-- sample binary U_k
    |-- expire previous-day blackboard according to tau_B
    |-- apply configured active-memory persistence for the new day
    v
DAWN
    |
    |-- if U_k = 0: coordinator posts nothing
    |-- if U_k = 1: seed exactly b DIRECTIVE messages
    v
DAY
    |
    |-- 24 ordinary agents evolve normally
    |-- each focal agent samples q live board messages
    |-- votes
    |-- may post REQUEST / REPORT / NONE
    |-- exact evidence may propagate through REPORTs
    |
    |-- coordinator is completely silent during the day
    v
END OF DAY
    |
    |-- measure n_{k+1}
    |-- snapshot K_active / K_hist / board / population
```

The critical causal ordering is:

\[
n_k
\rightarrow
Y_k
\rightarrow
P(U_k\mid Y_k)
\rightarrow
U_k
\rightarrow
\text{dawn board perturbation}
\rightarrow
\text{autonomous day}
\rightarrow
n_{k+1}.
\]

Do not interleave controller injections with microscopic agent updates in this blackboard mode.

---

# 2. Preserve the existing sensor and binary policy

The implementation before the actuation layer must remain unchanged.

For every round:

```text
q_c sampled ordinary agents
        ->
Y_k
        ->
existing stochastic soft-feedback policy
        ->
P(U_k = 1 | Y_k)
        ->
sample U_k in {NO_OP, ACT}
```

Preserve the existing:

```text
q_c
beta
theta
target Z
sensor sampling rule
randomness / seeding conventions
```

The intended policy remains the current project policy, e.g. the existing implementation of

\[
P(U_k=1\mid Y_k)
=
\sigma\left[
\beta\left(
\theta-\frac{y_Z}{q_c}
\right)
\right]
\]

when that is the configured policy.

Do **not** move this decision into an LLM prompt.

The runtime samples `U`.

### Hard regression requirement

For fixed synthetic `Y` values, run a large provider-free Monte Carlo test and verify that empirical action frequencies agree with the configured `P(U=1|Y)` within sampling error.

Log every round:

```text
n_k
sensor_agent_ids
sensor_votes
Y_k / y_Z
q_c
beta
theta
P(U=1 | Y_k)
sampled U_k
```

---

# 3. Redefine blackboard `b`: dawn control mass, not microscopic positions

For the new blackboard coordination mode:

```text
b = number of coordinator DIRECTIVE messages seeded at dawn
```

Therefore:

```text
U = 0  -> exactly 0 coordinator messages
U = 1  -> exactly b coordinator messages
```

All `b` messages must exist on the live board **before the first ordinary focal update of the day**.

The coordinator does not inject additional messages later in that day.

## Important compatibility point

The old direct-control modes may retain their historical `b` / microscopic-position semantics for backward compatibility.

But in the **new dawn-blackboard mode**, do not use:

```text
controlled microscopic positions
controller injection times during the day
replacement social slots
```

as the actuator.

No ordinary-agent message is replaced or edited.

No existing board message is modified.

The coordinator only adds its own dawn messages.

---

# 4. Keep the coordination aspect: use `DIRECTIVE`

Do not remove `DIRECTIVE`.

The simplified public speech-act vocabulary remains:

```text
ordinary agents:
    REQUEST
    REPORT
    NONE

coordinating participant:
    DIRECTIVE
```

This preserves the coordination narrative we want from the shared-blackboard setting.

`reply_to` remains a structural field, not a message type.

Examples:

```text
REQUEST -> REPORT(reply_to=request_id)

DIRECTIVE -> REPORT(reply_to=directive_id)

REPORT -> REPORT(reply_to=earlier_report_id)
```

Do not reintroduce the six-type ontology.

---

# 5. Standardize every visible message

Every public message visible to another ordinary agent must use the same basic schema:

```text
message_id
author_public_id
vote
type
text
shared_fact_id
reply_to
created_round
expires_round
```

## 5.1 Vote is mandatory for every posted message

Whenever a participant posts, the message carries that participant's current public vote.

For an ordinary agent:

```text
vote = its current population vote at posting time
```

For the coordinating participant:

```text
vote = its configured/current control preference
```

For the current truth-target pilot, that is the configured target/gold option.

The coordinator vote must be non-null on every DIRECTIVE.

## 5.2 Coordinator vote is never part of the population

The controlled population remains exactly:

```text
N = 24 ordinary agents
```

Define all population statistics only from them:

\[
n_k
=
\sum_{i=1}^{24}\mathbf 1[X_i(k)=Z].
\]

The coordinator's public vote must **never** enter:

```text
n_k
p_truth
vote counts
dominant share
consensus
sensor pool
q_c sampling
population thermodynamic observables
```

It is social input, not a 25th population vote.

Add explicit tests for this.

---

# 6. Socially, the coordinator should look like another participant

Internally, keep the special identity needed for analysis:

```text
author_kind = controller/coordinator
U provenance
controller policy provenance
```

But ordinary-agent prompts must **not** render labels such as:

```text
CONTROLLER
EXTERNAL CONTROLLER
CONTROL MESSAGE
```

Use a neutral participant identity, for example the repository's existing neutral coordinator alias or a dedicated non-population ID such as:

```text
Agent 25
```

Do not hard-code `Agent 25` if the repository already has a safe abstraction; use a stable neutral public alias.

The receiving agent may see:

```text
Agent 25
Current vote: B
Type: DIRECTIVE
Message: ...
```

That is intentional.

The `DIRECTIVE` speech-act tag remains visible.

The special experimental role does not.

---

# 7. Directive content: preserve the existing coordination policy

Do not redesign the semantic content of DIRECTIVEs unless required to support dawn generation.

The current coordination behavior can remain as before.

A DIRECTIVE may:

```text
prioritize an option
ask the group to surface discriminating evidence
focus attention on unresolved evidence
encourage reporting
coordinate work
```

It may use only information available to the controller at night:

```text
configured target/preference
Y_k / sensed vote information allowed by the existing policy
fixed experiment parameters
```

It must not inspect:

```text
K_active_i
K_hist_i
private reasons
private evidence assignments
hidden matrix values
future day messages
```

and must never carry:

```text
shared_fact_id
```

### Multiple dawn DIRECTIVEs

If `b > 1`, seed exactly `b` DIRECTIVE messages.

It is acceptable for them to be repeated or to vary naturally according to the existing directive-generation mechanism.

Do **not** introduce a new adaptive within-day policy.

All controller messages for the day must be fixed/generated at dawn before ordinary-agent evolution starts.

Increasing `b` should primarily increase the coordinator's representation in the live message pool, and therefore its opportunity to be sampled.

---

# 8. Ordinary-agent board interaction remains autonomous

After dawn, run the normal 24-agent day.

For each focal update:

```text
sample q eligible live board messages
render K_active
render current vote
render sampled public messages
LLM reasons autonomously
update vote
optionally emit REQUEST / REPORT / NONE
```

The coordinator does not force:

```text
a vote
a reply
a REPORT
an evidence transfer
```

Sampling a DIRECTIVE only adds it to the focal agent's social context.

For the current pilot keep:

```text
q = 1
```

The analysis dashboard is **never** exposed to the agents.

Agents see only the social context explicitly rendered by the runtime, not the accumulated analysis dashboard and not the entire board unless the normal `q` sampling rule gives it to them.

Add a test ensuring analysis/dashboard artifacts cannot leak into prompts.

---

# 9. Exact evidence semantics remain unchanged

Keep:

```text
K_hist_i(t)   = every exact evidence card ever acquired
K_active_i(t) = evidence currently available to the LLM
```

Only `K_active` is rendered as exact private evidence.

## REPORT

A `REPORT` may carry:

```text
shared_fact_id = one exact card in K_active(author)
```

When sampled by another agent:

```text
new card:
    -> K_hist(receiver)
    -> K_active(receiver)

historical but inactive card:
    -> K_active(receiver)
    -> record REFRESH
```

## REQUEST

Must have:

```text
shared_fact_id = null
```

## DIRECTIVE

Must have:

```text
shared_fact_id = null
```

An inactive historical card is not shareable until reactivated.

Preserve existing semantic-only influence: a REPORT without an exact card may affect reasoning but creates no exact evidence acquisition.

---

# 10. Memory and board persistence remain separate

Keep two different timescales.

## Public memory

```text
tau_B = board-message lifetime
```

For the current pilot:

```text
tau_B = 1 round
```

Previous-day messages expire at the night boundary before the new day's board is seeded.

## Private active memory

```text
rho = active evidence persistence
```

For the current pilot:

```text
rho = 0.50
```

Persistence acts on `K_active`, not `K_hist`.

Cards removed from `K_active` remain in `K_hist`.

Do not remove this memory mechanism.

Some delayed control influence through epistemic memory is acceptable and expected; the protocol is designed so that the strongest directly attributable control signal is nevertheless the same-round transition:

\[
U_k \rightarrow n_{k+1}.
\]

---

# 11. Explicitly measure nominal budget versus realized exposure

Because DIRECTIVEs are placed on the board rather than forced into focal contexts:

```text
b != number of realized controller exposures
```

Record separately.

For every round compute:

```text
U_k
b
number of dawn DIRECTIVEs posted
number of focal updates in which a DIRECTIVE was sampled
number of unique ordinary agents exposed to >=1 DIRECTIVE
total eligible board-message reads
DIRECTIVE share among eligible live messages
DIRECTIVE -> reply count
DIRECTIVE -> REPORT reply count
DIRECTIVE -> evidence-bearing REPORT count
downstream exact acquisitions
downstream refreshes
```

For `q=1`, also report the empirical realized exposure fraction:

\[
\hat e_k
=
\frac{\#\{\text{focal updates that sampled a DIRECTIVE}\}}
     {N}.
\]

Where straightforward, also log the instantaneous live-message composition before each focal sample:

```text
M_t = eligible live messages
D_t = eligible live DIRECTIVEs
D_t / M_t
```

This is the realized public-channel control pressure.

---

# 12. Fix causal attribution through replies

The first pilot's downstream controller metric was too local.

Implement transitive attribution at least through the immediate reply chain:

```text
DIRECTIVE
    ->
ordinary agent samples DIRECTIVE
    ->
ordinary agent emits REPORT(reply_to=DIRECTIVE)
    ->
that REPORT is sampled later
    ->
exact acquisition / refresh
```

The analysis must be able to attribute those downstream evidence events back to the originating DIRECTIVE.

At minimum save:

```text
origin_directive_id
reply_message_id
reply_author
reply_shared_fact_id
downstream_reader
event_type = acquisition | refresh
round
```

Do not claim a causal effect beyond the logged message/reply ancestry; this is a mechanistic provenance metric.

---

# 13. Current engineering rerun configuration

After implementation and provider-free tests pass, rerun exactly one short episode:

```text
task_id = task_001
model = gwdg/openai-gpt-oss-120b

N = 24
rounds = 5

q = 1
q_c = 12

rho = 0.50
tau_B = 1

b = 6

controller target = true/gold answer
controller public message type = DIRECTIVE
controller timing = dawn_only
```

Keep the existing frozen `task_001` F9 N=24 assignment:

```text
24 ordinary agents
exactly 1 canonical F9 card per agent
exactly 1 latent value per agent
population union = all 9 F9 cards
```

Do not redesign the task assignment in this handoff.

The high initial truth share observed in the previous pilot is a benchmark/assignment issue to revisit separately; it is not a reason to alter this protocol refactor.

---

# 14. Mandatory provider-free tests

Before real execution, add/verify tests for all of the following.

1. `Y_k` is sampled from exactly `q_c` ordinary-agent votes.
2. The coordinator is excluded from the sensor pool.
3. Existing `P(U|Y)` is numerically unchanged.
4. Empirical Monte Carlo action frequencies match configured `P(U|Y)`.
5. `U=0` produces zero coordinator board messages.
6. `U=1` produces exactly `b` DIRECTIVEs.
7. All `b` DIRECTIVEs exist before the first ordinary focal update.
8. No coordinator message is created during ordinary daytime updates.
9. No microscopic controlled-position scheduler is used in dawn-blackboard mode.
10. Coordinator messages never overwrite ordinary-agent social slots or messages.
11. Every posted REQUEST / REPORT / DIRECTIVE has a non-null public vote.
12. Coordinator vote is never counted in population statistics.
13. Coordinator vote is never sampled as one of the `q_c` sensed population votes.
14. Coordinator public identity is neutral in ordinary-agent prompts.
15. The word/label `controller` is not exposed to ordinary agents.
16. The `DIRECTIVE` type remains visible.
17. Ordinary agents cannot emit DIRECTIVE.
18. Coordinator cannot emit REQUEST or REPORT in this mode.
19. DIRECTIVE cannot carry `shared_fact_id`.
20. REQUEST cannot carry `shared_fact_id`.
21. REPORT exact transfer requires `shared_fact_id in K_active(author)`.
22. `K_hist` never decreases.
23. Only `K_active` is rendered as exact evidence.
24. Board messages from the previous day expire at the night boundary.
25. Acquired evidence is not deleted merely because its source message expired.
26. `rho` acts only on `K_active`.
27. Analysis/dashboard content is never included in agent prompts.
28. `n_k`, truth share, consensus, and all population metrics use only N=24 ordinary agents.
29. Existing historical runs remain loadable without relabeling/reinterpreting raw message types.
30. The first-pilot downstream-attribution example pattern is correctly captured:
    `DIRECTIVE -> REPORT reply -> later exact acquisition/refresh`.

Run the relevant regression suite.

---

# 15. Prompt archive requirements

For the new 5-round rerun, archive every rendered prompt.

For a focal update that samples a coordinator post, the human-readable prompt archive should make it obvious that the model saw something like:

```text
PUBLIC MESSAGE

Author: Agent 25
Current vote: B
Type: DIRECTIVE
Message: <coordination text>
```

It must not say:

```text
Controller
External controller
Control action
```

Also archive at least one example from a `U=0` round showing that no coordinator message exists on the board.

---

# 16. Dashboard changes

Regenerate the existing dashboard, but make the night/day protocol visually explicit.

For each round show:

```text
NIGHT
    n_k
    Y_k
    P(U=1|Y_k)
    sampled U_k

DAWN
    number of DIRECTIVEs seeded
    coordinator public vote
    directive texts

DAY
    board evolution
    realized directive exposures
    REQUEST / REPORT production
    evidence acquisitions / refreshes

END OF DAY
    n_{k+1}
    p_truth
    K_active summary
    K_hist summary
```

The dashboard is an analysis artifact only.

Visually distinguish coordinator-originated DIRECTIVEs for us, while preserving the neutral prompt rendering seen by agents.

---

# 17. Required raw outputs

Preserve or extend the current study artifacts so every transition is reconstructable.

At minimum save:

```text
config.yaml
initial_assignment.json

round_control_events.jsonl
messages.jsonl
message_reads.jsonl
directive_lineage.jsonl
evidence_transfers.jsonl
persistence_events.jsonl

agent_state_by_update.csv
agent_state_by_round.csv
population_by_round.csv

analysis/prompts/*
analysis/figures/*
analysis/dashboard/*
analysis/task001_pilot_report.md
```

Each `round_control_events.jsonl` record should contain at least:

```text
round
n_k
sensor_agent_ids
sensor_votes
Y_k
P_U1_given_Y
U
b
coordinator_public_vote
directive_message_ids
```

---

# 18. Rerun report questions

The updated 5-round engineering report should answer:

1. Is the temporal sequence truly `night sense -> U -> dawn actuation -> autonomous day -> n_{k+1}`?
2. Is `P(U|Y)` unchanged from the existing controller?
3. Does `U=0` produce a clean no-control day?
4. Does `U=1` produce exactly `b` dawn DIRECTIVEs?
5. Are all coordinator posts present before ordinary-agent evolution starts?
6. Does the coordinator disappear during the day?
7. Does the coordinator look like another participant in agent prompts?
8. Is its vote visible but excluded from all population counts?
9. What realized exposure does `b=6` produce under `q=1, tau_B=1`?
10. Do DIRECTIVEs elicit REQUEST/REPORT responses?
11. Can directive-triggered REPORTs produce later evidence acquisitions/refreshes?
12. Are `K_active`, `K_hist`, and board lifetime still behaving correctly?
13. Is the implementation now safe to freeze for the overnight control-efficiency experiment?

Do not infer scientific controller efficiency from this single rerun.

---

# 19. Stop condition

Complete:

```text
implementation
provider-free tests
regression tests
one 5-round task_001 rerun
dashboard/report regeneration
```

Then STOP.

Do not launch:

```text
b sweep
rho sweep
multiple seeds
additional tasks
overnight experiment
```

At completion print:

```text
implementation status
changed protocol/module summary
tests passed/failed

task
N
rounds
q
q_c
rho
tau_B
b

round-by-round:
    n_k
    Y_k
    P(U=1|Y_k)
    U
    dawn DIRECTIVE count
    realized DIRECTIVE exposure
    n_{k+1}

REQUEST count
REPORT count
DIRECTIVE count
exact acquisitions
refresh events
directive-attributed acquisitions
directive-attributed refreshes

results directory
prompt archive path
dashboard path
report path
wall-clock time
```

---

# 20. Final conceptual invariant

After this refactor, the game should be accurately summarized by:

\[
\boxed{
n_k
\rightarrow
Y_k
\rightarrow
P(U_k\mid Y_k)
\rightarrow
U_k
\rightarrow
B_k^{\mathrm{dawn}}
\rightarrow
\text{autonomous multi-agent communication}
\rightarrow
n_{k+1}
}
\]

with:

```text
U_k = 0:
    coordinator is absent from the day's board

U_k = 1:
    coordinator contributes exactly b factless DIRECTIVEs at dawn
    each DIRECTIVE exposes its public vote
    coordinator vote is social input only, never population state
```

and with the coordinator remaining silent after dawn.

This is the protocol to freeze before the main efficiency experiment.
