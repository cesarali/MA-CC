# Handoff: MuSR Blackboard Task001 Pilot — Simplified Messages, 5-Round Engineering Run

## Purpose

Implement the simplified blackboard communication model and run **one short 5-round episode on `task_001`**.

This is primarily an **engineering and inspection run**, not a scientific experiment.

The goals are:

1. simplify the message ontology;
2. verify that the new message types work through the actual runtime;
3. inspect the exact prompts shown to agents and the controller;
4. inspect the blackboard round by round;
5. verify evidence acquisition, persistence, expiry, and refresh;
6. confirm that the dashboard/trajectory outputs are correct;
7. stop after one 5-round episode.

Do not launch a parameter sweep.

---

# 1. Frozen pilot configuration

Use:

```text
task_id = task_001
model = gwdg/openai-gpt-oss-120b

N = 24
rounds = 5

q = 1
q_c = 12

tau_B = 1 round
rho = 0.50

b = 6
controller mode = coordination / DIRECTIVE
controller target = true/gold answer
```

This run is deliberately short.

The objective is to inspect mechanics, not estimate final controller efficiency.

---

# 2. Initial information distribution

For this pilot enforce:

```text
private latent breadth k = 1
```

Each agent starts with evidence about **exactly one latent value**.

Prefer the 9 canonical F9 cards already validated for `task_001`:

```text
9 latent values
1 canonical F9 card per latent value
```

Distribute the nine cards as evenly as possible across 24 agents:

```text
6 latent values -> 3 initial holders
3 latent values -> 2 initial holders
```

Therefore:

```text
24 agents
1 exact evidence card per agent
1 latent value per agent
population union = all 9 F9 cards
```

Save and validate the assignment before execution.

---

# 3. IMPORTANT: simplify the message ontology in the implementation

The current six message types are too complicated for the intended model.

Replace the game-facing ontology:

```text
CLAIM
QUESTION
REQUEST
RESULT
REPLY
CORRECTION
```

with the following minimal communication language.

## Ordinary agents

Ordinary agents may emit only:

```text
REQUEST
REPORT
NONE
```

## Controller

The controller may emit only:

```text
DIRECTIVE
```

This is an implementation change, not merely an analysis relabeling.

Update all relevant schemas, enums, prompt rendering, parsing, validation, serialization, analysis, tests, and dashboard code.

---

# 4. Semantics of the three effective speech acts

## REQUEST

An ordinary agent asks for information or work.

Examples:

```text
"Does anyone have evidence about Elena's interview ability?"

"Can someone report evidence that distinguishes B from C?"
```

Rules:

```text
shared_fact_id = null
reply_to = optional
```

A REQUEST cannot directly transfer exact evidence.

---

## REPORT

An ordinary agent reports information, an answer, a conclusion, or a correction.

Examples:

```text
"I have evidence that Elena is strong at interviews."

"That earlier report seems misleading; my evidence favors B instead."
```

Rules:

```text
shared_fact_id = optional
reply_to = optional
```

`REPORT` intentionally subsumes the old:

```text
CLAIM
RESULT
REPLY
CORRECTION
```

A correction is therefore just:

```text
type = REPORT
reply_to = earlier_message_id
text = correction text
```

A direct answer to a request is:

```text
type = REPORT
reply_to = request_message_id
```

---

## DIRECTIVE

Only the controller may issue a DIRECTIVE.

Examples:

```text
"Prioritize evidence that distinguishes B from A."

"Agents with evidence relevant to option B should report it."

"Focus this round on unresolved evidence about the interview assignment."
```

Rules:

```text
shared_fact_id = null
reply_to = optional/null
```

The controller is organizing collective attention.

It is not allowed to fabricate or transmit task evidence.

---

# 5. `reply_to` is structural, not a message class

Retain:

```text
reply_to: optional message_id
```

Examples:

```text
REQUEST
    ↓
REPORT(reply_to=request_id)
```

or:

```text
DIRECTIVE
    ↓
REPORT(reply_to=directive_id)
```

or:

```text
REPORT
    ↓
REPORT(reply_to=previous_report_id)
```

This preserves conversation structure without needing REPLY or CORRECTION as separate message categories.

---

# 6. Backward compatibility

Do not break historical stored runs unnecessarily.

Preferred implementation:

- runtime for this pilot uses the new simplified ontology;
- loaders for old result files may still recognize legacy message labels if needed;
- new prompts must expose only the simplified ontology;
- new analysis should map legacy labels only when reading old runs, never when generating new behavior.

If there is a clean versioned schema mechanism already in the repository, use it.

Do not silently reinterpret old raw data.

---

# 7. Update the actual prompt

The ordinary-agent prompt must clearly tell the LLM that it can choose:

```text
REQUEST
REPORT
NONE
```

and define them succinctly.

Do not mention:

```text
CLAIM
QUESTION
RESULT
REPLY
CORRECTION
```

in the new runtime prompt.

The public output schema should be approximately:

```text
vote
private_reason
public_message:
    type: REQUEST | REPORT | NONE
    text: string | null
    shared_fact_id: evidence_id | null
    reply_to: message_id | null
```

The controller prompt/action renderer must use:

```text
DIRECTIVE
```

only.

Archive the exact rendered prompts.

---

# 8. Evidence transfer rules

Semantic language and exact evidence remain different.

## Semantic-only REPORT

```text
type = REPORT
text = ...
shared_fact_id = null
```

May influence reasoning, but does not change exact evidence memory.

## Exact-evidence REPORT

```text
type = REPORT
text = ...
shared_fact_id = e_...
```

An agent may attach an evidence card only if:

```text
shared_fact_id ∈ K_active(author)
```

When another agent samples that live REPORT:

```text
shared_fact_id -> K_hist(receiver)
shared_fact_id -> K_active(receiver)
```

If the receiver already has the card historically but it is inactive, this event is a:

```text
refresh
```

and restores the card to `K_active`.

---

# 9. Persistence model

Keep separate:

```text
K_hist_i(t)
K_active_i(t)
B(t)
```

where:

```text
K_hist = every exact card ever acquired
K_active = cards currently available in the LLM prompt
B(t) = live public blackboard
```

For this pilot:

```text
rho = 0.50
```

At the configured round transition, each active card survives independently with probability `rho`.

If forgotten:

```text
remove from K_active
retain in K_hist
```

Historical-but-inactive evidence must not be rendered to the LLM.

Historical-but-inactive evidence must not be shareable.

It can return to active memory only through re-exposure to the same exact evidence card.

---

# 10. Blackboard lifetime

Use:

```text
tau_B = 1 round
```

Messages expire according to the current board semantics.

Message expiry affects only the public board.

It must not directly delete:

```text
K_hist
K_active
```

Active forgetting is controlled only by `rho`.

---

# 11. One five-round episode only

After implementation and tests pass, run:

```text
task_001
N=24
rounds=5
q=1
q_c=12
rho=0.50
tau_B=1
b=6
```

Stop after this episode.

Do not automatically run:

```text
more seeds
more tasks
more b values
more rho values
```

---

# 12. The main deliverable is inspectability

For this pilot, raw prompts and blackboard state are more important than aggregate statistics.

Archive every actual agent prompt and response.

For each focal update save:

```text
round
microscopic_update
agent_id
current_vote
K_hist ids
K_active ids
latent values represented in K_active
sampled board message ids
full rendered prompt
raw model response
parsed vote
parsed private reason
parsed public message
```

Also archive every controller action/prompt if the controller uses an LLM or rendered instruction path.

---

# 13. Blackboard dashboard

Prepare a simple dashboard/report that allows us to inspect the episode round by round.

For each round show:

## Population

```text
vote counts A/B/C
p_truth
```

## Agent state

```text
mean |K_active|
mean |K_hist|
active latent coverage
historical latent coverage
```

Prefer a 24 x 9 heatmap for:

```text
active evidence coverage
historical evidence coverage
```

## Blackboard

Render the full live/posting history with columns:

```text
message_id
round
author
author_role = agent/controller
type = REQUEST / REPORT / DIRECTIVE
text
shared_fact_id
reply_to
created_at
expires_at
```

The controller DIRECTIVEs must be visually distinct.

## Evidence flow

Show:

```text
exact transfers
refreshes
semantic-only reads
```

## Controller

Show:

```text
sensed votes
whether controller acted
DIRECTIVE text
which agents sampled it
which agents replied to it
whether downstream exact evidence moved
```

---

# 14. Exact prompt inspection files

Create a human-readable prompt archive such as:

```text
analysis/prompts/
    round_00_agent_*.md
    round_01_agent_*.md
    ...
```

or an equivalent indexed HTML/Markdown view.

For each prompt clearly mark:

```text
ACTIVE PRIVATE EVIDENCE
CURRENT VOTE
VISIBLE BLACKBOARD MESSAGE
AVAILABLE PUBLIC ACTION TYPES
```

This is one of the central outputs of the pilot.

We need to be able to answer:

> What exactly did Agent 7 see at this update?

without reconstructing it manually.

---

# 15. Required implementation checks

Before the 5-round run verify:

1. new prompts expose only REQUEST / REPORT / NONE to ordinary agents;
2. ordinary agents cannot emit DIRECTIVE;
3. controller can emit DIRECTIVE only;
4. old CLAIM / QUESTION / RESULT / REPLY / CORRECTION are not generated in new runs;
5. REPORT supports `reply_to`;
6. REPORT supports optional `shared_fact_id`;
7. REQUEST cannot transfer exact evidence;
8. DIRECTIVE cannot transfer exact evidence;
9. inactive evidence cannot be shared;
10. exact REPORT read adds evidence to K_hist and K_active;
11. re-reading historical-but-inactive evidence creates a refresh;
12. semantic-only REPORT creates no exact acquisition;
13. K_hist never decreases;
14. only K_active appears in the private evidence prompt;
15. board expiry works;
16. private reasoning never becomes public automatically;
17. initial assignment is exactly k=1;
18. population union is all 9 F9 cards;
19. no hidden matrix values leak into prompts;
20. legacy result loading still works if backward compatibility is required.

Run relevant regression tests.

---

# 16. Suggested output directory

```text
results/studies/musr_blackboard_task001_pilot_01/
```

Save at least:

```text
config.yaml
initial_assignment.json

episode.jsonl
messages.jsonl
controller_events.jsonl
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

---

# 17. Pilot report

The report should focus on mechanics.

Answer:

1. Did the simplified message ontology work cleanly?
2. What exact prompts did agents see?
3. Did agents actually emit REQUESTs?
4. Did agents emit REPORTs with and without evidence IDs?
5. Did agents respond to DIRECTIVEs?
6. Did exact evidence move correctly?
7. Did any refresh event occur?
8. Did `rho=0.5` prevent monotonic active-memory saturation?
9. Did historical memory and active memory visibly diverge?
10. Did board expiry behave correctly?
11. Did anything look confusing or overcomplicated in the prompt?
12. Is the implementation ready for a real \(b \times \rho\) experiment?

Do not make strong scientific claims from one 5-round episode.

---

# 18. Stop condition

This handoff is complete after:

```text
implementation
tests
one 5-round episode
dashboard/report generation
```

Then STOP.

At completion print:

```text
implementation status
tests passed/failed

task
N
rounds
rho
b

initial p_truth
final p_truth

REQUEST count
REPORT count
DIRECTIVE count

exact evidence transfers
refresh events

results directory
prompt archive path
dashboard path
report path
wall-clock time
```
