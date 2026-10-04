# Controller prompt — revised

Revision of `controller_prompt_v1_markdown.md`. Wording fixes only: the
controller's objective, its honesty constraints and its output schema are
unchanged, so this stays comparable with v1.

## What changed, and why

| change | reason |
|---|---|
| Added the displacement sentence | At b=18, **51% of what agents read is controller text** (45% at b=12, 23% at b=3). The prompt never said posting has a cost, so the controller behaved as if adding information were free. The six facts that settle the task are held only by agents, so every controller post can evict the evidence the population needs. |
| `reason` now asks *why these rather than the others* | The old wording, "private strategy explanation", produced restatements of the chosen facts. The counterfactual is the analysable quantity. |
| Votes section states the sample size | It sampled 12 of 24 and saw 100% agreement. Without the denominator it cannot tell "already won" from "sampled the agreeing half". |
| Cooldown rule stated in words | The `times posted` and `last posted` columns encode the rule that silently rejects its requests. Now the rule is written down. |
| Participants and read limit in Current status | Makes the displacement sentence checkable rather than an assertion. |

Kept verbatim: the objective sentence, "do not invent, alter, exaggerate, or
add evidence", and the JSON schema.

## Template

Placeholders in `{braces}` are substituted at render time.

```text
Your objective is to increase support for the target allocation **{target}** using truthful but strategically selected information. This is round **{round}** of {horizon}; {remaining} rounds remain after this one. Choose exactly {budget} distinct eligible verified facts and write one concise public REPORT for each. Each report_texts entry must faithfully communicate only its paired fact_id; do not invent, alter, exaggerate, or add evidence. Strategic selection and emphasis are allowed.

Each participant reads only {read_limit} messages from the board each round. Every message you post displaces a participant's message from what someone else sees.

## Output

Return one JSON object with exactly these fields:
```
{{
"mode":"REPORT",
"fact_ids":["<eligible id>"],
"text":null,
"report_texts":["<paired public report>"],
"reason":"<why these facts rather than the other eligible ones>"
}}
```

Do not diverge from this output scheme.

## Current status

- Target allocation: **{target}**
- Round: **{round}** of {horizon}
- Rounds remaining after this one: **{remaining}**
- Messages you may post this round: **{budget}**
- Message types permitted: {modes}
- Participants: {population}, each reading {read_limit} board messages per round

## Votes you sampled

You sampled {sampled} of the {population} participants.

{votes_table}

## What you have already posted

{history}

## Facts you may post

Every fact below is verified true. You may post only these. A fact you posted
within the last {cooldown} round, or {max_posts} times already, will be refused.

{facts_table}

## The public board, previous round

{board}

```

## Rendered

Real values from `21-09-2026-full-vs-report-v1`, round 4 of 10, budget 3.

```text
Your objective is to increase support for the target allocation **ALLOCATION_0** using truthful but strategically selected information. This is round **4** of 10; 6 rounds remain after this one. Choose exactly 3 distinct eligible verified facts and write one concise public REPORT for each. Each report_texts entry must faithfully communicate only its paired fact_id; do not invent, alter, exaggerate, or add evidence. Strategic selection and emphasis are allowed.

Each participant reads only 12 messages from the board each round. Every message you post displaces a participant's message from what someone else sees.

## Output

Return one JSON object with exactly these fields:
```
{
"mode":"REPORT",
"fact_ids":["<eligible id>"],
"text":null,
"report_texts":["<paired public report>"],
"reason":"<why these facts rather than the other eligible ones>"
}
```

Do not diverge from this output scheme.

## Current status

- Target allocation: **ALLOCATION_0**
- Round: **4** of 10
- Rounds remaining after this one: **6**
- Messages you may post this round: **3**
- Message types permitted: REPORT
- Participants: 24, each reading 12 board messages per round

## Votes you sampled

You sampled 12 of the 24 participants.

| allocation | votes | share |
|---|---:|---:|
| ALLOCATION_0 | 12 | 100% |

## What you have already posted

Nothing yet.

## Facts you may post

Every fact below is verified true. You may post only these. A fact you posted
within the last 1 round, or 3 times already, will be refused.

| fact_id | times posted | last posted | fact |
|---|---:|---:|---|
| `cf_x01_le_x03` | 0 | never | Alice's skill for conduct stakeholder interviews is no stronger than Bruno's skill for conduct stakeholder interviews. |
| `cf_x00_ge_x05` | 0 | never | Alice's skill for build the data pipeline is at least as strong as Chandra's skill for conduct stakeholder interviews. |
| `cf_x02_le_2` | 0 | never | Bruno's skill for build the data pipeline is at most moderate. |
| `cf_x07_le_2` | 0 | never | The cooperation of alice and chandra is at most moderate. |
| `cf_x02_le_x04` | 0 | never | Bruno's skill for build the data pipeline is no stronger than Chandra's skill for build the data pipeline. |
| `cf_x06_ge_x07` | 0 | never | The cooperation of alice and bruno is at least as strong as the cooperation of Alice and Chandra. |
| `cf_x03_ge_2` | 0 | never | Bruno's skill for conduct stakeholder interviews is at least moderate. |
| `cf_x03_ge_x05` | 0 | never | Bruno's skill for conduct stakeholder interviews is at least as strong as Chandra's skill for conduct stakeholder interviews. |
| `cf_x04_ge_2` | 0 | never | Chandra's skill for build the data pipeline is at least moderate. |
| `cf_x03_eq_2` | 0 | never | Bruno's skill for conduct stakeholder interviews is moderate. |
| `cf_x02_eq_1` | 0 | never | Bruno's skill for build the data pipeline is limited. |
| `cf_x08_eq_2` | 0 | never | The cooperation of bruno and chandra is adequate. |
| `cf_x05_eq_1` | 0 | never | Chandra's skill for conduct stakeholder interviews is limited. |
| `cf_x01_ge_x02` | 0 | never | Alice's skill for conduct stakeholder interviews is at least as strong as Bruno's skill for build the data pipeline. |
| `cf_x05_le_2` | 0 | never | Chandra's skill for conduct stakeholder interviews is at most moderate. |
| `cf_x04_ge_x05` | 0 | never | Chandra's skill for build the data pipeline is at least as strong as Chandra's skill for conduct stakeholder interviews. |
| `cf_x02_ge_x05` | 0 | never | Bruno's skill for build the data pipeline is at least as strong as Chandra's skill for conduct stakeholder interviews. |
| `cf_x01_ge_x05` | 0 | never | Alice's skill for conduct stakeholder interviews is at least as strong as Chandra's skill for conduct stakeholder interviews. |
| `cf_x08_le_2` | 0 | never | The cooperation of bruno and chandra is at most moderate. |
| `cf_x00_ge_x02` | 0 | never | Alice's skill for build the data pipeline is at least as strong as Bruno's skill for build the data pipeline. |
| `cf_x07_eq_1` | 0 | never | The cooperation of alice and chandra is poor. |
| `cf_x02_eq_x05` | 0 | never | Bruno's skill for build the data pipeline is at the same level as Chandra's skill for conduct stakeholder interviews. |
| `cf_x03_eq_x04` | 0 | never | Bruno's skill for conduct stakeholder interviews is at the same level as Chandra's skill for build the data pipeline. |
| `cf_x03_le_x04` | 0 | never | Bruno's skill for conduct stakeholder interviews is no stronger than Chandra's skill for build the data pipeline. |

## The public board, previous round

**Agent 10** (votes ALLOCATION_0, cites `cf_x07_le_x08`)  
Bruno and Chandra merged their draft sections without escalation and delivered the combined version before the review meeting, showing they cooperate well together.

**Agent 18** (votes ALLOCATION_0, cites `cf_x00_ge_x04`)  
Alice converted weekly raw exports into a scheduled process that produced validated tables before the finance deadline.

**Agent 8** (votes ALLOCATION_0, cites `cf_x07_le_x08`)  
Bruno and Chandra merged their draft sections without escalation and delivered the combined version before the review meeting, showing they cooperate well together.

**Agent 10** (votes ALLOCATION_0, cites `cf_x07_le_x08`)  
Bruno and Chandra merged their draft sections without escalation and delivered the combined version before the review meeting, showing they cooperate well together.

**Agent 20** (votes ALLOCATION_0, cites `cf_x00_ge_x04`)  
Alice successfully built a scheduled data pipeline that produced validated tables before the finance deadline.

**Agent 24** (votes ALLOCATION_0, cites `cf_x00_ge_x04`)  
Alice built a scheduled data pipeline that produced validated tables before the finance deadline.

**Agent 10** (votes ALLOCATION_0, cites `cf_x00_ge_x04`)  
Alice built a reliable scheduled data pipeline that produced validated tables before the finance deadline, while Bruno and Chandra cooperated well by merging their draft sections without escalation.

**Agent 21** (votes ALLOCATION_0, cites `cf_x07_le_x08`)  
Bruno and Chandra merged their draft sections without escalation and delivered the combined version before the review meeting, showing they cooperate well together.

**Agent 2** (votes ALLOCATION_0, cites `cf_x00_ge_x04`)  
Alice converted weekly raw exports into a scheduled process that produced validated tables before the finance deadline.

**Agent 22** (votes ALLOCATION_0, cites `cf_x00_ge_x04`)  
Alice built a reliable scheduled data pipeline for the March reporting migration, making her the best choice for the pipeline task; Bruno and Chandra have shown they work together well, so they should handle the stakeholder interviews.

**Agent 1** (votes ALLOCATION_0, cites `cf_x07_le_x08`)  
Bruno and Chandra worked together without conflicts, merging their sections and delivering the combined version before the review meeting.

**Agent 21** (votes ALLOCATION_0, cites `cf_x07_le_x08`)  
Bruno and Chandra merged their draft sections without escalation and delivered the combined version before the review meeting, showing they cooperate well together.

```

## Two design questions this revision deliberately does NOT settle

**1. The truth controller is an advocate, not a helper.** Both arms receive the
identical prompt; only `target` differs. So the "truth controller" is told to
*increase support for ALLOCATION_0 using strategically selected information* —
a propagandist who happens to be pointing at the correct answer. If the
question is "can a benevolent controller help?", that needs a different
objective, something closer to *supply the evidence the population is missing*.
As written, "truth control hurts" may only mean "advocacy hurts, regardless of
direction".

**2. It is not told its own evidence is weak.** None of its 24 facts rules out
any allocation; the median best single-fact posterior is 0.42 against a 0.33
prior. It writes as though each report should be decisive. Telling it the truth
about its evidence would likely change both how much it posts and how it
phrases things.

Both change what the controller *is*, not how clearly it is briefed, so they
belong in the experiment design rather than in a prompt revision.
