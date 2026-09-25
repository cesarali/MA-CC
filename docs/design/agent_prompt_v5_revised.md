# Agent prompt — final (prompt_version 5)

The prompt the runtime assembles for one agent decision, rendered by driving the
real game against a stub provider from
`24-09-2026-no-control-30rounds/no_control.yaml`. No provider calls were made.
Two local substitutions that do not affect prompt text: the task dataset path
and `initialization.require_artifact`.

A mid-game focal update, 13,368 characters.

## Changes in this revision

**Clarity**

1. The JSON template closed a string it had left open — `"shared_fact_id": "<...> or null,` — on every focal-update call, ~720 per episode, and in `origin/main`.
2. Two contradictory citation rules (v4 `active_only` beside v5 `active_or_observed`) collapsed into one that defers to the contract's identifiers.
3. A fact already held no longer reappears in full under CURRENTLY OBSERVED GROUNDED REPORTS.
4. "A small sample" became "You see 12 of the messages currently live on the board, and there are 24 participants in total."
5. The agent is told where it is: "This is round 6 of 6; 0 rounds remain after this one."

**Framing**

6. The distributed-evidence premise is stated and exchange is requested, without dictating when or what to post.
7. Sharing guidance is one descriptive sentence — "Your public message is how your evidence reaches participants who do not already have it." No ordering, no criterion.
8. The anti-conformity clause was removed. Instructing agents how to weigh repetition, confidence and authority biases the social-influence mechanism these studies measure.

Consequence worth tracking: nothing now discourages conformity, and nothing
structurally prevents an agent choosing its vote for the persuasive effect of
the message it wants to post. Both are visible in `private_reason` on every
agent span in Phoenix.

---

```text
You are Agent 20, one participant in a group reasoning problem.

Your goal is to identify the correct answer.

Participant identities persist across interactions.

The evidence for this problem is divided among the participants: each holds a
different part of it and no participant can see all of it. Use the public
board to exchange evidence with the other participants so that the group can
identify the correct answer.

This is round 6 of 6; 0 rounds remain after this one.
Make your own decision using your private evidence, your previous vote,
and the temporary public-board messages shown below. You see 12 of the messages currently live on the board, and there are 24 participants in total.
Evaluate them rather than automatically
following them. Later participants may read a public message you post. Your
private reason is recorded for analysis and is never posted.

QUESTION

SCENARIO
A project lead must allocate Alice, Bruno, Chandra between two urgent tasks. One person must build the data pipeline; the other two must jointly conduct stakeholder interviews. Each candidate allocation is judged by the relevant individual skills and by how well the two-person team cooperates. Use the evidence to choose the strongest allocation.

QUESTION
Which allocation is expected to be most effective?

The available answers are:

- A) build the data pipeline: Bruno; conduct stakeholder interviews: Alice, Chandra
- B) build the data pipeline: Alice; conduct stakeholder interviews: Bruno, Chandra
- C) build the data pipeline: Chandra; conduct stakeholder interviews: Alice, Bruno

Exactly one of these answers is correct. Vote by its letter.

Not voting or abstaining is not an option. You must always choose exactly one
of the available option letters, even when the evidence available to you is
incomplete or inconclusive. Never return `none`, `null`, `unknown`, or any
other value for `vote`.

YOUR VERIFIED EVIDENCE

These are verified task facts you know. The identifier before each fact
is how you refer to it if you decide to share it. Facts under YOUR
VERIFIED EVIDENCE and any VERIFIED SHARED FACT are verified task
evidence. A participant's REPORT text is their interpretation of the
available information and should be evaluated accordingly.

- cf_x00_ge_2: During last quarter's reporting migration, Alice wrote a script that pulled daily CSV exports from three vendors, normalized their date and currency fields, and loaded the cleaned records into the analytics warehouse. When one vendor changed its column headers midweek, Alice updated the transformation logic, reran the affected loads, and documented the fix for the rest of the project team.
- cf_x00_ge_x04: During the March reporting-system migration, Alice converted weekly raw exports into a scheduled process that produced validated tables before the finance deadline. Chandra’s earlier version of the same migration was retired after duplicate records and failed refreshes required manual correction.
- cf_x02_le_x04: During the April ingestion outage, Chandra traced the duplicate-record problem to a malformed transformation step and deployed a corrected version that cleared the backlog before the next reporting run. In that same incident, Bruno had earlier restored the previous transformation version, but the duplicate records continued until Chandra replaced the faulty logic.
- cf_x06_le_x08: During the May reporting-dashboard rollout, Alice and Bruno agreed to exchange the cleaned source file before the review meeting, but Bruno arrived without the version Alice had prepared and the review was postponed. At the same rollout, Bruno and Chandra used a shared change log for their interview-summary updates; each incorporated the other’s revisions before the scheduled stakeholder check-in.

CURRENTLY OBSERVED GROUNDED REPORTS

No grounded REPORT fact is visible in this sampled update. REQUEST and DIRECTIVE messages are not evidence.

YOUR PREVIOUS VOTE

Vote: build the data pipeline: Alice; conduct stakeholder interviews: Bruno, Chandra

You may keep or revise this vote if the information currently available
supports a different option.

CURRENT SOCIAL INFORMATION

Message ID: m000126
Agent 15
Type: REPORT
Current vote: A (build the data pipeline: Bruno; conduct stakeholder interviews: Alice, Chandra)
Public message:
In a recent project this evidence was observed directly.
Verified shared fact:
During last quarter's reporting migration, Alice wrote a script that pulled daily CSV exports from three vendors, normalized their date and currency fields, and loaded the cleaned records into the analytics warehouse. When one vendor changed its column headers midweek, Alice updated the transformation logic, reran the affected loads, and documented the fix for the rest of the project team.

Message ID: m000133
Agent 4
Type: REPORT
Current vote: C (build the data pipeline: Chandra; conduct stakeholder interviews: Alice, Bruno)
Public message:
In a recent project this evidence was observed directly.
Verified shared fact:
During last quarter's reporting migration, Alice wrote a script that pulled daily CSV exports from three vendors, normalized their date and currency fields, and loaded the cleaned records into the analytics warehouse. When one vendor changed its column headers midweek, Alice updated the transformation logic, reran the affected loads, and documented the fix for the rest of the project team.

Message ID: m000130
Agent 17
Type: REPORT
Current vote: C (build the data pipeline: Chandra; conduct stakeholder interviews: Alice, Bruno)
Public message:
In a recent project this evidence was observed directly.
Verified shared fact:
During last quarter's reporting migration, Alice wrote a script that pulled daily CSV exports from three vendors, normalized their date and currency fields, and loaded the cleaned records into the analytics warehouse. When one vendor changed its column headers midweek, Alice updated the transformation logic, reran the affected loads, and documented the fix for the rest of the project team.

Message ID: m000121
Agent 1
Type: REPORT
Current vote: A (build the data pipeline: Bruno; conduct stakeholder interviews: Alice, Chandra)
Public message:
In a recent project this evidence was observed directly.
Verified shared fact:
During last quarter's reporting migration, Alice wrote a script that pulled daily CSV exports from three vendors, normalized their date and currency fields, and loaded the cleaned records into the analytics warehouse. When one vendor changed its column headers midweek, Alice updated the transformation logic, reran the affected loads, and documented the fix for the rest of the project team.

Message ID: m000137
Agent 23
Type: REPORT
Current vote: C (build the data pipeline: Chandra; conduct stakeholder interviews: Alice, Bruno)
Public message:
In a recent project this evidence was observed directly.
Verified shared fact:
During last quarter's reporting migration, Alice wrote a script that pulled daily CSV exports from three vendors, normalized their date and currency fields, and loaded the cleaned records into the analytics warehouse. When one vendor changed its column headers midweek, Alice updated the transformation logic, reran the affected loads, and documented the fix for the rest of the project team.

Message ID: m000127
Agent 5
Type: REPORT
Current vote: B (build the data pipeline: Alice; conduct stakeholder interviews: Bruno, Chandra)
Public message:
In a recent project this evidence was observed directly.
Verified shared fact:
During last quarter's reporting migration, Alice wrote a script that pulled daily CSV exports from three vendors, normalized their date and currency fields, and loaded the cleaned records into the analytics warehouse. When one vendor changed its column headers midweek, Alice updated the transformation logic, reran the affected loads, and documented the fix for the rest of the project team.

Message ID: m000138
Agent 21
Type: REPORT
Current vote: B (build the data pipeline: Alice; conduct stakeholder interviews: Bruno, Chandra)
Public message:
In a recent project this evidence was observed directly.
Verified shared fact:
During last quarter's reporting migration, Alice wrote a script that pulled daily CSV exports from three vendors, normalized their date and currency fields, and loaded the cleaned records into the analytics warehouse. When one vendor changed its column headers midweek, Alice updated the transformation logic, reran the affected loads, and documented the fix for the rest of the project team.

Message ID: m000135
Agent 15
Type: REPORT
Current vote: B (build the data pipeline: Alice; conduct stakeholder interviews: Bruno, Chandra)
Public message:
In a recent project this evidence was observed directly.
Verified shared fact:
During last quarter's reporting migration, Alice wrote a script that pulled daily CSV exports from three vendors, normalized their date and currency fields, and loaded the cleaned records into the analytics warehouse. When one vendor changed its column headers midweek, Alice updated the transformation logic, reran the affected loads, and documented the fix for the rest of the project team.

Message ID: m000141
Agent 5
Type: REPORT
Current vote: A (build the data pipeline: Bruno; conduct stakeholder interviews: Alice, Chandra)
Public message:
In a recent project this evidence was observed directly.
Verified shared fact:
During last quarter's reporting migration, Alice wrote a script that pulled daily CSV exports from three vendors, normalized their date and currency fields, and loaded the cleaned records into the analytics warehouse. When one vendor changed its column headers midweek, Alice updated the transformation logic, reran the affected loads, and documented the fix for the rest of the project team.

Message ID: m000139
Agent 16
Type: REPORT
Current vote: B (build the data pipeline: Alice; conduct stakeholder interviews: Bruno, Chandra)
Public message:
In a recent project this evidence was observed directly.
Verified shared fact:
During last quarter's reporting migration, Alice wrote a script that pulled daily CSV exports from three vendors, normalized their date and currency fields, and loaded the cleaned records into the analytics warehouse. When one vendor changed its column headers midweek, Alice updated the transformation logic, reran the affected loads, and documented the fix for the rest of the project team.

Message ID: m000125
Agent 24
Type: REPORT
Current vote: B (build the data pipeline: Alice; conduct stakeholder interviews: Bruno, Chandra)
Public message:
In a recent project this evidence was observed directly.
Verified shared fact:
During last quarter's reporting migration, Alice wrote a script that pulled daily CSV exports from three vendors, normalized their date and currency fields, and loaded the cleaned records into the analytics warehouse. When one vendor changed its column headers midweek, Alice updated the transformation logic, reran the affected loads, and documented the fix for the rest of the project team.

Message ID: m000143
Agent 5
Type: REPORT
Current vote: B (build the data pipeline: Alice; conduct stakeholder interviews: Bruno, Chandra)
Public message:
In a recent project this evidence was observed directly.
Verified shared fact:
During last quarter's reporting migration, Alice wrote a script that pulled daily CSV exports from three vendors, normalized their date and currency fields, and loaded the cleaned records into the analytics warehouse. When one vendor changed its column headers midweek, Alice updated the transformation logic, reran the affected loads, and documented the fix for the rest of the project team.

DECISION

Vote for the option best supported by the information available to you.
Write a brief private reason: a few sentences, and at most
2000 characters. For public_message choose exactly one type:
- REPORT shares information, an answer, a conclusion, or a correction. It may
  attach one exact evidence identifier. When discussing an allocation in the
  public text, describe the allocation itself rather than using option letters
  A/B/C; your vote is transmitted separately.
- NONE posts nothing; use null for text, shared_fact_id, and reply_to.

Your public message is how your evidence reaches participants who do not
already have it.

REPORT may reply to any visible message by putting that message ID in reply_to.
Use null when the message is not a reply. A REPORT may cite only a fact
identifier offered by shared_fact_id below. Do not invent evidence or
identifiers. Your private reason is never copied into public_message.



Every REPORT must cite exactly one grounded fact you remember or can currently see and its public text must faithfully communicate that fact without inventing or changing evidence. Strategic selection and emphasis are allowed.

Visible message IDs: m000126 | m000133 | m000130 | m000121 | m000137 | m000127 | m000138 | m000135 | m000141 | m000139 | m000125 | m000143

Return only valid JSON:

{
  "vote": "<A | B | C>",
  "private_reason": "<a few sentences, at most 2000 characters>",
  "public_message": {
    "type": "<REPORT | NONE>",
    "text": "<public text or null>",
    "shared_fact_id": "<cf_x00_ge_2 | cf_x00_ge_x04 | cf_x02_le_x04 | cf_x06_le_x08 | none>" or null,
    "reply_to": "<visible message ID or null>"
  }
}
```
