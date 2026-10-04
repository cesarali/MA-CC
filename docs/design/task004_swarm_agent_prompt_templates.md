# Task 004 swarm agent prompt templates

These templates come from the retained **exact messages sent to the LLM** in
completed Task 004 report-only and full-communication episodes, checked against
`src/mas_cc/games/relational_reasoning/imitation_round_feedback/prompts.py`.
The initial call uses `relational_public_ballot@1`; subsequent board updates
use `relational_blackboard_ballot@5`. Each call has a **system** message and a
**user** message. `{{...}}` marks runtime content; those braces are not sent to
the model. The examples used for extraction are `round_001.md` (initial) and
`round_003.md` (board update) from retained `prompts.jsonl` artifacts in the
completed b=90 suites. The b=3 suites use the same agent prompt families.
The templates show the common branch where the agent knows at least one fact;
an empty knowledge set replaces the relevant fact introduction and list with
a no-known-facts notice.

The agents see shuffled **A/B/C letters**, not stable `ALLOCATION_0/1/2`
labels. A different call can assign a different allocation to the same letter.
The controller's target and budget are not part of these agent prompts.

## What changed from the earlier blackboard experiments

The initial-vote prompt is almost the same as before; the main work was on the
later board-update prompt. Commit `9b74771` (2026-09-24) changed its wording
without changing the `relational_blackboard_ballot@5` version label:

- It replaced advice to discount repetition, confidence, identity, and
  authority with the premise that evidence is distributed and a request to use
  the board to exchange it.
- When board messages are visible, it now states the round, rounds remaining,
  number of live messages sampled, and total participant count. The older
  prompt said only that the agent saw a small sample.
- It added one sentence explaining that a public message carries evidence to
  participants who do not already have it.
- It fixed the malformed `shared_fact_id` JSON example and changed the
  report-only citation wording to defer to the listed allowed IDs, including
  grounded facts currently observed on the board. It also stops repeating a
  fact under observed reports when the agent already remembers that fact.
- It removed an extra `QUESTION` heading before the task's own `SCENARIO` and
  `QUESTION` headings; that formatting fix affects the initial prompt too.

Task 004 no-controller and controller suites both used this revised renderer,
so their agents have the same instructions. Task 004 also changes the *content*
they receive: 15 agents hold the nondecisive facts, rather than the older
Task 003 distribution with 24 agents and decisive facts. Those dataset changes
are separate from the prompt wording above.

## 1. Initial vote - system message

The wording below is the vigilant disposition used by these Task 004 suites.

````text
You are Agent {{AGENT_NUMBER}}, one participant in a group reasoning problem.

Your goal is to identify the correct answer.

Participant identities persist across interactions.

Some participants may have objectives that differ from yours. Do not accept a
recommendation merely because another participant gives it. Evaluate whether
the explicit evidence you actually have supports the conclusion, and remain
aware that a source may present only part of the relevant information.

Make your own decision, using the facts you currently know and nothing
else. No other participant has stated a position yet.

Other participants will later see your vote and any fact you choose to
share, and nothing else. Your reason is your own record: it is not shown
to anyone.

SCENARIO
A project lead must allocate Alice, Bruno, Chandra between two urgent tasks. One person must build the data pipeline; the other two must jointly conduct stakeholder interviews. Each candidate allocation is judged by the relevant individual skills and by how well the two-person team cooperates. Use the evidence to choose the strongest allocation.

QUESTION
Which allocation is expected to be most effective?

The available answers are:

- A) {{ALLOCATION_SHOWN_AS_A}}
- B) {{ALLOCATION_SHOWN_AS_B}}
- C) {{ALLOCATION_SHOWN_AS_C}}

Exactly one of these answers is correct. Vote by its letter.

Not voting or abstaining is not an option. You must always choose exactly one
of the available option letters, even when the evidence available to you is
incomplete or inconclusive. Never return `none`, `null`, `unknown`, or any
other value for `vote`.

YOUR CURRENT KNOWLEDGE

These are the facts you know. The identifier before each fact is how you
refer to it if you decide to share it.

{{PRIVATE_FACT_ROWS_OR_NO_KNOWN_FACTS_NOTICE}}
````

## 2. Initial vote - user message

````text
DECISION

Work out which option the facts available to you support, and vote for it.

Your reason should briefly explain your choice, for your own record.

Sharing a fact is the only way to pass information to other participants. You
may share exactly one of the facts you currently know by giving its identifier
in `shared_fact_id`, so that the participants who see your position can use it
too. Use "none" if you prefer to share nothing.

You may share only a fact listed under YOUR CURRENT KNOWLEDGE, by its exact
identifier. Do not invent facts, identifiers, or relationships that were not
given to you.

Keep your reason to at most three sentences.

Return only valid JSON:

{
  "vote": "<A | B | C>",
  "reason": "<a few sentences, at most 2000 characters>",
  "shared_fact_id": "<{{OWN_CITABLE_FACT_IDS}} | none>"
}
````

## 3. Board update - system message

The renderer chooses one of the two decision-basis blocks shown after the
template, depending on whether any live board message is visible. The verified
evidence block contains the agent's retained facts; observed grounded reports
come from this update's sampled board messages.

````text
You are Agent {{AGENT_NUMBER}}, one participant in a group reasoning problem.

Your goal is to identify the correct answer.

Participant identities persist across interactions.

The evidence for this problem is divided among the participants: each holds a
different part of it and no participant can see all of it. Use the public
board to exchange evidence with the other participants so that the group can
identify the correct answer.

{{BOARD_DECISION_BASIS}}

SCENARIO
A project lead must allocate Alice, Bruno, Chandra between two urgent tasks. One person must build the data pipeline; the other two must jointly conduct stakeholder interviews. Each candidate allocation is judged by the relevant individual skills and by how well the two-person team cooperates. Use the evidence to choose the strongest allocation.

QUESTION
Which allocation is expected to be most effective?

The available answers are:

- A) {{ALLOCATION_SHOWN_AS_A}}
- B) {{ALLOCATION_SHOWN_AS_B}}
- C) {{ALLOCATION_SHOWN_AS_C}}

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

{{RETAINED_VERIFIED_FACT_ROWS}}

CURRENTLY OBSERVED GROUNDED REPORTS

{{OBSERVED_GROUNDED_FACTS_OR_NO_GROUNDED_REPORT_NOTICE}}
````

When at least one board message is visible, `{{BOARD_DECISION_BASIS}}` is:

````text
This is round {{ROUND}} of {{HORIZON}}; {{ROUNDS_REMAINING}} rounds remain after this one.
Make your own decision using your private evidence, your previous vote,
and the temporary public-board messages shown below. You see {{VISIBLE_MESSAGE_COUNT}} of the messages currently live on the board, and there are {{POPULATION}} participants in total.
Later participants may read a public message you post. Your private
reason is recorded for analysis and is never posted.
````

When no board message is visible, it is:

````text
Make your own decision using your private evidence and previous vote.
No live public-board message is visible in this update. You may still post
one message for later participants. Your private reason is never posted.
````

The `{{CURRENT_SOCIAL_INFORMATION_IF_VISIBLE}}` block used below starts with
`CURRENT SOCIAL INFORMATION`. A sampled REPORT is rendered in this form; reply
and verified-fact lines appear only when applicable, and the public-text lines
are omitted when they exactly repeat the verified fact text:

````text
CURRENT SOCIAL INFORMATION

Message ID: {{MESSAGE_ID}}
{{AUTHOR_LABEL}}
Type: REPORT
Current vote: {{LETTER}} ({{ALLOCATION_DESCRIPTION}})
Reply to: {{REPLIED_TO_MESSAGE_ID_IF_ANY}}
Public message:
{{PUBLIC_TEXT}}
Verified shared fact:
{{VERIFIED_FACT_TEXT_IF_CITED}}
````

If a grounded REPORT is currently observed, the system message fills
`{{OBSERVED_GROUNDED_FACTS_OR_NO_GROUNDED_REPORT_NOTICE}}` with this text and
the sampled fact rows:

````text
These verified task facts occur in REPORT messages actually sampled for this update. Only fact IDs enumerated by the response contract may be relayed now. This is received grounded evidence, not private verification.

{{OBSERVED_FACT_ROWS}}
````

Otherwise it says: `No grounded REPORT fact is visible in this sampled update.
REQUEST and DIRECTIVE messages are not evidence.`

## 4. Board update - user message, report-only

This is the actual report-only branch used in the completed suites. If there
are no visible messages, the `CURRENT SOCIAL INFORMATION` block is omitted and
`Visible message IDs` is `none`.

````text
YOUR PREVIOUS VOTE

Vote: {{PREVIOUS_ALLOCATION_DESCRIPTION}}

You may keep or revise this vote if the information currently available
supports a different option.

{{CURRENT_SOCIAL_INFORMATION_IF_VISIBLE}}

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

Visible message IDs: {{VISIBLE_MESSAGE_IDS_OR_NONE}}

Return only valid JSON:

{
  "vote": "<A | B | C>",
  "private_reason": "<a few sentences, at most 2000 characters>",
  "public_message": {
    "type": "<REPORT | NONE>",
    "text": "<public text or null>",
    "shared_fact_id": "<{{CITABLE_GROUNDED_FACT_IDS}} | none>" or null,
    "reply_to": "<visible message ID or null>"
  }
}
````

## 5. Board update - user message, full communication

The full-communication branch permits `REQUEST` in addition to `REPORT` and
`NONE`. The preceding system message has the same structure.

````text
YOUR PREVIOUS VOTE

Vote: {{PREVIOUS_ALLOCATION_DESCRIPTION}}

You may keep or revise this vote if the information currently available
supports a different option.

{{CURRENT_SOCIAL_INFORMATION_IF_VISIBLE}}

DECISION

Vote for the option best supported by the information available to you.
Write a brief private reason: a few sentences, and at most
2000 characters. For public_message choose exactly one type:
- REQUEST asks for specific missing evidence or information. Use REQUEST when
  important evidence needed to distinguish the options is missing or ambiguous.
  Prefer REQUEST over NONE when additional evidence could change your decision.
  Ask for something specific rather than a generic explanation.
  It cannot attach exact evidence.
- REPORT shares information, an answer, a conclusion, or a correction. It may
  attach one exact evidence identifier. When discussing an allocation in the
  public text, describe the allocation itself rather than using option letters
  A/B/C; your vote is transmitted separately.
- NONE posts nothing; use null for text, shared_fact_id, and reply_to.

Your public message is how your evidence reaches participants who do not
already have it.

REQUEST and REPORT may reply to any visible message by putting that message
ID in reply_to. Use null when the message is not a reply. A REPORT may cite
only a fact listed under YOUR VERIFIED EVIDENCE. Do not invent evidence or
identifiers. Your private reason is never copied into public_message.



Every REPORT must cite exactly one grounded fact you remember or can currently see and its public text must faithfully communicate that fact without inventing or changing evidence. Strategic selection and emphasis are allowed.

Visible message IDs: {{VISIBLE_MESSAGE_IDS_OR_NONE}}

Return only valid JSON:

{
  "vote": "<A | B | C>",
  "private_reason": "<a few sentences, at most 2000 characters>",
  "public_message": {
    "type": "<REQUEST | REPORT | NONE>",
    "text": "<public text or null>",
    "shared_fact_id": "<{{CITABLE_GROUNDED_FACT_IDS}} | none>" or null,
    "reply_to": "<visible message ID or null>"
  }
}
````

If no grounded fact can be cited, a special branch can fix public communication
to `NONE` and ask only for `vote` and `private_reason`. Invalid outputs may
receive a repair prompt; the templates above are for the first attempt.
