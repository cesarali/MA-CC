# TDD: Blackboard Prompt Patch — Recipient-Specific Vote Rendering + Stronger REQUEST Use

**Date:** 2026-09-04  
**Status:** implementation handoff  
**Scope:** minimal prompt/rendering patch for `relational_blackboard_ballot`  
**Priority:** high — fixes semantic inconsistency caused by option shuffling

## 1. Goal

Make two small, targeted changes without redesigning the current prompt:

1. **Fix option-letter rendering for retrieved board messages.**
2. **Make REQUEST behavior more explicit and useful.**

Do **not** otherwise restructure the prompt. DeepSeek and other models are sensitive to prompt changes, so preserve the existing block order, wording style, response schema, and runtime behavior as much as possible.

This should be implemented as a new prompt/version fingerprint. Existing episodes and historical results remain immutable.

## 2. Problem A — stale option letters in public messages

The game randomizes the displayed answer order per LLM call. That is fine.

The bug is that a sender may write free text such as:

```text
"So I stick with C."
```

but when the same message is shown later to another agent, the recipient may have a different letter-to-allocation mapping.

The structured semantic vote may render correctly, while the free-text letter becomes stale and contradictory.

Example:

```text
sender mapping:
C -> ALLOCATION_1

recipient mapping:
A -> ALLOCATION_1
```

The recipient must see the sender's vote as:

```text
Current vote: A (<semantic allocation text>)
```

and must never reuse the sender's old letter as the authoritative rendered vote.

## 3. Required representation invariant

The hard invariant is:

\[
\boxed{\text{votes are stored semantically; letters exist only at render time}}
\]

Internally, board/runtime records should store:

```text
ALLOCATION_0
ALLOCATION_1
ALLOCATION_2
```

or the existing semantic relation identifier.

Do not store a letter as the authoritative vote identity.

When rendering any retrieved message to a recipient:

```text
semantic_vote
    ->
recipient's current option mapping
    ->
recipient-visible letter
```

The displayed letter must always correspond to the recipient's current randomized option order.

## 4. Message rendering rule

For every board message visible to an agent:

```text
Current vote: <recipient-specific letter> (<semantic allocation text>)
```

The letter must be computed from the current recipient prompt's option map.

The semantic allocation text should remain available as the unambiguous human-readable referent.

No sender-specific letter should be reused.

## 5. Public free-text rule

Prevent new stale-letter contamination at the source.

Add one small instruction to REPORT behavior:

> When discussing an allocation in public_message text, describe the allocation itself rather than using option letters A/B/C; your vote is transmitted separately.

Examples:

### Allowed

```text
I think Alice should analyze while Bruno and Chandra coordinate because ...
```

### Not allowed

```text
I think B is best.
```

or:

```text
I stick with C.
```

The structured `vote` field remains the only place where the agent returns A/B/C.

The runtime then resolves that letter immediately to the semantic allocation as it already does.

## 6. Do not sanitize historical text by guesswork

Do not attempt to reinterpret stale A/B/C references in already-created historical public text.

Historical runs remain unchanged.

For the new prompt version:

- prevent new public-message letter references through the prompt;
- store semantic votes;
- render structured message votes using recipient-specific mapping.

If a newly generated public message still includes an option letter despite instructions, preserve it in raw logs for audit but consider adding a lightweight validator/warning.

Do not silently rewrite model prose unless there is already an approved normalization layer.

## 7. Problem B — REQUEST is underused

The current protocol already supports:

```text
REQUEST
REPORT
NONE
```

but REQUEST is not behaviorally salient enough.

Do not add a new message type.

Do not change board mechanics.

Only strengthen the existing REQUEST instruction.

## 8. Minimal prompt patch

Keep the current `DECISION` block structure.

Replace the current short REQUEST/REPORT descriptions with:

```text
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
```

Everything else in the current prompt should remain as close as possible to the existing version.

## 9. Scope firewall

This change may modify only:

- prompt text for REQUEST/REPORT guidance;
- semantic vote storage if any stale-letter storage remains;
- board-message rendering of structured votes;
- associated validation/tests;
- prompt/version fingerprint.

Do **not** change:

- task content;
- option randomization;
- semantic answer resolution;
- blackboard sampling;
- `q`;
- `q_c`;
- controller logic;
- night/dawn/day timing;
- persistence;
- message lifetime;
- evidence acquisition;
- controller report logic;
- estimators;
- analysis definitions;
- previous study outputs.

## 10. Required implementation checks

### 10.1 Semantic vote storage

Inspect the complete path:

```text
LLM vote letter
-> response parser
-> semantic allocation resolution
-> board message record
-> later board retrieval
-> recipient prompt rendering
```

Confirm that after parsing, the authoritative stored vote is semantic.

If both letter and semantic forms are stored for diagnostics, the semantic form must be authoritative.

### 10.2 Recipient-specific rendering

Create a deterministic regression fixture.

Example:

```text
Sender sees:
A -> ALLOCATION_0
B -> ALLOCATION_2
C -> ALLOCATION_1

Sender votes:
C
```

The stored vote must become:

```text
ALLOCATION_1
```

Later:

```text
Recipient sees:
A -> ALLOCATION_1
B -> ALLOCATION_0
C -> ALLOCATION_2
```

The rendered board message must say:

```text
Current vote: A (<ALLOCATION_1 semantic text>)
```

It must not say:

```text
Current vote: C
```

## 11. Public-message tests

Add tests that verify the prompt explicitly instructs:

```text
do not use A/B/C in public message prose
```

The JSON `vote` field must still allow:

```text
A | B | C
```

The public message schema remains unchanged.

Optional but useful:

- flag public text containing isolated option-letter patterns such as `A`, `B`, `C`, `option A`, `option B`, `option C`;
- do not fail runtime solely on this initially unless already consistent with existing validation philosophy;
- retain raw content for audit.

## 12. REQUEST tests

Add prompt-level tests confirming the decision block now says:

```text
REQUEST when important evidence is missing or ambiguous
prefer REQUEST over NONE when more evidence could change the decision
ask for something specific
```

Do not force REQUEST mechanically.

The LLM still decides among:

```text
REQUEST
REPORT
NONE
```

based on the current state.

## 13. Prompt versioning

Create a new prompt version, e.g.:

```text
relational_blackboard_ballot@3
```

or the next repository-consistent version.

Do not mutate the historical prompt definition in a way that changes reproduction of old runs.

The new prompt hash/fingerprint must reflect the change.

## 14. Smoke test

Run a small provider-free/fake-provider smoke test that exercises:

1. two different option permutations;
2. one sender message stored semantically;
3. later retrieval by another recipient;
4. correct recipient-specific letter rendering;
5. unchanged semantic allocation text;
6. unchanged JSON response schema;
7. updated REQUEST/REPORT instructions.

Then inspect one exact rendered prompt manually.

## 15. Optional small real-model probe

Only after provider-free tests pass, run a very small prompt-only probe if convenient.

Goal:

- verify models still parse the prompt correctly;
- verify REQUEST can actually occur;
- verify public REPORT text tends to use semantic allocation descriptions rather than letters.

This is not a scientific study and should not alter any population-run configuration.

## 16. Acceptance criteria

Implementation passes if:

- retrieved board-message votes always match the recipient's current option ordering;
- no stale sender letter is used for the structured displayed vote;
- authoritative stored vote identity is semantic;
- prompt explicitly discourages A/B/C inside public prose;
- REQUEST guidance is stronger but the message protocol is otherwise unchanged;
- response JSON contract is unchanged;
- existing game mechanics are unchanged;
- old prompt/run reproducibility is preserved;
- regression and smoke tests pass.

## 17. Final report

At completion provide:

```text
files changed
prompt version
tests added
test results

confirmation that:
- structured votes are stored semantically
- recipient-specific rendering is correct
- public-message letter contamination is prevented by prompt guidance
- REQUEST guidance was strengthened
- no other game/runtime/scientific behavior changed

one exact before/after rendered example
```

Keep this implementation deliberately small and surgical.
