# Instructions for Claude

## How to talk to me

Explain everything in the simplest words that are still accurate.
Assume I am smart but new to this part of the code, so I understand ideas
quickly but I do not yet know the vocabulary.

### Define every technical term the first time you use it

Never use a technical term, an abbreviation, or a name from this codebase
without saying what it means in the same sentence or the one right after.
This applies to:

- Programming words (for example: closure, mutex, idempotent, monad).
- Library and tool names (for example: pyarrow, SLURM, ruff).
- Names invented inside this repo (for example: a study, a round, a game,
  a recorder). These are the ones I am most likely to be missing, because
  I cannot look them up anywhere outside this project.
- Abbreviations. Write the full words once, then the short form:
  "MI (mutual information, a number saying how much one thing tells you
  about another)".

If a definition would take more than a sentence, give the one-sentence
version first and offer the longer one afterwards.

### Prefer plain words over jargon

If a plain word says the same thing, use the plain word.

- "runs again on its own" instead of "recurses"
- "safe to run twice" instead of "idempotent"
- "the setting is read once when the program starts" instead of
  "resolved at import time"

Use the technical term when it is genuinely more precise, or when I will
see that exact word in the code, the logs, or the documentation. In that
case, give the plain meaning next to it so I can connect the two.

### Explain the why, not only the what

When you describe a change, say what problem it solves and what would go
wrong without it. A change I cannot explain to someone else is a change I
cannot review.

### Use concrete examples

Show a small, real example instead of describing a shape in the abstract.
One sample input and its output beats a paragraph about the input format.

### Keep the structure simple too

- Short sentences. One idea per sentence.
- Short paragraphs. Lists where a list fits.
- Say the answer first, then the reasoning. Do not build up to it.
- No filler like "as you know" or "simply" or "just" — if it were
  obvious, I would not have asked.

### When you are unsure

Say so plainly: "I am not sure whether X, because I have not checked Y."
Do not hide uncertainty behind confident-sounding technical language.

## Repository instructions

The environment, experiment, and workflow rules for this repository live in
[AGENTS.md](AGENTS.md). Follow those as well — this file governs *how you
explain things*, `AGENTS.md` governs *what you do*.
