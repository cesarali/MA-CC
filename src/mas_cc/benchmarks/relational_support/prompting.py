"""The minimal single-model prompt: facts, question, options, one answer.

Everything that makes the multi-agent game a *game* is absent by construction -
there is no participant, no vote, no peer, no controller, no round, no shared
history.  A prompt here is a reasoning item and nothing else.

Two presentation choices matter for validity:

**Fact identifiers are not shown.**  The game shows them because an agent has to
cite one.  Here they would be a leak: a list reading ``f1, f3, f4`` announces
that ``f2`` was withheld, which is precisely the manipulation under test.  The
identifiers stay in the results table, where they belong.

**Facts are shuffled once per task, not once per condition.**  A single
presentation order is drawn from the task seed and every condition of that task
renders its subset in that same order.  Conditions therefore differ by deletion
only - never by reordering - and no condition can be identified from the shape
of its fact list.

**The answer options carry their own permutation** (see :mod:`.presentation`).
The relation-to-letter assignment is rebuilt per item so that position can be
controlled for; the FACTS and QUESTION blocks are byte-identical across the
permutations of a task, so a permutation moves the options and nothing else.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .conditions import EvidenceCondition
from .presentation import OptionPresentation

PROMPT_FAMILY = "relational_support_probe"
PROMPT_VERSION = 1

SYSTEM_PROMPT = (
    "You answer spatial reasoning questions.\n"
    "\n"
    "You are given a list of facts about where objects are, a question, and a\n"
    "numbered set of answer options. Work out which option follows from the\n"
    "facts you were given.\n"
    "\n"
    "The facts use the eight compass directions. \"X is north of Y\" means X is\n"
    "one step north of Y; \"X is northeast of Y\" means one step north and one\n"
    "step east, and so on. Directions compose: the answer is the overall\n"
    "direction obtained by following the facts from one object to the other.\n"
    "\n"
    "Some of the facts may be irrelevant to the question. You may not have\n"
    "every fact you would need; answer with the option best supported by what\n"
    "you were given, and always answer with exactly one option.\n"
    "\n"
    "End your reply with a final line in exactly this form:\n"
    "ANSWER: <label>"
)

NO_FACTS_LINE = "(You have not been given any facts about the objects in the question.)"

_ANSWER_PATTERN = re.compile(r"ANSWER\s*[:\-]?\s*\**\s*([A-H])\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class BenchmarkPrompt:
    """One rendered item, plus the audit trail the validator needs."""

    system: str
    user: str
    task_id: str
    condition: EvidenceCondition
    presentation: OptionPresentation
    shown_fact_ids: tuple[str, ...]
    facts_block: str
    options_block: str
    question: str

    def to_markdown(self) -> str:
        return (
            f"# {self.task_id} - {self.condition.condition_id}"
            f" ({self.condition.condition}) - {self.presentation.permutation_id}\n\n"
            f"- correct relation: displayed at "
            f"**{self.presentation.correct_display_position}**"
            " (the relation itself is never named here)\n\n"
            "## System\n\n```text\n"
            f"{self.system}\n```\n\n"
            "## User\n\n```text\n"
            f"{self.user}\n```\n"
        )


def presentation_order(task: Any) -> tuple[str, ...]:
    """A per-task shuffle of *all* fact ids, shared by every condition."""

    order = list(task.fact_order)
    random.Random(f"{task.task_id}|{task.seed}|presentation").shuffle(order)
    return tuple(order)


def render_prompt(
    task: Any,
    condition: EvidenceCondition,
    presentation: OptionPresentation,
    *,
    order: Sequence[str] | None = None,
) -> BenchmarkPrompt:
    """Render one item.  Distractors are always shown; support is filtered."""

    resolved_order = tuple(order) if order is not None else presentation_order(task)
    shown_support = set(condition.shown_supporting_fact_ids)
    distractors = set(task.distractor_fact_ids)
    shown_fact_ids = tuple(
        fact_id
        for fact_id in resolved_order
        if fact_id in shown_support or fact_id in distractors
    )
    fact_lines = [f"- {task.fact_text(fact_id)}" for fact_id in shown_fact_ids]
    facts_block = "\n".join(fact_lines) if fact_lines else NO_FACTS_LINE
    options_block = presentation.options_block
    user = (
        "FACTS\n"
        "\n"
        f"{facts_block}\n"
        "\n"
        "QUESTION\n"
        "\n"
        f"{task.question}\n"
        "\n"
        "OPTIONS\n"
        "\n"
        f"{options_block}\n"
        "\n"
        "Answer with one option label. End your reply with a final line in\n"
        "exactly this form:\n"
        "ANSWER: <label>"
    )
    return BenchmarkPrompt(
        system=SYSTEM_PROMPT,
        user=user,
        task_id=task.task_id,
        condition=condition,
        presentation=presentation,
        shown_fact_ids=shown_fact_ids,
        facts_block=facts_block,
        options_block=options_block,
        question=task.question,
    )


def parse_answer(content: str, option_labels: Sequence[str]) -> str | None:
    """The label the model settled on, or ``None`` if it never gave one.

    The declared ``ANSWER:`` line wins and the *last* one wins, so a reasoning
    model that thinks out loud and restates its choice is read correctly.  The
    fallback - a bare label standing alone on the last non-empty line - exists
    because that is the only other shape seen in practice; anything else is
    recorded as an unparsed response rather than guessed at.
    """

    labels = {label.upper() for label in option_labels}
    matches = _ANSWER_PATTERN.findall(content or "")
    for candidate in reversed(matches):
        if candidate.upper() in labels:
            return candidate.upper()
    for line in reversed([line.strip() for line in (content or "").splitlines()]):
        if not line:
            continue
        stripped = line.strip("*_`.,:;() ").upper()
        return stripped if stripped in labels else None
    return None
