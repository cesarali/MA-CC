"""Decoupling the semantic answer from the letter it is displayed under.

The first run of this benchmark found that under insufficient information the
model answered ``A`` in 89% of zero-evidence items.  That makes the stored
``correct_option`` letter a confound rather than a label: ``accuracy_zero`` stops
measuring "chance" and starts measuring "how often A happened to be correct".

The fix does **not** touch the generated worlds.  Facts, supporting chains,
distractors and the correct semantic relation are exactly what the generator
froze.  Only the relation-to-letter assignment is rebuilt at prompt time, and
correctness is then scored on the **relation** the model picked, never on the
letter.

Rather than trusting a random assignment to come out balanced, the design is
crossed: every task and evidence condition is presented three times, with the
correct relation at ``A``, at ``B``, and at ``C``.  Position is therefore
balanced by construction within every cell of the analysis, and "accuracy at
this position" becomes a directly measurable quantity instead of a subgroup
whose size is an accident.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

FROZEN = "frozen"
ALL_POSITIONS = "all_positions"
PRESENTATION_MODES = (ALL_POSITIONS, FROZEN)


@dataclass(frozen=True, slots=True)
class OptionPresentation:
    """One assignment of the task's relations to the displayed labels."""

    permutation_id: str
    correct_display_position: str
    labels: tuple[str, ...]
    relation_by_label: Mapping[str, str]
    correct_relation: str

    @property
    def displayed_order(self) -> tuple[str, ...]:
        return tuple(self.relation_by_label[label] for label in self.labels)

    @property
    def options_block(self) -> str:
        return "\n".join(f"{label}) {self.relation_by_label[label]}" for label in self.labels)

    def relation_for(self, label: str | None) -> str | None:
        return self.relation_by_label.get(label) if label else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "permutation_id": self.permutation_id,
            "correct_display_position": self.correct_display_position,
            "displayed_option_order": "|".join(self.displayed_order),
        }


def build_presentations(
    task: Any, *, seed: int, mode: str = ALL_POSITIONS
) -> tuple[OptionPresentation, ...]:
    """Every presentation of one task's options, in label order.

    ``all_positions`` returns one presentation per label, each placing the
    correct relation at that label - the balanced crossing.  ``frozen`` returns
    the generator's own assignment unchanged, which reproduces the pre-control
    behaviour and exists so the two can be compared rather than argued about.

    The two incorrect relations keep a single per-task order across all
    presentations, so the only thing that moves between them is where the
    correct relation sits.
    """

    if mode not in PRESENTATION_MODES:
        raise ValueError(f"unknown presentation mode {mode!r}; expected {PRESENTATION_MODES}")

    labels = tuple(task.option_labels)
    correct = task.correct_relation

    if mode == FROZEN:
        return (
            OptionPresentation(
                permutation_id="frozen",
                correct_display_position=task.correct_option,
                labels=labels,
                relation_by_label=dict(task.option_relations),
                correct_relation=correct,
            ),
        )

    others = sorted(
        relation for label, relation in task.option_relations.items() if relation != correct
    )
    if len(others) != len(labels) - 1:
        raise ValueError(
            f"task {task.task_id}: expected exactly one option equal to the correct relation"
        )
    # One shuffle per task, reused by every placement: the incorrect relations
    # hold their relative order so that the placements differ in exactly one way.
    random.Random(f"{seed}|{task.task_id}|{task.seed}|options").shuffle(others)

    presentations: list[OptionPresentation] = []
    for index, position in enumerate(labels):
        remaining = iter(others)
        relation_by_label = {
            label: (correct if slot == index else next(remaining))
            for slot, label in enumerate(labels)
        }
        presentations.append(
            OptionPresentation(
                permutation_id=f"pos{position}",
                correct_display_position=position,
                labels=labels,
                relation_by_label=relation_by_label,
                correct_relation=correct,
            )
        )
    return tuple(presentations)


def position_balance(presentations: Sequence[OptionPresentation]) -> dict[str, int]:
    """How many presentations place the correct relation at each label."""

    counts: dict[str, int] = {}
    for presentation in presentations:
        counts[presentation.correct_display_position] = (
            counts.get(presentation.correct_display_position, 0) + 1
        )
    return counts
