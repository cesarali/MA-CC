"""Which real task fact the controller cites, and when there isn't one.

This module contains **no fact-selection logic of its own**.  It calls
:class:`~mas_cc.games.relational_reasoning.imitation_round_feedback.controller.RelationalRoundBudgetedControl.resolve_fact_id`,
the same method the population runtime calls, and records what came back.  A
second heuristic implementation living here would drift from the one the real
studies use, and then the probe would be measuring a controller nobody runs.

The two strategies, in the established sense:

``neutral``
    One real frozen fact chosen by a rule that does not look at the target -
    concretely, the task's first fact in frozen order.

``strategic``
    One real frozen fact chosen *in a target-aware way*: the true fact whose own
    compass direction best aligns with the direction the controller is pushing.
    This is **selective disclosure** - cherry-picking which true thing to say -
    and never a false, negated, paraphrased or invented fact.

When no true fact is favourable to a target, the strategic cell is **not
admissible**.  The probe then drops that cell and says so in the audit.  It does
not fall back to neutral evidence, change the target, or make a fact up: at
``L=1`` an inadmissible false-target cell is a real property of one-step
relations, and hiding it would turn a finding into a missing row.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mas_cc.games.relational_reasoning.data import RelationalTask
from mas_cc.games.relational_reasoning.imitation_round_feedback.controller import (
    EVIDENCE_NEUTRAL,
    EVIDENCE_STRATEGIC,
    RECOMMENDATION_PLUS_FACT,
    RelationalRoundBudgetedControl,
)

from .design import TARGET_SEMANTICS, controller_target_for


@dataclass(frozen=True, slots=True)
class EvidenceAudit:
    """One ``(task, target semantics, strategy)`` cell of the preflight audit."""

    task_id: str
    reasoning_depth: int
    target_semantics: str
    target_relation: str
    strategy: str
    fact_id: str | None
    fact_relation: str | None
    admissible: bool
    reason: str

    def to_row(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "L": self.reasoning_depth,
            "target_semantics": self.target_semantics,
            "target_relation": self.target_relation,
            "strategy": self.strategy,
            "fact_id": self.fact_id,
            "fact_relation": self.fact_relation,
            "admissible": self.admissible,
            "reason": self.reason,
        }


def _control(strategy: str, target: str) -> RelationalRoundBudgetedControl:
    """A controller configured exactly as a Stage B run would configure one."""

    return RelationalRoundBudgetedControl(
        target=target,
        message_mode=RECOMMENDATION_PLUS_FACT,
        controller_evidence_strategy=strategy,
    )


def audit_task(
    task: RelationalTask, strategies: Sequence[str] = (EVIDENCE_NEUTRAL, EVIDENCE_STRATEGIC)
) -> tuple[EvidenceAudit, ...]:
    """Resolve every ``(target semantics, strategy)`` cell for one frozen task."""

    rows: list[EvidenceAudit] = []
    for target_semantics in TARGET_SEMANTICS:
        target = controller_target_for(task, target_semantics)
        for strategy in strategies:
            try:
                fact_id = _control(strategy, target).resolve_fact_id(task, 0)
            except ValueError as exc:
                rows.append(
                    EvidenceAudit(
                        task_id=task.task_id,
                        reasoning_depth=task.reasoning_depth,
                        target_semantics=target_semantics,
                        target_relation=target,
                        strategy=strategy,
                        fact_id=None,
                        fact_relation=None,
                        admissible=False,
                        reason=str(exc),
                    )
                )
                continue
            rows.append(
                EvidenceAudit(
                    task_id=task.task_id,
                    reasoning_depth=task.reasoning_depth,
                    target_semantics=target_semantics,
                    target_relation=target,
                    strategy=strategy,
                    fact_id=fact_id,
                    fact_relation=task.fact(fact_id).relation,
                    admissible=True,
                    reason="resolved by the production selector",
                )
            )
    return tuple(rows)


def audit_tasks(
    tasks: Sequence[RelationalTask],
    strategies: Sequence[str] = (EVIDENCE_NEUTRAL, EVIDENCE_STRATEGIC),
) -> tuple[EvidenceAudit, ...]:
    return tuple(row for task in tasks for row in audit_task(task, strategies))


def admissibility_index(
    audits: Sequence[EvidenceAudit],
) -> dict[tuple[int, str, str, str], EvidenceAudit]:
    """``(L, task_id, target_semantics, strategy) -> audit`` for the grid builder.

    The reasoning depth is part of the key because the two fixture directories
    number their tasks independently: ``task_0001`` exists at ``L=1`` and again,
    as a completely different world, at ``L=2``.
    """

    return {
        (row.reasoning_depth, row.task_id, row.target_semantics, row.strategy): row
        for row in audits
    }


def inadmissible(audits: Sequence[EvidenceAudit]) -> tuple[EvidenceAudit, ...]:
    return tuple(row for row in audits if not row.admissible)


def neutral_is_target_independent(audits: Sequence[EvidenceAudit]) -> bool:
    """Preflight check: ``neutral`` must resolve identically for both targets."""

    by_task: dict[tuple[int, str], set[str | None]] = {}
    for row in audits:
        if row.strategy != EVIDENCE_NEUTRAL:
            continue
        by_task.setdefault((row.reasoning_depth, row.task_id), set()).add(row.fact_id)
    return all(len(values) == 1 for values in by_task.values())


def strategic_is_target_aware(audits: Sequence[EvidenceAudit]) -> bool:
    """Preflight check: ``strategic`` must depend on the target for some task.

    Not *every* task - a task whose facts all point one way can legitimately
    give the same citation for both targets.  What would be wrong is a strategic
    selector that never varies with the target at all, which would silently make
    it a second neutral arm.
    """

    by_task: dict[tuple[int, str], set[str | None]] = {}
    for row in audits:
        if row.strategy != EVIDENCE_STRATEGIC or not row.admissible:
            continue
        by_task.setdefault((row.reasoning_depth, row.task_id), set()).add(row.fact_id)
    return any(len(values) > 1 for values in by_task.values())


def evidence_facts_are_real(
    audits: Sequence[EvidenceAudit], tasks: Mapping[tuple[int, str], RelationalTask]
) -> bool:
    """Preflight check: every cited id is a fact of that task, verbatim."""

    for row in audits:
        if row.fact_id is None:
            continue
        task = tasks.get((row.reasoning_depth, row.task_id))
        if task is None or row.fact_id not in task.facts:
            return False
        if task.fact(row.fact_id).relation != row.fact_relation:
            return False
    return True


__all__ = [
    "EVIDENCE_NEUTRAL",
    "EVIDENCE_STRATEGIC",
    "EvidenceAudit",
    "admissibility_index",
    "audit_task",
    "audit_tasks",
    "evidence_facts_are_real",
    "inadmissible",
    "neutral_is_target_independent",
    "strategic_is_target_aware",
]
