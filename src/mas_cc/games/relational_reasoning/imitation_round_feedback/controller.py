"""Round-level budgeted controller for the relational reasoning game.

The **sensing and policy machinery is reused unchanged** from the HiddenBench
round-feedback controller: one decision per population round, a hypergeometric
vote sensor of size ``q_c``, a soft (logistic) policy over ``{NO_OP,
ADVOCATE_Z}``, and an exact budget ``b`` of randomly placed controlled
positions.  The controller senses **votes only** and never receives any agent's
knowledge state.

What this subclass adds is *what the controller may say*.  Because a relational
task's symbolic content is exact, the linguistic content of an intervention can
be manipulated as a controlled variable:

``message_mode: recommendation_only``
    The controller occupies its social slot with a recommendation and no
    evidence.  No fact enters anybody's ``K_i`` from the controller, ever.

``message_mode: recommendation_plus_fact``
    The same recommendation, plus exactly one fact of the frozen task, chosen
    **before the episode runs** and fixed for its whole duration.

Fact selection is deterministic and never delegated to a model:
``controller_fact_id`` names one explicitly, or ``controller_fact_selector:
supporting`` resolves to the task's first supporting fact in task order.  One
fixed fact per episode is the default on purpose - varying the citation per slot
would add an uncontrolled stochastic channel to an experiment whose whole point
is measuring what one message does.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from mas_cc.config import ControlConfig
from mas_cc.control import Control
from mas_cc.llm_runtime.validation import ValidationIssue

from ...hidden_bench.imitation.controller import ADVOCATE_TARGET, NO_OP, SoftTargetControl
from ...hidden_bench.imitation_round_feedback.controller import (
    EVIDENCE_NONE,
    RoundSoftTargetBudgetedControl,
)
from ..data import RelationalTask

RECOMMENDATION_ONLY = "recommendation_only"
RECOMMENDATION_PLUS_FACT = "recommendation_plus_fact"
MESSAGE_MODES = (RECOMMENDATION_ONLY, RECOMMENDATION_PLUS_FACT)

SELECTOR_SUPPORTING = "supporting"
FACT_SELECTORS = (SELECTOR_SUPPORTING,)

SCHEDULE_SOFT = "soft"
SCHEDULE_ALWAYS = "always"
SCHEDULE_NEVER = "never"
ADVOCACY_SCHEDULES = (SCHEDULE_SOFT, SCHEDULE_ALWAYS, SCHEDULE_NEVER)
"""``control.options.advocacy_schedule``.

``soft`` is the default and the closed loop: the sensed target share feeds the
logistic policy, so whether the controller acts depends on the population.
``always`` is open loop - it advocates every round regardless of what it sensed.
``never`` is the corresponding open-loop no-op schedule.

Open loop is what a *controllability* study wants. Under the soft policy the
actuation a population receives is a function of that population's own state,
so a cell where the population converged early is a cell where the controller
mostly stopped, and "did control move it" and "did it need moving" are
confounded. ``always`` fixes the intervention and lets the population vary.

The sensor still runs and is still logged under ``always``; only the decision
ignores it, which keeps the sensing columns available for comparison."""


@dataclass(frozen=True, slots=True)
class RelationalRoundBudgetedControl(RoundSoftTargetBudgetedControl):
    """Soft target policy, exact per-round slot budget, explicit evidence choice."""

    message_mode: str = RECOMMENDATION_ONLY
    controller_fact_id: str | None = None
    controller_fact_selector: str | None = None
    advocacy_schedule: str = SCHEDULE_SOFT

    policy: ClassVar[str] = "soft_target"
    default_template_version: ClassVar[int] = 3

    def select_action(self, sampled_target_share: float, rng: Any) -> tuple[str, float]:
        """The policy step, with an open-loop escape hatch.

        Under ``always`` the sensed share is deliberately not consulted and the
        seeded stream is deliberately not drawn from, so the schedule is exactly
        reproducible and independent of the population trajectory.
        """

        if self.advocacy_schedule == SCHEDULE_ALWAYS:
            return ADVOCATE_TARGET, 1.0
        if self.advocacy_schedule == SCHEDULE_NEVER:
            return NO_OP, 0.0
        # Explicit rather than `super()`: `@dataclass(slots=True)` rebuilds the
        # class, which leaves the zero-argument form's `__class__` cell pointing
        # at the discarded original.  The parent does the same for the same
        # reason.
        return SoftTargetControl.select_action(self, sampled_target_share, rng)

    @property
    def transmits_fact(self) -> bool:
        return self.message_mode == RECOMMENDATION_PLUS_FACT

    def resolve_fact_id(self, task: RelationalTask) -> str | None:
        """The one fact this controller cites for the whole episode, or ``None``.

        Resolution is deterministic and validated against the frozen task, so a
        configuration that names a fact the task does not contain fails at
        launch rather than quietly advocating without evidence.
        """

        if not self.transmits_fact:
            return None
        if self.controller_fact_id is not None:
            if self.controller_fact_id not in task.facts:
                raise ValueError(
                    f"control.options.controller_fact_id {self.controller_fact_id!r} is "
                    f"not a fact of task {task.task_id!r}"
                )
            return self.controller_fact_id
        if self.controller_fact_selector == SELECTOR_SUPPORTING:
            if not task.supporting_fact_ids:
                raise ValueError(
                    f"task {task.task_id!r} has no supporting facts for "
                    "controller_fact_selector 'supporting'"
                )
            return task.supporting_fact_ids[0]
        raise ValueError(
            "message_mode 'recommendation_plus_fact' requires either "
            "control.options.controller_fact_id or control.options.controller_fact_selector"
        )

    @classmethod
    def _extra_from_options(
        cls, options: Mapping[str, Any], issues: list[ValidationIssue]
    ) -> dict[str, Any]:
        # `beta` from the soft policy, deliberately *not* the HiddenBench
        # round controller's `_extra_from_options`: that one validates
        # `evidence_mode` and pins `template_version`, neither of which means
        # anything on a relational task.
        values = SoftTargetControl._extra_from_options(options, issues)

        budget = options.get("intervention_budget", 0)
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            issues.append(
                ValidationIssue(
                    "control.options.intervention_budget",
                    "must be a non-negative integer",
                )
            )
        else:
            values["intervention_budget"] = budget

        if "evidence_mode" in options:
            issues.append(
                ValidationIssue(
                    "control.options.evidence_mode",
                    "is a HiddenBench option; this controller uses message_mode "
                    f"(one of {list(MESSAGE_MODES)})",
                )
            )
        values["evidence_mode"] = EVIDENCE_NONE

        mode = options.get("message_mode", RECOMMENDATION_ONLY)
        if mode not in MESSAGE_MODES:
            issues.append(
                ValidationIssue(
                    "control.options.message_mode", f"must be one of {list(MESSAGE_MODES)}"
                )
            )
            mode = RECOMMENDATION_ONLY
        values["message_mode"] = str(mode)

        fact_id = options.get("controller_fact_id")
        if fact_id is not None and (not isinstance(fact_id, str) or not fact_id.strip()):
            issues.append(
                ValidationIssue(
                    "control.options.controller_fact_id",
                    "must be a fact identifier such as f1",
                )
            )
            fact_id = None
        values["controller_fact_id"] = None if fact_id is None else str(fact_id).strip()

        schedule = options.get("advocacy_schedule", SCHEDULE_SOFT)
        if schedule not in ADVOCACY_SCHEDULES:
            issues.append(
                ValidationIssue(
                    "control.options.advocacy_schedule",
                    f"must be one of {list(ADVOCACY_SCHEDULES)}",
                )
            )
            schedule = SCHEDULE_SOFT
        values["advocacy_schedule"] = str(schedule)

        selector = options.get("controller_fact_selector")
        if selector is not None and selector not in FACT_SELECTORS:
            issues.append(
                ValidationIssue(
                    "control.options.controller_fact_selector",
                    f"must be one of {list(FACT_SELECTORS)}",
                )
            )
            selector = None
        values["controller_fact_selector"] = None if selector is None else str(selector)

        if values["controller_fact_id"] and values["controller_fact_selector"]:
            issues.append(
                ValidationIssue(
                    "control.options.controller_fact_id",
                    "cannot be combined with controller_fact_selector; the citation "
                    "must have exactly one deterministic source",
                )
            )
        if mode == RECOMMENDATION_PLUS_FACT and not (
            values["controller_fact_id"] or values["controller_fact_selector"]
        ):
            issues.append(
                ValidationIssue(
                    "control.options.message_mode",
                    "'recommendation_plus_fact' requires controller_fact_id or "
                    "controller_fact_selector",
                )
            )
        if mode == RECOMMENDATION_ONLY and (
            values["controller_fact_id"] or values["controller_fact_selector"]
        ):
            issues.append(
                ValidationIssue(
                    "control.options.message_mode",
                    "'recommendation_only' transmits no fact; remove "
                    "controller_fact_id/controller_fact_selector or switch mode",
                )
            )
        return values


def create_relational_round_budgeted_control(config: ControlConfig) -> Control:
    return RelationalRoundBudgetedControl.from_options(config.options)


__all__ = [
    "ADVOCACY_SCHEDULES",
    "FACT_SELECTORS",
    "MESSAGE_MODES",
    "RECOMMENDATION_ONLY",
    "RECOMMENDATION_PLUS_FACT",
    "SCHEDULE_ALWAYS",
    "SCHEDULE_NEVER",
    "SCHEDULE_SOFT",
    "SELECTOR_SUPPORTING",
    "RelationalRoundBudgetedControl",
    "create_relational_round_budgeted_control",
]
