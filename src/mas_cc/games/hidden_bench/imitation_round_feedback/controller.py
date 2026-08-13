"""One-sense/one-action budgeted feedback controller."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from mas_cc.config import ControlConfig
from mas_cc.control import Control, RoundControlSignal
from mas_cc.games.protocols import GameState
from mas_cc.llm_runtime.validation import ValidationIssue

from ..imitation.controller import SoftTargetControl, ThresholdTargetControl


@dataclass(frozen=True, slots=True)
class RoundSoftTargetBudgetedControl(SoftTargetControl):
    """Soft target policy sampled once, with an exact per-round slot budget."""

    intervention_budget: int = 0
    policy: ClassVar[str] = "soft_target"
    default_template_version: ClassVar[int] = 3

    def interaction_signal(self, **_: Any) -> None:
        """This controller deliberately has no event-clock behavior."""

        return None

    def round_signal(
        self,
        *,
        round_index: int,
        state: GameState,
        rng: Any,
    ) -> RoundControlSignal:
        # Call the event controller's implementation as a pure sensor/policy
        # helper.  The round runtime invokes this method once and owns all
        # schedule/actuation semantics after it returns.
        signal = ThresholdTargetControl.interaction_signal(
            self,
            agent_id=state.agents[0].agent_id,
            interaction_index=round_index + 1,
            state=state,
            rng=rng,
        )
        return RoundControlSignal(
            action=signal.action,
            target=signal.target,
            message=signal.message,
            observation=signal.observation,
            metadata={
                **dict(signal.metadata),
                "intervention_budget": self.intervention_budget,
                "round_index": round_index,
            },
        )

    @classmethod
    def _extra_from_options(
        cls, options: Mapping[str, Any], issues: list[ValidationIssue]
    ) -> dict[str, Any]:
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
        if options.get("template_version", cls.default_template_version) != 3:
            issues.append(
                ValidationIssue(
                    "control.options.template_version",
                    "round feedback requires fixed peer-style template version 3",
                )
            )
        return values


def create_round_soft_target_budgeted_control(config: ControlConfig) -> Control:
    return RoundSoftTargetBudgetedControl.from_options(config.options)


__all__ = [
    "RoundSoftTargetBudgetedControl",
    "create_round_soft_target_budgeted_control",
]
