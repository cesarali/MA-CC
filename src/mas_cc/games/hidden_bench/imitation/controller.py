"""Provider-free measurement and feedback control for the imitation game."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mas_cc.config import ControlConfig
from mas_cc.control import Control, InteractionControlSignal
from mas_cc.core import AgentId
from mas_cc.games.protocols import GameState
from mas_cc.llm_runtime.exceptions import ConfigurationError
from mas_cc.llm_runtime.validation import ValidationIssue

NO_OP = "NO_OP"
ADVOCATE_TARGET = "ADVOCATE_Z"
ADVOCACY_TEMPLATE_VERSION = 1


def advocacy_message(target: str) -> str:
    """Fixed intervention text: target advocacy without task evidence."""

    return (
        f"The external controller currently advocates option {target}. "
        "Reconsider your current position before committing your next vote."
    )


@dataclass(frozen=True, slots=True)
class ThresholdTargetControl(Control):
    """Hypergeometric opinion sensor followed by a deterministic threshold policy."""

    target: str | int = "correct"
    sensor_sample_size: int = 1
    threshold: float = 0.5
    policy: str = "threshold_target"

    def override(
        self, *, agent_id: AgentId, interaction_index: int, state: GameState
    ) -> str | None:
        # Feedback shapes the interaction.  It never overwrites the vote.
        return None

    def _resolved_target(self, state: GameState) -> str:
        if isinstance(self.target, int):
            task = state.data.get("task", {})
            options = task.get("possible_answers", ()) if isinstance(task, Mapping) else ()
            if self.target < 0 or self.target >= len(options):
                raise ValueError(
                    f"controller target index {self.target} is outside [0, {len(options)})"
                )
            return str(options[self.target])
        if self.target != "correct":
            return self.target
        task = state.data.get("task", {})
        if not isinstance(task, Mapping) or not task.get("correct_answer"):
            raise ValueError("target 'correct' requires state.data.task.correct_answer")
        return str(task["correct_answer"])

    def interaction_signal(
        self,
        *,
        agent_id: AgentId,
        interaction_index: int,
        state: GameState,
        rng: Any,
    ) -> InteractionControlSignal:
        if self.sensor_sample_size > len(state.agents):
            raise ValueError(
                "controller sensor_sample_size cannot exceed the population size"
            )
        target = self._resolved_target(state)
        sampled = rng.sample(list(state.agents), self.sensor_sample_size)
        opinions = [
            None
            if agent.attributes.get("committed_action") is None
            else str(agent.attributes["committed_action"])
            for agent in sampled
        ]
        counts = Counter(value for value in opinions if value is not None)
        support = counts.get(target, 0) / self.sensor_sample_size
        action = ADVOCATE_TARGET if support < self.threshold else NO_OP
        message = advocacy_message(target) if action == ADVOCATE_TARGET else None
        return InteractionControlSignal(
            action=action,
            target=target,
            message=message,
            observation={
                "sampled_agent_ids": [str(agent.agent_id) for agent in sampled],
                "sampled_opinions": opinions,
                "sampled_opinion_counts": dict(counts),
                "sample_size": self.sensor_sample_size,
            },
            metadata={
                "policy": self.policy,
                "threshold": self.threshold,
                "target_support": support,
                "message_template_version": ADVOCACY_TEMPLATE_VERSION,
            },
        )

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> "ThresholdTargetControl":
        target = options.get("target", "correct")
        sample_size = options.get("sensor_sample_size", 1)
        threshold = options.get("threshold", 0.5)
        policy = options.get("policy", "threshold_target")
        issues: list[ValidationIssue] = []
        if (
            isinstance(target, bool)
            or not isinstance(target, (str, int))
            or (isinstance(target, str) and not target.strip())
            or (isinstance(target, int) and target < 0)
        ):
            issues.append(
                ValidationIssue(
                    "control.options.target",
                    "must be 'correct', an option label, or a non-negative zero-based index",
                )
            )
        if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
            issues.append(
                ValidationIssue("control.options.sensor_sample_size", "must be a positive integer")
            )
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0 <= float(threshold) <= 1
        ):
            issues.append(ValidationIssue("control.options.threshold", "must be between 0 and 1"))
        if policy != "threshold_target":
            issues.append(
                ValidationIssue(
                    "control.options.policy", "only 'threshold_target' is implemented"
                )
            )
        if issues:
            raise ConfigurationError(issues, context="control creation")
        return cls(
            target=target,
            sensor_sample_size=int(sample_size),
            threshold=float(threshold),
            policy=str(policy),
        )


def create_threshold_target_control(config: ControlConfig) -> Control:
    return ThresholdTargetControl.from_options(config.options)


def controller_from_game_options(options: Mapping[str, Any]) -> ThresholdTargetControl | None:
    """Compatibility adapter for the plan's nested ``game.options.controller`` form."""

    raw = options.get("controller")
    if not isinstance(raw, Mapping) or not bool(raw.get("enabled", False)):
        return None
    return ThresholdTargetControl.from_options(raw)
