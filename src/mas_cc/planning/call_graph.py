"""Provider-independent logical-call specifications.

Games describe demand here without importing prices or provider adapters.  The
preflight layer can then combine the same plan with any provider quote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

from mas_cc.prompts import PromptContext


@dataclass(frozen=True, slots=True)
class LogicalCallSpec:
    logical_calls: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.logical_calls, bool)
            or not isinstance(self.logical_calls, int)
            or self.logical_calls < 1
        ):
            raise ValueError("logical_calls must be a positive integer")


@dataclass(frozen=True, slots=True)
class InteractionCount:
    """Lower, expected, and maximum interaction counts for one game run."""

    lower: int
    expected: int
    maximum: int
    fixed: int | None = None

    def __post_init__(self) -> None:
        values = (self.lower, self.expected, self.maximum)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("interaction counts must be integers")
        if self.lower < 0 or not self.lower <= self.expected <= self.maximum:
            raise ValueError("interaction counts must satisfy 0 <= lower <= expected <= maximum")
        if self.fixed is not None and (
            isinstance(self.fixed, bool) or not isinstance(self.fixed, int)
        ):
            raise TypeError("fixed interaction count must be an integer or None")
        if self.fixed is not None and self.fixed != self.lower:
            raise ValueError("fixed interactions must equal lower, expected, and maximum")
        if self.fixed is not None and not (
            self.fixed == self.expected == self.maximum
        ):
            raise ValueError("fixed interactions must equal lower, expected, and maximum")

    def to_dict(self) -> dict[str, int | None]:
        return {
            "fixed": self.fixed,
            "lower": self.lower,
            "expected": self.expected,
            "maximum": self.maximum,
        }


@dataclass(frozen=True, slots=True)
class PromptContextScenario:
    """A bounded prompt context supplied by a game to the prompt layer."""

    name: str
    context: PromptContext
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("prompt context scenario name must be non-empty")
        object.__setattr__(self, "assumptions", tuple(self.assumptions))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "context": self.context.to_dict(),
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True, slots=True)
class DecisionStagePlan:
    """Logical demand for one decision stage in every interaction."""

    name: str
    requests_per_interaction: int
    forced_decisions_per_interaction: int = 0
    provider_free_decisions_per_interaction: int = 0
    retry_bound: int = 0
    representative_prompt: PromptContextScenario | None = None
    maximum_prompt: PromptContextScenario | None = None
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("decision stage name must be non-empty")
        for name in (
            "requests_per_interaction",
            "forced_decisions_per_interaction",
            "provider_free_decisions_per_interaction",
            "retry_bound",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "assumptions", tuple(self.assumptions))

    def requests(self, interactions: int, *, include_retries: bool = False) -> int:
        multiplier = 1 + self.retry_bound if include_retries else 1
        return interactions * self.requests_per_interaction * multiplier

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requests_per_interaction": self.requests_per_interaction,
            "forced_decisions_per_interaction": self.forced_decisions_per_interaction,
            "provider_free_decisions_per_interaction": self.provider_free_decisions_per_interaction,
            "retry_bound": self.retry_bound,
            "representative_prompt": (
                None if self.representative_prompt is None else self.representative_prompt.to_dict()
            ),
            "maximum_prompt": (
                None if self.maximum_prompt is None else self.maximum_prompt.to_dict()
            ),
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True, slots=True)
class ProviderRequestCount:
    lower: int
    expected: int
    maximum: int

    def to_dict(self) -> dict[str, int]:
        return {"lower": self.lower, "expected": self.expected, "maximum": self.maximum}


@dataclass(frozen=True, slots=True)
class GameCallPlan:
    """Provider- and price-neutral demand emitted by a game implementation."""

    game_type: str
    game_version: int
    interactions: InteractionCount
    decision_stages: tuple[DecisionStagePlan, ...]
    stopping_condition_assumptions: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.game_type.strip():
            raise ValueError("game_type must be non-empty")
        if self.game_version < 1:
            raise ValueError("game_version must be positive")
        stages = tuple(self.decision_stages)
        if not stages:
            raise ValueError("a game call plan must contain at least one decision stage")
        if len({stage.name for stage in stages}) != len(stages):
            raise ValueError("decision stage names must be unique")
        object.__setattr__(self, "decision_stages", stages)
        object.__setattr__(
            self, "stopping_condition_assumptions", tuple(self.stopping_condition_assumptions)
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def provider_requests(self) -> ProviderRequestCount:
        return ProviderRequestCount(
            lower=sum(stage.requests(self.interactions.lower) for stage in self.decision_stages),
            expected=sum(stage.requests(self.interactions.expected) for stage in self.decision_stages),
            maximum=sum(
                stage.requests(self.interactions.maximum, include_retries=True)
                for stage in self.decision_stages
            ),
        )

    def logical_calls(
        self, scenario: Literal["lower", "expected", "maximum"] = "expected"
    ) -> LogicalCallSpec:
        calls = getattr(self.provider_requests, scenario)
        if calls < 1:
            raise ValueError(f"{scenario} game demand contains no provider requests")
        return LogicalCallSpec(calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "game_type": self.game_type,
            "game_version": self.game_version,
            "interactions": self.interactions.to_dict(),
            "decision_stages": [stage.to_dict() for stage in self.decision_stages],
            "provider_requests": self.provider_requests.to_dict(),
            "stopping_condition_assumptions": list(self.stopping_condition_assumptions),
            "metadata": dict(self.metadata),
        }
