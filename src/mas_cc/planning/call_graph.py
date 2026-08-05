"""Provider-independent logical-call specifications.

Games describe demand here without importing prices or provider adapters.  The
preflight layer can then combine the same plan with any provider quote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
import math
from typing import Any, Literal, Mapping

from mas_cc.llm_runtime.prompts import CompilablePrompt


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
class PromptScenario:
    """A bounded prompt supplied by a game to compilation and planning."""

    name: str
    bound_prompt: CompilablePrompt
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("prompt scenario name must be non-empty")
        if not isinstance(self.bound_prompt, CompilablePrompt):
            raise TypeError("PromptScenario.bound_prompt must satisfy CompilablePrompt")
        object.__setattr__(self, "assumptions", tuple(self.assumptions))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prompt_family": self.bound_prompt.family,
            "prompt_version": self.bound_prompt.version,
            "definition_hash": self.bound_prompt.compile().definition_hash,
            "instance_hash": self.bound_prompt.compile().instance_hash,
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
    expected_attempts_per_request: float = 1.0
    concurrency_within_stage: int = 1
    state_barrier_after_stage: bool = True
    lower_prompt: PromptScenario | None = None
    representative_prompt: PromptScenario | None = None
    maximum_prompt: PromptScenario | None = None
    prompt_scenarios: tuple[PromptScenario, ...] = ()
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
        if (
            not isinstance(self.expected_attempts_per_request, (int, float))
            or isinstance(self.expected_attempts_per_request, bool)
            or not math.isfinite(self.expected_attempts_per_request)
            or not 1 <= self.expected_attempts_per_request <= 1 + self.retry_bound
        ):
            raise ValueError(
                "expected_attempts_per_request must be between 1 and 1 + retry_bound"
            )
        if self.concurrency_within_stage < 1:
            raise ValueError("concurrency_within_stage must be positive")
        scenarios = tuple(self.prompt_scenarios)
        if len({scenario.name for scenario in scenarios}) != len(scenarios):
            raise ValueError("prompt scenario names must be unique within a stage")
        object.__setattr__(self, "expected_attempts_per_request", float(self.expected_attempts_per_request))
        object.__setattr__(self, "prompt_scenarios", scenarios)
        object.__setattr__(self, "assumptions", tuple(self.assumptions))

    def requests(
        self,
        interactions: int,
        *,
        scenario: Literal["lower", "expected", "maximum"] = "lower",
        include_retries: bool = False,
    ) -> int:
        if include_retries:
            scenario = "maximum"
        multiplier = {
            "lower": 1.0,
            "expected": self.expected_attempts_per_request,
            "maximum": float(1 + self.retry_bound),
        }[scenario]
        return math.ceil(interactions * self.requests_per_interaction * multiplier)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requests_per_interaction": self.requests_per_interaction,
            "forced_decisions_per_interaction": self.forced_decisions_per_interaction,
            "provider_free_decisions_per_interaction": self.provider_free_decisions_per_interaction,
            "retry_bound": self.retry_bound,
            "expected_attempts_per_request": self.expected_attempts_per_request,
            "concurrency_within_stage": self.concurrency_within_stage,
            "state_barrier_after_stage": self.state_barrier_after_stage,
            "lower_prompt": (
                None if self.lower_prompt is None else self.lower_prompt.to_dict()
            ),
            "representative_prompt": (
                None if self.representative_prompt is None else self.representative_prompt.to_dict()
            ),
            "maximum_prompt": (
                None if self.maximum_prompt is None else self.maximum_prompt.to_dict()
            ),
            "prompt_scenarios": [
                scenario.to_dict() for scenario in self.prompt_scenarios
            ],
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
            lower=sum(
                stage.requests(self.interactions.lower, scenario="lower")
                for stage in self.decision_stages
            ),
            expected=sum(
                stage.requests(self.interactions.expected, scenario="expected")
                for stage in self.decision_stages
            ),
            maximum=sum(
                stage.requests(self.interactions.maximum, scenario="maximum")
                for stage in self.decision_stages
            ),
        )

    @property
    def logical_decisions(self) -> ProviderRequestCount:
        def total(interactions: int) -> int:
            return sum(
                interactions * stage.requests_per_interaction
                for stage in self.decision_stages
            )

        return ProviderRequestCount(
            total(self.interactions.lower),
            total(self.interactions.expected),
            total(self.interactions.maximum),
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
            "logical_decisions": self.logical_decisions.to_dict(),
            "provider_requests": self.provider_requests.to_dict(),
            "stopping_condition_assumptions": list(self.stopping_condition_assumptions),
            "metadata": dict(self.metadata),
        }
