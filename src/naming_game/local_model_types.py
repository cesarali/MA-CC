"""Public, dependency-free types for constrained local-model decisions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from .models import TokenUsage

DecisionOutputFormat: TypeAlias = Literal["json_reason", "choice_reason", "choice_only"]
TextDecisionOutputFormat: TypeAlias = Literal["choice_reason", "choice_only"]
ChoiceSelectionPolicy: TypeAlias = Literal["argmax", "sample"]


@dataclass(frozen=True)
class ChoiceScore:
    choice: str
    token_ids: tuple[int, ...]
    log_likelihood: float
    probability: float


@dataclass(frozen=True)
class ConstrainedLLMResponse:
    selected_choice: str
    scores: tuple[ChoiceScore, ...]
    model: str
    latency_seconds: float
    usage: TokenUsage
    temperature: float


@dataclass(frozen=True)
class ConstrainedDecisionResponse:
    selected_choice: str
    scores: tuple[ChoiceScore, ...]
    content: str
    reason: str | None
    reason_valid: bool | None
    output_format: TextDecisionOutputFormat
    model: str
    latency_seconds: float
    usage: TokenUsage
    choice_temperature: float
    selection_policy: ChoiceSelectionPolicy


@runtime_checkable
class ConstrainedLLMClient(Protocol):
    async def complete_constrained(
        self,
        messages: list[dict[str, str]],
        *,
        choices: Sequence[str],
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> ConstrainedLLMResponse: ...


@runtime_checkable
class ConstrainedDecisionClient(Protocol):
    async def complete_decision(
        self, messages: list[dict[str, str]], *, choices: Sequence[str],
        output_format: TextDecisionOutputFormat, choice_temperature: float = 1.0,
        selection_policy: ChoiceSelectionPolicy = "argmax",
        generation_temperature: float = 0.0, max_reason_tokens: int = 32,
        seed: int | None = None,
    ) -> ConstrainedDecisionResponse: ...
