"""Public, dependency-free types for constrained local-model decisions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .models import TokenUsage


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
