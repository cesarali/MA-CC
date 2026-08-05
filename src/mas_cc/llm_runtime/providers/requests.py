"""Immutable, provider-independent completion requests."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from ..messages import Message


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): item for key, item in value.items()})


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_thaw(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """The single request shape accepted by every Phase 4 adapter."""

    messages: tuple[Message, ...]
    temperature: float = 0.0
    max_output_tokens: int = 256
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.messages, tuple):
            object.__setattr__(self, "messages", tuple(self.messages))
        if not self.messages or any(not isinstance(item, Message) for item in self.messages):
            raise ValueError("CompletionRequest.messages must contain at least one Message")
        if not isinstance(self.temperature, (int, float)) or not math.isfinite(
            self.temperature
        ) or self.temperature < 0:
            raise ValueError("CompletionRequest.temperature must be finite and non-negative")
        object.__setattr__(self, "temperature", float(self.temperature))
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens < 1
        ):
            raise ValueError("CompletionRequest.max_output_tokens must be positive")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("CompletionRequest.seed must be an integer or None")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("CompletionRequest.metadata must be a mapping")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def wire_messages(self) -> list[dict[str, str]]:
        """Return only provider wire fields; prompt metadata never leaks."""

        return [
            {"role": message.role.value, "content": message.content}
            for message in self.messages
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [message.to_dict() for message in self.messages],
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "seed": self.seed,
            "metadata": _thaw(self.metadata),
        }
