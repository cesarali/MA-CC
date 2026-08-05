"""Chat-message vocabulary: the roles and message shape LLM providers speak.

Lives at the top of llm_runtime (not inside providers/ or prompts/) because
both subpackages use it and must remain independent siblings — neither
imports the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_thaw(item) for item in value)
    return value


class MessageRole(str, Enum):
    """Roles accepted by a provider-independent conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class Message:
    """One normalized message, independent of any LLM wire format."""

    role: MessageRole
    content: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.role, str):
            try:
                object.__setattr__(self, "role", MessageRole(self.role))
            except ValueError as exc:
                raise ValueError(f"Message.role is invalid: {self.role!r}") from exc
        if not isinstance(self.role, MessageRole):
            raise TypeError("Message.role must be a MessageRole")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("Message.content must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("Message.metadata must be a mapping")
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "metadata": _thaw(self.metadata),
        }
