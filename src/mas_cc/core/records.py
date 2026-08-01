"""Immutable provider-independent message and timestamp records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .ids import MessageId


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


@dataclass(frozen=True, order=True, slots=True)
class Timestamp:
    """A timezone-aware timestamp normalized to UTC."""

    value: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.value, datetime):
            raise TypeError("Timestamp.value must be a datetime")
        if self.value.tzinfo is None or self.value.utcoffset() is None:
            raise ValueError("Timestamp.value must include a timezone")
        object.__setattr__(self, "value", self.value.astimezone(timezone.utc))

    @classmethod
    def now(cls) -> "Timestamp":
        return cls(datetime.now(timezone.utc))

    @classmethod
    def parse(cls, value: str) -> "Timestamp":
        if not isinstance(value, str):
            raise TypeError("timestamp must be a string")
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(f"invalid ISO 8601 timestamp: {value!r}") from exc
        return cls(parsed)

    def isoformat(self) -> str:
        return self.value.isoformat().replace("+00:00", "Z")

    def __str__(self) -> str:
        return self.isoformat()


@dataclass(frozen=True, slots=True)
class Message:
    """One normalized message, independent of any LLM wire format."""

    role: MessageRole
    content: str
    message_id: MessageId | None = None
    created_at: Timestamp | None = None
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
        result: dict[str, Any] = {
            "role": self.role.value,
            "content": self.content,
            "metadata": _thaw(self.metadata),
        }
        if self.message_id is not None:
            result["message_id"] = str(self.message_id)
        if self.created_at is not None:
            result["created_at"] = self.created_at.isoformat()
        return result
