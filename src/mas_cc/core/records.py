"""Immutable timestamp record.

Message/MessageRole moved to mas_cc.llm_runtime.messages — they're LLM
chat-message vocabulary, not general framework primitives. Timestamp stays
here since it's genuinely generic (not LLM-specific) and mas_cc.llm_runtime
imports it back for Message.created_at.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


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
