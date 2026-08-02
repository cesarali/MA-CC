"""Normalized provider usage and completion records."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


_SENSITIVE_KEYS = frozenset(
    {"authorization", "api_key", "apikey", "access_token", "token", "headers", "cookie"}
)


def redact_raw_response(value: Any) -> Any:
    """Recursively remove common credential-bearing keys from an artifact."""

    if isinstance(value, Mapping):
        return {
            str(key): "<redacted>"
            if str(key).lower() in _SENSITIVE_KEYS
            else redact_raw_response(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [redact_raw_response(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"ProviderUsage.{name} must be a non-negative integer or None")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ProviderUsage":
        if not isinstance(value, Mapping):
            return cls()
        details = value.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, Mapping) else None
        return cls(
            input_tokens=_optional_int(value.get("prompt_tokens", value.get("input_tokens"))),
            output_tokens=_optional_int(
                value.get("completion_tokens", value.get("output_tokens"))
            ),
            total_tokens=_optional_int(value.get("total_tokens")),
            cached_input_tokens=_optional_int(cached),
        )

    def to_dict(self) -> dict[str, int | None]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
        }


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


@dataclass(frozen=True, slots=True)
class CompletionResponse:
    content: str
    provider: str
    model: str
    usage: ProviderUsage = field(default_factory=ProviderUsage)
    finish_reason: str | None = None
    request_id: str | None = None
    latency_seconds: float = 0.0
    retries: int = 0
    status_code: int | None = None
    load_seconds: float | None = None
    inference_seconds: float | None = None
    raw_response: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("CompletionResponse.content must be a string")
        if not self.provider or not self.model:
            raise ValueError("CompletionResponse provider and model must be non-empty")
        if self.latency_seconds < 0 or self.retries < 0:
            raise ValueError("CompletionResponse timing and retries cannot be negative")
        safe = redact_raw_response(self.raw_response)
        object.__setattr__(self, "raw_response", MappingProxyType(safe))

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "provider": self.provider,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "finish_reason": self.finish_reason,
            "request_id": self.request_id,
            "latency_seconds": self.latency_seconds,
            "retries": self.retries,
            "status_code": self.status_code,
            "load_seconds": self.load_seconds,
            "inference_seconds": self.inference_seconds,
        }

    def redacted_raw_response(self) -> dict[str, Any]:
        return dict(redact_raw_response(self.raw_response))
