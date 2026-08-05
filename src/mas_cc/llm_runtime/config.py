"""Immutable, connection-free configuration models for providers and prompts.

The models intentionally use only the standard library.  They describe what a
provider or prompt needs without constructing a client, reading credentials,
or opening any external service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True, slots=True)
class LLMProviderConfig:
    """Connection-free settings for one LLM provider adapter."""

    type: str
    model: str
    schema_version: int = 1
    credentials_env: str | None = None
    base_url_env: str | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 2
    request_concurrency: int = 1
    temperature: float = 0.0
    max_output_tokens: int = 256
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", _freeze(self.options))

    @property
    def provider_type(self) -> str:
        return self.type

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "type": self.type,
            "model": self.model,
            "credentials_env": self.credentials_env,
            "base_url_env": self.base_url_env,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "request_concurrency": self.request_concurrency,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "options": _thaw(self.options),
        }


# A short alias is convenient in user code while the longer name makes the
# architecture boundary explicit.
ProviderConfig = LLMProviderConfig


@dataclass(frozen=True, slots=True)
class PromptConfig:
    """A Version 2 family selection plus permitted presentation policy."""

    prompt_family: str
    prompt_version: int
    blocks: tuple[str, ...] = ()
    response_contract: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 2
    message_mode: str | None = None
    block_separator: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version not in {1, 2}
        ):
            raise ValueError("PromptConfig.schema_version must be 1 or 2")
        if not isinstance(self.prompt_family, str) or not self.prompt_family.strip():
            raise ValueError("PromptConfig.prompt_family must be non-empty")
        if (
            isinstance(self.prompt_version, bool)
            or not isinstance(self.prompt_version, int)
            or self.prompt_version < 1
        ):
            raise ValueError("PromptConfig.prompt_version must be positive")
        if isinstance(self.blocks, (str, bytes)) or any(
            not isinstance(name, str) or not name for name in self.blocks
        ):
            raise ValueError("PromptConfig.blocks must contain non-empty strings")
        blocks = tuple(self.blocks)
        if self.schema_version == 2 and blocks:
            raise ValueError(
                "PromptConfig.blocks is forbidden in schema version 2; "
                "the registered FullPrompt owns authoritative order"
            )
        if not isinstance(self.response_contract, Mapping):
            raise TypeError("PromptConfig.response_contract must be a mapping")
        if self.message_mode not in {None, "per_block", "merge_consecutive_roles"}:
            raise ValueError("PromptConfig.message_mode is invalid")
        if self.block_separator is not None and not isinstance(self.block_separator, str):
            raise TypeError("PromptConfig.block_separator must be a string or None")
        if not isinstance(self.options, Mapping):
            raise TypeError("PromptConfig.options must be a mapping")
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "response_contract", _freeze(self.response_contract))
        object.__setattr__(self, "options", _freeze(self.options))

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "prompt_family": self.prompt_family,
            "prompt_version": self.prompt_version,
            "response_contract": _thaw(self.response_contract),
            "message_mode": self.message_mode,
            "block_separator": self.block_separator,
            "options": _thaw(self.options),
        }
        if self.schema_version == 1:
            result["blocks"] = list(self.blocks)
        return result

    @property
    def is_legacy(self) -> bool:
        return self.schema_version == 1

    def migration_diagnostics(self) -> tuple[str, ...]:
        if not self.is_legacy:
            return ()
        return (
            "prompt.schema_version 1 is legacy; migrate to 2",
            "remove prompt.blocks because the registered FullPrompt owns authoritative order",
            "move prompt.options.message_mode/block_separator to prompt top-level fields",
        )
