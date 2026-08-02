"""Immutable, provider-independent configuration models.

The models intentionally use only the standard library.  They describe what a
run needs without constructing providers, reading credentials, or opening any
external service.
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


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Generic game selection and structural settings."""

    type: str
    population_size: int
    horizon: int
    schema_version: int = 1
    topology: str = "complete"
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", _freeze(self.options))

    @property
    def game_type(self) -> str:
        return self.type

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "type": self.type,
            "population_size": self.population_size,
            "horizon": self.horizon,
            "topology": self.topology,
            "options": _thaw(self.options),
        }


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Runtime policy that remains independent from a concrete executor."""

    schema_version: int = 1
    seed: int = 0
    repetitions: int = 1
    parallelism: int = 1
    fail_fast: bool = True
    timeout_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "repetitions": self.repetitions,
            "parallelism": self.parallelism,
            "fail_fast": self.fail_fast,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Audit and monitoring settings; credentials are never represented here."""

    schema_version: int = 1
    level: str = "INFO"
    console: bool = True
    audit: bool = True
    comet: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", _freeze(self.options))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "level": self.level,
            "console": self.console,
            "audit": self.audit,
            "comet": self.comet,
            "options": _thaw(self.options),
        }


@dataclass(frozen=True, slots=True)
class StorageConfig:
    """Artifact destinations and checkpoint policy."""

    schema_version: int = 1
    output_dir: str = "results"
    format: str = "jsonl"
    checkpoints: bool = True
    overwrite: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", _freeze(self.options))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "output_dir": self.output_dir,
            "format": self.format,
            "checkpoints": self.checkpoints,
            "overwrite": self.overwrite,
            "options": _thaw(self.options),
        }


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Offline analysis selection for a completed run."""

    schema_version: int = 1
    enabled: bool = False
    estimators: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "estimators", tuple(self.estimators))
        object.__setattr__(self, "options", _freeze(self.options))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "enabled": self.enabled,
            "estimators": list(self.estimators),
            "options": _thaw(self.options),
        }


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Human-facing experiment identity and grouping metadata."""

    schema_version: int = 1
    name: str = "unnamed-experiment"
    description: str = ""
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "metadata": _thaw(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PricingConfig:
    """Explicit pricing provenance/freshness policy; resolving it performs no I/O."""

    schema_version: int = 1
    mode: str = "offline"
    cache_path: str | None = None
    max_age_seconds: float = 86400.0
    require_fresh_at_launch: bool = True
    fallback_policy: str = "deny"
    explicit_unknown_price_override: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "cache_path": self.cache_path,
            "max_age_seconds": self.max_age_seconds,
            "require_fresh_at_launch": self.require_fresh_at_launch,
            "fallback_policy": self.fallback_policy,
            "explicit_unknown_price_override": self.explicit_unknown_price_override,
        }


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    """MAS-CC run limits, separate from any provider account budget."""

    schema_version: int = 1
    accounting_unit: str = "unknown"
    system_max_cost_per_run: float | None = None
    max_cost_per_run: float | None = None
    max_provider_requests: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    allow_unbounded_paid_requests: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "accounting_unit": self.accounting_unit,
            "system_max_cost_per_run": self.system_max_cost_per_run,
            "max_cost_per_run": self.max_cost_per_run,
            "max_provider_requests": self.max_provider_requests,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "allow_unbounded_paid_requests": self.allow_unbounded_paid_requests,
        }


@dataclass(frozen=True, slots=True)
class RunConfig:
    """A fully resolved run configuration with no component references."""

    llm_provider: LLMProviderConfig
    prompt: PromptConfig
    game: GameConfig
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    schema_version: int = 1

    @property
    def provider(self) -> LLMProviderConfig:
        """Alias matching the concise YAML section name."""

        return self.llm_provider

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, secret-free serialization structure."""

        return {
            "schema_version": self.schema_version,
            "llm_provider": self.llm_provider.to_dict(),
            "prompt": self.prompt.to_dict(),
            "game": self.game.to_dict(),
            "execution": self.execution.to_dict(),
            "logging": self.logging.to_dict(),
            "storage": self.storage.to_dict(),
            "analysis": self.analysis.to_dict(),
            "experiment": self.experiment.to_dict(),
            "pricing": self.pricing.to_dict(),
            "budget": self.budget.to_dict(),
        }
