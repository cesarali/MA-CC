"""Immutable, provider-independent configuration models.

The models intentionally use only the standard library.  They describe what a
run needs without constructing providers, reading credentials, or opening any
external service.

``LLMProviderConfig`` and ``PromptConfig`` live in
:mod:`mas_cc.llm_runtime.config` as part of the portable ``llm_runtime``
component and are re-exported here for compatibility, since ``RunConfig``
composes them alongside the repository-wide sections below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from mas_cc.llm_runtime.config import LLMProviderConfig, PromptConfig, ProviderConfig
from mas_cc.metrics.interactions import PARTIAL_BIN_POLICIES, BinnedTrajectoryPolicy

__all_reexported__ = ("LLMProviderConfig", "PromptConfig", "ProviderConfig")


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
class ControlConfig:
    """Selection of a provider-independent intervention/control mechanism.

    Mirrors ``GameConfig``'s ``type``/``options`` idiom: the shared schema
    only names a mechanism and carries a free-form options mapping, so a
    concrete ``Control`` implementation owns validating its own options
    rather than every mechanism's fields living in this shared model.
    """

    schema_version: int = 1
    mechanism: str = "none"
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", _freeze(self.options))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mechanism": self.mechanism,
            "options": _thaw(self.options),
        }


@dataclass(frozen=True, slots=True)
class MetricsConfig:
    """Whether to compute the game's declared metrics, and what may reach Comet.

    Metric instances themselves live in code (``games/<game>/metrics.py``);
    this section only controls whether they run and which metric names are
    allowed off the machine, mirroring LoggingConfig's Comet privacy stance.

    ``available`` is the per-metric way to say this: ``{metric_name: {comet:
    true}}`` puts that metric's Comet routing right next to its name, rather
    than in a separately-maintained flat list you have to cross-reference by
    hand. ``comet_export`` (a flat list of names) still works too, for
    backward compatibility — ``comet_export_names()`` returns the union of
    both.

    ``bin_size_interactions`` / ``partial_final_bin`` / ``exclude_committed_outputs``
    configure the binned trajectory metrics (success rate and production
    probability). They are explicit configuration rather than constants buried
    in code because changing any of them changes what a reported number means,
    so a run's own config has to record which policy produced it.
    """

    schema_version: int = 1
    enabled: bool = True
    comet_export: tuple[str, ...] = ()
    available: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    bin_size_interactions: int | None = None
    partial_final_bin: str = "drop"
    exclude_committed_outputs: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "comet_export", tuple(self.comet_export))
        object.__setattr__(self, "available", _freeze(self.available))
        if self.bin_size_interactions is not None and self.bin_size_interactions < 1:
            raise ValueError("metrics.bin_size_interactions must be a positive integer or null")
        if self.partial_final_bin not in PARTIAL_BIN_POLICIES:
            raise ValueError(
                f"metrics.partial_final_bin must be one of {PARTIAL_BIN_POLICIES}, "
                f"got {self.partial_final_bin!r}"
            )

    def resolved_bin_size(self, population_size: int) -> int:
        """Interactions per trajectory bin, defaulting to one population round.

        The Ashery convention is that a population round *is* N pair
        interactions for population size N, so leaving this unset tracks the
        population instead of silently pinning a bin size that stops matching
        when the population changes.
        """

        return self.bin_size_interactions or population_size

    def binning_policy(self, population_size: int) -> "BinnedTrajectoryPolicy | None":
        """The binned-trajectory policy this config asks for, or None when metrics are off."""

        if not self.enabled:
            return None
        return BinnedTrajectoryPolicy(
            bin_size=self.resolved_bin_size(population_size),
            partial_final_bin=self.partial_final_bin,
            exclude_committed_outputs=self.exclude_committed_outputs,
        )

    def comet_export_names(self) -> tuple[str, ...]:
        """Every metric name allowed to reach Comet, from either spelling."""

        per_metric = {
            name
            for name, settings in self.available.items()
            if isinstance(settings, Mapping) and settings.get("comet") is True
        }
        return tuple(sorted({*self.comet_export, *per_metric}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "enabled": self.enabled,
            "comet_export": list(self.comet_export),
            "available": _thaw(self.available),
            "bin_size_interactions": self.bin_size_interactions,
            "partial_final_bin": self.partial_final_bin,
            "exclude_committed_outputs": self.exclude_committed_outputs,
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
    control: ControlConfig = field(default_factory=ControlConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
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
            "control": self.control.to_dict(),
            "metrics": self.metrics.to_dict(),
            "experiment": self.experiment.to_dict(),
            "pricing": self.pricing.to_dict(),
            "budget": self.budget.to_dict(),
        }
