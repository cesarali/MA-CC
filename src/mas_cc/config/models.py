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

from mas_cc.llm_runtime.config import (  # noqa: F401 - public compatibility re-export
    LLMProviderConfig,
    PromptConfig,
    ProviderConfig,
)
from mas_cc.metrics.interactions import PARTIAL_BIN_POLICIES, BinnedTrajectoryPolicy

__all_reexported__ = ("LLMProviderConfig", "PromptConfig", "ProviderConfig")

ARTIFACT_PROFILES = ("full", "results_only")
CHECKPOINT_MODES = ("off", "episode")
PROMPT_EXAMPLE_SCOPES = ("episode", "cell")


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
    artifact_profile: str = "full"
    checkpoint_mode: str = "episode"
    overwrite: bool = False
    wipe_and_recompute: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.artifact_profile not in ARTIFACT_PROFILES:
            raise ValueError(
                f"artifact_profile must be one of {', '.join(ARTIFACT_PROFILES)}"
            )
        if self.checkpoint_mode not in CHECKPOINT_MODES:
            raise ValueError(
                f"checkpoint_mode must be one of {', '.join(CHECKPOINT_MODES)}"
            )
        object.__setattr__(self, "options", _freeze(self.options))

    @property
    def checkpoints(self) -> bool:
        """Compatibility view for callers not yet migrated to the named mode."""

        return self.checkpoint_mode == "episode"

    @property
    def retention_policy(self) -> "RetentionPolicy":
        return RetentionPolicy.for_profile(self.artifact_profile)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "output_dir": self.output_dir,
            "format": self.format,
            "artifact_profile": self.artifact_profile,
            "checkpoint_mode": self.checkpoint_mode,
            "overwrite": self.overwrite,
            "wipe_and_recompute": self.wipe_and_recompute,
            "options": _thaw(self.options),
        }


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """Resolved local-retention behavior used by artifact writers.

    Keeping this policy typed prevents individual writers from growing their
    own interpretations of ``artifact_profile``.
    """

    profile: str
    verbose_episode_history: bool
    rich_analysis_intermediates: bool
    per_episode_prompt_files: bool

    @classmethod
    def for_profile(cls, profile: str) -> "RetentionPolicy":
        if profile not in ARTIFACT_PROFILES:
            raise ValueError(f"unknown artifact profile {profile!r}")
        full = profile == "full"
        return cls(
            profile=profile,
            verbose_episode_history=full,
            rich_analysis_intermediates=full,
            per_episode_prompt_files=full,
        )


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Offline analysis selection for a completed run."""

    schema_version: int = 1
    enabled: bool = False
    estimators: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)
    comet_export: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "estimators", tuple(self.estimators))
        object.__setattr__(self, "options", _freeze(self.options))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "enabled": self.enabled,
            "estimators": list(self.estimators),
            "options": _thaw(self.options),
            "comet_export": self.comet_export,
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


DEFAULT_CELL_METRICS = (
    "dominant_action_share",
    "action_share_relabelled",
    "active_fraction",
    "consensus_round",
    "converged_fraction",
)
"""The cell-level metrics computed at every cell completion unless overridden."""


@dataclass(frozen=True, slots=True)
class AggregationConfig:
    """How a cell's episodes are combined, and which cross-episode metrics run.

    Separate from ``metrics:`` on purpose. That section governs what each
    *episode* records; this one governs how many finished episodes become one
    answer, which is a different decision made at a different time — after the
    episodes exist, and re-makeable without re-running them.

    The first four fields are correctness rules rather than preferences; each
    one's reasoning lives on :class:`mas_cc.metrics.AggregationPolicy`, which
    is what they resolve into. ``horizons`` and ``null_permutations`` are here
    because the sweep metrics that need them (`lagged_cmi`, `mi_null_band`)
    have no other place to be told, and both change what the reported number
    means.
    """

    schema_version: int = 1
    forward_fill: str = "absorbing"
    relabel_by_winner: bool = True
    percentiles: tuple[int, ...] = (10, 50, 90)
    rolling_window: int = 20
    cell_metrics: tuple[str, ...] = DEFAULT_CELL_METRICS
    sweep_metrics: tuple[str, ...] = ()
    horizons: tuple[int, ...] = (1,)
    null_permutations: int = 200

    def __post_init__(self) -> None:
        object.__setattr__(self, "percentiles", tuple(int(p) for p in self.percentiles))
        object.__setattr__(self, "cell_metrics", tuple(self.cell_metrics))
        object.__setattr__(self, "sweep_metrics", tuple(self.sweep_metrics))
        object.__setattr__(self, "horizons", tuple(int(h) for h in self.horizons))

    def policy(self) -> "Any":
        """The resolved :class:`mas_cc.metrics.AggregationPolicy`.

        Imported inside the method so this module keeps its "standard library
        only, constructs nothing" property.
        """

        from mas_cc.metrics import AggregationPolicy

        return AggregationPolicy(
            forward_fill=self.forward_fill,
            relabel_by_winner=self.relabel_by_winner,
            percentiles=self.percentiles,
            rolling_window=self.rolling_window,
        )

    def resolved_cell_metrics(self) -> tuple[str, ...]:
        """``cell_metrics``, plus the count tables the configured sweep metrics need.

        A `SweepMetric` reads only what the cell tier put in
        ``AggregateResult.counts``, so asking for `terminal_mi` without also
        asking for `macrostate_counts` would silently produce NaN. Adding the
        dependency here rather than making the user list it keeps the YAML
        matching the spec while making the broken pair unconfigurable.
        """

        if not self.sweep_metrics or "macrostate_counts" in self.cell_metrics:
            return self.cell_metrics
        return (*self.cell_metrics, "macrostate_counts")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "forward_fill": self.forward_fill,
            "relabel_by_winner": self.relabel_by_winner,
            "percentiles": list(self.percentiles),
            "rolling_window": self.rolling_window,
            "cell_metrics": list(self.cell_metrics),
            "sweep_metrics": list(self.sweep_metrics),
            "horizons": list(self.horizons),
            "null_permutations": self.null_permutations,
        }


COMET_WRITERS = ("master_only",)
"""Who may talk to Comet. Only the master ever does — see `CometObservability`."""

CELL_REPORTING_MODES = ("experiments", "master", "disabled")
"""Where a finished cell's curves and plots go. See `CometObservability`.

``disabled`` rather than the more natural ``off``: YAML 1.1 resolves a bare
``off`` (and ``no``) to boolean false, so ``cell_reporting: off`` would arrive
as ``False`` and fail type validation for a reason the config does not show.
"""


@dataclass(frozen=True, slots=True)
class CometObservability:
    """Shape of the master's remote reporting; ``logging.comet`` is the on/off switch.

    ``writer`` has exactly one legal value, and that is the point. Workers
    write episode files and nothing else, so there is one writer per Comet
    experiment key and therefore no race on its step counter. Comet becomes a
    *view* rather than a store: if it fails, the run is unaffected and the same
    numbers can be re-logged later from the episode files.

    ``heartbeat_seconds`` is a timer, not a completion hook, because the
    question it answers is "is this job alive" — a metric that only moves when
    an episode finishes cannot distinguish a dead master from a slow one.

    ``metric_plots`` opts completed cells into uploading the same aggregate
    PNGs that are always rendered locally. It defaults off so image upload is
    explicit even when cell reporting is on.

    ``cell_reporting`` says *where* a finished cell's curves, scalars and PNGs
    go — one setting with three values rather than two booleans, because two
    booleans describe four states when only three exist, and the fourth has to
    resolve silently in favour of one of them:

    - ``experiments`` (default) — one child experiment per cell, named
      ``<run>/<cell_id>``. Metric names stay bare (``m_ctrl``), so Comet can
      overlay one cell against another. This is the comparison a sweep exists
      to make, and it is what a grid usually wants.
    - ``master`` — everything lands on the master instead, prefixed by cell id
      (``cell-0000_m_ctrl``). One experiment holds the whole run: the grid
      image, every cell's curves and PNGs, and the post-run analysis report.
      Right for a single-cell run, which has nothing to overlay against and
      would otherwise scatter its output over three near-empty experiments —
      and right for any run you would rather read in one place.
    - ``disabled`` — the master keeps its progress series and grid image, and
      no cell curves are uploaded at all.

    Cell reporting is orthogonal to ``sweep_experiment``, which is what decides
    whether the master exists; the grid image belongs to the master and is
    published under every one of these three modes.
    """

    writer: str = "master_only"
    heartbeat_seconds: float = 60.0
    grid_image_every_n_episodes: int = 25
    sweep_experiment: bool = True
    cell_reporting: str = "experiments"
    metric_plots: bool = False
    progress_metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.writer not in COMET_WRITERS:
            raise ValueError(
                f"observability.comet.writer must be one of {COMET_WRITERS}, got {self.writer!r}"
            )
        if self.cell_reporting not in CELL_REPORTING_MODES:
            raise ValueError(
                "observability.comet.cell_reporting must be one of "
                f"{CELL_REPORTING_MODES}, got {self.cell_reporting!r}"
            )
        if self.heartbeat_seconds <= 0:
            raise ValueError("observability.comet.heartbeat_seconds must be positive")
        if self.grid_image_every_n_episodes < 1:
            raise ValueError("observability.comet.grid_image_every_n_episodes must be at least 1")
        object.__setattr__(self, "progress_metrics", tuple(self.progress_metrics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "writer": self.writer,
            "heartbeat_seconds": self.heartbeat_seconds,
            "grid_image_every_n_episodes": self.grid_image_every_n_episodes,
            "sweep_experiment": self.sweep_experiment,
            "cell_reporting": self.cell_reporting,
            "metric_plots": self.metric_plots,
            "progress_metrics": list(self.progress_metrics),
        }


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    """Live-monitoring policy for a long unattended run."""

    schema_version: int = 1
    comet: CometObservability = field(default_factory=CometObservability)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "comet": self.comet.to_dict()}


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
    live_spend_poll_seconds: int | None = None

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
            "live_spend_poll_seconds": self.live_spend_poll_seconds,
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
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
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
            "aggregation": self.aggregation.to_dict(),
            "observability": self.observability.to_dict(),
            "experiment": self.experiment.to_dict(),
            "pricing": self.pricing.to_dict(),
            "budget": self.budget.to_dict(),
        }
