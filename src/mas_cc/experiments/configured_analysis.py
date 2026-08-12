"""Post-run analyses selected by the resolved ``analysis:`` configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from mas_cc.config import RunConfig


def _integer_option(options: Mapping[str, Any], name: str, default: int) -> int:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"analysis.options.{name} must be a non-negative integer")
    return value


def _number_option(options: Mapping[str, Any], name: str, default: float) -> float:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"analysis.options.{name} must be a number")
    return float(value)


def _configured_arguments(config: RunConfig) -> dict[str, Any] | None:
    analysis = config.analysis
    if not analysis.enabled:
        return None
    if not analysis.estimators:
        raise ValueError("analysis.enabled is true but analysis.estimators is empty")
    if config.game.type != "hidden_bench_imitation":
        raise ValueError(
            f"configured post-run analysis is not supported for game.type {config.game.type!r}"
        )

    from mas_cc.games.hidden_bench.imitation.analysis import (
        CONTROLLER_DIAGNOSTIC_STATISTICS,
        INFORMATION_STATISTICS,
    )

    # One list in the config, two arguments downstream: the MI/CMI channels and
    # the controller diagnostics that normalize and sign them are computed by
    # different machinery, but a run author thinks of them as one request.
    requested = tuple(analysis.estimators)
    unknown = sorted(
        set(requested) - set(INFORMATION_STATISTICS) - set(CONTROLLER_DIAGNOSTIC_STATISTICS)
    )
    if unknown:
        raise ValueError(
            "analysis.estimators contains unsupported HiddenBench imitation statistic(s): "
            + ", ".join(unknown)
        )
    statistics = tuple(name for name in requested if name in INFORMATION_STATISTICS)
    diagnostics = tuple(
        name for name in requested if name in CONTROLLER_DIAGNOSTIC_STATISTICS
    )

    options = dict(analysis.options)
    allowed_options = {"bootstrap_resamples", "null_permutations", "confidence", "seed"}
    unknown_options = sorted(set(options) - allowed_options)
    if unknown_options:
        raise ValueError(
            "unknown HiddenBench imitation analysis option(s): " + ", ".join(unknown_options)
        )

    confidence = _number_option(options, "confidence", 0.95)
    if not 0 < confidence < 1:
        raise ValueError("analysis.options.confidence must be between zero and one")
    run_id = f"{config.experiment.name}-{config.execution.seed}"
    return {
        "bootstrap_resamples": _integer_option(options, "bootstrap_resamples", 1000),
        "null_permutations": _integer_option(options, "null_permutations", 1000),
        "confidence": confidence,
        "seed": _integer_option(options, "seed", config.execution.seed),
        "statistics": statistics,
        "diagnostics": diagnostics,
        # `analysis.comet_export` opts the report in; the logging master switch
        # can still veto it, same as every other Comet integration.
        "comet_export": analysis.comet_export and config.logging.comet,
        "comet_project": str(config.logging.options.get("comet_project", "mas-cc")),
        "comet_run_name": f"{run_id}/analysis",
    }


def validate_configured_analysis(config: RunConfig) -> None:
    """Fail before launch when a requested post-run analysis is invalid."""

    _configured_arguments(config)


def run_configured_analysis(
    config: RunConfig, run_dir: str | Path, comet_sink: Any | None = None
) -> dict[str, Any] | None:
    """Run configured analysis after rich or compact scientific data is persisted.

    ``comet_sink`` is the run's already-open master experiment when the config
    asked for a single consolidated one; ``None`` keeps the historical
    behaviour of opening a dedicated ``<run>/analysis`` experiment.
    """

    arguments = _configured_arguments(config)
    if arguments is None:
        return None
    from mas_cc.games.hidden_bench.imitation.analysis import analyze_hidden_bench_imitation
    from mas_cc.storage import canonical_hash

    root = Path(run_dir)
    return analyze_hidden_bench_imitation(
        root,
        root / "hidden_bench_imitation_analysis",
        comet_sink=comet_sink,
        artifact_profile=config.storage.artifact_profile,
        resolved_config_hash=canonical_hash(config.to_dict()),
        **arguments,
    )


__all__ = ["run_configured_analysis", "validate_configured_analysis"]
