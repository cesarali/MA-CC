"""Post-run analyses selected by the resolved ``analysis:`` configuration."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from mas_cc.config import RunConfig


RELATIONAL_ROUND_FEEDBACK = "relational_imitation_round_feedback"
BLACKBOARD_EPISODE_INSPECTION = "blackboard_episode_inspection"
RELATIONAL_THEORETICAL_REFERENCES = frozenset(
    {"single_affinity_revised", "none", "matched_qvoter_null"}
)
ROUND_FEEDBACK_GAME_TYPES = frozenset(
    {"hidden_bench_imitation_round_feedback", RELATIONAL_ROUND_FEEDBACK}
)
"""Games whose configured analysis is the round-feedback pipeline.

They share one statistic vocabulary (`ROUND_ANALYSIS_STATISTICS`) because they
share the estimators; only the record adapter and the report destination
differ."""


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
    if config.game.type not in {
        "hidden_bench_imitation",
        "hidden_bench_imitation_round_feedback",
        "relational_imitation_round_feedback",
    }:
        raise ValueError(
            f"configured post-run analysis is not supported for game.type {config.game.type!r}"
        )

    if config.game.type in ROUND_FEEDBACK_GAME_TYPES:
        from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import (
            ROUND_ANALYSIS_STATISTICS,
        )

        requested = tuple(analysis.estimators)
        inspection = (
            config.game.type == RELATIONAL_ROUND_FEEDBACK
            and BLACKBOARD_EPISODE_INSPECTION in requested
        )
        statistical = tuple(
            name for name in requested if name != BLACKBOARD_EPISODE_INSPECTION
        )
        unknown = sorted(set(statistical) - set(ROUND_ANALYSIS_STATISTICS))
        if unknown:
            raise ValueError(
                "analysis.estimators contains unsupported round-feedback statistic(s): "
                + ", ".join(unknown)
            )
        # Validated here rather than at analysis time so a typo in the estimator
        # list fails at preflight, before the run spends anything.
        statistics = statistical
        diagnostics: tuple[str, ...] = ()
        current_statistics: tuple[str, ...] = ()
    else:
        from mas_cc.games.hidden_bench.imitation.analysis import (
            CONTROLLER_DIAGNOSTIC_STATISTICS,
            CURRENT_STATISTICS,
            INFORMATION_STATISTICS,
        )

        # One list in the config, two arguments downstream: the MI/CMI channels and
        # the controller diagnostics that normalize and sign them are computed by
        # different machinery, but a run author thinks of them as one request.
        requested = tuple(analysis.estimators)
        unknown = sorted(
            set(requested)
            - set(INFORMATION_STATISTICS)
            - set(CONTROLLER_DIAGNOSTIC_STATISTICS)
            - set(CURRENT_STATISTICS)
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
        current_statistics = tuple(
            name for name in requested if name in CURRENT_STATISTICS
        )

    options = dict(analysis.options)
    allowed_options = {
        "bootstrap_resamples",
        "null_permutations",
        "confidence",
        "seed",
        "per_cell_reports",
    }
    if config.game.type == RELATIONAL_ROUND_FEEDBACK:
        # Bin count for the joint (kappa, phi) diagnostic conditioning. The
        # three scalar epistemic conditionings are fixed at three bins by
        # design and deliberately have no dial.
        allowed_options.add("epistemic_bins")
        allowed_options.add("theoretical_reference")
    unknown_options = sorted(set(options) - allowed_options)
    if unknown_options:
        raise ValueError(
            "unknown HiddenBench imitation analysis option(s): "
            + ", ".join(unknown_options)
        )
    if not isinstance(options.get("per_cell_reports", False), bool):
        raise ValueError("analysis.options.per_cell_reports must be a boolean")

    confidence = _number_option(options, "confidence", 0.95)
    if not 0 < confidence < 1:
        raise ValueError("analysis.options.confidence must be between zero and one")
    run_id = f"{config.experiment.name}-{config.execution.seed}"
    if config.game.type == RELATIONAL_ROUND_FEEDBACK:
        theoretical_reference = options.get(
            "theoretical_reference", "single_affinity_revised"
        )
        if not isinstance(theoretical_reference, str) or theoretical_reference not in (
            RELATIONAL_THEORETICAL_REFERENCES
        ):
            raise ValueError(
                "analysis.options.theoretical_reference must be one of: "
                + ", ".join(sorted(RELATIONAL_THEORETICAL_REFERENCES))
            )
        persistence = config.game.options.get("epistemic_persistence", 1.0)
        if isinstance(persistence, bool) or not isinstance(persistence, (int, float)):
            raise ValueError("game.options.epistemic_persistence must be a number")
        if float(persistence) < 1.0 and theoretical_reference != "none":
            raise ValueError(
                "analysis.options.theoretical_reference must be 'none' when "
                "game.options.epistemic_persistence is below 1.0"
            )
        if (
            config.game.options.get("social_mode", "peer") == "board"
            and theoretical_reference != "none"
        ):
            raise ValueError(
                "analysis.options.theoretical_reference must be 'none' for "
                "game.options.social_mode 'board'; q-voter theory assumes current peers"
            )
        return {
            "bootstrap_resamples": _integer_option(
                options, "bootstrap_resamples", 1000
            ),
            "null_permutations": _integer_option(options, "null_permutations", 1000),
            "confidence": confidence,
            "seed": _integer_option(options, "seed", config.execution.seed),
            "statistics": statistics,
            "epistemic_bins": _integer_option(options, "epistemic_bins", 4),
            "theoretical_reference": theoretical_reference,
            # Current analysis is aggregate post-processing, so it follows the
            # same master-only Comet gate as the other configured analyzers.
            "comet_export": analysis.comet_export and config.logging.comet,
            "comet_project": str(config.logging.options.get("comet_project", "mas-cc")),
            "comet_run_name": f"{run_id}/analysis",
            "blackboard_episode_inspection": inspection,
        }
    return {
        "bootstrap_resamples": _integer_option(options, "bootstrap_resamples", 1000),
        "null_permutations": _integer_option(options, "null_permutations", 1000),
        "confidence": confidence,
        "seed": _integer_option(options, "seed", config.execution.seed),
        "statistics": statistics,
        "diagnostics": diagnostics,
        "current_statistics": current_statistics,
        # `analysis.comet_export` opts the report in; the logging master switch
        # can still veto it, same as every other Comet integration.
        "comet_export": analysis.comet_export and config.logging.comet,
        "comet_project": str(config.logging.options.get("comet_project", "mas-cc")),
        "comet_run_name": f"{run_id}/analysis",
    }


def per_cell_reports_enabled(config: RunConfig) -> bool:
    """Whether a grid should render configured analysis as each cell closes."""

    if not config.analysis.enabled:
        return False
    value = config.analysis.options.get("per_cell_reports", False)
    if not isinstance(value, bool):
        raise ValueError("analysis.options.per_cell_reports must be a boolean")
    return value


def _report_slug(config: RunConfig, cell_id: str) -> str:
    """Readable, filesystem-safe identity for one HiddenBench q/q_c cell."""

    task = config.game.options.get("task_id", "task-unspecified")
    q = config.game.options.get("social_group_size", 1)
    q_c = config.control.options.get("sensor_sample_size", "none")
    raw = f"{cell_id}__task-{task}__N-{config.game.population_size}__q-{q}__qc-{q_c}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-.")


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
    from mas_cc.storage import canonical_hash

    root = Path(run_dir)
    if config.game.type == RELATIONAL_ROUND_FEEDBACK:
        inspection = bool(arguments.pop("blackboard_episode_inspection", False))
        if inspection:
            from mas_cc.games.relational_reasoning.imitation_round_feedback.pilot_artifacts import (
                build_blackboard_pilot_artifacts,
            )

            inspection_summary = build_blackboard_pilot_artifacts(config, root)
        else:
            inspection_summary = None
        if not arguments["statistics"]:
            return inspection_summary
        from mas_cc.games.relational_reasoning.imitation_round_feedback.analysis import (
            analyze_relational_imitation_round_feedback,
        )

        summary = analyze_relational_imitation_round_feedback(
            root,
            root / "relational_imitation_round_feedback_analysis",
            comet_sink=comet_sink,
            **arguments,
        )
        return (
            summary
            if inspection_summary is None
            else {**summary, "blackboard_episode_inspection": inspection_summary}
        )

    if config.game.type == "hidden_bench_imitation_round_feedback":
        from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import (
            analyze_hidden_bench_imitation_round_feedback,
        )

        round_arguments = dict(arguments)
        round_arguments.pop("diagnostics", None)
        round_arguments.pop("current_statistics", None)
        return analyze_hidden_bench_imitation_round_feedback(
            root,
            root / "hidden_bench_imitation_round_feedback_analysis",
            comet_sink=comet_sink,
            artifact_profile=config.storage.artifact_profile,
            resolved_config_hash=canonical_hash(config.to_dict()),
            **round_arguments,
        )

    from mas_cc.games.hidden_bench.imitation.analysis import (
        analyze_hidden_bench_imitation,
    )

    return analyze_hidden_bench_imitation(
        root,
        root / "hidden_bench_imitation_analysis",
        comet_sink=comet_sink,
        artifact_profile=config.storage.artifact_profile,
        resolved_config_hash=canonical_hash(config.to_dict()),
        **arguments,
    )


def run_configured_cell_analysis(
    config: RunConfig,
    cell_dir: str | Path,
    cell_id: str,
    comet_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Render the two compact human-readable reports for one completed grid cell.

    The normal analyzer also creates machine-readable tables and plots.  Those
    remain useful in the final whole-grid analysis, but retaining a second copy
    under every cell would defeat ``results_only``.  Cell completion therefore
    runs the exact configured estimators in a temporary directory and keeps
    only the MI/controller-diagnostic and truth-current Markdown reports.

    When analysis export is enabled, a consolidated grid passes its already
    open master sink here.  Otherwise the analyzer opens a uniquely named
    per-cell analysis experiment.  In either case there is still no worker or
    episode-level Comet writer.
    """

    arguments = _configured_arguments(config)
    if arguments is None:
        return None
    from mas_cc.storage import canonical_hash

    root = Path(cell_dir)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    slug = _report_slug(config, cell_id)
    arguments["comet_run_name"] = f"{arguments['comet_run_name']}/{cell_id}"
    retained: list[str] = []
    with tempfile.TemporaryDirectory(prefix=".cell-analysis-", dir=root) as temporary:
        destination = Path(temporary)
        if config.game.type == RELATIONAL_ROUND_FEEDBACK:
            from mas_cc.games.relational_reasoning.imitation_round_feedback.analysis import (
                analyze_relational_imitation_round_feedback,
            )

            summary = analyze_relational_imitation_round_feedback(
                root,
                destination,
                comet_sink=comet_sink,
                comet_name_suffix=slug,
                **arguments,
            )
            source_names = (
                "round_information_estimates.md",
                "analysis_summary.json",
                Path("currents") / "current_analysis.md",
            )
        elif config.game.type == "hidden_bench_imitation_round_feedback":
            from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import (
                analyze_hidden_bench_imitation_round_feedback,
            )

            round_arguments = dict(arguments)
            round_arguments.pop("diagnostics", None)
            round_arguments.pop("current_statistics", None)
            summary = analyze_hidden_bench_imitation_round_feedback(
                root,
                destination,
                artifact_profile=config.storage.artifact_profile,
                resolved_config_hash=canonical_hash(config.to_dict()),
                comet_sink=comet_sink,
                comet_name_suffix=slug,
                **round_arguments,
            )
            source_names = (
                "round_information_estimates.md",
                "analysis_summary.json",
            )
        else:
            from mas_cc.games.hidden_bench.imitation.analysis import (
                analyze_hidden_bench_imitation,
            )

            summary = analyze_hidden_bench_imitation(
                root,
                destination,
                artifact_profile=config.storage.artifact_profile,
                resolved_config_hash=canonical_hash(config.to_dict()),
                comet_sink=comet_sink,
                comet_name_suffix=slug,
                **arguments,
            )
            source_names = (
                "information_estimates.md",
                "truth_current_estimates.md",
            )
        for source_name in source_names:
            source = destination / source_name
            if not source.is_file():
                continue
            target = reports / f"{source.stem}__{slug}{source.suffix}"
            source.replace(target)
            retained.append(str(target))
    return {**summary, "cell_report_slug": slug, "cell_reports": retained}


__all__ = [
    "per_cell_reports_enabled",
    "run_configured_analysis",
    "run_configured_cell_analysis",
    "validate_configured_analysis",
]
