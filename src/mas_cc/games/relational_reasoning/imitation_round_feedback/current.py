"""Finite-horizon controller-target current analysis for completed episodes.

This module is post-processing only.  It reads the same ``RoundEvent`` objects
as the relational information analysis, resamples them by whole episode with
the shared bootstrap helper, and evaluates the already-established exact
finite-N q-voter kernels.  It never imports or invokes a provider runtime.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...hidden_bench.imitation.controller import ADVOCATE_TARGET, NO_OP
from ...hidden_bench.imitation_round_feedback.analysis import (
    RoundEvent,
    bootstrap_episode_rows,
)
from .theory import (
    ClassicalReference,
    classical_reference,
    finite_horizon_current_moments_for_episodes,
    q1_current_closed_forms,
    q1_mean_response,
    theory_parameters_from_record,
)


CURRENT_VALUE_NAMES = (
    "current_mean",
    "current_variance",
    "current_fano_dispersion",
    "current_precision_irisarri",
    "current_snr2",
)

CURRENT_SUMMARY_REQUIRED_FIELDS = (
    "cell_id",
    "task_id",
    "analysis_target",
    "N",
    "q",
    "q_c",
    "b",
    "c",
    "beta",
    "theta",
    "K",
    "n_repetitions",
    "current_mean_empirical",
    "current_variance_empirical",
    "current_fano_dispersion_empirical",
    "current_precision_irisarri_empirical",
    "current_snr2_empirical",
    "current_mean_theory",
    "current_variance_theory",
    "current_fano_dispersion_theory",
    "current_precision_irisarri_theory",
    "current_snr2_theory",
    "current_mean_empirical_minus_theory",
    "current_variance_empirical_minus_theory",
    "current_fano_dispersion_empirical_minus_theory",
    "current_precision_irisarri_empirical_minus_theory",
    "current_snr2_empirical_minus_theory",
    "current_precision_support",
    "current_snr2_zero_variance_nonzero_mean",
    "current_snr2_degenerate_zero_current",
)


def _safe_metric_ratios(mean: float, variance: float, *, suffix: str) -> dict[str, Any]:
    """Fano, inverse-Fano and SNR² with explicit unsmoothed degeneracies."""

    fano = precision = snr2 = math.nan
    zero_mean = bool(math.isfinite(mean) and mean == 0.0)
    zero_variance = bool(math.isfinite(variance) and variance == 0.0)
    zero_variance_nonzero_mean = bool(zero_variance and not zero_mean)
    degenerate_zero_current = bool(zero_variance and zero_mean)
    if math.isfinite(mean) and math.isfinite(variance):
        if mean == 0.0:
            fano = math.nan if variance == 0.0 else math.inf
        else:
            fano = variance / abs(mean)
        if variance == 0.0:
            precision = math.nan if mean == 0.0 else math.inf
            snr2 = math.nan if mean == 0.0 else math.inf
        else:
            precision = abs(mean) / variance
            snr2 = mean * mean / variance
    return {
        f"current_fano_dispersion_{suffix}": fano,
        f"current_precision_irisarri_{suffix}": precision,
        f"current_snr2_{suffix}": snr2,
        f"current_fano_zero_mean_{suffix}": zero_mean,
        f"current_snr2_zero_variance_nonzero_mean_{suffix}": (
            zero_variance_nonzero_mean
        ),
        f"current_snr2_degenerate_zero_current_{suffix}": degenerate_zero_current,
    }


def empirical_current_statistics(currents: Sequence[float]) -> dict[str, Any]:
    """Mean and sample variance across repeated complete episode currents."""

    values = np.asarray(currents, dtype=float)
    mean = float(values.mean()) if len(values) else math.nan
    variance = float(values.var(ddof=1)) if len(values) >= 2 else math.nan
    return {
        "current_mean_empirical": mean,
        "current_variance_empirical": variance,
        **_safe_metric_ratios(mean, variance, suffix="empirical"),
    }


def _episode_row(
    ordered: Sequence[RoundEvent],
    micro_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if not ordered:
        raise ValueError("an episode current requires at least one completed round")
    first, last = ordered[0], ordered[-1]
    target = str(first.event.get("analysis_target") or first.event.get("correct_answer"))
    initial = first.target_before
    final = last.target_after
    terminal_current = final - initial
    increments = [
        value
        for value in (_micro_target_increment(row, target) for row in micro_rows)
        if value is not None
    ]
    microscopic = sum(increments) if increments else None
    return {
        "cell_id": first.cell_id,
        "task_id": str(first.event.get("task_id") or "task-unspecified"),
        "episode_id": first.episode_id,
        "analysis_target": target,
        "initial_target_count": initial,
        "final_target_count": final,
        "episode_current": terminal_current,
        "K": len(ordered),
        "microscopic_current_available": bool(increments),
        "microscopic_current": microscopic,
        "microscopic_current_matches_terminal": (
            None if microscopic is None else microscopic == terminal_current
        ),
    }


def episode_current_rows(
    rounds: Sequence[RoundEvent],
    micro: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """One terminal-count current per ``(cell, task, episode)``."""

    grouped: dict[tuple[str, str, str], list[RoundEvent]] = defaultdict(list)
    for row in rounds:
        task_id = str(row.event.get("task_id") or "task-unspecified")
        grouped[(row.cell_id, task_id, row.episode_id)].append(row)
    micro_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in micro:
        micro_groups[(str(row.get("cell_id", "run")), str(row.get("episode_id")))].append(row)
    result = []
    for (cell_id, _task_id, episode_id), group in grouped.items():
        ordered = sorted(group, key=lambda row: row.round_index)
        result.append(
            _episode_row(ordered, micro_groups.get((cell_id, episode_id), ()))
        )
    return sorted(result, key=lambda row: (row["cell_id"], row["task_id"], row["episode_id"]))


def _micro_target_increment(row: Mapping[str, Any], target: str) -> int | None:
    before = row.get("focal_opinion_before")
    after = row.get("focal_opinion_after")
    if before is not None and after is not None:
        return int(str(after) == target) - int(str(before) == target)
    options = row.get("possible_answers")
    before_counts = row.get("occupation_counts_before")
    after_counts = row.get("occupation_counts_after")
    if options is None or before_counts is None or after_counts is None:
        return None

    def count(counts: Any) -> int:
        if isinstance(counts, Mapping):
            return int(counts.get(target, 0))
        labels = [str(value) for value in options]
        return int(counts[labels.index(target)])

    return count(after_counts) - count(before_counts)


def read_relational_micro_events(root: str | Path) -> list[dict[str, Any]]:
    """Read optional full or compact microscopic records for telescoping checks."""

    source = Path(root)
    if not source.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(
        [*source.rglob("trajectory.jsonl"), *source.rglob("micro_slot_trajectory.jsonl")]
    ):
        cell_id = "run"
        for parent in path.parents:
            if (parent / "overrides.json").is_file():
                cell_id = parent.name
                break
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            event = payload.get("event", payload)
            if not isinstance(event, Mapping):
                continue
            if (
                event.get("microscopic_event_index") is None
                and event.get("within_round_index") is None
            ):
                continue
            rows.append(
                {
                    "cell_id": str(payload.get("cell_id", cell_id)),
                    "episode_id": str(
                        payload.get("episode_id", event.get("episode_id", "episode"))
                    ),
                    **dict(event),
                }
            )
    return rows


def _support_label(repetitions: int) -> str:
    if repetitions <= 2:
        return "descriptive_only"
    if repetitions < 10:
        return "limited"
    return "adequate"


def _analysis_target(episodes: Sequence[Mapping[str, Any]]) -> str:
    targets = {str(row["analysis_target"]) for row in episodes}
    return next(iter(targets)) if len(targets) == 1 else "episode_specific_controller_target"


def _initial_distribution(episodes: Sequence[Mapping[str, Any]], N: int) -> np.ndarray:
    distribution = np.zeros(N + 1, dtype=float)
    for row in episodes:
        distribution[int(row["initial_target_count"])] += 1.0
    distribution /= len(episodes)
    return distribution


def _theory_kernel(
    rows: Sequence[RoundEvent], reference: ClassicalReference
) -> tuple[str, np.ndarray]:
    actions = [row.U_k for row in rows if row.U_k in {ADVOCATE_TARGET, NO_OP}]
    probabilities = [row.p_k for row in rows if row.U_k in {ADVOCATE_TARGET, NO_OP}]
    if actions and all(action == ADVOCATE_TARGET for action in actions) and all(
        value is not None and math.isclose(float(value), 1.0, abs_tol=1e-12)
        for value in probabilities
    ):
        return "always_ADVOCATE", reference.R1
    if actions and all(action == NO_OP for action in actions) and all(
        value is not None and math.isclose(float(value), 0.0, abs_tol=1e-12)
        for value in probabilities
    ):
        return "always_NO_OP", reference.R0
    return "stochastic_feedback", reference.closed_loop_kernel


def _current_summary_without_bootstrap(
    rows: Sequence[RoundEvent],
    episodes: Sequence[Mapping[str, Any]],
    *,
    theory_enabled: bool,
) -> dict[str, Any]:
    first = rows[0]
    parameters = None
    found = {
        item.key: item
        for item in (theory_parameters_from_record(row.event) for row in rows)
        if item is not None
    }
    if len(found) == 1:
        parameters = next(iter(found.values()))
    N = parameters.N if parameters is not None else sum(first.N_k)
    repetitions = len(episodes)
    empirical = empirical_current_statistics(
        [float(row["episode_current"]) for row in episodes]
    )
    horizons = sorted({int(row["K"]) for row in episodes})
    row: dict[str, Any] = {
        "cell_id": first.cell_id,
        "task_id": str(first.event.get("task_id") or "task-unspecified"),
        "analysis_target": _analysis_target(episodes),
        "N": N,
        "q": parameters.q if parameters is not None else first.event.get("social_group_size"),
        "q_c": parameters.q_c if parameters is not None else first.event.get("sensor_sample_size"),
        "b": parameters.b if parameters is not None else first.event.get("intervention_budget"),
        "c": (
            parameters.actuation_fraction
            if parameters is not None
            else first.event.get("actuation_fraction")
        ),
        "beta": parameters.beta if parameters is not None else first.event.get("controller_beta"),
        "theta": (
            parameters.theta
            if parameters is not None
            else first.event.get("controller_threshold")
        ),
        "K": horizons[0] if len(horizons) == 1 else "mixed:" + ",".join(map(str, horizons)),
        "n_repetitions": repetitions,
        **empirical,
        "current_precision_support": _support_label(repetitions),
        # Required unsuffixed flags describe the empirical repeated-episode
        # statistic; suffixed theory flags are also retained below.
        "current_snr2_zero_variance_nonzero_mean": empirical[
            "current_snr2_zero_variance_nonzero_mean_empirical"
        ],
        "current_snr2_degenerate_zero_current": empirical[
            "current_snr2_degenerate_zero_current_empirical"
        ],
        "theory_mode": "unavailable",
        "theory_applicable": False,
        "theory_skip_reason": None,
    }

    theory_mean = theory_variance = math.nan
    if not theory_enabled:
        row["theory_skip_reason"] = "matched theory comparison disabled"
    elif parameters is None:
        row["theory_skip_reason"] = (
            "no single matched (N, q, q_c, b, beta, theta) tuple applies"
        )
    else:
        reference = classical_reference(parameters)
        mode, kernel = _theory_kernel(rows, reference)
        moments = finite_horizon_current_moments_for_episodes(
            kernel,
            [int(item["initial_target_count"]) for item in episodes],
            [int(item["K"]) for item in episodes],
        )
        theory_mean = moments["mean"]
        theory_variance = moments["variance"]
        row.update(
            {
                "theory_mode": mode,
                "theory_applicable": True,
                "current_second_moment_theory": moments["second_moment"],
            }
        )
        if parameters.q == 1:
            initial = _initial_distribution(episodes, N)
            row.update(q1_current_closed_forms(initial, N=N, b=parameters.b))
            row["q1_current_closed_form_matches_kernel"] = bool(
                np.allclose(
                    reference.mean_response,
                    [q1_mean_response(n / N, N=N, b=parameters.b) for n in range(N + 1)],
                    atol=1e-12,
                    rtol=1e-12,
                )
            )

    theory = {
        "current_mean_theory": theory_mean,
        "current_variance_theory": theory_variance,
        **_safe_metric_ratios(theory_mean, theory_variance, suffix="theory"),
    }
    row.update(theory)
    for name in CURRENT_VALUE_NAMES:
        empirical_value = float(row[f"{name}_empirical"])
        theory_value = float(row[f"{name}_theory"])
        row[f"{name}_empirical_minus_theory"] = empirical_value - theory_value
    return row


def _episode_occurrences(rows: Sequence[RoundEvent]) -> list[list[RoundEvent]]:
    """Split a bootstrap draw, including adjacent duplicate episode blocks."""

    result: list[list[RoundEvent]] = []
    current: list[RoundEvent] = []
    for row in rows:
        boundary = bool(
            current
            and (
                row.episode_id != current[-1].episode_id
                or row.round_index <= current[-1].round_index
            )
        )
        if boundary:
            result.append(current)
            current = []
        current.append(row)
    if current:
        result.append(current)
    return result


def current_cell_summary(
    rows: Sequence[RoundEvent],
    *,
    bootstrap_resamples: int,
    confidence: float,
    seed: int,
    theory_enabled: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One task/cell summary and its episode-current rows."""

    if not rows:
        raise ValueError("current summary requires at least one round")
    episodes = episode_current_rows(rows)
    summary = _current_summary_without_bootstrap(
        rows, episodes, theory_enabled=theory_enabled
    )
    draws = bootstrap_episode_rows(
        sorted(rows, key=lambda row: (row.episode_id, row.round_index)),
        resamples=bootstrap_resamples,
        seed=seed,
    )
    bootstrap_rows: list[dict[str, Any]] = []
    for draw in draws:
        occurrences = _episode_occurrences(draw)
        drawn_episodes = [_episode_row(block) for block in occurrences]
        bootstrap_rows.append(
            _current_summary_without_bootstrap(
                list(draw), drawn_episodes, theory_enabled=theory_enabled
            )
        )
    alpha = (1.0 - confidence) / 2.0
    for field in (
        *(f"{name}_empirical" for name in CURRENT_VALUE_NAMES),
        *(f"{name}_theory" for name in CURRENT_VALUE_NAMES),
        *(f"{name}_empirical_minus_theory" for name in CURRENT_VALUE_NAMES),
    ):
        values = [
            float(row[field])
            for row in bootstrap_rows
            if isinstance(row.get(field), (int, float))
            and not isinstance(row.get(field), bool)
            and math.isfinite(float(row[field]))
        ]
        summary[f"{field}_bootstrap_ci_low"] = (
            math.nan if not values else float(np.quantile(values, alpha))
        )
        summary[f"{field}_bootstrap_ci_high"] = (
            math.nan if not values else float(np.quantile(values, 1.0 - alpha))
        )
    summary["current_bootstrap_unit"] = "episode"
    summary["current_bootstrap_resamples"] = bootstrap_resamples
    return summary, episodes


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return "undefined"
        if math.isinf(float(value)):
            return "infinite" if float(value) > 0 else "-infinite"
        return format(value, ".8g")
    return str(value)


def write_current_report(row: Mapping[str, Any], path: Path) -> None:
    """Write the single empirical-and-theory Markdown report for a task/cell."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Controller-target current analysis",
        "",
        "This report measures the net population current toward the controller "
        "target over repeated LLM episodes and compares it with the exact "
        "finite-N controlled q-voter at the same matched parameters. The primary "
        "quantities are the mean current, current variance, Fano-like dispersion, "
        "and the squared signal-to-noise ratio SNR². The report also gives the "
        "inverse-Fano / Irisarri-style precision explicitly so there is no "
        "convention ambiguity.",
        "",
        f"Cell: `{row['cell_id']}`  ",
        f"Task: `{row['task_id']}`  ",
        f"Analysis target: `{row['analysis_target']}` (target vs not-target)",
        "",
        "## Empirical repeated-episode current",
        "",
        "```text",
        f"episodes: {row['n_repetitions']}",
        f"mean current: {_format(row['current_mean_empirical'])}",
        f"variance: {_format(row['current_variance_empirical'])}",
        "Fano-like dispersion Var/|mean|: "
        + _format(row["current_fano_dispersion_empirical"]),
        "Irisarri-style precision |mean|/Var: "
        + _format(row["current_precision_irisarri_empirical"]),
        f"SNR²: {_format(row['current_snr2_empirical'])}",
        f"support/repetition warning: {row['current_precision_support']}",
        "```",
        "",
        "## Exact matched q-voter current",
        "",
        "```text",
        f"theory mode: {row['theory_mode']}",
        f"N={_format(row['N'])}",
        f"q={_format(row['q'])}",
        f"q_c={_format(row['q_c'])}",
        f"b={_format(row['b'])}",
        f"c={_format(row['c'])}",
        f"beta={_format(row['beta'])}",
        f"theta={_format(row['theta'])}",
        f"K={_format(row['K'])}",
        "",
        f"mean current: {_format(row['current_mean_theory'])}",
        f"variance: {_format(row['current_variance_theory'])}",
        "Fano-like dispersion Var/|mean|: "
        + _format(row["current_fano_dispersion_theory"]),
        "Irisarri-style precision |mean|/Var: "
        + _format(row["current_precision_irisarri_theory"]),
        f"SNR²: {_format(row['current_snr2_theory'])}",
        "```",
        "",
        "## Direct comparison",
        "",
        "| Quantity | Empirical | Exact q-voter | Empirical - theory |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "current_mean": "Mean current",
        "current_variance": "Current variance",
        "current_fano_dispersion": "Fano-like dispersion Var/|mean|",
        "current_precision_irisarri": "Irisarri-style precision |mean|/Var",
        "current_snr2": "Current SNR²",
    }
    for name in CURRENT_VALUE_NAMES:
        lines.append(
            f"| {labels[name]} | {_format(row[f'{name}_empirical'])} | "
            f"{_format(row[f'{name}_theory'])} | "
            f"{_format(row[f'{name}_empirical_minus_theory'])} |"
        )
    if row.get("q") == 1 and "q1_current_response_closed_form_theory" in row:
        lines.extend(
            [
                "",
                "## q=1 genuine one-round closed form",
                "",
                "These values average the closed form over the empirical initial "
                "target-count distribution. The finite-horizon values above remain "
                "the exact matrix result for all K rounds.",
                "",
                "```text",
                "NO_OP mean current: "
                + _format(row["q1_current_mean_noop_closed_form_theory"]),
                "ADVOCATE mean current: "
                + _format(row["q1_current_mean_advocate_closed_form_theory"]),
                "ADVOCATE - NO_OP response: "
                + _format(row["q1_current_response_closed_form_theory"]),
                "response in population-fraction coordinates: "
                + _format(row["q1_current_response_fraction_closed_form_theory"]),
                "closed form matches exact one-round kernels: "
                + _format(row["q1_current_closed_form_matches_kernel"]),
                "```",
            ]
        )
    if row.get("theory_skip_reason"):
        lines.extend(["", f"Theory note: {row['theory_skip_reason']}."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _slug(row: Mapping[str, Any]) -> str:
    raw = f"{row['cell_id']}__{row['task_id']}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-.") or "cell"


def write_current_analysis(
    rounds: Sequence[RoundEvent],
    output_dir: str | Path,
    *,
    bootstrap_resamples: int,
    confidence: float,
    seed: int,
    theory_enabled: bool = True,
    micro: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    """Write episode CSV, one summary CSV, and one report per task/cell."""

    groups: dict[tuple[str, str], list[RoundEvent]] = defaultdict(list)
    for row in rounds:
        groups[(row.cell_id, str(row.event.get("task_id") or "task-unspecified"))].append(row)
    summaries: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for index, group in enumerate(groups.values()):
        summary, _ = current_cell_summary(
            group,
            bootstrap_resamples=bootstrap_resamples,
            confidence=confidence,
            seed=seed + index,
            theory_enabled=theory_enabled,
        )
        summaries.append(summary)
    # Reconstruct once with optional microscopic records so the authoritative
    # episode table contains the independent telescoping check.
    episode_rows.extend(episode_current_rows(rounds, micro))

    destination = Path(output_dir) / "currents"
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(episode_rows).to_csv(destination / "episode_currents.csv", index=False)
    extra_fields = sorted(
        set().union(*(row.keys() for row in summaries))
        - set(CURRENT_SUMMARY_REQUIRED_FIELDS)
    )
    pd.DataFrame(summaries).reindex(
        columns=[*CURRENT_SUMMARY_REQUIRED_FIELDS, *extra_fields]
    ).to_csv(destination / "cell_current_summary.csv", index=False)
    report_paths: list[Path] = []
    for row in summaries:
        path = (
            destination / "current_analysis.md"
            if len(summaries) == 1
            else destination / _slug(row) / "current_analysis.md"
        )
        write_current_report(row, path)
        report_paths.append(path)
    return summaries, episode_rows, report_paths


def current_analysis_comet_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Finite current scalars for the master/post-hoc Comet writer."""

    metrics: dict[str, float] = {}
    field_names = {
        "current_mean_empirical": "current/mean_empirical",
        "current_mean_theory": "current/mean_theory",
        "current_variance_empirical": "current/variance_empirical",
        "current_variance_theory": "current/variance_theory",
        "current_fano_dispersion_empirical": "current/fano_dispersion_empirical",
        "current_fano_dispersion_theory": "current/fano_dispersion_theory",
        "current_precision_irisarri_empirical": "current/precision_irisarri_empirical",
        "current_precision_irisarri_theory": "current/precision_irisarri_theory",
        "current_snr2_empirical": "current/snr2_empirical",
        "current_snr2_theory": "current/snr2_theory",
        "n_repetitions": "current/n_repetitions",
        "q1_current_response_closed_form_theory": "current/q1_response_closed_form_theory",
        "q1_current_closed_form_matches_kernel": "current/q1_closed_form_matches_kernel",
    }
    many = len(rows) > 1
    for row in rows:
        prefix = f"{_slug(row)}/" if many else ""
        for field, key in field_names.items():
            value = row.get(field)
            if isinstance(value, bool):
                metrics[prefix + key] = float(value)
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                metrics[prefix + key] = float(value)
    return metrics


__all__ = [
    "CURRENT_SUMMARY_REQUIRED_FIELDS",
    "CURRENT_VALUE_NAMES",
    "current_analysis_comet_metrics",
    "current_cell_summary",
    "empirical_current_statistics",
    "episode_current_rows",
    "read_relational_micro_events",
    "write_current_analysis",
    "write_current_report",
]
