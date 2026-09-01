"""The standard discover → validate → estimate → report → package workflow."""

from __future__ import annotations

import json
import hashlib
import math
import os
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.storage import canonical_hash, file_sha256
from mas_cc.storage.scientific import compact_row_to_imitation_event

from .canonical import build_canonical_tables
from .discovery import DiscoveredRun, discover_cells, discover_runs
from .submission import read_submission_manifest
from .table_io import (
    CANONICAL_TABLE_FORMAT,
    read_scientific_table,
    retained_table_path,
    write_scientific_table,
)
from .validation import (
    paired_initialization_diagnostics,
    validate_study,
    validation_markdown,
)

ESTIMATOR_ALIASES = {
    "round_target_actuation_cmi_memory": "round_memory_target_actuation_cmi",
    "round_target_actuation_cmi_memory_phi": "round_phi_target_actuation_cmi",
    "target_signed_actuation": "round_target_signed_actuation",
}

PRIMARY_COLUMNS = (
    "study_id",
    "source_run_id",
    "cell_id",
    "metric",
    "estimator_version",
    "estimator_variant",
    "grouping_json",
    "conditioning_json",
    "estimate",
    "ci_low",
    "ci_high",
    "confidence",
    "null_type",
    "null_mean",
    "null_std",
    "p_value",
    "null_permutations",
    "bootstrap_resamples",
    "n_observations",
    "n_episodes",
    "action_entropy_ceiling_bits",
    "dual_action_support_fraction",
    "units",
    "support_status",
    "p_plus",
    "p_minus",
    "non_target_exposures",
    "non_target_to_target",
    "target_exposures",
    "target_to_non_target",
    "analysis_hash",
)

ETA_IR_JOIN_KEYS = (
    "study_id",
    "source_run_id",
    "cell_id",
    "grouping_json",
    "conditioning_json",
)


def _conditioning_json(metric: str) -> str:
    """Describe the estimator state represented by one aggregate row."""

    target_state = {"state": ["target_count_before"]}
    augmented = {
        "round_memory_target": "conditioning_memory_state",
        "round_epistemic_target": "conditioning_epistemic_state",
        "round_phi_target": "conditioning_phi_bin",
        "round_susceptible_target": "conditioning_susceptible_bin",
        "round_kappa_target": "conditioning_kappa_bin",
    }
    if metric in {
        "round_target_actuation_cmi",
        "round_target_signed_actuation",
        # The canonical chi is matched on exactly the CMI's state, which is
        # what makes the two joinable into a state-local eta_IR at all.
        "round_target_susceptibility",
    }:
        return json.dumps(target_state, sort_keys=True)
    if metric == "round_target_sensing_mi":
        return json.dumps({"state": [], "channel": ["target_count"]}, sort_keys=True)
    if metric == "round_target_signed_response_share":
        return json.dumps({"state": ["marginal"]}, sort_keys=True)
    for stem, extra_state in augmented.items():
        if metric.startswith(stem):
            return json.dumps(
                {"state": ["target_count_before", extra_state]}, sort_keys=True
            )
    return "{}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _recipe(study_manifest: Mapping[str, Any]) -> tuple[dict[str, Any], Path | None]:
    value = study_manifest.get("analysis_recipe")
    if value is None:
        return {}, None
    path = Path(str(value))
    if not path.is_file():
        raise ValueError(f"analysis recipe recorded by the study is missing: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid analysis recipe {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("analysis recipe must contain a mapping")
    return dict(raw), path


def _resampling(recipe: Mapping[str, Any]) -> dict[str, Any]:
    raw = recipe.get("resampling", {})
    if not isinstance(raw, Mapping):
        raise ValueError("analysis resampling must be a mapping")
    result = {
        "bootstrap_resamples": int(raw.get("bootstrap_resamples", 1000)),
        "null_permutations": int(raw.get("null_permutations", 1000)),
        "confidence": float(raw.get("confidence", 0.95)),
        "seed": int(raw.get("seed", 1)),
    }
    if result["bootstrap_resamples"] < 0 or result["null_permutations"] < 0:
        raise ValueError("bootstrap_resamples and null_permutations cannot be negative")
    if not 0 < result["confidence"] < 1:
        raise ValueError("analysis confidence must be between zero and one")
    return result


def _requested_statistics(recipe: Mapping[str, Any]) -> tuple[str, ...]:
    raw = recipe.get("estimators", ())
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("analysis estimators must be a list")
    excluded = {
        "effective_affinity",
        "kinetic_compliance",
        "episode_current",
        "cell_current",
    }
    return tuple(
        ESTIMATOR_ALIASES.get(str(name), str(name))
        for name in raw
        if str(name) not in excluded
    )


def _round_events(
    runs: tuple[DiscoveredRun, ...], scientific_frames: Mapping[str, pd.DataFrame]
) -> list[Any]:
    """Use established adapters and qualify execution-local identities study-wide."""

    events: list[Any] = []
    loaded_cells: set[str] = set()
    loaded_indices: set[tuple[int, str]] = set()
    for run in runs:
        identity = (run.entry.array_index, str(run.path))
        if identity in loaded_indices:
            continue
        loaded_indices.add(identity)
        local: list[Any] = []
        try:
            if run.game_type == "relational_imitation_round_feedback":
                from mas_cc.games.relational_reasoning.imitation_round_feedback.analysis import (
                    read_relational_round_records,
                )

                local = read_relational_round_records(run.path)
            elif run.game_type == "hidden_bench_imitation_round_feedback":
                from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import (
                    read_round_records,
                )

                local = read_round_records(run.path)
        except (FileNotFoundError, ValueError):
            local = []
        prefix = f"config-{run.entry.array_index:04d}"
        for row in local:
            qualified_cell = f"{prefix}/{row.cell_id}"
            loaded_cells.add(qualified_cell)
            events.append(
                replace(
                    row,
                    cell_id=qualified_cell,
                    episode_id=f"{prefix}/{row.cell_id}/{row.episode_id}",
                )
            )

    supported_prefixes = {
        f"config-{run.entry.array_index:04d}"
        for run in runs
        if run.game_type
        in {
            "relational_imitation_round_feedback",
            "hidden_bench_imitation_round_feedback",
        }
    }
    if not supported_prefixes:
        return events

    # Results-only artifacts preserve the exact transition encoding. Feed it
    # through the existing generic round adapter rather than estimating here.
    from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import (
        adapt_round_record,
    )

    for cell_key, frame in scientific_frames.items():
        if (
            frame.empty
            or cell_key in loaded_cells
            or cell_key.split("/", 1)[0] not in supported_prefixes
        ):
            continue
        for raw in frame.to_dict(orient="records"):
            if raw.get("possible_answers") is None:
                continue
            event = dict(compact_row_to_imitation_event(raw))
            event.update(
                {
                    "record_type": "imitation_round_feedback",
                    "round_index": int(raw["interaction_index"]),
                }
            )
            adapted = adapt_round_record(
                event,
                cell_id=cell_key,
                episode_id=f"{cell_key}/{raw['episode_id']}",
            )
            events.append(adapted)
    return events


def _round_events_from_canonical(frame: pd.DataFrame) -> list[Any]:
    """Rebuild estimator inputs exclusively from retained canonical rounds."""

    if frame.empty:
        return []
    from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import (
        adapt_round_record,
    )
    from mas_cc.games.relational_reasoning.imitation_round_feedback.analysis import (
        adapt_relational_round_record,
    )

    events: list[Any] = []
    for raw in frame.to_dict(orient="records"):
        record = dict(raw)
        for key, value in tuple(record.items()):
            if isinstance(value, str) and value[:1] in {"[", "{"}:
                try:
                    record[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass
            elif isinstance(value, float) and math.isnan(value):
                record[key] = None
        cell_id = str(record["cell_id"])
        episode_id = str(record["episode_id"])
        record_type = record.get("record_type")
        if record_type == "relational_imitation_round_feedback":
            events.append(
                adapt_relational_round_record(
                    record, cell_id=cell_id, episode_id=episode_id
                )
            )
        elif record_type in {None, "imitation_round_feedback"} and all(
            key in record
            for key in (
                "occupation_counts_before",
                "occupation_counts_after",
                "target_count_before",
                "target_count_after",
                "truth_count_before",
                "truth_count_after",
            )
        ):
            events.append(
                adapt_round_record(record, cell_id=cell_id, episode_id=episode_id)
            )
    return events


def _support_status(row: Mapping[str, Any]) -> str:
    def finite(value: Any, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if math.isfinite(number) else default

    actions = int(finite(row.get("number_of_actions_observed"), 0.0))
    dual = finite(row.get("round_dual_action_state_fraction"), 0.0)
    sparse = finite(row.get("round_singleton_fraction"), 1.0)
    if actions < 2 or dual == 0:
        return "unsupported"
    if dual < 0.25 or sparse > 0.5:
        return "limited"
    return "adequate"


def _run_information_group(payload: tuple[Any, ...]) -> tuple[list[Any], list[Any]]:
    """Process-safe call into the established estimator engine."""

    rows, statistics, settings, seed = payload
    from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import (
        round_information_analysis,
    )

    return round_information_analysis(
        rows,
        statistics=statistics,
        bootstrap_resamples=int(settings["bootstrap_resamples"]),
        null_permutations=int(settings["null_permutations"]),
        confidence=float(settings["confidence"]),
        seed=seed,
    )


def _write_analysis_progress(
    path: Path, *, completed: int, total: int, active: str | None = None
) -> None:
    _write_json(
        path,
        {
            "stage": "information_resampling",
            "completed_groups": completed,
            "total_groups": total,
            "remaining_groups": total - completed,
            "active_group": active,
            "updated_at": _now(),
        },
    )


def _information_tables(
    study_id: str,
    events: Sequence[Any],
    statistics: tuple[str, ...],
    settings: Mapping[str, Any],
    analysis_hash: str,
    source_run_ids: Mapping[str, str],
    *,
    progress_path: Path | None = None,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import (
        ROUND_ANALYSIS_STATISTICS,
    )

    unknown = sorted(set(statistics) - set(ROUND_ANALYSIS_STATISTICS))
    if unknown:
        raise ValueError(
            "unknown study information estimator(s): " + ", ".join(unknown)
        )
    if not events or not statistics:
        return (
            pd.DataFrame(columns=PRIMARY_COLUMNS),
            pd.DataFrame(columns=["study_id", "cell_id", "metric", "analysis_hash"]),
        )

    grouped: dict[str, list[Any]] = {}
    for event in events:
        grouped.setdefault(str(event.cell_id), []).append(event)
    estimates: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    confidence = float(settings["confidence"])
    ordered = sorted(grouped.items())
    completed_results: dict[str, tuple[list[Any], list[Any]]] = {}
    pending: dict[str, tuple[Any, ...]] = {}
    for cell_id, rows in ordered:
        group_seed = int(settings["seed"])
        group_seed += int(hashlib.sha256(cell_id.encode("utf-8")).hexdigest()[:8], 16)
        pending[cell_id] = (rows, statistics, dict(settings), group_seed)

    total = len(ordered)
    done = len(completed_results)
    if progress_path is not None:
        _write_analysis_progress(progress_path, completed=done, total=total)
    if pending:
        with ProcessPoolExecutor(
            max_workers=max(1, min(workers, len(pending)))
        ) as pool:
            futures = {
                pool.submit(_run_information_group, payload): cell_id
                for cell_id, payload in pending.items()
            }
            for future in as_completed(futures):
                cell_id = futures[future]
                value = future.result()
                completed_results[cell_id] = value
                done += 1
                print(
                    f"[analysis] information groups {done}/{total}: {cell_id}",
                    flush=True,
                )
                if progress_path is not None:
                    _write_analysis_progress(
                        progress_path, completed=done, total=total, active=cell_id
                    )

    for cell_id, rows in ordered:
        result_rows, null_rows = completed_results[cell_id]
        source_run_id = source_run_ids.get(
            cell_id.split("/", 1)[0], cell_id.split("/", 1)[0]
        )
        null_by_metric: dict[str, list[float]] = {}
        for item in null_rows:
            value = float(item["estimate"])
            null_by_metric.setdefault(str(item["statistic"]), []).append(value)
        for item in result_rows:
            metric = str(item["statistic"])
            finite = [
                value
                for value in null_by_metric.get(metric, ())
                if math.isfinite(value)
            ]
            estimate = float(item["estimate"])
            p_value = (
                math.nan
                if not finite
                else (1 + sum(value >= estimate for value in finite))
                / (len(finite) + 1)
            )
            estimates.append(
                {
                    "study_id": study_id,
                    "source_run_id": source_run_id,
                    "cell_id": cell_id,
                    "metric": metric,
                    "estimator_version": "round-feedback-v1",
                    "estimator_variant": item.get(
                        "main_estimator_variant", item.get("estimator_variant")
                    ),
                    "grouping_json": json.dumps({"cell_id": cell_id}, sort_keys=True),
                    "conditioning_json": _conditioning_json(metric),
                    "estimate": estimate,
                    "ci_low": item.get("bootstrap_ci_low"),
                    "ci_high": item.get("bootstrap_ci_high"),
                    "confidence": confidence,
                    "null_type": item.get("null_type"),
                    "null_mean": item.get("null_mean"),
                    "null_std": math.nan if not finite else float(np.std(finite)),
                    "p_value": p_value,
                    "null_permutations": len(finite),
                    "bootstrap_resamples": int(settings["bootstrap_resamples"]),
                    "n_observations": item.get("n_rounds"),
                    "n_episodes": item.get("n_episodes"),
                    "action_entropy_ceiling_bits": item.get(
                        "conditional_action_entropy_bits"
                    ),
                    "dual_action_support_fraction": item.get(
                        "round_dual_action_state_fraction"
                    ),
                    "units": item.get("units"),
                    "support_status": _support_status(item),
                    "analysis_hash": analysis_hash,
                }
            )
            support.append(
                {
                    "study_id": study_id,
                    "source_run_id": source_run_id,
                    "cell_id": cell_id,
                    "metric": metric,
                    "analysis_hash": analysis_hash,
                    **{
                        key: value
                        for key, value in item.items()
                        if key.startswith("round_")
                        or key
                        in {
                            "n_episodes",
                            "n_rounds",
                            "number_of_actions_observed",
                            "unique_population_states",
                            "unique_sensor_states",
                            "min_rounds_per_population_state",
                            "median_rounds_per_population_state",
                            "max_rounds_per_population_state",
                        }
                    },
                    "support_status": _support_status(item),
                    "action_0_count": sum(str(event.U_k) == "NO_OP" for event in rows),
                    "action_1_count": sum(
                        str(event.U_k) != "NO_OP" and event.U_k is not None
                        for event in rows
                    ),
                    "action_entropy_bits": item.get(
                        "round_controller_action_entropy",
                        item.get("conditional_action_entropy_bits"),
                    ),
                    "dual_action_support_fraction": item.get(
                        "round_dual_action_state_fraction"
                    ),
                    "occupied_conditioning_states": item.get(
                        "round_conditioning_state_count",
                        item.get("unique_population_states"),
                    ),
                    "singleton_fraction": item.get("round_singleton_fraction"),
                    "sparse_state_fraction": item.get("round_singleton_fraction"),
                }
            )
    return pd.DataFrame(estimates, columns=PRIMARY_COLUMNS), pd.DataFrame(support)


def _state_local_primary(
    study_id: str,
    events: Sequence[Any],
    recipe: Mapping[str, Any],
    analysis_hash: str,
    source_run_ids: Mapping[str, str],
) -> pd.DataFrame:
    """Estimate requested state slices through the authoritative estimator."""

    requested = recipe.get("state_local", ())
    if isinstance(requested, (str, bytes)) or not isinstance(requested, Sequence):
        raise ValueError("analysis state_local must be a list")
    coordinates = {
        "x": None,
        "x_phi": "conditioning_phi_bin",
        "x_kappa": "conditioning_kappa_bin",
    }
    unknown = sorted(set(map(str, requested)) - set(coordinates))
    if unknown:
        raise ValueError(
            "unknown state-local analysis resolution(s): " + ", ".join(unknown)
        )
    if not requested:
        return pd.DataFrame(columns=PRIMARY_COLUMNS)
    raw_x_bins = recipe.get("state_local_x_bins")
    if raw_x_bins is None:
        x_bins = None
    elif (
        isinstance(raw_x_bins, bool)
        or not isinstance(raw_x_bins, int)
        or raw_x_bins < 1
    ):
        raise ValueError("analysis state_local_x_bins must be a positive integer")
    else:
        x_bins = int(raw_x_bins)
    from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import (
        round_information_analysis,
    )

    by_cell: dict[str, list[Any]] = {}
    for event in events:
        by_cell.setdefault(str(event.cell_id), []).append(event)
    rows: list[dict[str, Any]] = []
    statistics = (
        "round_target_actuation_cmi",
        "round_target_information_fraction",
        "round_target_signed_actuation",
        # The canonical chi. `round_target_signed_actuation` stays beside it as
        # the legacy magnetization diagnostic; the two differ by K/(K-1) and
        # only this one may enter eta_IR.
        "round_target_susceptibility",
    )
    for cell_id, cell_rows in sorted(by_cell.items()):
        for resolution in map(str, requested):
            extra = coordinates[resolution]
            slices: dict[tuple[Any, ...], list[Any]] = {}
            for event in cell_rows:
                x = event.event.get("target_count_before")
                value = None if extra is None else event.event.get(extra)
                if x is None or (extra is not None and value is None):
                    continue
                if resolution == "x" and x_bins is not None:
                    population = int(event.event.get("N") or sum(event.N_k))
                    fraction = float(x) / population
                    bin_index = min(int(fraction * x_bins), x_bins - 1)
                    slice_key = (bin_index,)
                else:
                    slice_key = (x,) if extra is None else (x, value)
                slices.setdefault(slice_key, []).append(event)
            for slice_values, sample in sorted(
                slices.items(), key=lambda item: str(item[0])
            ):
                estimates, _ = round_information_analysis(
                    sample,
                    statistics=statistics,
                    bootstrap_resamples=0,
                    null_permutations=0,
                    confidence=0.95,
                    seed=1,
                )
                grouping = {"cell_id": cell_id, "resolution": resolution}
                if resolution == "x" and x_bins is not None:
                    bin_index = int(slice_values[0])
                    grouping.update(
                        {
                            "target_fraction_bin_index": bin_index,
                            "target_fraction_bin_lower": bin_index / x_bins,
                            "target_fraction_bin_upper": (bin_index + 1) / x_bins,
                            "target_fraction_bin_center": (bin_index + 0.5) / x_bins,
                            "target_fraction_bin_count": x_bins,
                        }
                    )
                else:
                    grouping["target_count_before"] = slice_values[0]
                if extra is not None and not (resolution == "x" and x_bins is not None):
                    grouping[extra] = slice_values[1]
                for item in estimates:
                    rows.append(
                        {
                            "study_id": study_id,
                            "source_run_id": source_run_ids.get(
                                cell_id.split("/", 1)[0], cell_id.split("/", 1)[0]
                            ),
                            "cell_id": cell_id,
                            "metric": str(item["statistic"]),
                            "estimator_version": "round-feedback-v1",
                            "estimator_variant": item.get(
                                "main_estimator_variant", item.get("estimator_variant")
                            ),
                            "grouping_json": json.dumps(grouping, sort_keys=True),
                            "conditioning_json": _conditioning_json(
                                str(item["statistic"])
                            ),
                            "estimate": item["estimate"],
                            "ci_low": math.nan,
                            "ci_high": math.nan,
                            "confidence": 0.95,
                            "null_type": None,
                            "null_mean": math.nan,
                            "null_std": math.nan,
                            "p_value": math.nan,
                            "null_permutations": 0,
                            "bootstrap_resamples": 0,
                            "n_observations": item.get("n_rounds"),
                            "n_episodes": item.get("n_episodes"),
                            "action_entropy_ceiling_bits": item.get(
                                "conditional_action_entropy_bits"
                            ),
                            "dual_action_support_fraction": item.get(
                                "round_dual_action_state_fraction"
                            ),
                            "units": item.get("units"),
                            "support_status": _support_status(item),
                            "analysis_hash": analysis_hash,
                            **grouping,
                        }
                    )
    return pd.DataFrame(rows)


def _current_primary(
    study_id: str,
    events: Sequence[Any],
    requested: set[str],
    settings: Mapping[str, Any],
    analysis_hash: str,
    source_run_ids: Mapping[str, str],
) -> pd.DataFrame:
    if not events or not requested.intersection({"episode_current", "cell_current"}):
        return pd.DataFrame(columns=PRIMARY_COLUMNS)
    from mas_cc.games.relational_reasoning.imitation_round_feedback.current import (
        current_cell_summary,
    )

    grouped: dict[str, list[Any]] = {}
    for event in events:
        grouped.setdefault(str(event.cell_id), []).append(event)
    rows: list[dict[str, Any]] = []
    confidence = float(settings["confidence"])
    for index, (cell_id, group) in enumerate(sorted(grouped.items())):
        try:
            summary, episodes = current_cell_summary(
                group,
                bootstrap_resamples=int(settings["bootstrap_resamples"]),
                confidence=confidence,
                seed=int(settings["seed"]) + index,
                theoretical_reference="none",
            )
        except (KeyError, TypeError, ValueError):
            continue
        source_run_id = source_run_ids.get(
            cell_id.split("/", 1)[0], cell_id.split("/", 1)[0]
        )
        if "episode_current" in requested:
            for episode in episodes:
                rows.append(
                    {
                        "study_id": study_id,
                        "source_run_id": source_run_id,
                        "cell_id": cell_id,
                        "metric": "episode_current",
                        "estimator_version": "relational-current-v1",
                        "estimator_variant": "terminal_target_count_difference",
                        "grouping_json": json.dumps(
                            {"episode_id": episode["episode_id"]}, sort_keys=True
                        ),
                        "conditioning_json": "{}",
                        "estimate": episode["episode_current"],
                        "ci_low": math.nan,
                        "ci_high": math.nan,
                        "confidence": confidence,
                        "null_type": None,
                        "null_mean": math.nan,
                        "null_std": math.nan,
                        "p_value": math.nan,
                        "null_permutations": 0,
                        "bootstrap_resamples": 0,
                        "n_observations": episode["K"],
                        "n_episodes": 1,
                        "units": "target_count",
                        "support_status": "descriptive_only",
                        "analysis_hash": analysis_hash,
                    }
                )
        if "cell_current" in requested:
            rows.append(
                {
                    "study_id": study_id,
                    "source_run_id": source_run_id,
                    "cell_id": cell_id,
                    "metric": "cell_current",
                    "estimator_version": "relational-current-v1",
                    "estimator_variant": "mean_episode_current",
                    "grouping_json": json.dumps({"cell_id": cell_id}, sort_keys=True),
                    "conditioning_json": "{}",
                    "estimate": summary["current_mean_empirical"],
                    "ci_low": summary.get("current_mean_empirical_bootstrap_ci_low"),
                    "ci_high": summary.get("current_mean_empirical_bootstrap_ci_high"),
                    "confidence": confidence,
                    "null_type": None,
                    "null_mean": math.nan,
                    "null_std": math.nan,
                    "p_value": math.nan,
                    "null_permutations": 0,
                    "bootstrap_resamples": int(settings["bootstrap_resamples"]),
                    "n_observations": sum(int(episode["K"]) for episode in episodes),
                    "n_episodes": len(episodes),
                    "units": "target_count",
                    "support_status": summary.get(
                        "current_precision_support", "unknown"
                    ),
                    "analysis_hash": analysis_hash,
                }
            )
    return pd.DataFrame(rows, columns=PRIMARY_COLUMNS)


def _affinity_primary(
    study_id: str,
    micro_slots: pd.DataFrame,
    events: Sequence[Any],
    requested: set[str],
    settings: Mapping[str, Any],
    analysis_hash: str,
    source_run_ids: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = requested.intersection({"effective_affinity", "kinetic_compliance"})
    if not names or micro_slots.empty:
        return pd.DataFrame(columns=PRIMARY_COLUMNS), pd.DataFrame()
    from mas_cc.analysis.effective_affinity import effective_affinity_analysis

    target_by_round: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for event in events:
        local_episode = str(
            event.event.get("episode_id", event.episode_id.rsplit("/", 1)[-1])
        )
        target_by_round[(str(event.cell_id), local_episode, int(event.round_index))] = (
            event.event
        )
    records = []
    for row in micro_slots.to_dict(orient="records"):
        try:
            key = (str(row["cell_id"]), str(row["episode_id"]), int(row["round_index"]))
        except (KeyError, TypeError, ValueError):
            continue
        round_event = target_by_round.get(key, {})
        records.append(
            {
                **row,
                "analysis_target": row.get("analysis_target")
                or row.get("round_controller_target")
                or round_event.get("analysis_target")
                or round_event.get("controller_target"),
                "round_controller_action": row.get("round_controller_action")
                or round_event.get("controller_action"),
            }
        )
    summaries = effective_affinity_analysis(
        records,
        bootstrap_resamples=int(settings["bootstrap_resamples"]),
        confidence=float(settings["confidence"]),
        seed=int(settings["seed"]),
    )
    primary: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    for summary in summaries:
        cell_id = str(summary["cell_id"])
        source_run_id = source_run_ids.get(
            cell_id.split("/", 1)[0], cell_id.split("/", 1)[0]
        )
        status = (
            "unsupported"
            if not summary["n_plus"] or not summary["n_minus"]
            else "limited"
            if int(summary["n_episodes"]) < 10
            else "adequate"
        )
        for metric in sorted(names):
            primary.append(
                {
                    "study_id": study_id,
                    "source_run_id": source_run_id,
                    "cell_id": cell_id,
                    "metric": metric,
                    "estimator_version": "study05-effective-affinity-v1",
                    "estimator_variant": (
                        "raw_transition_rate_ratio"
                        if metric == "effective_affinity"
                        else "raw_transition_activity"
                    ),
                    "grouping_json": json.dumps({"cell_id": cell_id}, sort_keys=True),
                    "conditioning_json": json.dumps(
                        {"controlled_slot": True, "round_action": "ADVOCATE_TARGET"},
                        sort_keys=True,
                    ),
                    "estimate": summary[metric],
                    "ci_low": summary[f"{metric}_ci_low"],
                    "ci_high": summary[f"{metric}_ci_high"],
                    "confidence": float(settings["confidence"]),
                    "null_type": None,
                    "null_mean": math.nan,
                    "null_std": math.nan,
                    "p_value": math.nan,
                    "null_permutations": 0,
                    "bootstrap_resamples": int(settings["bootstrap_resamples"]),
                    "n_observations": summary["n_observations"],
                    "n_episodes": summary["n_episodes"],
                    "units": (
                        "nats" if metric == "effective_affinity" else "probability"
                    ),
                    "support_status": status,
                    "p_plus": summary["p_plus"],
                    "p_minus": summary["p_minus"],
                    "non_target_exposures": summary["n_plus"],
                    "non_target_to_target": summary["k_plus"],
                    "target_exposures": summary["n_minus"],
                    "target_to_non_target": summary["k_minus"],
                    "analysis_hash": analysis_hash,
                }
            )
        support.append(
            {
                "study_id": study_id,
                "source_run_id": source_run_id,
                "cell_id": cell_id,
                "metric": "effective_affinity_and_kinetic_compliance",
                "n_episodes": summary["n_episodes"],
                "n_observations": summary["n_observations"],
                "non_target_exposures": summary["n_plus"],
                "non_target_to_target": summary["k_plus"],
                "target_exposures": summary["n_minus"],
                "target_to_non_target": summary["k_minus"],
                "p_plus": summary["p_plus"],
                "p_minus": summary["p_minus"],
                "support_status": status,
                "analysis_hash": analysis_hash,
            }
        )
    return pd.DataFrame(primary, columns=PRIMARY_COLUMNS), pd.DataFrame(support)


def _ingest_existing_information(
    study_id: str,
    runs: tuple[DiscoveredRun, ...],
    analysis_hash: str,
    settings: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimates: list[pd.DataFrame] = []
    support: list[pd.DataFrame] = []
    for run in runs:
        candidates = sorted(run.path.glob("*_analysis/round_information_estimates.csv"))
        candidates.extend(
            sorted((run.path / "analysis").glob("round_information_estimates.csv"))
        )
        if not candidates:
            continue
        estimate_path = candidates[0]
        prefix = f"config-{run.entry.array_index:04d}"
        raw = pd.read_csv(estimate_path)
        raw = raw[raw["cell_id"].astype(str) != "pooled"].copy()
        normalized = pd.DataFrame(
            {
                "study_id": study_id,
                "source_run_id": run.run_id,
                "cell_id": raw["cell_id"].map(lambda value: f"{prefix}/{value}"),
                "metric": raw["statistic"],
                "estimator_version": "round-feedback-v1",
                "estimator_variant": raw.get(
                    "main_estimator_variant", raw.get("estimator_variant")
                ),
                "grouping_json": raw["cell_id"].map(
                    lambda value: json.dumps({"cell_id": str(value)})
                ),
                "conditioning_json": "{}",
                "estimate": raw["estimate"],
                "ci_low": raw.get("bootstrap_ci_low"),
                "ci_high": raw.get("bootstrap_ci_high"),
                "confidence": np.nan,
                "null_type": raw.get("null_type"),
                "null_mean": raw.get("null_mean"),
                "null_std": np.nan,
                "p_value": np.nan,
                "null_permutations": int(settings["null_permutations"]),
                "bootstrap_resamples": int(settings["bootstrap_resamples"]),
                "n_observations": raw.get("n_rounds"),
                "n_episodes": raw.get("n_episodes"),
                "units": raw.get("units"),
                "support_status": raw.apply(lambda row: _support_status(row), axis=1),
                "analysis_hash": analysis_hash,
            }
        )
        estimates.append(normalized)
        support_path = estimate_path.with_name("round_support_diagnostics.csv")
        if support_path.is_file():
            item = pd.read_csv(support_path)
            item = item[item["cell_id"].astype(str) != "pooled"].copy()
            item.insert(0, "study_id", study_id)
            item.insert(1, "source_run_id", run.run_id)
            item["cell_id"] = item["cell_id"].map(lambda value: f"{prefix}/{value}")
            item["analysis_hash"] = analysis_hash
            support.append(item)
    return (
        (
            pd.concat(estimates, ignore_index=True)
            if estimates
            else pd.DataFrame(columns=PRIMARY_COLUMNS)
        ),
        pd.concat(support, ignore_index=True) if support else pd.DataFrame(),
    )


DERIVED_COLUMNS = (
    "study_id",
    "source_run_id",
    "cell_id",
    "metric",
    "grouping_json",
    "conditioning_json",
    "estimate",
    "ci_low",
    "ci_high",
    "confidence",
    "units",
    "dependencies_json",
    "support_status",
    "analysis_hash",
)

SINGLE_AFFINITY_DERIVED = (
    "round_target_susceptibility",
    "eta_ir",
    "target_sensing_information_nats",
    "controlled_current",
    "affinity_weighted_current_nats",
    "thermodynamic_control_expenditure_nats",
    "eta_th",
)
"""Naming any one of these in `derived:` requests the whole coupled family.

They are ingredients of one estimator, not seven independent numbers: the same
state-matched `chi(n)`, the same action weights and the same episode bootstrap
feed all of them.  Emitting only a subset would let a reader join an `eta_th`
to a `J_c` that came from a different resample.  The recipe still lists them
individually so the intent is readable."""


def _eta_ir_state_local(
    study_id: str,
    requested: Sequence[Any],
    estimates: pd.DataFrame,
    events: Sequence[Any],
    analysis_hash: str,
) -> pd.DataFrame:
    """State-resolved `eta_IR(n)` - the theory's own state-local ratio.

    Repaired relative to the first implementation: the response is now
    `round_target_susceptibility` (target fraction), not
    `round_target_signed_actuation` (aligned magnetization).  On the K=3 tasks
    the old numerator was inflated by `(3/2)^2 = 2.25`, because the Pinsker
    bound is stated in the fraction coordinate.  The fix routes through the
    fraction estimator rather than dividing by 2.25 afterwards, so it is
    correct for any K.

    This is the state-resolved surface the heatmaps read.  The headline
    per-cell number is the occupancy-weighted `eta_ir` from
    :func:`_build_eta_ir`, which is a ratio of sums over these same states.
    """

    columns = list(DERIVED_COLUMNS)
    names = {str(value) for value in requested}
    if "eta_ir" not in names or estimates.empty:
        return pd.DataFrame(columns=columns)
    event_groups: dict[str, list[Any]] = {}
    for event in events:
        event_groups.setdefault(str(event.cell_id), []).append(event)
    rows: list[dict[str, Any]] = []

    cmi = estimates[estimates["metric"] == "round_target_actuation_cmi"].copy()
    response = estimates[estimates["metric"] == "round_target_susceptibility"].copy()
    for dependency, frame in (("CMI", cmi), ("signed response", response)):
        duplicates = frame.duplicated(list(ETA_IR_JOIN_KEYS), keep=False)
        if duplicates.any():
            raise ValueError(
                f"eta_ir has duplicate {dependency} rows at one state/grouping resolution"
            )
    joined = cmi.merge(
        response,
        on=list(ETA_IR_JOIN_KEYS),
        how="inner",
        suffixes=("_cmi", "_response"),
        validate="one_to_one",
    )
    for item in joined.to_dict(orient="records"):
        cell_id = str(item["cell_id"])
        group = event_groups.get(cell_id, [])
        try:
            grouping = json.loads(str(item["grouping_json"]))
        except (TypeError, json.JSONDecodeError):
            grouping = {}
        if "target_count_before" in grouping:
            group = [
                event
                for event in group
                if event.event.get("target_count_before")
                == grouping["target_count_before"]
            ]
        if "target_fraction_bin_index" in grouping:
            bins = int(grouping["target_fraction_bin_count"])
            selected_bin = int(grouping["target_fraction_bin_index"])
            group = [
                event
                for event in group
                if min(
                    int(
                        (float(event.event.get("target_count_before")) / sum(event.N_k))
                        * bins
                    ),
                    bins - 1,
                )
                == selected_bin
            ]
        for key in ("conditioning_phi_bin", "conditioning_kappa_bin"):
            if key in grouping:
                group = [
                    event for event in group if event.event.get(key) == grouping[key]
                ]
        controlled = [
            event
            for event in group
            if str(event.U_k) in {"ADVOCATE_Z", "ADVOCATE_TARGET", "NO_OP"}
        ]
        if not controlled:
            continue
        a = sum(str(event.U_k) != "NO_OP" for event in controlled) / len(controlled)
        transfer = float(item["estimate_cmi"])
        chi = float(item["estimate_response"])
        value = (
            math.nan
            if transfer <= 0
            else 2 * a * (1 - a) * chi * chi / (math.log(2) * transfer)
        )
        rows.append(
            {
                "study_id": study_id,
                "source_run_id": item["source_run_id"],
                "cell_id": cell_id,
                "metric": "eta_ir_state_local",
                "grouping_json": item["grouping_json"],
                "conditioning_json": item["conditioning_json"],
                "estimate": value,
                "ci_low": math.nan,
                "ci_high": math.nan,
                "confidence": math.nan,
                "units": "dimensionless",
                "dependencies_json": json.dumps(
                    {
                        "join_keys": list(ETA_IR_JOIN_KEYS),
                        "metrics": [
                            "round_target_actuation_cmi",
                            "round_target_susceptibility",
                            "action_frequency",
                        ],
                        "response_coordinate": "target_fraction",
                    },
                    sort_keys=True,
                ),
                "support_status": str(item["support_status_cmi"]),
                "n_observations": len(controlled),
                "n_episodes": len({str(event.episode_id) for event in controlled}),
                "analysis_hash": analysis_hash,
                **grouping,
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)


def _micro_rows_by_cell(micro_slots: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Micro-slot records grouped by study cell, with the round's action attached.

    `h` and `gamma` are read off individual controlled vote slots, but whether
    a slot was inside an advocating round is a property of the round record.
    The two are joined here, once, so every builder below sees the same table.
    """

    if micro_slots.empty:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in micro_slots.to_dict(orient="records"):
        cell_id = str(row.get("cell_id", "run"))
        grouped.setdefault(cell_id, []).append(row)
    return grouped


def _single_affinity_by_cell(
    events: Sequence[Any],
    micro_slots: pd.DataFrame,
    settings: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Run the whole coupled estimator once per cell.

    One call, not four: the bootstrap has to resample episodes once and
    recompute every ingredient inside that resample, so `eta_ir`, `J_c`,
    `I_sens` and `eta_th` cannot be estimated in separate passes without
    losing the correlations their intervals depend on.
    """

    from mas_cc.analysis.single_affinity import single_affinity_analysis

    by_cell: dict[str, list[Any]] = {}
    for event in events:
        by_cell.setdefault(str(event.cell_id), []).append(event)
    micro_by_cell = _micro_rows_by_cell(micro_slots)
    round_targets: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for event in events:
        local = str(
            event.event.get("episode_id", str(event.episode_id).rsplit("/", 1)[-1])
        )
        round_targets[(str(event.cell_id), local, int(event.round_index))] = event.event

    result: dict[str, dict[str, Any]] = {}
    for index, (cell_id, rows) in enumerate(sorted(by_cell.items())):
        micro: list[dict[str, Any]] = []
        for row in micro_by_cell.get(cell_id, ()):
            try:
                key = (cell_id, str(row["episode_id"]), int(row["round_index"]))
            except (KeyError, TypeError, ValueError):
                continue
            round_event = round_targets.get(key, {})
            micro.append(
                {
                    **row,
                    "analysis_target": row.get("analysis_target")
                    or row.get("round_controller_target")
                    or round_event.get("analysis_target")
                    or round_event.get("controller_target"),
                    "round_controller_action": row.get("round_controller_action")
                    or round_event.get("controller_action"),
                }
            )
        bundle = single_affinity_analysis(
            rows,
            micro,
            bootstrap_resamples=int(settings["bootstrap_resamples"]),
            confidence=float(settings["confidence"]),
            seed=int(settings["seed"]) + index,
        )
        bundle["_rows"], bundle["_micro"] = rows, micro
        result[cell_id] = bundle
    return result


def _single_affinity_theory_comparison(
    study_id: str,
    bundles: Mapping[str, Mapping[str, Any]],
    source_run_ids: Mapping[str, str],
    analysis_hash: str,
) -> pd.DataFrame:
    """Empirical and exact single-affinity values side by side, per cell.

    The theory column is evaluated on the cell's own empirical occupancy, so a
    gap between the columns means the LLM population departed from the
    single-affinity kernel - not that the two were averaged over different
    states.  A cell that does not pin down one protocol or one finite affinity
    gets a row saying why instead of a table of NaNs.  The matched q-voter
    reference is a separate classical null and never appears here.
    """

    from mas_cc.analysis.single_affinity import PROVENANCE, theory_comparison

    rows: list[dict[str, Any]] = []
    for cell_id, bundle in sorted(bundles.items()):
        prefix = cell_id.split("/", 1)[0]
        comparison = theory_comparison(
            bundle.get("_rows", ()), bundle.get("_micro", ())
        )
        common = {
            "study_id": study_id,
            "source_run_id": source_run_ids.get(prefix, prefix),
            "cell_id": cell_id,
            "analysis_hash": analysis_hash,
            **PROVENANCE,
        }
        if not comparison["available"]:
            rows.append(
                {
                    **common,
                    "quantity": None,
                    "units": None,
                    "empirical": math.nan,
                    "single_affinity_theory": math.nan,
                    "residual": math.nan,
                    "reference": "single_affinity_revised",
                    "available": False,
                    "reason": comparison["reason"],
                }
            )
            continue
        for item in comparison["rows"]:
            rows.append({**common, **item, "available": True, "reason": None})
    return pd.DataFrame(rows)


_SINGLE_AFFINITY_UNITS: Mapping[str, str] = {
    "susceptibility_occupancy_weighted": "target_fraction_per_cycle",
    "eta_ir": "dimensionless",
    "eta_ir_pinsker_numerator_bits": "bits",
    "eta_ir_denominator_T_bits": "bits",
    "target_sensing_information_nats": "nats_per_cycle",
    "target_sensing_information_horizon_nats": "nats",
    "controlled_current": "target_count_per_cycle",
    "controlled_current_horizon": "target_count",
    "affinity_weighted_current_nats": "nats_per_horizon",
    "thermodynamic_control_expenditure_nats": "nats_per_horizon",
    "eta_th": "dimensionless",
    "eta_th_signed": "dimensionless",
    "eta_th_bounded": "dimensionless",
}
"""Every emitted quantity's units, in one table, so a bits/nats mix-up has to
be made here in the open rather than inferred from a column name."""

_SINGLE_AFFINITY_AUDIT = (
    "chi_state_count",
    "chi_identified_state_count",
    "chi_dual_action_state_fraction",
    "chi_dual_action_event_fraction",
    "chi_identified_occupancy_mass",
    "eta_ir_identified_occupancy_mass",
    "eta_ir_support_mass",
    "eta_ir_valid",
    "eta_th_identified_occupancy_mass",
    "eta_th_target_directed",
    "eta_th_valid",
    "eta_th_signed",
    "eta_th_bounded",
    "eta_th_numeric_defined",
    "eta_th_has_bounded_interpretation",
    "eta_th_undefined_reason",
    "eta_th_signed_ci_low",
    "eta_th_signed_ci_high",
    "target_sensing_valid",
    "affinity_valid",
    "effective_affinity",
    "kinetic_compliance",
    "p_plus",
    "p_minus",
    "plus_transitions",
    "plus_eligible",
    "minus_transitions",
    "minus_eligible",
    "controlled_current_rounds",
    "controlled_current",
    "controlled_current_horizon",
    "affinity_weighted_current_nats",
    "target_sensing_information_horizon_nats",
    "thermodynamic_control_expenditure_nats",
    "target_sensing_information_rounds",
    "n_rounds",
    "n_episodes",
    "n_micro_slots",
)
"""Support, identifiability and validity carried on every row of the family.

Repeated on each row on purpose: a `eta_th` read out of the table on its own
still says how much occupancy mass it was identified on and whether the
bounded reading applied."""


def _single_affinity_rows(
    study_id: str,
    metrics: Sequence[str],
    bundles: Mapping[str, Mapping[str, Any]],
    source_run_ids: Mapping[str, str],
    dependencies: Mapping[str, Sequence[str]],
    analysis_hash: str,
) -> list[dict[str, Any]]:
    """One long-form derived row per (cell, quantity), with its interval."""

    from mas_cc.analysis.single_affinity import PROVENANCE

    rows: list[dict[str, Any]] = []
    for cell_id, bundle in sorted(bundles.items()):
        prefix = cell_id.split("/", 1)[0]
        audit = {
            key: bundle.get(key) for key in _SINGLE_AFFINITY_AUDIT if key in bundle
        }
        # `_rows`/`_micro` are handles for the theory comparison, not data.
        for metric in metrics:
            rows.append(
                {
                    "study_id": study_id,
                    "source_run_id": source_run_ids.get(prefix, prefix),
                    "cell_id": cell_id,
                    "metric": metric,
                    "grouping_json": json.dumps({"cell_id": cell_id}, sort_keys=True),
                    "conditioning_json": json.dumps(
                        {"state": ["target_count_before"]}, sort_keys=True
                    ),
                    "estimate": bundle.get(metric, math.nan),
                    "ci_low": bundle.get(f"{metric}_ci_low", math.nan),
                    "ci_high": bundle.get(f"{metric}_ci_high", math.nan),
                    "confidence": bundle.get("confidence", math.nan),
                    "units": _SINGLE_AFFINITY_UNITS.get(metric, "dimensionless"),
                    "dependencies_json": json.dumps(
                        {"metrics": list(dependencies.get(metric, ()))}, sort_keys=True
                    ),
                    "support_status": _single_affinity_support(metric, bundle),
                    "analysis_hash": analysis_hash,
                    **audit,
                    **PROVENANCE,
                }
            )
    return rows


def _single_affinity_support(metric: str, bundle: Mapping[str, Any]) -> str:
    """`adequate` / `limited` / `unsupported` for one quantity of the family.

    A quantity is unsupported when its own validity flag is false - no
    identified state, no positive information denominator, or a transition
    direction that never fired - and limited when it is identified on less
    than a quarter of the visited occupancy mass.
    """

    def flag(name: str) -> bool:
        return bool(bundle.get(name, False))

    if metric.startswith("eta_ir") and not flag("eta_ir_valid"):
        return "unsupported"
    if metric == "eta_th_bounded" and not flag("eta_th_has_bounded_interpretation"):
        return "unsupported"
    if metric in {"eta_th", "eta_th_signed"} and not flag("eta_th_numeric_defined"):
        return "unsupported"
    if metric in {
        "affinity_weighted_current_nats",
        "thermodynamic_control_expenditure_nats",
    } and not (flag("affinity_valid") and flag("controlled_current_valid")):
        return "unsupported"
    if metric.startswith("controlled_current") and not flag("controlled_current_valid"):
        return "unsupported"
    if metric.startswith("target_sensing"):
        # Sensing needs an occupancy and the known sensor kernel, nothing from
        # the response - so it is not gated on chi's identifiability below.
        return "adequate" if flag("target_sensing_valid") else "unsupported"
    mass = bundle.get("chi_identified_occupancy_mass")
    try:
        mass = float(mass)
    except (TypeError, ValueError):
        return "limited"
    if not math.isfinite(mass) or mass <= 0.0:
        return "unsupported"
    return "adequate" if mass >= 0.25 else "limited"


def _build_susceptibility(study_id, bundles, source_run_ids, analysis_hash):
    """Occupancy-weighted `chi`, as a summary of the state-resolved estimator."""

    return _single_affinity_rows(
        study_id,
        ("susceptibility_occupancy_weighted",),
        bundles,
        source_run_ids,
        {"susceptibility_occupancy_weighted": ("round_target_susceptibility",)},
        analysis_hash,
    )


def _build_eta_ir(study_id, bundles, source_run_ids, analysis_hash):
    """Headline `eta_IR` plus the numerator and denominator it is a ratio of."""

    dependencies = {
        "eta_ir": (
            "round_target_susceptibility",
            "round_target_actuation_cmi",
            "empirical_action_weight",
            "empirical_occupancy",
        ),
        "eta_ir_pinsker_numerator_bits": (
            "round_target_susceptibility",
            "empirical_action_weight",
            "empirical_occupancy",
        ),
        "eta_ir_denominator_T_bits": ("round_target_actuation_cmi",),
    }
    return _single_affinity_rows(
        study_id,
        tuple(dependencies),
        bundles,
        source_run_ids,
        dependencies,
        analysis_hash,
    )


def _build_target_sensing_information(study_id, bundles, source_run_ids, analysis_hash):
    """Scalar-channel `I_sens` in nats, per cycle and over the horizon."""

    dependencies = {
        "target_sensing_information_nats": (
            "empirical_occupancy",
            "exact_hypergeometric_sensor_kernel",
        ),
        "target_sensing_information_horizon_nats": (
            "empirical_occupancy",
            "exact_hypergeometric_sensor_kernel",
        ),
    }
    return _single_affinity_rows(
        study_id,
        tuple(dependencies),
        bundles,
        source_run_ids,
        dependencies,
        analysis_hash,
    )


def _build_controlled_current(study_id, bundles, source_run_ids, analysis_hash):
    """Response-based `J_c`. Not `cell_current`, which is a terminal difference."""

    dependencies = {
        "controlled_current": (
            "round_target_susceptibility",
            "empirical_action_weight",
            "empirical_occupancy",
        ),
        "controlled_current_horizon": (
            "round_target_susceptibility",
            "empirical_action_weight",
            "empirical_occupancy",
        ),
    }
    return _single_affinity_rows(
        study_id,
        tuple(dependencies),
        bundles,
        source_run_ids,
        dependencies,
        analysis_hash,
    )


def _build_eta_th(study_id, bundles, source_run_ids, analysis_hash):
    """`h J_c / (h J_c + I_sens)` over the horizon, with its two terms beside it."""

    dependencies = {
        "affinity_weighted_current_nats": (
            "effective_affinity",
            "controlled_current_horizon",
        ),
        "thermodynamic_control_expenditure_nats": (
            "effective_affinity",
            "controlled_current_horizon",
            "target_sensing_information_horizon_nats",
        ),
        "eta_th": (
            "affinity_weighted_current_nats",
            "thermodynamic_control_expenditure_nats",
        ),
        "eta_th_signed": (
            "affinity_weighted_current_nats",
            "thermodynamic_control_expenditure_nats",
        ),
        "eta_th_bounded": (
            "affinity_weighted_current_nats",
            "thermodynamic_control_expenditure_nats",
        ),
    }
    return _single_affinity_rows(
        study_id,
        tuple(dependencies),
        bundles,
        source_run_ids,
        dependencies,
        analysis_hash,
    )


def _thermodynamic_efficiency_diagnostics(
    study_id: str,
    bundles: Mapping[str, Mapping[str, Any]],
    source_run_ids: Mapping[str, str],
) -> pd.DataFrame:
    """One complete, explainable thermodynamic record per scientific cell."""

    rows = []
    support_fields = (
        "affinity_valid",
        "controlled_current_valid",
        "target_sensing_valid",
        "eta_th_identified_occupancy_mass",
        "plus_transitions",
        "plus_eligible",
        "minus_transitions",
        "minus_eligible",
        "p_plus",
        "p_minus",
        "n_rounds",
        "n_episodes",
        "n_micro_slots",
    )
    for cell_id, bundle in sorted(bundles.items()):
        config_id = cell_id.split("/", 1)[0]
        rows.append(
            {
                "study_id": study_id,
                "config_id": config_id,
                "source_run_id": source_run_ids.get(config_id, config_id),
                "cell_id": cell_id,
                "h": bundle.get("effective_affinity"),
                "controlled_current": bundle.get("controlled_current_horizon"),
                "affinity_weighted_current_nats": bundle.get(
                    "affinity_weighted_current_nats"
                ),
                "target_sensing_information_horizon_nats": bundle.get(
                    "target_sensing_information_horizon_nats"
                ),
                "thermodynamic_control_expenditure_nats": bundle.get(
                    "thermodynamic_control_expenditure_nats"
                ),
                "eta_th_signed": bundle.get("eta_th_signed"),
                "eta_th_bounded": bundle.get("eta_th_bounded"),
                "eta_th_has_bounded_interpretation": bundle.get(
                    "eta_th_has_bounded_interpretation"
                ),
                "eta_th_numeric_defined": bundle.get("eta_th_numeric_defined"),
                "eta_th_undefined_reason": bundle.get("eta_th_undefined_reason"),
                "bootstrap_ci_low": bundle.get("eta_th_signed_ci_low"),
                "bootstrap_ci_high": bundle.get("eta_th_signed_ci_high"),
                **{field: bundle.get(field) for field in support_fields},
            }
        )
    return pd.DataFrame(rows)


def _thermodynamic_diagnostics_from_derived(derived: pd.DataFrame) -> pd.DataFrame:
    rows = derived[
        derived.get("metric", pd.Series(dtype=str)) == "eta_th_signed"
    ].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["config_id"] = rows["cell_id"].astype(str).str.split("/", n=1).str[0]
    rows["h"] = rows["effective_affinity"]
    rows["controlled_current"] = rows["controlled_current_horizon"]
    rows["bootstrap_ci_low"] = rows["eta_th_signed_ci_low"]
    rows["bootstrap_ci_high"] = rows["eta_th_signed_ci_high"]
    columns = [
        "study_id",
        "config_id",
        "source_run_id",
        "cell_id",
        "social_group_size",
        "controller_evidence_strategy",
        "target_semantics",
        "epistemic_persistence",
        "intervention_budget",
        "h",
        "controlled_current",
        "affinity_weighted_current_nats",
        "target_sensing_information_horizon_nats",
        "thermodynamic_control_expenditure_nats",
        "eta_th_signed",
        "eta_th_bounded",
        "eta_th_has_bounded_interpretation",
        "eta_th_numeric_defined",
        "eta_th_undefined_reason",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "support_status",
        "affinity_valid",
        "controlled_current_valid",
        "target_sensing_valid",
        "eta_th_identified_occupancy_mass",
        "plus_transitions",
        "plus_eligible",
        "minus_transitions",
        "minus_eligible",
        "p_plus",
        "p_minus",
        "n_rounds",
        "n_episodes",
        "n_micro_slots",
    ]
    return rows.reindex(columns=columns)


def _derived(
    study_id: str,
    requested: Sequence[Any],
    estimates: pd.DataFrame,
    events: Sequence[Any],
    analysis_hash: str,
    *,
    micro_slots: pd.DataFrame | None = None,
    settings: Mapping[str, Any] | None = None,
    source_run_ids: Mapping[str, str] | None = None,
    theoretical_reference: str = "single_affinity_revised",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(derived observables, single-affinity theory comparison)`."""

    names = {str(value) for value in requested}
    frames = [
        _eta_ir_state_local(study_id, requested, estimates, events, analysis_hash)
    ]
    comparison = pd.DataFrame()
    if names.intersection(SINGLE_AFFINITY_DERIVED) and events:
        bundles = _single_affinity_by_cell(
            events,
            pd.DataFrame() if micro_slots is None else micro_slots,
            settings or {"bootstrap_resamples": 0, "confidence": 0.95, "seed": 1},
        )
        run_ids = dict(source_run_ids or {})
        rows: list[dict[str, Any]] = []
        rows.extend(_build_susceptibility(study_id, bundles, run_ids, analysis_hash))
        rows.extend(_build_eta_ir(study_id, bundles, run_ids, analysis_hash))
        rows.extend(
            _build_target_sensing_information(study_id, bundles, run_ids, analysis_hash)
        )
        rows.extend(
            _build_controlled_current(study_id, bundles, run_ids, analysis_hash)
        )
        rows.extend(_build_eta_th(study_id, bundles, run_ids, analysis_hash))
        if rows:
            frames.append(pd.DataFrame(rows))
        if theoretical_reference == "single_affinity_revised":
            comparison = _single_affinity_theory_comparison(
                study_id, bundles, run_ids, analysis_hash
            )
    frames = [frame for frame in frames if not frame.empty]
    derived = (
        pd.DataFrame(columns=list(DERIVED_COLUMNS))
        if not frames
        else pd.concat(frames, ignore_index=True, sort=False)
    )
    return derived, comparison


def _attach_coordinates(frame: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "cell_id" not in frame or cells.empty:
        return frame
    drop = {
        "study_id",
        "source_config_index",
        "source_run_id",
        "source_run_path",
        "source_cell_id",
        "config_hash",
        "resolved_config_hash",
        "expected_episodes",
        "completed_episodes",
        "failed_episodes",
        "sealed",
    }
    coordinates = cells[
        [
            column
            for column in cells.columns
            if column == "cell_id" or column not in drop
        ]
    ]
    return frame.merge(
        coordinates, on="cell_id", how="left", suffixes=("", "_coordinate")
    )


def _expected_cell_coordinates(entries: Sequence[Any]) -> list[dict[str, Any]]:
    """Resolve the declared structural grid, including cells not yet run."""

    rows: list[dict[str, Any]] = []
    for entry in entries:
        source = load_run_config_or_grid(entry.config_path)
        cells = source.cells if isinstance(source, GridSpec) else ()
        if not cells:
            continue
        for cell in cells:
            config = cell.config
            record = {
                "cell_id": f"config-{entry.array_index:04d}/{cell.cell_id}",
                **dict(cell.overrides),
            }
            for path, value in cell.overrides.items():
                leaf = str(path).rsplit(".", 1)[-1]
                if leaf not in record:
                    record[leaf] = value
            record.update(
                {
                    "population_size": config.game.population_size,
                    "social_group_size": config.game.options.get("social_group_size"),
                    "epistemic_persistence": config.game.options.get(
                        "epistemic_persistence"
                    ),
                    "receiver_epistemic_disposition": config.game.options.get(
                        "receiver_epistemic_disposition"
                    ),
                    "controller_evidence_strategy": config.control.options.get(
                        "controller_evidence_strategy"
                    ),
                    "intervention_budget": config.control.options.get(
                        "intervention_budget"
                    ),
                    "target_semantics": config.experiment.metadata.get(
                        "target_semantics"
                    ),
                }
            )
            rows.append(record)
    return rows


def _state_local_phase_tables(
    entries: Sequence[Any],
    cells: pd.DataFrame,
    primary: pd.DataFrame,
    derived: pd.DataFrame,
    events: Sequence[Any],
    *,
    bins: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Complete binned phase grids with explicit absence/support semantics."""

    if bins is None:
        return pd.DataFrame(), pd.DataFrame()
    expected = _expected_cell_coordinates(entries)
    found = set(cells.get("cell_id", pd.Series(dtype=str)).astype(str))
    occupancy_counts: dict[tuple[str, int], int] = {}
    occupancy_episodes: dict[tuple[str, int], set[str]] = {}
    for event in events:
        population = int(event.event.get("N") or sum(event.N_k))
        fraction = float(event.event["target_count_before"]) / population
        bin_index = min(int(fraction * bins), bins - 1)
        key = (str(event.cell_id), bin_index)
        occupancy_counts[key] = occupancy_counts.get(key, 0) + 1
        occupancy_episodes.setdefault(key, set()).add(str(event.episode_id))

    estimates = pd.concat([primary, derived], ignore_index=True, sort=False)
    estimate_lookup: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    if not estimates.empty and "target_fraction_bin_index" in estimates:
        selected = estimates.dropna(subset=["target_fraction_bin_index"])
        for row in selected.to_dict(orient="records"):
            estimate_lookup[
                (
                    str(row.get("cell_id")),
                    int(row["target_fraction_bin_index"]),
                    str(row.get("metric")),
                )
            ] = row

    occupancy_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    metrics = {
        "chi": "round_target_susceptibility",
        "T_pi": "round_target_actuation_cmi",
        "eta_IF": "round_target_information_fraction",
        "eta_IR": "eta_ir_state_local",
    }
    for coordinate in expected:
        cell_id = str(coordinate["cell_id"])
        structural_present = cell_id in found
        for bin_index in range(bins):
            key = (cell_id, bin_index)
            n_observations = occupancy_counts.get(key, 0)
            common = {
                **coordinate,
                "target_fraction_bin_index": bin_index,
                "target_fraction_bin_lower": bin_index / bins,
                "target_fraction_bin_upper": (bin_index + 1) / bins,
                "target_fraction_bin_center": (bin_index + 0.5) / bins,
                "target_fraction_bin_count": bins,
                "n_observations": n_observations,
                "n_episodes": len(occupancy_episodes.get(key, set())),
            }
            occupancy_status = (
                "structural_cell_not_run"
                if not structural_present
                else "state_not_visited"
                if n_observations == 0
                else "visited"
            )
            occupancy_rows.append(
                {**common, "phase_status": occupancy_status, "estimate": n_observations}
            )
            for label, source_metric in metrics.items():
                estimate = estimate_lookup.get((cell_id, bin_index, source_metric))
                if not structural_present:
                    status = "structural_cell_not_run"
                elif n_observations == 0:
                    status = "state_not_visited"
                elif estimate is None:
                    status = "insufficient_estimator_support"
                else:
                    raw_status = str(estimate.get("support_status", "unsupported"))
                    try:
                        finite = math.isfinite(float(estimate.get("estimate")))
                    except (TypeError, ValueError):
                        finite = False
                    status = (
                        raw_status
                        if raw_status in {"adequate", "limited"} and finite
                        else "insufficient_estimator_support"
                    )
                phase_rows.append(
                    {
                        **common,
                        "metric": label,
                        "source_metric": source_metric,
                        "estimate": (
                            math.nan if estimate is None else estimate.get("estimate")
                        ),
                        "units": None if estimate is None else estimate.get("units"),
                        "source_support_status": (
                            None if estimate is None else estimate.get("support_status")
                        ),
                        "phase_status": status,
                    }
                )
    return pd.DataFrame(phase_rows), pd.DataFrame(occupancy_rows)


def _phi_conditioning_comparison(primary: pd.DataFrame) -> pd.DataFrame:
    """Cellwise raw/null-adjusted comparison of T_pi and T_pi_phi."""

    if primary.empty:
        return pd.DataFrame()
    wanted = primary[
        primary["metric"].isin(
            {"round_target_actuation_cmi", "round_phi_target_actuation_cmi"}
        )
    ].copy()
    if wanted.empty:
        return pd.DataFrame()
    keys = ["study_id", "source_run_id", "cell_id"]
    rows: list[dict[str, Any]] = []
    for coordinates, group in wanted.groupby(keys, dropna=False):
        by_metric = {str(row["metric"]): row for row in group.to_dict(orient="records")}
        base = by_metric.get("round_target_actuation_cmi")
        conditioned = by_metric.get("round_phi_target_actuation_cmi")
        if base is None or conditioned is None:
            continue

        def finite_or_nan(value: Any) -> float:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return math.nan
            return number if math.isfinite(number) else math.nan

        t_pi = finite_or_nan(base.get("estimate"))
        t_phi = finite_or_nan(conditioned.get("estimate"))
        null_pi = finite_or_nan(base.get("null_mean"))
        null_phi = finite_or_nan(conditioned.get("null_mean"))
        rows.append(
            {
                **dict(zip(keys, coordinates, strict=True)),
                "T_pi": t_pi,
                "T_pi_phi": t_phi,
                "Delta_T_phi": t_phi - t_pi,
                "T_pi_null_mean": null_pi,
                "T_pi_phi_null_mean": null_phi,
                "T_pi_null_adjusted": t_pi - null_pi,
                "T_pi_phi_null_adjusted": t_phi - null_phi,
                "T_pi_ci_low": base.get("ci_low"),
                "T_pi_ci_high": base.get("ci_high"),
                "T_pi_phi_ci_low": conditioned.get("ci_low"),
                "T_pi_phi_ci_high": conditioned.get("ci_high"),
                "T_pi_action_entropy_ceiling_bits": base.get(
                    "action_entropy_ceiling_bits"
                ),
                "T_pi_phi_action_entropy_ceiling_bits": conditioned.get(
                    "action_entropy_ceiling_bits"
                ),
                "T_pi_dual_action_support_fraction": base.get(
                    "dual_action_support_fraction"
                ),
                "T_pi_phi_dual_action_support_fraction": conditioned.get(
                    "dual_action_support_fraction"
                ),
                "T_pi_support_status": base.get("support_status"),
                "T_pi_phi_support_status": conditioned.get("support_status"),
                "T_pi_n_observations": base.get("n_observations"),
                "T_pi_phi_n_observations": conditioned.get("n_observations"),
                "descriptive_caution": (
                    "Raw conditioned CMI can rise with finite-sample null bias; "
                    "compare the separate null-adjusted values."
                ),
            }
        )
    return pd.DataFrame(rows)


def _rho_aggregated_state_local_maps(
    phase: pd.DataFrame, occupancy: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Observation-weighted descriptive maps over rho, never pooled estimators."""

    if phase.empty or occupancy.empty:
        return pd.DataFrame(), pd.DataFrame()
    keys = [
        column
        for column in (
            "social_group_size",
            "controller_evidence_strategy",
            "receiver_epistemic_disposition",
            "target_semantics",
            "intervention_budget",
            "target_fraction_bin_index",
            "target_fraction_bin_lower",
            "target_fraction_bin_upper",
            "target_fraction_bin_center",
            "target_fraction_bin_count",
        )
        if column in phase.columns
    ]
    map_rows: list[dict[str, Any]] = []
    for coordinates, group in phase.groupby([*keys, "metric"], dropna=False):
        valid = group[
            group["phase_status"].isin(["adequate", "limited"])
            & pd.to_numeric(group["estimate"], errors="coerce").notna()
        ].copy()
        weights = pd.to_numeric(valid.get("n_observations"), errors="coerce")
        values = pd.to_numeric(valid.get("estimate"), errors="coerce")
        estimate = (
            math.nan
            if valid.empty or weights.sum() <= 0
            else float(np.average(values, weights=weights))
        )
        status = (
            "state_not_visited"
            if int(pd.to_numeric(group["n_observations"], errors="coerce").sum()) == 0
            else "insufficient_estimator_support"
            if valid.empty
            else "limited"
            if (valid["phase_status"] == "limited").any()
            else "adequate"
        )
        coordinate_values = coordinates[:-1]
        map_rows.append(
            {
                **dict(zip(keys, coordinate_values, strict=True)),
                "metric": coordinates[-1],
                "estimate": estimate,
                "n_observations": int(weights.sum()) if not valid.empty else 0,
                "n_rho": int(group["epistemic_persistence"].nunique()),
                "rho_values_json": json.dumps(
                    sorted(
                        float(value)
                        for value in group["epistemic_persistence"].dropna().unique()
                    )
                ),
                "phase_status": status,
                "aggregation_scope": "rho-aggregated descriptive state-local summaries",
                "aggregation_weight": "n_observations",
                "descriptive_only": True,
            }
        )
    occupancy_rows = occupancy.groupby(keys, dropna=False, as_index=False).agg(
        n_observations=("n_observations", "sum"),
        n_episodes=("n_episodes", "sum"),
        n_rho=("epistemic_persistence", "nunique"),
    )
    occupancy_rows["estimate"] = occupancy_rows["n_observations"]
    occupancy_rows["phase_status"] = np.where(
        occupancy_rows["n_observations"] > 0, "visited", "state_not_visited"
    )
    occupancy_rows["aggregation_scope"] = (
        "rho-aggregated descriptive state-local summaries"
    )
    occupancy_rows["aggregation_weight"] = "n_observations"
    occupancy_rows["descriptive_only"] = True
    return pd.DataFrame(map_rows), occupancy_rows


def _rho_aggregated_descriptive_summary(
    primary: pd.DataFrame,
    derived: pd.DataFrame,
    endpoints: pd.DataFrame,
) -> pd.DataFrame:
    """Clearly labelled descriptive marginals; never an estimator super-pool."""

    group_columns = [
        "social_group_size",
        "controller_evidence_strategy",
        "target_semantics",
        "intervention_budget",
    ]
    rows: list[dict[str, Any]] = []
    estimator_specs = (
        (derived, "susceptibility_occupancy_weighted", "chi"),
        (primary, "round_target_actuation_cmi", "T_pi"),
        (derived, "eta_ir", "eta_IR"),
        (derived, "eta_th_signed", "eta_th_signed"),
    )
    for frame, source_metric, label in estimator_specs:
        if frame.empty or not set(group_columns + ["metric", "estimate"]).issubset(
            frame.columns
        ):
            continue
        selected = frame[frame["metric"].astype(str) == source_metric].copy()
        if "resolution" in selected:
            selected = selected[selected["resolution"].isna()]
        selected["estimate"] = pd.to_numeric(selected["estimate"], errors="coerce")
        selected = selected[np.isfinite(selected["estimate"])]
        for coordinates, group in selected.groupby(group_columns, dropna=False):
            values = group["estimate"]
            rows.append(
                {
                    **dict(zip(group_columns, coordinates, strict=True)),
                    "metric": label,
                    "source_metric": source_metric,
                    "estimate": float(values.mean()),
                    "std_across_rho_cells": float(values.std(ddof=1)),
                    "minimum_across_rho_cells": float(values.min()),
                    "maximum_across_rho_cells": float(values.max()),
                    "n_rho": int(group["epistemic_persistence"].nunique()),
                    "rho_values_json": json.dumps(
                        sorted(
                            {
                                float(value)
                                for value in group["epistemic_persistence"].dropna()
                            }
                        )
                    ),
                    "n_cells": int(group["cell_id"].nunique()),
                    "n_episodes": pd.to_numeric(
                        group.get("n_episodes", pd.Series(dtype=float)),
                        errors="coerce",
                    ).sum(min_count=1),
                    "aggregation_scope": "rho_aggregated_descriptive",
                    "descriptive_only": True,
                }
            )

    endpoint_specs = {
        "late_target_share": (
            "late_time_mean_controller_target_share"
            if "late_time_mean_controller_target_share" in endpoints
            else "late_time_mean_false_target_share"
        ),
        "late_truth_share": "late_time_mean_truth_share",
        "late_active_phi": "late_time_mean_active_phi",
    }
    if not endpoints.empty and set(group_columns).issubset(endpoints.columns):
        for label, column in endpoint_specs.items():
            if column not in endpoints:
                continue
            prepared = endpoints.copy()
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
            prepared = prepared[np.isfinite(prepared[column])]
            for coordinates, group in prepared.groupby(group_columns, dropna=False):
                values = group[column]
                rows.append(
                    {
                        **dict(zip(group_columns, coordinates, strict=True)),
                        "metric": label,
                        "source_metric": column,
                        "estimate": float(values.mean()),
                        "std_across_rho_cells": float(values.std(ddof=1)),
                        "minimum_across_rho_cells": float(values.min()),
                        "maximum_across_rho_cells": float(values.max()),
                        "n_rho": int(group["epistemic_persistence"].nunique()),
                        "rho_values_json": json.dumps(
                            sorted(
                                {
                                    float(value)
                                    for value in group["epistemic_persistence"].dropna()
                                }
                            )
                        ),
                        "n_cells": int(group["cell_id"].nunique()),
                        "n_episodes": int(group["episode_id"].nunique()),
                        "aggregation_scope": "rho_aggregated_descriptive",
                        "descriptive_only": True,
                    }
                )
    return pd.DataFrame(rows)


def _factorial_contrasts(
    primary: pd.DataFrame,
    derived: pd.DataFrame,
    requested: Sequence[Any],
    analysis_hash: str,
) -> pd.DataFrame:
    """Matched descriptive factorial differences, not a new estimator."""

    if "factorial_contrasts" not in {str(value) for value in requested}:
        return pd.DataFrame()
    frames = [frame for frame in (primary, derived) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    source = pd.concat(frames, ignore_index=True, sort=False)
    needed = {
        "receiver_epistemic_disposition",
        "controller_evidence_strategy",
        "target_semantics",
        "intervention_budget",
        "task_id",
        "metric",
        "estimate",
    }
    if not needed.issubset(source.columns):
        return pd.DataFrame()
    source = source[
        source["metric"].isin(
            {
                "round_target_actuation_cmi",
                "round_target_signed_actuation",
                "round_target_susceptibility",
                "susceptibility_occupancy_weighted",
                "eta_ir",
                "eta_th",
                "eta_th_signed",
                "controlled_current",
                "target_sensing_information_nats",
            }
        )
    ].copy()
    keys = ["metric", "intervention_budget", "task_id"]
    if (
        "epistemic_persistence" in source
        and source["epistemic_persistence"].notna().any()
    ):
        keys.append("epistemic_persistence")
    rows: list[dict[str, Any]] = []

    def differences(frame, axis, low, high, kind, extra):
        index = [
            *keys,
            *[
                column
                for column in extra
                if column in frame
                and frame[column].notna().any()
                and column not in keys
                and column != axis
            ],
        ]
        pivot = frame.pivot_table(
            index=index, columns=axis, values="estimate", aggfunc="mean"
        )
        if low not in pivot or high not in pivot:
            return
        for coordinates, value in (pivot[high] - pivot[low]).items():
            values = coordinates if isinstance(coordinates, tuple) else (coordinates,)
            record = dict(zip(index, values, strict=True))
            base_metric = str(record.pop("metric"))
            if record.get("receiver_epistemic_disposition") and record.get(
                "controller_evidence_strategy"
            ):
                record["derived_epistemic_condition"] = (
                    f"{record['receiver_epistemic_disposition']}_"
                    f"{record['controller_evidence_strategy']}"
                )
            rows.append(
                {
                    **record,
                    "study_id": str(frame["study_id"].iloc[0]),
                    "source_run_id": "matched-factorial-contrast",
                    "cell_id": f"contrast-{kind}-{len(rows):06d}",
                    "metric": f"delta_{kind}_{base_metric}",
                    "contrast_type": kind,
                    "estimate": float(value),
                    "units": "descriptive_difference",
                    "support_status": "descriptive_only",
                    "grouping_json": json.dumps(record, sort_keys=True, default=str),
                    "conditioning_json": "{}",
                    "dependencies_json": json.dumps(
                        {"high": high, "low": low, "axis": axis}
                    ),
                    "analysis_hash": analysis_hash,
                }
            )

    differences(
        source,
        "receiver_epistemic_disposition",
        "naive",
        "vigilant",
        "vigilance",
        [
            "controller_evidence_strategy",
            "target_semantics",
            "social_group_size",
        ],
    )
    differences(
        source,
        "controller_evidence_strategy",
        "neutral",
        "strategic",
        "evidence_strategy",
        [
            "receiver_epistemic_disposition",
            "target_semantics",
            "social_group_size",
        ],
    )
    differences(
        source,
        "social_group_size",
        1,
        2,
        "q",
        [
            "receiver_epistemic_disposition",
            "controller_evidence_strategy",
            "target_semantics",
        ],
    )
    differences(
        source,
        "target_semantics",
        "false",
        "truth",
        "truth_false",
        [
            "receiver_epistemic_disposition",
            "controller_evidence_strategy",
            "social_group_size",
        ],
    )
    return pd.DataFrame(rows)


def _value_label(metric: str, subset: pd.DataFrame) -> str:
    """`metric [units]`, with the units taken from the row being plotted.

    Bits and nats appear on the same page in this family, and a fraction
    response and a magnetization response look identical on an unlabelled
    axis. The estimator already records units per row, so the label is read
    from there rather than guessed from the metric name.
    """

    if "units" not in subset:
        return metric
    units = sorted({str(value) for value in subset["units"].dropna().unique()})
    return metric if len(units) != 1 or not units[0] else f"{metric} [{units[0]}]"


def _render_plots(
    recipe: Mapping[str, Any], tables: Mapping[str, pd.DataFrame], destination: Path
) -> list[str]:
    raw = recipe.get("plots", {})
    destination.mkdir(parents=True, exist_ok=True)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        builtins = {
            "target_cmi_x_b": {
                "source": "primary_estimates",
                "metric": "round_target_actuation_cmi",
            },
            "eta_ir_x_b": {
                "source": "derived_observables",
                "metric": "eta_ir_state_local",
            },
            "memory_conditioning": {
                "source": "primary_estimates",
                "metric": "round_memory_target_actuation_cmi",
            },
            "h_eff_phi_b": {
                "source": "primary_estimates",
                "metric": "effective_affinity",
            },
            "gamma_eff_phi_b": {
                "source": "primary_estimates",
                "metric": "kinetic_compliance",
            },
        }
        raw = {
            str(name): builtins.get(
                str(name), {"source": "primary_estimates", "metric": str(name)}
            )
            for name in raw
        }
    if not isinstance(raw, Mapping):
        raise ValueError("analysis plots must be a list or mapping")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    paths: list[str] = []
    for name, spec in raw.items():
        if not isinstance(spec, Mapping):
            continue
        source_name = str(spec.get("source", "primary_estimates"))
        frame = tables.get(source_name)
        if frame is None or frame.empty:
            continue
        metric = str(spec.get("metric", ""))
        subset = (
            frame[frame["metric"].astype(str) == metric]
            if metric and "metric" in frame
            else frame
        )
        filters = spec.get("filters", {})
        if isinstance(filters, Mapping):
            for column, expected in filters.items():
                if str(column) in subset:
                    subset = subset[subset[str(column)].astype(str) == str(expected)]
        value_column = str(spec.get("value", ""))
        if value_column and value_column in subset:
            subset = subset.copy()
            subset["estimate"] = pd.to_numeric(subset[value_column], errors="coerce")
        if "support_status" in subset:
            subset = subset[subset["support_status"].astype(str) != "unsupported"]
        x, y = str(spec.get("x", "")), str(spec.get("y", ""))
        if subset.empty or "estimate" not in subset:
            continue
        kind = str(spec.get("kind", "heatmap"))
        if x not in subset or (kind != "line" and y not in subset):
            figure, axis = plt.subplots(figsize=(max(5, len(subset) * 0.5), 4))
            labels = subset.get("cell_id", pd.Series(range(len(subset)))).astype(str)
            axis.bar(range(len(subset)), subset["estimate"].astype(float))
            axis.set_xticks(range(len(subset)), labels=labels, rotation=45, ha="right")
            axis.set_ylabel(_value_label(metric, subset))
            axis.set_title(str(name))
            figure.tight_layout()
            path = destination / f"{name}.png"
            figure.savefig(path, dpi=150)
            plt.close(figure)
            paths.append(str(path))
            continue
        facet = spec.get("facet")
        groups = (
            [("all", subset)]
            if facet not in subset
            else list(subset.groupby(str(facet), dropna=False))
        )
        finite_estimates = pd.to_numeric(subset["estimate"], errors="coerce")
        finite_estimates = finite_estimates[np.isfinite(finite_estimates)]
        shared_vmin = float(finite_estimates.min()) if len(finite_estimates) else None
        shared_vmax = float(finite_estimates.max()) if len(finite_estimates) else None
        two_by_two = str(spec.get("layout", "")) == "2x2" and len(groups) == 4
        nrows, ncols = (2, 2) if two_by_two else (1, len(groups))
        figure, axes = plt.subplots(
            nrows, ncols, squeeze=False, figsize=(5 * ncols, 4 * nrows)
        )
        for axis, (label, group) in zip(axes.flat, groups, strict=True):
            if kind == "line":
                series = str(spec.get("series", ""))
                line_groups = (
                    [("all", group)]
                    if series not in group
                    else group.groupby(series, dropna=False)
                )
                for series_label, line_group in line_groups:
                    grouped_values = line_group.groupby(x, dropna=False)["estimate"]
                    values = grouped_values.mean().sort_index()
                    if bool(spec.get("show_variability", False)):
                        deviations = (
                            grouped_values.std().reindex(values.index).fillna(0)
                        )
                        axis.errorbar(
                            values.index,
                            values.values,
                            yerr=deviations.values,
                            marker="o",
                            capsize=3,
                            label=str(series_label),
                        )
                    else:
                        axis.plot(
                            values.index,
                            values.values,
                            marker="o",
                            label=str(series_label),
                        )
                if series in group:
                    axis.legend()
                axis.set_xlabel(x)
                axis.set_ylabel(_value_label(value_column or metric, subset))
                axis.set_title(str(label))
                continue
            pivot = group.pivot_table(
                index=y, columns=x, values="estimate", aggfunc="mean"
            )
            y_values = sorted(group[y].dropna().unique())
            x_values = sorted(group[x].dropna().unique())
            pivot = pivot.reindex(index=y_values, columns=x_values)
            if str(spec.get("color_scale", "shared")) == "independent":
                group_finite = pd.to_numeric(group["estimate"], errors="coerce")
                group_finite = group_finite[np.isfinite(group_finite)]
                panel_vmin = float(group_finite.min()) if len(group_finite) else None
                panel_vmax = float(group_finite.max()) if len(group_finite) else None
            else:
                panel_vmin, panel_vmax = shared_vmin, shared_vmax
            image = axis.imshow(
                pivot.to_numpy(dtype=float),
                aspect="auto",
                origin="lower",
                vmin=panel_vmin,
                vmax=panel_vmax,
            )
            status_column = str(spec.get("status_column", ""))
            if status_column and status_column in group:
                status = group.pivot_table(
                    index=y, columns=x, values=status_column, aggfunc="first"
                ).reindex(index=y_values, columns=x_values)
                colors = {
                    "structural_cell_not_run": "#4d4d4d",
                    "state_not_visited": "#d9d9d9",
                    "insufficient_estimator_support": "#f4a261",
                }
                for row_index in range(len(status.index)):
                    for column_index in range(len(status.columns)):
                        value = status.iat[row_index, column_index]
                        if value in colors:
                            axis.add_patch(
                                Rectangle(
                                    (column_index - 0.5, row_index - 0.5),
                                    1,
                                    1,
                                    facecolor=colors[value],
                                    edgecolor="white",
                                    linewidth=0.4,
                                    zorder=3,
                                )
                            )
                axis.legend(
                    handles=[
                        Patch(facecolor=color, label=label.replace("_", " "))
                        for label, color in colors.items()
                    ],
                    loc="upper left",
                    bbox_to_anchor=(1.02, 0),
                    fontsize=7,
                )
            axis.set_xticks(
                range(len(pivot.columns)),
                labels=[str(value) for value in pivot.columns],
            )
            axis.set_yticks(
                range(len(pivot.index)), labels=[str(value) for value in pivot.index]
            )
            axis.set_xlabel(x)
            axis.set_ylabel(y)
            axis.set_title(str(label) if facet in subset else metric)
            figure.colorbar(image, ax=axis, label=_value_label(metric, subset))
        figure.tight_layout()
        path = destination / f"{name}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(str(path))
    return paths


def _render_false_takeover_plots(
    rounds: Sequence[Any],
    endpoints: pd.DataFrame,
    destination: Path,
) -> list[str]:
    """Render only the three descriptive figures requested by the pilot."""

    if not rounds or endpoints.empty:
        return []
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    endpoint_lookup = endpoints.set_index(["cell_id", "episode_id"])
    trajectory_rows = []
    for row in rounds:
        total = sum(row.N_k1)
        trajectory_rows.append(
            {
                "cell_id": row.cell_id,
                "episode_id": row.episode_id,
                "round_index": row.round_index,
                "controller_target_share": row.target_after / total,
                "truth_vote_share": row.truth_after / total,
            }
        )
    prepared = pd.DataFrame(trajectory_rows)
    for value, title, filename in (
        (
            "controller_target_share",
            "False-target vote share",
            "false_target_share_over_rounds.png",
        ),
        ("truth_vote_share", "Truth vote share", "truth_share_over_rounds.png"),
    ):
        if value not in prepared:
            continue
        figure, axis = plt.subplots(figsize=(8, 5))
        for (cell_id, episode_id), frame in prepared.groupby(
            ["cell_id", "episode_id"], sort=True
        ):
            frame = frame.sort_values("round_index")
            endpoint = endpoint_lookup.loc[(cell_id, episode_id)]
            label = f"q={endpoint['social_group_size']}, b={endpoint['intervention_budget']}, {endpoint['task_id']}"
            axis.plot(
                frame["round_index"],
                pd.to_numeric(frame[value], errors="coerce"),
                marker="o",
                label=label,
            )
        axis.axhline(
            0.5, color="black", linestyle="--", linewidth=1, label="majority threshold"
        )
        axis.set(xlabel="round", ylabel="vote share", ylim=(-0.02, 1.02), title=title)
        axis.legend(fontsize=7, ncol=2)
        axis.grid(alpha=0.2)
        figure.tight_layout()
        path = destination / filename
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(str(path))

    figure, axis = plt.subplots(figsize=(7, 4))
    labels, false_means, truth_means = [], [], []
    for (q, budget), frame in endpoints.groupby(
        ["social_group_size", "intervention_budget"], sort=True
    ):
        labels.append(f"q={q}, b={budget}")
        false_means.append(float(frame["final_false_target_share"].mean()))
        truth_means.append(float(frame["final_truth_share"].mean()))
    x = np.arange(len(labels))
    width = 0.38
    axis.bar(x - width / 2, false_means, width, label="false target")
    axis.bar(x + width / 2, truth_means, width, label="truth")
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(x, labels=labels)
    axis.set(
        ylabel="mean final vote share",
        ylim=(0, 1),
        title="Final outcomes (two tasks per regime)",
    )
    axis.legend()
    figure.tight_layout()
    path = destination / "final_outcomes.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(str(path))
    return paths


SINGLE_AFFINITY_METHODS = (
    "## Single-affinity derived observables",
    "",
    "`eta_ir`, `eta_th` and their components are stated in the coordinates of",
    "the revised single-affinity theory, so an empirical number and the exact",
    "theory number of the same name are directly comparable.",
    "",
    "**Corrected `eta_ir`.** The response entering the Pinsker numerator is",
    "`round_target_susceptibility`: the state-matched difference of mean",
    "target-*fraction* changes, `E[dx|ADVOCATE,n] - E[dx|NO_OP,n]`. The first",
    "implementation used `round_target_signed_actuation`, the same difference",
    "in aligned-magnetization units, which is larger by `K/(K-1)` and so",
    "inflated the squared numerator by `(K/(K-1))^2` - 2.25x on a three-option",
    "task. Rows carrying the corrected definition are stamped",
    "`theory_semantics_version = single_affinity_v1`; rows without that stamp",
    "predate the correction and are not comparable to them.",
    "",
    "**Headline `eta_ir` is occupancy weighted.** It is a ratio of sums,",
    "`sum_n p(n) B_IR(n) / I(U;n'|n)`, not a mean of state-local ratios. The",
    "state-resolved surface is published separately as `eta_ir_state_local`.",
    "",
    "**`controlled_current` is not `cell_current`.** The thermodynamic current",
    "is `N sum_n p_k(n) a(n) chi(n)`, evaluated state by state and summed over",
    "the horizon. `cell_current` remains available and remains the terminal",
    "episode difference `n_Z,H - n_Z,0`, a behavioural outcome that also",
    "contains the ordinary social dynamics.",
    "",
    "**`target_sensing_information_nats` is not `round_sensing_mi`.** The",
    "thermodynamic sensing term is the scalar channel `I(n_Z;Y_Z)` in nats,",
    "computed from the empirical occupancy and the exact hypergeometric sensor",
    "kernel, summed round by round. `round_sensing_mi` remains available and",
    "remains the full K-option vector channel `I(N;Y)` in bits.",
    "",
    "**Units.** `chi` target fraction; `T_pi` and the Pinsker numerator bits;",
    "`h`, `h*J_c` and `I_sens` nats; `J_c` target counts per cycle.",
    "",
    "**Intervals.** Both efficiencies are bootstrapped by resampling whole",
    "episodes and recomputing every ingredient inside each replicate, never by",
    "combining separately bootstrapped intervals for `h`, `J_c` and `I_sens`.",
    "",
    "**Separate references.** The matched q-voter theory is a classical null",
    "reported elsewhere and is never substituted into these formulas.",
    "",
)
"""What a reader of the derived table needs in order to trust or reject it.

Written into `reports/methods.md` whenever derived observables exist, because
the one thing a corrected estimator must not do is arrive silently under the
old name."""


def _derived_semantics(derived: pd.DataFrame) -> dict[str, Any]:
    """The provenance stamp, lifted out of the table into the manifest."""

    from mas_cc.analysis.single_affinity import PROVENANCE

    if derived.empty or "theory_semantics_version" not in derived:
        return {}
    return {
        **PROVENANCE,
        "metrics": sorted(
            {
                str(value)
                for value in derived.loc[
                    derived["theory_semantics_version"].notna(), "metric"
                ]
            }
        ),
    }


def _reset_analysis_handoff(analysis_dir: Path, study_id: str) -> None:
    """Remove obsolete products before writing the sole output contract."""

    analysis_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "cache",
        "cell_cache",
        "tables",
        "plots",
        "plots-debug",
        "reports",
        "provenance",
    ):
        path = analysis_dir / name
        if path.is_dir():
            shutil.rmtree(path)
    for name in (
        "analysis_manifest.json",
        "analysis_recipe.yaml",
        "progress.json",
        "validation.json",
        "validation.md",
        f"{study_id}_analysis.zip",
        "theory_comparison.csv",
        "theory_state_curves.csv",
    ):
        (analysis_dir / name).unlink(missing_ok=True)
    for path in analysis_dir.rglob("*.pickle"):
        path.unlink(missing_ok=True)


def _package(analysis_dir: Path, study_id: str) -> Path:
    destination = analysis_dir / f"{study_id}_analysis.zip"
    temporary = destination.with_suffix(".zip.tmp")
    root_names = {
        "validation.json",
        "validation.md",
        "analysis_manifest.json",
        "analysis_recipe.yaml",
    }
    directory_names = {"tables", "plots", "reports", "provenance"}
    paths = [path for name in root_names if (path := analysis_dir / name).is_file()]
    for name in directory_names:
        directory = analysis_dir / name
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            relative = path.relative_to(analysis_dir)
            if (
                "cache" in relative.parts
                or path.suffix in {".pickle", ".tmp"}
                or path.name == "information_nulls.parquet"
                or path.name.endswith(":Zone.Identifier")
            ):
                continue
            info = zipfile.ZipInfo(str(relative).replace("\\", "/"))
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    temporary.replace(destination)
    return destination


def _scientific_identity(
    entries: Sequence[Any], cells: Sequence[Any], canonical: Mapping[str, pd.DataFrame]
) -> str:
    artifacts: list[dict[str, Any]] = []
    for cell in cells:
        paths: list[Path] = []
        for name in ("scientific_events.parquet", "cell_complete.json"):
            candidate = cell.path / name
            if candidate.is_file():
                paths.append(candidate)
        paths.extend(sorted(cell.path.rglob("round_trajectory.jsonl")))
        paths.extend(sorted(cell.path.rglob("micro_slot_trajectory.jsonl")))
        for path in paths:
            artifacts.append(
                {
                    "cell_id": cell.cell_key,
                    "kind": path.name,
                    "sha256": file_sha256(path),
                }
            )
    return canonical_hash(
        {
            "config_hashes": [entry.config_hash for entry in entries],
            "artifacts": artifacts,
            "cells": canonical["cells"].to_dict(orient="records"),
            "episodes": canonical["episodes"][
                [
                    column
                    for column in (
                        "cell_id",
                        "episode_id",
                        "interaction_count",
                        "status",
                    )
                    if column in canonical["episodes"]
                ]
            ].to_dict(orient="records"),
        }
    )


def aggregate_study(
    study_dir: str | Path, *, allow_incomplete: bool = False
) -> dict[str, Any]:
    """Create the complete canonical analysis package for one submitted study."""

    root = Path(study_dir).expanduser().resolve()
    manifest_path = root / "study_manifest.json"
    submission_path = root / "submission_manifest.csv"
    if not manifest_path.is_file() or not submission_path.is_file():
        raise ValueError(f"not a submitted MA-CC study directory: {root}")
    study_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    study_id = str(study_manifest["study_id"])
    lineage_mode = (root / "study_lineage.json").is_file()
    target_manifest = None
    if lineage_mode:
        from .extension import extension_aggregation_context

        target_manifest, entries = extension_aggregation_context(root)
    else:
        entries = read_submission_manifest(submission_path)
    recipe, recipe_path = _recipe(study_manifest)
    from mas_cc.analysis.single_affinity import PROVENANCE as theory_provenance

    theoretical_reference = recipe.get(
        "theoretical_reference", "single_affinity_revised"
    )
    if theoretical_reference not in {"single_affinity_revised", "none"}:
        raise ValueError(
            "analysis theoretical_reference must be single_affinity_revised or none"
        )
    settings = _resampling(recipe)

    analysis_dir = root / "analysis"
    retained_input_identity: str | None = None
    runs = discover_runs(entries)
    cells = discover_cells(runs)
    retained_paths = {
        name: retained_table_path(analysis_dir / "tables", name)
        for name in ("cells", "episodes", "rounds", "micro_slots")
    }
    if cells:
        canonical, canonical_metadata = build_canonical_tables(study_id, cells)
        if target_manifest is not None:
            from .extension import consolidate_extension_tables

            canonical, validation = consolidate_extension_tables(
                canonical, target_manifest
            )
        else:
            validation = validate_study(entries, runs, cells, canonical)
    elif all(path is not None for path in retained_paths.values()):
        canonical = {
            name: read_scientific_table(path) for name, path in retained_paths.items()
        }
        prior_validation_path = analysis_dir / "validation.json"
        if not prior_validation_path.is_file():
            raise ValueError(
                "canonical study tables exist without validation metadata: "
                + str(analysis_dir)
            )
        validation = json.loads(prior_validation_path.read_text(encoding="utf-8"))
        prior_manifest_path = analysis_dir / "analysis_manifest.json"
        if prior_manifest_path.is_file():
            prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
            retained_input_identity = prior_manifest.get("scientific_input_identity")
        validation.setdefault("warnings", []).append(
            "source run trees unavailable; reaggregated from retained canonical tables"
        )
    else:
        canonical, canonical_metadata = build_canonical_tables(study_id, cells)
        validation = validate_study(entries, runs, cells, canonical)
    if cells:
        selection = canonical_metadata.get("record_selection", {})
        excluded = sum(
            int(item.get("excluded_incomplete_records", 0))
            for item in selection.values()
        )
        superseded = sum(
            int(item.get("superseded_retry_records", 0)) for item in selection.values()
        )
        validation["canonical_record_selection"] = selection
        if excluded:
            validation.setdefault("warnings", []).append(
                f"excluded {excluded} trajectory records from incomplete episodes"
            )
        if superseded:
            validation.setdefault("warnings", []).append(
                f"excluded {superseded} superseded trajectory records from safe retries"
            )
    _reset_analysis_handoff(analysis_dir, study_id)
    tables_dir = analysis_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if lineage_mode:
        provenance = analysis_dir / "provenance"
        provenance.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "study_lineage.json", provenance / "study_lineage.json")
        for path in sorted((root / "extensions").glob("extension-*/*.json")):
            if path.name in {
                "target_manifest.json",
                "compatibility_report.json",
                "state.json",
                "migration.json",
            }:
                shutil.copy2(
                    path,
                    provenance / f"{path.parent.name}_{path.name}",
                )

    validation["allow_incomplete"] = bool(allow_incomplete)
    if not validation["valid"] and allow_incomplete:
        validation["complete"] = False
        validation["warnings"].append("aggregation continued under --allow-incomplete")
    _write_json(analysis_dir / "validation.json", validation)
    (analysis_dir / "validation.md").write_text(
        validation_markdown(validation), encoding="utf-8"
    )
    if not validation["valid"] and not allow_incomplete:
        raise ValueError(
            "study validation failed; inspect " + str(analysis_dir / "validation.json")
        )

    for name, frame in canonical.items():
        write_scientific_table(tables_dir, name, frame)

    input_identity = retained_input_identity or _scientific_identity(
        entries, cells, canonical
    )
    statistics = _requested_statistics(recipe)
    raw_estimators = recipe.get("estimators", ())
    requested_estimators = {str(name) for name in raw_estimators}
    analysis_hash = canonical_hash(
        {
            "scientific_input_identity": input_identity,
            "estimator": "existing_round_information_analysis",
            "estimator_version": "round-feedback-v1",
            "statistics": statistics,
            "settings": settings,
            "theoretical_reference": theoretical_reference,
            "theory_provenance": dict(theory_provenance),
        }
    )
    events = _round_events_from_canonical(canonical["rounds"])
    if theoretical_reference != "none" and any(
        event.event.get("record_type") == "relational_imitation_round_feedback"
        and float(event.event.get("epistemic_persistence", 1.0)) < 1.0
        for event in events
    ):
        raise ValueError(
            "analysis theoretical_reference must be none for finite epistemic "
            "persistence"
        )
    endpoint_recipe = recipe.get("episode_endpoints")
    episode_endpoints = pd.DataFrame()
    episode_endpoint_summary = pd.DataFrame()
    if endpoint_recipe is not None:
        if not isinstance(endpoint_recipe, Mapping):
            raise ValueError("analysis episode_endpoints must be a mapping")
        classifier = str(endpoint_recipe.get("classifier", ""))
        if classifier not in {
            "relational_false_takeover_v1",
            "relational_persistence_exploratory_v1",
            "relational_persistence_refinement_v1",
            "relational_persistence_truth_aligned_v1",
        }:
            raise ValueError("unsupported episode endpoint classifier")
        if classifier == "relational_false_takeover_v1":
            from .episode_endpoints import relational_false_takeover_tables

            episode_endpoints, episode_endpoint_summary = (
                relational_false_takeover_tables(events, canonical["cells"])
            )
        elif classifier == "relational_persistence_exploratory_v1":
            from .episode_endpoints import relational_persistence_tables

            episode_endpoints, episode_endpoint_summary = relational_persistence_tables(
                events, canonical["cells"]
            )
        elif classifier == "relational_persistence_refinement_v1":
            from .episode_endpoints import relational_persistence_refinement_tables

            episode_endpoints, episode_endpoint_summary = (
                relational_persistence_refinement_tables(events, canonical["cells"])
            )
        else:
            from .episode_endpoints import relational_persistence_truth_tables

            episode_endpoints, episode_endpoint_summary = (
                relational_persistence_truth_tables(events, canonical["cells"])
            )
        write_scientific_table(tables_dir, "episode_endpoints", episode_endpoints)
        write_scientific_table(
            tables_dir, "episode_endpoint_summary", episode_endpoint_summary
        )
    source_run_ids = {
        str(row["cell_id"]).split("/", 1)[0]: str(row["source_run_id"])
        for row in canonical["cells"].to_dict(orient="records")
        if row.get("cell_id") is not None and row.get("source_run_id") is not None
    }
    source_run_ids.update(
        {f"config-{run.entry.array_index:04d}": run.run_id for run in runs}
    )
    if statistics:
        information, support = _information_tables(
            study_id,
            events,
            statistics,
            settings,
            analysis_hash,
            source_run_ids,
            progress_path=analysis_dir / "progress.json",
            workers=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
        )
    else:
        information, support = _ingest_existing_information(
            study_id, runs, analysis_hash, settings
        )

    information = _attach_coordinates(information, canonical["cells"])
    support = _attach_coordinates(support, canonical["cells"])
    auxiliary_hash = canonical_hash(
        {
            "scientific_input_identity": input_identity,
            "estimators": sorted(
                requested_estimators.intersection(
                    {
                        "episode_current",
                        "cell_current",
                        "effective_affinity",
                        "kinetic_compliance",
                    }
                )
            ),
            "settings": settings,
            "version": 1,
        }
    )
    current_primary = _current_primary(
        study_id,
        events,
        requested_estimators,
        settings,
        auxiliary_hash,
        source_run_ids,
    )
    affinity_primary, affinity_support = _affinity_primary(
        study_id,
        canonical["micro_slots"],
        events,
        requested_estimators,
        settings,
        auxiliary_hash,
        source_run_ids,
    )
    state_local = _attach_coordinates(
        _state_local_primary(study_id, events, recipe, analysis_hash, source_run_ids),
        canonical["cells"],
    )
    primary_frames = [
        frame
        for frame in (
            information,
            _attach_coordinates(current_primary, canonical["cells"]),
            _attach_coordinates(affinity_primary, canonical["cells"]),
            state_local,
        )
        if not frame.empty
    ]
    primary = (
        pd.concat(primary_frames, ignore_index=True)
        if primary_frames
        else pd.DataFrame(columns=PRIMARY_COLUMNS)
    )
    if {"target_count_before", "population_size"}.issubset(primary.columns):
        primary["target_fraction_before"] = pd.to_numeric(
            primary["target_count_before"], errors="coerce"
        ) / pd.to_numeric(primary["population_size"], errors="coerce")
    if not affinity_support.empty:
        support = pd.concat(
            [support, _attach_coordinates(affinity_support, canonical["cells"])],
            ignore_index=True,
        )
    derived_raw = recipe.get("derived", ())
    if isinstance(derived_raw, (str, bytes)) or not isinstance(derived_raw, Sequence):
        raise ValueError("analysis derived must be a list")
    derived_hash = canonical_hash(
        {
            "primary_analysis_hash": analysis_hash,
            "derived": list(derived_raw),
            "version": 1,
            "theoretical_reference": theoretical_reference,
            "theory_provenance": dict(theory_provenance),
        }
    )
    derived, theory_comparison = _derived(
        study_id,
        derived_raw,
        primary,
        events,
        derived_hash,
        micro_slots=canonical["micro_slots"],
        settings=settings,
        source_run_ids=source_run_ids,
        theoretical_reference=str(theoretical_reference),
    )
    derived = _attach_coordinates(derived, canonical["cells"])
    if {
        "target_count_before",
        "population_size",
    }.issubset(derived.columns):
        derived["target_fraction_before"] = pd.to_numeric(
            derived["target_count_before"], errors="coerce"
        ) / pd.to_numeric(derived["population_size"], errors="coerce")
    contrasts = _factorial_contrasts(primary, derived, derived_raw, derived_hash)
    if not contrasts.empty:
        derived = pd.concat([derived, contrasts], ignore_index=True, sort=False)

    outputs = {
        "primary_estimates": primary,
        "information_estimates": information,
        "support_diagnostics": support,
        "derived_observables": derived,
    }
    phi_comparison = _phi_conditioning_comparison(primary)
    if not phi_comparison.empty:
        outputs["phi_conditioning_comparison"] = _attach_coordinates(
            phi_comparison, canonical["cells"]
        )
    initialization_summary, initialization_audit, _ = paired_initialization_diagnostics(
        canonical
    )
    if not initialization_summary.empty:
        outputs["initialization_diagnostics"] = initialization_summary
    if not initialization_audit.empty:
        outputs["matched_initialization_audit"] = initialization_audit
    raw_state_bins = recipe.get("state_local_x_bins")
    state_bins = int(raw_state_bins) if raw_state_bins is not None else None
    phase_maps, binned_occupancy = _state_local_phase_tables(
        entries,
        canonical["cells"],
        primary,
        derived,
        events,
        bins=state_bins,
    )
    if not phase_maps.empty:
        outputs["state_local_phase_maps"] = phase_maps
    if not binned_occupancy.empty:
        outputs["state_occupancy_binned"] = binned_occupancy
    if bool(recipe.get("rho_aggregated_descriptive", False)):
        rho_phase, rho_occupancy = _rho_aggregated_state_local_maps(
            phase_maps, binned_occupancy
        )
        if not rho_phase.empty:
            outputs["rho_aggregated_state_local_maps"] = rho_phase
        if not rho_occupancy.empty:
            outputs["rho_aggregated_state_occupancy"] = rho_occupancy
    if bool(recipe.get("rho_aggregated_descriptive", False)):
        rho_summary = _rho_aggregated_descriptive_summary(
            primary, derived, episode_endpoints
        )
        if not rho_summary.empty:
            outputs["rho_aggregated_descriptive_summary"] = rho_summary
    if {
        "cell_id",
        "episode_id",
        "target_count_before",
    }.issubset(canonical["rounds"].columns):
        observed = canonical["rounds"].dropna(subset=["target_count_before"]).copy()
        state_occupancy = observed.groupby(
            ["cell_id", "target_count_before"], dropna=False, as_index=False
        ).agg(
            n_observations=("episode_id", "size"),
            n_episodes=("episode_id", "nunique"),
        )
        state_occupancy = _attach_coordinates(state_occupancy, canonical["cells"])
        if "population_size" in state_occupancy:
            state_occupancy["target_fraction_before"] = pd.to_numeric(
                state_occupancy["target_count_before"], errors="coerce"
            ) / pd.to_numeric(state_occupancy["population_size"], errors="coerce")
        outputs["state_occupancy"] = state_occupancy
    thermodynamic_diagnostics = _thermodynamic_diagnostics_from_derived(derived)
    if not thermodynamic_diagnostics.empty:
        outputs["thermodynamic_efficiency_diagnostics"] = thermodynamic_diagnostics
    if theoretical_reference == "single_affinity_revised":
        outputs["single_affinity_theory_comparison"] = _attach_coordinates(
            theory_comparison, canonical["cells"]
        )
    for name, frame in outputs.items():
        write_scientific_table(tables_dir, name, frame)

    plot_tables = {
        **canonical,
        "rounds": _attach_coordinates(canonical["rounds"], canonical["cells"]),
        "episode_endpoints": episode_endpoints,
        "episode_endpoint_summary": episode_endpoint_summary,
        **outputs,
    }
    plots = _render_plots(recipe, plot_tables, analysis_dir / "plots")
    if (
        not episode_endpoints.empty
        and str((endpoint_recipe or {}).get("classifier", ""))
        == "relational_false_takeover_v1"
    ):
        plots.extend(
            _render_false_takeover_plots(
                events,
                episode_endpoints,
                analysis_dir / "plots",
            )
        )
    reports = analysis_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    counts = validation["counts"]
    (reports / "summary.md").write_text(
        "\n".join(
            [
                f"# {study_id}",
                "",
                f"Aggregation status: **{'complete' if validation['complete'] else 'incomplete'}**.",
                "",
                f"- Runs: {counts['found_configs']} / {counts['expected_configs']}",
                f"- Cells: {counts['found_cells']} / {counts['expected_cells']}",
                f"- Completed episodes: {counts['completed_episodes']} / {counts['expected_episodes']}",
                f"- Round records: {counts['round_rows']}",
                f"- Micro-slot records: {counts['micro_slot_rows']}",
                f"- Primary estimates: {len(primary)}",
                f"- Derived observables: {len(derived)}",
                f"- Single-affinity theory comparison rows: {len(theory_comparison)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (reports / "methods.md").write_text(
        "\n".join(
            [
                "# Methods",
                "",
                "Scientific identities were recovered from resolved configs, cell overrides, and compact scientific records.",
                "Information estimates use the repository's established direct-counting round-feedback estimator, whole-episode bootstrap, configured nulls, and support diagnostics.",
                "Execution shards were reconstructed into scientific cells before per-cell estimation.",
                "",
                f"Analysis hash: `{analysis_hash}`.",
                "",
                *(SINGLE_AFFINITY_METHODS if not derived.empty else ()),
            ]
        ),
        encoding="utf-8",
    )
    state_occupancy = outputs.get("state_occupancy", pd.DataFrame())
    if not state_occupancy.empty:
        state_lines = [
            "# Empirical state-space support",
            "",
            "`state_occupancy.csv` is counted directly from retained round records before estimator support filtering.",
            "Absent target-count states are genuinely unvisited in the retained trajectories; they are not interpolated.",
            "A visited state may still be absent from a state-local estimator when it lacks both controller-action values or fails another estimator support requirement.",
            "This distinction is separate from missing structural `(rho,b)` cells, which strict validation rejects.",
            "",
            "| cell_id | rho | b | target_count_before | target_fraction_before | n_observations | n_episodes |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        ordered_state = state_occupancy.sort_values(["cell_id", "target_count_before"])
        for row in ordered_state.to_dict(orient="records"):
            state_lines.append(
                "| {cell} | {rho} | {budget} | {count} | {fraction:.6g} | {observations} | {episodes} |".format(
                    cell=row.get("cell_id", ""),
                    rho=row.get("epistemic_persistence", ""),
                    budget=row.get("intervention_budget", ""),
                    count=row.get("target_count_before", ""),
                    fraction=float(row.get("target_fraction_before", math.nan)),
                    observations=int(row.get("n_observations", 0)),
                    episodes=int(row.get("n_episodes", 0)),
                )
            )
        (reports / "state_space_support.md").write_text(
            "\n".join(state_lines) + "\n", encoding="utf-8"
        )
    if (
        not episode_endpoints.empty
        and str((endpoint_recipe or {}).get("classifier", ""))
        == "relational_false_takeover_v1"
    ):
        from .episode_endpoints import false_takeover_markdown

        (reports / "false_takeover.md").write_text(
            false_takeover_markdown(episode_endpoints, episode_endpoint_summary),
            encoding="utf-8",
        )
    provenance_dir = analysis_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, provenance_dir / "study_manifest.json")
    shutil.copy2(submission_path, provenance_dir / "submission_manifest.csv")
    submission_metadata = root / "submission.json"
    if submission_metadata.is_file():
        shutil.copy2(submission_metadata, provenance_dir / "submission.json")
    for recovery in sorted(root.glob("recovery_submission_*.json")):
        shutil.copy2(recovery, provenance_dir / recovery.name)
    for recovery_manifest in sorted(root.glob("execution_manifest_recovery_*.csv")):
        shutil.copy2(recovery_manifest, provenance_dir / recovery_manifest.name)
    for entry in entries:
        source_config = Path(entry.config_path)
        if source_config.is_file():
            shutil.copy2(
                source_config,
                provenance_dir / f"config-{entry.array_index:04d}-{source_config.name}",
            )
    if recipe_path is not None:
        shutil.copy2(recipe_path, analysis_dir / "analysis_recipe.yaml")
    analysis_manifest = {
        "schema_version": 2,
        "study_id": study_id,
        "status": "complete" if validation["complete"] else "incomplete",
        "created_at": _now(),
        "scientific_input_identity": input_identity,
        "analysis_hash": analysis_hash,
        "derived_hash": derived_hash,
        "auxiliary_analysis_hash": auxiliary_hash,
        "theory": dict(theory_provenance),
        "theoretical_reference": theoretical_reference,
        "estimator_engine": "mas_cc.games.hidden_bench.imitation_round_feedback.analysis.round_information_analysis",
        "requested_statistics": list(statistics),
        "resampling": settings,
        "retention_contract": {
            "canonical_table_format": CANONICAL_TABLE_FORMAT,
            "csv_tables": True,
            "parquet_tables": False,
            "compact_estimator_summaries": True,
            "persistent_analysis_cache": False,
            "individual_null_draws": False,
            "individual_bootstrap_draws": False,
        },
        "plots": plots,
        "tables": sorted(path.name for path in tables_dir.glob("*.csv")),
        "derived_semantics": _derived_semantics(derived),
    }
    _write_json(analysis_dir / "analysis_manifest.json", analysis_manifest)
    (analysis_dir / "progress.json").unlink(missing_ok=True)
    archive = _package(analysis_dir, study_id)
    return {
        "study_id": study_id,
        "study_dir": str(root),
        "analysis_dir": str(analysis_dir),
        "valid": validation["valid"],
        "complete": validation["complete"],
        "archive": str(archive),
        "counts": counts,
    }


__all__ = [
    "DERIVED_COLUMNS",
    "ESTIMATOR_ALIASES",
    "SINGLE_AFFINITY_DERIVED",
    "aggregate_study",
]
