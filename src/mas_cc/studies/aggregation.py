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

from mas_cc.storage import canonical_hash, file_sha256
from mas_cc.storage.scientific import compact_row_to_imitation_event

from .canonical import build_canonical_tables, parquet_safe
from .discovery import DiscoveredRun, discover_cells, discover_runs
from .submission import read_submission_manifest
from .validation import validate_study, validation_markdown


ESTIMATOR_ALIASES = {
    "round_target_actuation_cmi_memory": "round_memory_target_actuation_cmi",
    "round_target_actuation_cmi_memory_phi": "round_phi_target_actuation_cmi",
    "target_signed_actuation": "round_target_signed_actuation",
}

PRIMARY_COLUMNS = (
    "study_id", "source_run_id", "cell_id", "metric", "estimator_version",
    "estimator_variant", "grouping_json", "conditioning_json", "estimate",
    "ci_low", "ci_high", "confidence", "null_type", "null_mean", "null_std",
    "p_value", "null_permutations", "bootstrap_resamples", "n_observations",
    "n_episodes", "units", "support_status", "p_plus", "p_minus",
    "non_target_exposures", "non_target_to_target", "target_exposures",
    "target_to_non_target", "analysis_hash",
)

ETA_IR_JOIN_KEYS = (
    "study_id", "source_run_id", "cell_id", "grouping_json", "conditioning_json",
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
    excluded = {"effective_affinity", "kinetic_compliance", "episode_current", "cell_current"}
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
                from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import read_round_records

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
        if run.game_type in {
            "relational_imitation_round_feedback",
            "hidden_bench_imitation_round_feedback",
        }
    }
    if not supported_prefixes:
        return events

    # Results-only artifacts preserve the exact transition encoding. Feed it
    # through the existing generic round adapter rather than estimating here.
    from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import adapt_round_record

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
                "occupation_counts_before", "occupation_counts_after",
                "target_count_before", "target_count_after",
                "truth_count_before", "truth_count_after",
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
        raise ValueError("unknown study information estimator(s): " + ", ".join(unknown))
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
        with ProcessPoolExecutor(max_workers=max(1, min(workers, len(pending)))) as pool:
            futures = {
                pool.submit(_run_information_group, payload): cell_id
                for cell_id, payload in pending.items()
            }
            for future in as_completed(futures):
                cell_id = futures[future]
                value = future.result()
                completed_results[cell_id] = value
                done += 1
                print(f"[analysis] information groups {done}/{total}: {cell_id}", flush=True)
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
            finite = [value for value in null_by_metric.get(metric, ()) if math.isfinite(value)]
            estimate = float(item["estimate"])
            p_value = (
                math.nan
                if not finite
                else (1 + sum(value >= estimate for value in finite)) / (len(finite) + 1)
            )
            estimates.append(
                {
                    "study_id": study_id,
                    "source_run_id": source_run_id,
                    "cell_id": cell_id,
                    "metric": metric,
                    "estimator_version": "round-feedback-v1",
                    "estimator_variant": item.get("main_estimator_variant", item.get("estimator_variant")),
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
                        or key in {
                            "n_episodes", "n_rounds", "number_of_actions_observed",
                            "unique_population_states", "unique_sensor_states",
                            "min_rounds_per_population_state", "median_rounds_per_population_state",
                            "max_rounds_per_population_state",
                        }
                    },
                    "support_status": _support_status(item),
                    "action_0_count": sum(str(event.U_k) == "NO_OP" for event in rows),
                    "action_1_count": sum(str(event.U_k) != "NO_OP" and event.U_k is not None for event in rows),
                    "action_entropy_bits": item.get("round_controller_action_entropy", item.get("conditional_action_entropy_bits")),
                    "dual_action_support_fraction": item.get("round_dual_action_state_fraction"),
                    "occupied_conditioning_states": item.get("round_conditioning_state_count", item.get("unique_population_states")),
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
        raise ValueError("unknown state-local analysis resolution(s): " + ", ".join(unknown))
    if not requested:
        return pd.DataFrame(columns=PRIMARY_COLUMNS)
    from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import round_information_analysis

    by_cell: dict[str, list[Any]] = {}
    for event in events:
        by_cell.setdefault(str(event.cell_id), []).append(event)
    rows: list[dict[str, Any]] = []
    statistics = (
        "round_target_actuation_cmi",
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
                slices.setdefault((x,) if extra is None else (x, value), []).append(event)
            for slice_values, sample in sorted(slices.items(), key=lambda item: str(item[0])):
                if len({str(event.U_k) for event in sample if event.U_k is not None}) < 2:
                    continue
                estimates, _ = round_information_analysis(
                    sample, statistics=statistics, bootstrap_resamples=0,
                    null_permutations=0, confidence=0.95, seed=1,
                )
                grouping = {"cell_id": cell_id, "resolution": resolution, "target_count_before": slice_values[0]}
                if extra is not None:
                    grouping[extra] = slice_values[1]
                for item in estimates:
                    rows.append({
                        "study_id": study_id,
                        "source_run_id": source_run_ids.get(cell_id.split("/", 1)[0], cell_id.split("/", 1)[0]),
                        "cell_id": cell_id,
                        "metric": str(item["statistic"]),
                        "estimator_version": "round-feedback-v1",
                        "estimator_variant": item.get("main_estimator_variant", item.get("estimator_variant")),
                        "grouping_json": json.dumps(grouping, sort_keys=True),
                        "conditioning_json": _conditioning_json(str(item["statistic"])),
                        "estimate": item["estimate"],
                        "ci_low": math.nan, "ci_high": math.nan, "confidence": 0.95,
                        "null_type": None, "null_mean": math.nan, "null_std": math.nan,
                        "p_value": math.nan, "null_permutations": 0, "bootstrap_resamples": 0,
                        "n_observations": item.get("n_rounds"), "n_episodes": item.get("n_episodes"),
                        "units": item.get("units"), "support_status": _support_status(item),
                        "analysis_hash": analysis_hash,
                        **grouping,
                    })
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
        source_run_id = source_run_ids.get(cell_id.split("/", 1)[0], cell_id.split("/", 1)[0])
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
                        "grouping_json": json.dumps({"episode_id": episode["episode_id"]}, sort_keys=True),
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
                    "support_status": summary.get("current_precision_support", "unknown"),
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
        local_episode = str(event.event.get("episode_id", event.episode_id.rsplit("/", 1)[-1]))
        target_by_round[(str(event.cell_id), local_episode, int(event.round_index))] = event.event
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
                "analysis_target": row.get("analysis_target") or row.get("round_controller_target") or round_event.get("analysis_target") or round_event.get("controller_target"),
                "round_controller_action": row.get("round_controller_action") or round_event.get("controller_action"),
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
            else "limited" if int(summary["n_episodes"]) < 10 else "adequate"
        )
        for metric in sorted(names):
            primary.append(
                {
                    "study_id": study_id,
                    "source_run_id": source_run_id,
                    "cell_id": cell_id,
                    "metric": metric,
                    "estimator_version": "study05-effective-affinity-v1",
                    "estimator_variant": "raw_transition_rate_ratio" if metric == "effective_affinity" else "raw_transition_activity",
                    "grouping_json": json.dumps({"cell_id": cell_id}, sort_keys=True),
                    "conditioning_json": json.dumps({"controlled_slot": True, "round_action": "ADVOCATE_TARGET"}, sort_keys=True),
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
                    "units": "nats" if metric == "effective_affinity" else "probability",
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
    study_id: str, runs: tuple[DiscoveredRun, ...], analysis_hash: str,
    settings: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimates: list[pd.DataFrame] = []
    support: list[pd.DataFrame] = []
    for run in runs:
        candidates = sorted(run.path.glob("*_analysis/round_information_estimates.csv"))
        candidates.extend(sorted((run.path / "analysis").glob("round_information_estimates.csv")))
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
                "estimator_variant": raw.get("main_estimator_variant", raw.get("estimator_variant")),
                "grouping_json": raw["cell_id"].map(lambda value: json.dumps({"cell_id": str(value)})),
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
        pd.concat(estimates, ignore_index=True) if estimates else pd.DataFrame(columns=PRIMARY_COLUMNS),
        pd.concat(support, ignore_index=True) if support else pd.DataFrame(),
    )


DERIVED_COLUMNS = (
    "study_id", "source_run_id", "cell_id", "metric", "grouping_json",
    "conditioning_json", "estimate", "ci_low", "ci_high", "confidence",
    "units", "dependencies_json", "support_status", "analysis_hash",
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
    response = estimates[
        estimates["metric"] == "round_target_susceptibility"
    ].copy()
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
            group = [event for event in group if event.event.get("target_count_before") == grouping["target_count_before"]]
        for key in ("conditioning_phi_bin", "conditioning_kappa_bin"):
            if key in grouping:
                group = [event for event in group if event.event.get(key) == grouping[key]]
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
        value = math.nan if transfer <= 0 else 2 * a * (1 - a) * chi * chi / (math.log(2) * transfer)
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
            rows.append(
                {**common, **item, "available": True, "reason": None}
            )
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
            key: bundle.get(key)
            for key in _SINGLE_AFFINITY_AUDIT
            if key in bundle
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
    if metric == "eta_th" and not flag("eta_th_valid"):
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
        study_id, tuple(dependencies), bundles, source_run_ids, dependencies, analysis_hash
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
        study_id, tuple(dependencies), bundles, source_run_ids, dependencies, analysis_hash
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
        study_id, tuple(dependencies), bundles, source_run_ids, dependencies, analysis_hash
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
    }
    return _single_affinity_rows(
        study_id, tuple(dependencies), bundles, source_run_ids, dependencies, analysis_hash
    )


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
        rows.extend(_build_controlled_current(study_id, bundles, run_ids, analysis_hash))
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
        "study_id", "source_config_index", "source_run_id", "source_run_path",
        "source_cell_id", "config_hash", "resolved_config_hash", "expected_episodes",
        "completed_episodes", "failed_episodes", "sealed",
    }
    coordinates = cells[[column for column in cells.columns if column == "cell_id" or column not in drop]]
    return frame.merge(coordinates, on="cell_id", how="left", suffixes=("", "_coordinate"))


def _factorial_contrasts(
    primary: pd.DataFrame, derived: pd.DataFrame, requested: Sequence[Any], analysis_hash: str
) -> pd.DataFrame:
    """Matched descriptive factorial differences, not a new estimator."""

    if "factorial_contrasts" not in {str(value) for value in requested}:
        return pd.DataFrame()
    frames = [frame for frame in (primary, derived) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    source = pd.concat(frames, ignore_index=True, sort=False)
    needed = {
        "receiver_epistemic_disposition", "controller_evidence_strategy",
        "target_semantics", "intervention_budget", "task_id", "metric", "estimate",
    }
    if not needed.issubset(source.columns):
        return pd.DataFrame()
    source = source[source["metric"].isin({
        "round_target_actuation_cmi",
        "round_target_signed_actuation",
        "round_target_susceptibility",
        "eta_ir",
        "eta_ir_state_local",
        "eta_th",
        "controlled_current",
        "target_sensing_information_nats",
    })].copy()
    keys = ["metric", "intervention_budget", "task_id"]
    rows: list[dict[str, Any]] = []

    def differences(frame, axis, low, high, kind, extra):
        index = [*keys, *extra]
        pivot = frame.pivot_table(index=index, columns=axis, values="estimate", aggfunc="mean")
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
            rows.append({
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
                "dependencies_json": json.dumps({"high": high, "low": low, "axis": axis}),
                "analysis_hash": analysis_hash,
            })

    differences(source, "receiver_epistemic_disposition", "naive", "vigilant", "vigilance", ["controller_evidence_strategy", "target_semantics"])
    differences(source, "controller_evidence_strategy", "neutral", "strategic", "evidence_strategy", ["receiver_epistemic_disposition", "target_semantics"])
    differences(source, "target_semantics", "false", "truth", "truth_false", ["receiver_epistemic_disposition", "controller_evidence_strategy"])
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


def _render_plots(recipe: Mapping[str, Any], tables: Mapping[str, pd.DataFrame], destination: Path) -> list[str]:
    raw = recipe.get("plots", {})
    destination.mkdir(parents=True, exist_ok=True)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        builtins = {
            "target_cmi_x_b": {"source": "primary_estimates", "metric": "round_target_actuation_cmi"},
            "eta_ir_x_b": {"source": "derived_observables", "metric": "eta_ir_state_local"},
            "memory_conditioning": {"source": "primary_estimates", "metric": "round_memory_target_actuation_cmi"},
            "h_eff_phi_b": {"source": "primary_estimates", "metric": "effective_affinity"},
            "gamma_eff_phi_b": {"source": "primary_estimates", "metric": "kinetic_compliance"},
        }
        raw = {str(name): builtins.get(str(name), {"source": "primary_estimates", "metric": str(name)}) for name in raw}
    if not isinstance(raw, Mapping):
        raise ValueError("analysis plots must be a list or mapping")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    for name, spec in raw.items():
        if not isinstance(spec, Mapping):
            continue
        source_name = str(spec.get("source", "primary_estimates"))
        frame = tables.get(source_name)
        if frame is None or frame.empty:
            continue
        metric = str(spec.get("metric", ""))
        subset = frame[frame["metric"].astype(str) == metric] if metric and "metric" in frame else frame
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
        groups = [("all", subset)] if facet not in subset else list(subset.groupby(str(facet), dropna=False))
        finite_estimates = pd.to_numeric(subset["estimate"], errors="coerce")
        finite_estimates = finite_estimates[np.isfinite(finite_estimates)]
        shared_vmin = float(finite_estimates.min()) if len(finite_estimates) else None
        shared_vmax = float(finite_estimates.max()) if len(finite_estimates) else None
        two_by_two = str(spec.get("layout", "")) == "2x2" and len(groups) == 4
        nrows, ncols = (2, 2) if two_by_two else (1, len(groups))
        figure, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(5 * ncols, 4 * nrows))
        for axis, (label, group) in zip(axes.flat, groups, strict=True):
            if kind == "line":
                series = str(spec.get("series", ""))
                line_groups = [("all", group)] if series not in group else group.groupby(series, dropna=False)
                for series_label, line_group in line_groups:
                    values = line_group.groupby(x, dropna=False)["estimate"].mean().sort_index()
                    axis.plot(values.index, values.values, marker="o", label=str(series_label))
                if series in group:
                    axis.legend()
                axis.set_xlabel(x)
                axis.set_ylabel(_value_label(value_column or metric, subset))
                axis.set_title(str(label))
                continue
            pivot = group.pivot_table(index=y, columns=x, values="estimate", aggfunc="mean")
            image = axis.imshow(
                pivot.to_numpy(dtype=float), aspect="auto", origin="lower",
                vmin=shared_vmin, vmax=shared_vmax,
            )
            axis.set_xticks(range(len(pivot.columns)), labels=[str(value) for value in pivot.columns])
            axis.set_yticks(range(len(pivot.index)), labels=[str(value) for value in pivot.index])
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


SINGLE_AFFINITY_METHODS = (
    "## Single-affinity derived observables", "",
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
    "reported elsewhere and is never substituted into these formulas.", "",
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
        "cache", "cell_cache", "tables", "plots", "plots-debug", "reports", "provenance"
    ):
        path = analysis_dir / name
        if path.is_dir():
            shutil.rmtree(path)
    for name in (
        "analysis_manifest.json", "analysis_recipe.yaml", "progress.json",
        "validation.json", "validation.md", f"{study_id}_analysis.zip",
        "theory_comparison.csv", "theory_state_curves.csv",
    ):
        (analysis_dir / name).unlink(missing_ok=True)
    for path in analysis_dir.rglob("*.pickle"):
        path.unlink(missing_ok=True)


def _package(analysis_dir: Path, study_id: str) -> Path:
    destination = analysis_dir / f"{study_id}_analysis.zip"
    temporary = destination.with_suffix(".zip.tmp")
    root_names = {
        "validation.json", "validation.md", "analysis_manifest.json",
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
                or (path.suffix == ".csv" and "provenance" not in relative.parts)
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
            "episodes": canonical["episodes"][[
                column
                for column in ("cell_id", "episode_id", "interaction_count", "status")
                if column in canonical["episodes"]
            ]].to_dict(orient="records"),
        }
    )


def aggregate_study(study_dir: str | Path, *, allow_incomplete: bool = False) -> dict[str, Any]:
    """Create the complete canonical analysis package for one submitted study."""

    root = Path(study_dir).expanduser().resolve()
    manifest_path = root / "study_manifest.json"
    submission_path = root / "submission_manifest.csv"
    if not manifest_path.is_file() or not submission_path.is_file():
        raise ValueError(f"not a submitted MA-CC study directory: {root}")
    study_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    study_id = str(study_manifest["study_id"])
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
        name: analysis_dir / "tables" / f"{name}.parquet"
        for name in ("cells", "episodes", "rounds", "micro_slots")
    }
    if cells:
        canonical, _ = build_canonical_tables(study_id, cells)
        validation = validate_study(entries, runs, cells, canonical)
    elif all(path.is_file() for path in retained_paths.values()):
        canonical = {
            name: pd.read_parquet(path, engine="pyarrow")
            for name, path in retained_paths.items()
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
        canonical, _ = build_canonical_tables(study_id, cells)
        validation = validate_study(entries, runs, cells, canonical)
    _reset_analysis_handoff(analysis_dir, study_id)
    tables_dir = analysis_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    validation["allow_incomplete"] = bool(allow_incomplete)
    if not validation["valid"] and allow_incomplete:
        validation["complete"] = False
        validation["warnings"].append("aggregation continued under --allow-incomplete")
    _write_json(analysis_dir / "validation.json", validation)
    (analysis_dir / "validation.md").write_text(validation_markdown(validation), encoding="utf-8")
    if not validation["valid"] and not allow_incomplete:
        raise ValueError(
            "study validation failed; inspect " + str(analysis_dir / "validation.json")
        )

    for name, frame in canonical.items():
        parquet_safe(frame).to_parquet(tables_dir / f"{name}.parquet", index=False, engine="pyarrow")

    input_identity = retained_input_identity or _scientific_identity(entries, cells, canonical)
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
                    {"episode_current", "cell_current", "effective_affinity", "kinetic_compliance"}
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
    contrasts = _factorial_contrasts(primary, derived, derived_raw, derived_hash)
    if not contrasts.empty:
        derived = pd.concat([derived, contrasts], ignore_index=True, sort=False)

    outputs = {
        "primary_estimates": primary,
        "information_estimates": information,
        "support_diagnostics": support,
        "derived_observables": derived,
    }
    if theoretical_reference == "single_affinity_revised":
        outputs["single_affinity_theory_comparison"] = _attach_coordinates(
            theory_comparison, canonical["cells"]
        )
    for name, frame in outputs.items():
        parquet_safe(frame).to_parquet(tables_dir / f"{name}.parquet", index=False, engine="pyarrow")

    plot_tables = {
        **canonical,
        "rounds": _attach_coordinates(canonical["rounds"], canonical["cells"]),
        **outputs,
    }
    plots = _render_plots(recipe, plot_tables, analysis_dir / "plots")
    reports = analysis_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    counts = validation["counts"]
    (reports / "summary.md").write_text(
        "\n".join(
            [
                f"# {study_id}", "",
                f"Aggregation status: **{'complete' if validation['complete'] else 'incomplete'}**.", "",
                f"- Runs: {counts['found_configs']} / {counts['expected_configs']}",
                f"- Cells: {counts['found_cells']} / {counts['expected_cells']}",
                f"- Completed episodes: {counts['completed_episodes']} / {counts['expected_episodes']}",
                f"- Round records: {counts['round_rows']}",
                f"- Micro-slot records: {counts['micro_slot_rows']}",
                f"- Primary estimates: {len(primary)}",
                f"- Derived observables: {len(derived)}",
                f"- Single-affinity theory comparison rows: {len(theory_comparison)}", "",
            ]
        ),
        encoding="utf-8",
    )
    (reports / "methods.md").write_text(
        "\n".join(
            [
                "# Methods", "",
                "Scientific identities were recovered from resolved configs, cell overrides, and compact scientific records.",
                "Information estimates use the repository's established direct-counting round-feedback estimator, whole-episode bootstrap, configured nulls, and support diagnostics.",
                "Execution shards were reconstructed into scientific cells before per-cell estimation.", "",
                f"Analysis hash: `{analysis_hash}`.", "",
                *(SINGLE_AFFINITY_METHODS if not derived.empty else ()),
            ]
        ),
        encoding="utf-8",
    )
    provenance_dir = analysis_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, provenance_dir / "study_manifest.json")
    shutil.copy2(submission_path, provenance_dir / "submission_manifest.csv")
    submission_metadata = root / "submission.json"
    if submission_metadata.is_file():
        shutil.copy2(submission_metadata, provenance_dir / "submission.json")
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
        "schema_version": 1,
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
            "canonical_parquet": True,
            "compact_estimator_summaries": True,
            "persistent_analysis_cache": False,
            "individual_null_draws": False,
            "individual_bootstrap_draws": False,
            "csv_table_mirrors": False,
        },
        "plots": plots,
        "tables": sorted(path.name for path in tables_dir.glob("*.parquet")),
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
