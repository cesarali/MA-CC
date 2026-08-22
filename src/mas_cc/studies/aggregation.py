"""The standard discover → validate → estimate → report → package workflow."""

from __future__ import annotations

import json
import hashlib
import math
import shutil
import zipfile
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
    "p_value", "n_observations", "n_episodes", "units", "support_status",
    "analysis_hash",
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
    }:
        return json.dumps(target_state, sort_keys=True)
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


def _information_tables(
    study_id: str,
    events: Sequence[Any],
    statistics: tuple[str, ...],
    settings: Mapping[str, Any],
    analysis_hash: str,
    source_run_ids: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import (
        ROUND_ANALYSIS_STATISTICS,
        round_information_analysis,
    )

    unknown = sorted(set(statistics) - set(ROUND_ANALYSIS_STATISTICS))
    if unknown:
        raise ValueError("unknown study information estimator(s): " + ", ".join(unknown))
    if not events or not statistics:
        return (
            pd.DataFrame(columns=PRIMARY_COLUMNS),
            pd.DataFrame(columns=["study_id", "cell_id", "metric", "permutation", "null_type", "estimate", "analysis_hash"]),
            pd.DataFrame(columns=["study_id", "cell_id", "metric", "analysis_hash"]),
        )

    grouped: dict[str, list[Any]] = {}
    for event in events:
        grouped.setdefault(str(event.cell_id), []).append(event)
    grouped["pooled"] = list(events)
    estimates: list[dict[str, Any]] = []
    nulls: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    confidence = float(settings["confidence"])
    for cell_id, rows in sorted(grouped.items()):
        group_seed = int(settings["seed"])
        if cell_id != "pooled":
            group_seed += int(hashlib.sha256(cell_id.encode("utf-8")).hexdigest()[:8], 16)
        result_rows, null_rows = round_information_analysis(
            rows,
            statistics=statistics,
            bootstrap_resamples=int(settings["bootstrap_resamples"]),
            null_permutations=int(settings["null_permutations"]),
            confidence=confidence,
            seed=group_seed,
        )
        source_run_id = "pooled"
        if cell_id != "pooled":
            source_run_id = source_run_ids.get(cell_id.split("/", 1)[0], cell_id.split("/", 1)[0])
        null_by_metric: dict[str, list[float]] = {}
        for item in null_rows:
            value = float(item["estimate"])
            null_by_metric.setdefault(str(item["statistic"]), []).append(value)
            nulls.append(
                {
                    "study_id": study_id,
                    "source_run_id": source_run_id,
                    "cell_id": cell_id,
                    "metric": item["statistic"],
                    "permutation": item["permutation"],
                    "null_type": item["null_type"],
                    "estimate": value,
                    "analysis_hash": analysis_hash,
                }
            )
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
    return pd.DataFrame(estimates, columns=PRIMARY_COLUMNS), pd.DataFrame(nulls), pd.DataFrame(support)


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
        source_run_id = "pooled" if cell_id == "pooled" else source_run_ids.get(
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
                    "n_observations": summary["n_observations"],
                    "n_episodes": summary["n_episodes"],
                    "units": "nats" if metric == "effective_affinity" else "probability",
                    "support_status": status,
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


def _ingest_existing_information(study_id: str, runs: tuple[DiscoveredRun, ...], analysis_hash: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    estimates: list[pd.DataFrame] = []
    nulls: list[pd.DataFrame] = []
    support: list[pd.DataFrame] = []
    for run in runs:
        candidates = sorted(run.path.glob("*_analysis/round_information_estimates.csv"))
        candidates.extend(sorted((run.path / "analysis").glob("round_information_estimates.csv")))
        if not candidates:
            continue
        estimate_path = candidates[0]
        prefix = f"config-{run.entry.array_index:04d}"
        raw = pd.read_csv(estimate_path)
        normalized = pd.DataFrame(
            {
                "study_id": study_id,
                "source_run_id": run.run_id,
                "cell_id": raw["cell_id"].map(lambda value: "pooled" if str(value) == "pooled" else f"{prefix}/{value}"),
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
                "n_observations": raw.get("n_rounds"),
                "n_episodes": raw.get("n_episodes"),
                "units": raw.get("units"),
                "support_status": raw.apply(lambda row: _support_status(row), axis=1),
                "analysis_hash": analysis_hash,
            }
        )
        estimates.append(normalized)
        null_path = estimate_path.with_name("round_information_nulls.csv")
        if null_path.is_file():
            item = pd.read_csv(null_path).rename(columns={"statistic": "metric"})
            item.insert(0, "study_id", study_id)
            item.insert(1, "source_run_id", run.run_id)
            item["cell_id"] = item["cell_id"].map(lambda value: "pooled" if str(value) == "pooled" else f"{prefix}/{value}")
            item["analysis_hash"] = analysis_hash
            nulls.append(item)
        support_path = estimate_path.with_name("round_support_diagnostics.csv")
        if support_path.is_file():
            item = pd.read_csv(support_path)
            item.insert(0, "study_id", study_id)
            item.insert(1, "source_run_id", run.run_id)
            item["cell_id"] = item["cell_id"].map(lambda value: "pooled" if str(value) == "pooled" else f"{prefix}/{value}")
            item["analysis_hash"] = analysis_hash
            support.append(item)
    return (
        pd.concat(estimates, ignore_index=True) if estimates else pd.DataFrame(columns=PRIMARY_COLUMNS),
        pd.concat(nulls, ignore_index=True) if nulls else pd.DataFrame(),
        pd.concat(support, ignore_index=True) if support else pd.DataFrame(),
    )


def _derived(
    study_id: str,
    requested: Sequence[Any],
    estimates: pd.DataFrame,
    events: Sequence[Any],
    analysis_hash: str,
) -> pd.DataFrame:
    columns = [
        "study_id", "source_run_id", "cell_id", "metric", "grouping_json",
        "conditioning_json", "estimate", "units", "dependencies_json",
        "support_status", "analysis_hash",
    ]
    names = {str(value) for value in requested}
    if "eta_ir" not in names or estimates.empty:
        return pd.DataFrame(columns=columns)
    event_groups: dict[str, list[Any]] = {}
    for event in events:
        event_groups.setdefault(str(event.cell_id), []).append(event)
    event_groups["pooled"] = list(events)
    rows: list[dict[str, Any]] = []

    # eta_IR's response is the target-state-matched signed actuation.  The
    # marginal share response is intentionally ineligible: it does not have
    # the CMI's target-before conditioning and cannot be silently joined to it.
    cmi = estimates[estimates["metric"] == "round_target_actuation_cmi"].copy()
    response = estimates[estimates["metric"] == "round_target_signed_actuation"].copy()
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
                "metric": "eta_ir",
                "grouping_json": item["grouping_json"],
                "conditioning_json": item["conditioning_json"],
                "estimate": value,
                "units": "dimensionless",
                "dependencies_json": json.dumps(
                    {
                        "join_keys": list(ETA_IR_JOIN_KEYS),
                        "metrics": [
                            "round_target_actuation_cmi",
                            "round_target_signed_actuation",
                            "action_frequency",
                        ],
                    },
                    sort_keys=True,
                ),
                "support_status": str(item["support_status_cmi"]),
                "analysis_hash": analysis_hash,
            }
        )
    return pd.DataFrame(rows, columns=columns)


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


def _render_plots(recipe: Mapping[str, Any], tables: Mapping[str, pd.DataFrame], destination: Path) -> list[str]:
    raw = recipe.get("plots", {})
    destination.mkdir(parents=True, exist_ok=True)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        builtins = {
            "target_cmi_x_b": {"source": "primary_estimates", "metric": "round_target_actuation_cmi"},
            "eta_ir_x_b": {"source": "derived_observables", "metric": "eta_ir"},
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
        if "support_status" in subset:
            subset = subset[subset["support_status"].astype(str) != "unsupported"]
        x, y = str(spec.get("x", "")), str(spec.get("y", ""))
        if subset.empty or "estimate" not in subset:
            continue
        if x not in subset or y not in subset:
            figure, axis = plt.subplots(figsize=(max(5, len(subset) * 0.5), 4))
            labels = subset.get("cell_id", pd.Series(range(len(subset)))).astype(str)
            axis.bar(range(len(subset)), subset["estimate"].astype(float))
            axis.set_xticks(range(len(subset)), labels=labels, rotation=45, ha="right")
            axis.set_ylabel(metric)
            axis.set_title(str(name))
            figure.tight_layout()
            path = destination / f"{name}.png"
            figure.savefig(path, dpi=150)
            plt.close(figure)
            paths.append(str(path))
            continue
        facet = spec.get("facet")
        groups = [("all", subset)] if facet not in subset else list(subset.groupby(str(facet), dropna=False))
        figure, axes = plt.subplots(1, len(groups), squeeze=False, figsize=(5 * len(groups), 4))
        for axis, (label, group) in zip(axes[0], groups, strict=True):
            pivot = group.pivot_table(index=y, columns=x, values="estimate", aggfunc="mean")
            image = axis.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower")
            axis.set_xticks(range(len(pivot.columns)), labels=[str(value) for value in pivot.columns])
            axis.set_yticks(range(len(pivot.index)), labels=[str(value) for value in pivot.index])
            axis.set_xlabel(x)
            axis.set_ylabel(y)
            axis.set_title(str(label) if facet in subset else metric)
            figure.colorbar(image, ax=axis)
        figure.tight_layout()
        path = destination / f"{name}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(str(path))
    return paths


def _package(analysis_dir: Path, study_id: str) -> Path:
    destination = analysis_dir / f"{study_id}_analysis.zip"
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(analysis_dir.rglob("*")):
            if not path.is_file() or path in {destination, temporary} or path.name.endswith(":Zone.Identifier"):
                continue
            info = zipfile.ZipInfo(str(path.relative_to(analysis_dir)).replace("\\", "/"))
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
    settings = _resampling(recipe)

    runs = discover_runs(entries)
    cells = discover_cells(runs)
    canonical, context = build_canonical_tables(study_id, cells)
    analysis_dir = root / "analysis"
    tables_dir = analysis_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_study(entries, runs, cells, canonical)
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
        if name in {"cells", "episodes"}:
            parquet_safe(frame).to_csv(tables_dir / f"{name}.csv", index=False)

    input_identity = _scientific_identity(entries, cells, canonical)
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
        }
    )
    cache_dir = analysis_dir / "cache" / analysis_hash
    cache_files = {
        "estimates": cache_dir / "information_estimates.parquet",
        "nulls": cache_dir / "information_nulls.parquet",
        "support": cache_dir / "support_diagnostics.parquet",
    }
    cache_hit = all(path.is_file() for path in cache_files.values())
    events = _round_events(runs, context["scientific_frames"])
    source_run_ids = {
        f"config-{run.entry.array_index:04d}": run.run_id for run in runs
    }
    if cache_hit:
        information = pd.read_parquet(cache_files["estimates"])
        nulls = pd.read_parquet(cache_files["nulls"])
        support = pd.read_parquet(cache_files["support"])
    else:
        if statistics:
            information, nulls, support = _information_tables(
                study_id, events, statistics, settings, analysis_hash, source_run_ids
            )
        else:
            information, nulls, support = _ingest_existing_information(
                study_id, runs, analysis_hash
            )
        cache_dir.mkdir(parents=True, exist_ok=True)
        parquet_safe(information).to_parquet(cache_files["estimates"], index=False)
        parquet_safe(nulls).to_parquet(cache_files["nulls"], index=False)
        parquet_safe(support).to_parquet(cache_files["support"], index=False)

    information = _attach_coordinates(information, canonical["cells"])
    support = _attach_coordinates(support, canonical["cells"])
    nulls = _attach_coordinates(nulls, canonical["cells"])
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
    primary = pd.concat(
        [information, _attach_coordinates(current_primary, canonical["cells"]), _attach_coordinates(affinity_primary, canonical["cells"])],
        ignore_index=True,
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
        {"primary_analysis_hash": analysis_hash, "derived": list(derived_raw), "version": 1}
    )
    derived = _derived(study_id, derived_raw, information, events, derived_hash)
    derived = _attach_coordinates(derived, canonical["cells"])

    outputs = {
        "primary_estimates": primary,
        "information_estimates": information,
        "information_nulls": nulls,
        "support_diagnostics": support,
        "derived_observables": derived,
    }
    for name, frame in outputs.items():
        parquet_safe(frame).to_parquet(tables_dir / f"{name}.parquet", index=False, engine="pyarrow")
        if len(frame) <= 100_000:
            parquet_safe(frame).to_csv(tables_dir / f"{name}.csv", index=False)

    plot_tables = {**canonical, **outputs}
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
                f"- Derived observables: {len(derived)}", "",
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
                "Execution shards were merged before pooled estimation; shard-level CMIs were not averaged.", "",
                f"Analysis hash: `{analysis_hash}`.", "",
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
        "cache_hit": cache_hit,
        "estimator_engine": "mas_cc.games.hidden_bench.imitation_round_feedback.analysis.round_information_analysis",
        "plots": plots,
        "tables": sorted(path.name for path in tables_dir.glob("*.parquet")),
    }
    _write_json(analysis_dir / "analysis_manifest.json", analysis_manifest)
    archive = _package(analysis_dir, study_id)
    return {
        "study_id": study_id,
        "study_dir": str(root),
        "analysis_dir": str(analysis_dir),
        "valid": validation["valid"],
        "complete": validation["complete"],
        "cache_hit": cache_hit,
        "archive": str(archive),
        "counts": counts,
    }


__all__ = ["ESTIMATOR_ALIASES", "aggregate_study"]
