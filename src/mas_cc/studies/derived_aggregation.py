"""Derived study-level summaries of already estimated physical control cells.

This module coordinates the established round-feedback and single-affinity
estimators.  It never pools observations from distinct physical cells into a
new CMI estimator: each point, bootstrap draw, and policy-null draw is first
computed inside its cell and only then combined with declared weights.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from mas_cc.analysis.single_affinity import controlled_rows, eta_ir, susceptibility_summary
from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import (
    MAIN_ESTIMATOR_VARIANT,
    ROUND_CONDITIONING_STATE,
    _estimate_for,
    bootstrap_episode_rows,
    conditional_action_entropy_bits,
    policy_resampling_null,
    round_overlap_diagnostics,
)

ANALYSIS_SEMANTICS_VERSION = "study-control-aggregation-v1"
TARGET_CMI = "round_target_actuation_cmi"
SUPPORTED_METRICS = (
    TARGET_CMI,
    "susceptibility_occupancy_weighted",
    "round_target_information_fraction",
    "eta_ir",
)
PROTECTED_COORDINATES = (
    "task_id",
    "epistemic_persistence",
    "intervention_budget",
    "target_semantics",
    "model",
    "model_name",
    "provider",
    "population_size",
    "social_group_size",
    "sensor_sample_size",
    "beta",
    "threshold",
    "controller_communication_policy",
    "controller_communication_policy_version",
    "controller_actuation_mode",
    "message_mode",
    "receiver_epistemic_disposition",
    "controller_evidence_strategy",
)


@dataclass(frozen=True)
class StudyAggregateOutputs:
    study_metrics: pd.DataFrame
    state_local_metrics: pd.DataFrame
    stability: pd.DataFrame
    state_local_reconstruction: pd.DataFrame


@dataclass
class _CellCalculation:
    cell_id: str
    coordinates: dict[str, Any]
    n_observations: int
    n_episodes: int
    point: dict[str, float]
    support: dict[str, float]
    bootstrap: list[dict[str, float]]
    null: tuple[float, ...]


def _stable_seed(seed: int, *parts: Any) -> int:
    payload = "\x1f".join(map(str, (seed, *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _normalize_semantics(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"true", "truth", "correct"}:
        return "truth"
    if text in {"false", "incorrect", "adversarial"}:
        return "false"
    return text


def _components(rows: Sequence[Any]) -> dict[str, float]:
    eligible = controlled_rows(rows)
    if not eligible:
        return {name: math.nan for name in (
            "T", "H", "chi", "eta_if", "ir_numerator", "ir_denominator", "eta_ir"
        )}
    response = susceptibility_summary(eligible)
    information_response = eta_ir(eligible)
    transfer = float(
        getattr(_estimate_for(TARGET_CMI, eligible), MAIN_ESTIMATOR_VARIANT)
    )
    entropy = conditional_action_entropy_bits(
        [str(row.U_k) for row in eligible],
        [ROUND_CONDITIONING_STATE[TARGET_CMI](row) for row in eligible],
    )
    return {
        "T": transfer,
        "H": entropy,
        "chi": _finite(response.get("susceptibility_occupancy_weighted")),
        "eta_if": transfer / entropy if math.isfinite(entropy) and entropy > 1e-12 else math.nan,
        "ir_numerator": _finite(information_response.get("eta_ir_pinsker_numerator_bits")),
        "ir_denominator": _finite(information_response.get("eta_ir_denominator_T_bits")),
        "eta_ir": _finite(information_response.get("eta_ir")),
    }


def _status(support: Mapping[str, Any]) -> str:
    actions = int(_finite(support.get("number_of_actions_observed")) or 0)
    dual = _finite(support.get("round_dual_action_state_fraction"))
    singleton = _finite(support.get("round_singleton_fraction"))
    if actions < 2 or not math.isfinite(dual) or dual <= 0:
        return "unsupported"
    if dual < 0.25 or (math.isfinite(singleton) and singleton > 0.5):
        return "limited"
    return "adequate"


def _cell_calculation(
    cell_id: str,
    rows: Sequence[Any],
    coordinates: Mapping[str, Any],
    *,
    bootstrap_resamples: int,
    null_permutations: int,
    seed: int,
) -> _CellCalculation:
    eligible = controlled_rows(rows)
    support = round_overlap_diagnostics(
        eligible, state=ROUND_CONDITIONING_STATE[TARGET_CMI]
    )
    support["number_of_actions_observed"] = len(
        {str(row.U_k) for row in eligible if row.U_k is not None}
    )
    bootstrap = [
        _components(draw)
        for draw in bootstrap_episode_rows(
            eligible,
            resamples=bootstrap_resamples,
            seed=_stable_seed(seed, "bootstrap", cell_id),
        )
    ]
    null = policy_resampling_null(
        TARGET_CMI,
        eligible,
        permutations=null_permutations,
        seed=_stable_seed(seed, "null", cell_id),
    ) if eligible else ()
    return _CellCalculation(
        cell_id=cell_id,
        coordinates=dict(coordinates),
        n_observations=len(eligible),
        n_episodes=len({str(row.episode_id) for row in eligible}),
        point=_components(eligible),
        support={key: _finite(value) for key, value in support.items()},
        bootstrap=bootstrap,
        null=tuple(map(float, null)),
    )


def _weighted(values: Sequence[float], weights: Sequence[float]) -> float:
    pairs = [(float(v), float(w)) for v, w in zip(values, weights, strict=True)
             if math.isfinite(float(v)) and math.isfinite(float(w)) and float(w) > 0]
    if not pairs:
        return math.nan
    total = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / total


def _metric_value(metric: str, components: Mapping[str, float]) -> tuple[float, float, float, str]:
    if metric == TARGET_CMI:
        return components["T"], components["T"], math.nan, "bits"
    if metric == "susceptibility_occupancy_weighted":
        return components["chi"], components["chi"], math.nan, "target_fraction_per_cycle"
    if metric == "round_target_information_fraction":
        numerator, denominator = components["T"], components["H"]
        value = numerator / denominator if math.isfinite(denominator) and denominator > 1e-12 else math.nan
        return value, numerator, denominator, "dimensionless"
    if metric == "eta_ir":
        numerator, denominator = components["ir_numerator"], components["ir_denominator"]
        value = numerator / denominator if math.isfinite(denominator) and denominator > 0 else math.nan
        return value, numerator, denominator, "dimensionless"
    raise ValueError(f"unsupported derived study metric {metric!r}")


def _aggregate_components(cells: Sequence[_CellCalculation], weights: Sequence[float], *, draw: int | None = None) -> dict[str, float]:
    source = [cell.point if draw is None else cell.bootstrap[draw] for cell in cells]
    return {
        name: _weighted([item[name] for item in source], weights)
        for name in ("T", "H", "chi", "ir_numerator", "ir_denominator")
    }


def _varying_coordinates(cells: pd.DataFrame) -> set[str]:
    ignored = {
        "cell_id", "source_cell_id", "source_run_id", "source_run_path",
        "source_config_index", "source_extension_index", "source_submission_attempt",
        "study_id", "config_hash", "resolved_config_hash", "recorded_resolved_config_hash",
        "expected_episodes", "completed_episodes", "failed_episodes", "sealed",
    }
    return {
        column for column in cells.columns if column not in ignored
        and cells[column].dropna().astype(str).nunique() > 1
    }


def _validate_groupings(config: Mapping[str, Any], cells: pd.DataFrame) -> list[dict[str, Any]]:
    raw = config.get("groupings", ())
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("derived_study_aggregates.groupings must be a list")
    varying = _varying_coordinates(cells)
    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError("each derived study grouping must be a mapping")
        group_by = tuple(map(str, item.get("group_by", ())))
        marginalize = tuple(map(str, item.get("marginalize", ())))
        if set(group_by) & set(marginalize):
            raise ValueError("group_by and marginalize cannot overlap")
        missing = sorted((set(group_by) | set(marginalize)) - set(cells.columns))
        if missing:
            raise ValueError("unknown derived aggregation coordinate(s): " + ", ".join(missing))
        unaccounted = sorted(
            coordinate for coordinate in varying.intersection(PROTECTED_COORDINATES)
            if coordinate not in group_by and coordinate not in marginalize
        )
        if unaccounted:
            raise ValueError(
                "derived aggregation would silently mix scientific coordinate(s): "
                + ", ".join(unaccounted)
            )
        weighting = str(item.get("weighting", "balanced_cell"))
        if weighting != "balanced_cell":
            raise ValueError("study-level weighting must be balanced_cell")
        result.append({
            "name": str(item.get("name", f"grouping_{index}")),
            "group_by": group_by,
            "marginalize": marginalize,
            "weighting": weighting,
        })
    return result


def _support_rollup(cells: Sequence[_CellCalculation], weights: Sequence[float], expected: int) -> dict[str, Any]:
    statuses = [_status(cell.support) for cell in cells]
    return {
        "n_contributing_cells": len(cells),
        "n_expected_cells": int(expected),
        "cell_coverage_fraction": len(cells) / expected if expected else math.nan,
        "n_episodes": sum(cell.n_episodes for cell in cells),
        "n_round_observations": sum(cell.n_observations for cell in cells),
        "n_observations": sum(cell.n_observations for cell in cells),
        "min_cell_observations": min((cell.n_observations for cell in cells), default=0),
        "max_cell_observations": max((cell.n_observations for cell in cells), default=0),
        "weighted_dual_action_state_fraction": _weighted([cell.support.get("round_dual_action_state_fraction", math.nan) for cell in cells], weights),
        "weighted_dual_action_event_fraction": _weighted([cell.support.get("round_dual_action_event_fraction", math.nan) for cell in cells], weights),
        "weighted_single_action_slice_fraction": _weighted([cell.support.get("round_single_action_slice_fraction", math.nan) for cell in cells], weights),
        "weighted_singleton_fraction": _weighted([cell.support.get("round_singleton_fraction", math.nan) for cell in cells], weights),
        "conditioning_state_count_sum": int(sum(cell.support.get("round_conditioning_state_count", 0) for cell in cells)),
        "support_status": "unsupported" if not cells or all(value == "unsupported" for value in statuses) else "limited" if len(cells) < expected or any(value != "adequate" for value in statuses) else "adequate",
    }


def _study_rows(
    calculations: Mapping[str, _CellCalculation],
    expected_cells: pd.DataFrame,
    groupings: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    *, confidence: float,
    bootstrap_resamples: int,
    null_permutations: int,
    analysis_hash: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    alpha = (1.0 - confidence) / 2.0
    for spec in groupings:
        keys = list(spec["group_by"])
        source = expected_cells.copy()
        source = source[source["cell_id"].astype(str).isin(calculations)]
        grouped = source.groupby(keys, dropna=False, sort=True) if keys else [((), source)]
        expected_grouped = expected_cells.groupby(keys, dropna=False, sort=True) if keys else [((), expected_cells)]
        expected_lookup = {
            tuple(value) if isinstance(value, tuple) else (value,): len(group)
            for value, group in expected_grouped
        }
        for key_values, frame in grouped:
            key_tuple = tuple(key_values) if isinstance(key_values, tuple) else (key_values,)
            group_cells = [calculations[str(cell_id)] for cell_id in sorted(frame["cell_id"].astype(str))]
            supported = [cell for cell in group_cells if _status(cell.support) != "unsupported"]
            weights = [1.0] * len(supported)
            expected = expected_lookup.get(key_tuple, len(group_cells))
            support = _support_rollup(supported, weights, expected)
            point = _aggregate_components(supported, weights) if supported else {}
            draws = [
                _aggregate_components(supported, weights, draw=index)
                for index in range(bootstrap_resamples)
            ] if supported and all(len(cell.bootstrap) >= bootstrap_resamples for cell in supported) else []
            null_draws = [
                _weighted([cell.null[index] for cell in supported], weights)
                for index in range(null_permutations)
            ] if supported and all(len(cell.null) >= null_permutations for cell in supported) else []
            finite_null = [value for value in null_draws if math.isfinite(value)]
            grouping_values = dict(zip(keys, key_tuple, strict=True))
            for metric in metrics:
                value, numerator, denominator, units = _metric_value(metric, point) if point else (math.nan, math.nan, math.nan, "dimensionless")
                bootstrap_values = [_metric_value(metric, draw)[0] for draw in draws]
                bootstrap_values = [value for value in bootstrap_values if math.isfinite(value)]
                is_transfer = metric == TARGET_CMI
                null_mean = float(np.mean(finite_null)) if is_transfer and finite_null else math.nan
                null_adjusted_bootstrap = (
                    [draw - null_mean for draw in bootstrap_values]
                    if is_transfer and math.isfinite(null_mean)
                    else []
                )
                rows.append({
                    "metric": metric,
                    "estimate": value,
                    "ci_low": float(np.quantile(bootstrap_values, alpha)) if bootstrap_values else math.nan,
                    "ci_high": float(np.quantile(bootstrap_values, 1.0 - alpha)) if bootstrap_values else math.nan,
                    "confidence": confidence,
                    "units": units,
                    "aggregation_name": spec["name"],
                    "aggregation_level": "+".join(spec["marginalize"]) + "_marginalized",
                    "aggregation_weight": spec["weighting"],
                    "marginalized_dimensions": json.dumps(list(spec["marginalize"])),
                    "grouping_json": json.dumps(grouping_values, sort_keys=True, default=str),
                    **grouping_values,
                    **{dimension: math.nan for dimension in spec["marginalize"]},
                    "estimate_raw_bits": value if is_transfer else math.nan,
                    "null_mean_bits": null_mean,
                    "null_sd_bits": float(np.std(finite_null, ddof=1)) if is_transfer and len(finite_null) > 1 else math.nan,
                    "null_ci_low_bits": float(np.quantile(finite_null, alpha)) if is_transfer and finite_null else math.nan,
                    "null_ci_high_bits": float(np.quantile(finite_null, 1.0 - alpha)) if is_transfer and finite_null else math.nan,
                    "null_adjusted_bits": value - null_mean if is_transfer and math.isfinite(value) and math.isfinite(null_mean) else math.nan,
                    "null_adjusted_ci_low_bits": float(np.quantile(null_adjusted_bootstrap, alpha)) if null_adjusted_bootstrap else math.nan,
                    "null_adjusted_ci_high_bits": float(np.quantile(null_adjusted_bootstrap, 1.0 - alpha)) if null_adjusted_bootstrap else math.nan,
                    "permutation_p_value": ((1 + sum(draw >= value for draw in finite_null)) / (len(finite_null) + 1)) if is_transfer and finite_null and math.isfinite(value) else math.nan,
                    "n_null_draws": len(finite_null) if is_transfer else 0,
                    "component_numerator": numerator,
                    "component_denominator": denominator,
                    **support,
                    "bootstrap_unit": "episode",
                    "bootstrap_scope": "stratified_by_physical_cell",
                    "bootstrap_resamples": len(draws),
                    "analysis_semantics_version": ANALYSIS_SEMANTICS_VERSION,
                    "analysis_hash": analysis_hash,
                })
    return rows


def _state_local_calculations(
    events: Sequence[Any],
    cells: pd.DataFrame,
    *,
    bins: int,
    bootstrap_resamples: int,
    null_permutations: int,
    seed: int,
) -> pd.DataFrame:
    coordinates = cells.set_index(cells["cell_id"].astype(str)).to_dict(orient="index")
    def bin_index(event: Any) -> int:
        population = int(event.event.get("N") or sum(event.N_k))
        return min(int((event.target_before / population) * bins), bins - 1)

    by_cell: dict[str, list[Any]] = {}
    for event in events:
        by_cell.setdefault(str(event.cell_id), []).append(event)
    rows: list[dict[str, Any]] = []
    for cell_id, cell_events in sorted(by_cell.items()):
        eligible_cell = controlled_rows(cell_events)
        if not eligible_cell:
            continue
        point_bins: dict[int, list[Any]] = {}
        for event in eligible_cell:
            point_bins.setdefault(bin_index(event), []).append(event)
        bootstrap_bins: list[dict[int, list[Any]]] = []
        for draw in bootstrap_episode_rows(
            eligible_cell,
            resamples=bootstrap_resamples,
            seed=_stable_seed(seed, "state-local-bootstrap", cell_id),
        ):
            grouped_draw: dict[int, list[Any]] = {}
            for event in controlled_rows(draw):
                grouped_draw.setdefault(bin_index(event), []).append(event)
            bootstrap_bins.append(grouped_draw)
        for index, eligible in sorted(point_bins.items()):
            components = _components(eligible)
            overlap = round_overlap_diagnostics(
                eligible, state=lambda row: row.target_before
            )
            action_count = len(
                {str(row.U_k) for row in eligible if row.U_k is not None}
            )
            overlap["number_of_actions_observed"] = action_count
            null = policy_resampling_null(
                TARGET_CMI,
                eligible,
                permutations=null_permutations,
                seed=_stable_seed(seed, "state-local-null", cell_id, index),
            )
            bootstrap = [
                {
                    **_components(draw.get(index, ())),
                    "n_observations": float(len(draw.get(index, ()))),
                }
                for draw in bootstrap_bins
            ]
            common = {
                "cell_id": cell_id,
                **coordinates.get(cell_id, {}),
                "target_fraction_bin_index": index,
                "target_fraction_bin_lower": index / bins,
                "target_fraction_bin_upper": (index + 1) / bins,
                "target_fraction_bin_center": (index + 0.5) / bins,
                "target_fraction_bin_count": bins,
                "n_observations": len(eligible),
                "n_episodes": len({str(row.episode_id) for row in eligible}),
                **overlap,
            }
            for metric in SUPPORTED_METRICS:
                estimate, numerator, denominator, units = _metric_value(
                    metric, components
                )
                rows.append({
                    **common,
                    "metric": metric,
                    "estimate": estimate,
                    "units": units,
                    "component_numerator": numerator,
                    "component_denominator": denominator,
                    "bootstrap_draws": tuple(bootstrap),
                    "null_draws": tuple(null) if metric == TARGET_CMI else (),
                })
    return pd.DataFrame(rows)


def _validate_state_local_groupings(
    config: Mapping[str, Any], cells: pd.DataFrame
) -> list[dict[str, Any]]:
    raw = config.get("groupings", ())
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("derived_study_aggregates.state_local.groupings must be a list")
    varying = _varying_coordinates(cells)
    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError("each state-local grouping must be a mapping")
        group_by = tuple(map(str, item.get("group_by", ())))
        marginalize = tuple(map(str, item.get("marginalize", ())))
        if set(group_by) & set(marginalize):
            raise ValueError("state-local group_by and marginalize cannot overlap")
        missing = sorted((set(group_by) | set(marginalize)) - set(cells.columns))
        if missing:
            raise ValueError(
                "unknown state-local aggregation coordinate(s): " + ", ".join(missing)
            )
        unaccounted = sorted(
            coordinate
            for coordinate in varying.intersection(PROTECTED_COORDINATES)
            if coordinate not in group_by and coordinate not in marginalize
        )
        if unaccounted:
            raise ValueError(
                "state-local aggregation would silently mix scientific coordinate(s): "
                + ", ".join(unaccounted)
            )
        weighting = str(item.get("weighting", config.get("weighting", "n_observations")))
        if weighting not in {"n_observations", "balanced_cell"}:
            raise ValueError(
                "state-local weighting must be n_observations or balanced_cell"
            )
        result.append({
            "name": str(item.get("name", f"state_local_{index}")),
            "group_by": group_by,
            "marginalize": marginalize,
            "weighting": weighting,
        })
    return result


def _state_local_rows(
    local: pd.DataFrame,
    groupings: Sequence[Mapping[str, Any]],
    *,
    confidence: float,
    bootstrap_resamples: int,
    analysis_hash: str,
) -> pd.DataFrame:
    if local.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    alpha = (1.0 - confidence) / 2.0
    for item in groupings:
        group_by = list(item["group_by"])
        marginalize = list(item["marginalize"])
        weighting = str(item["weighting"])
        keys = [*group_by, "target_fraction_bin_index", "target_fraction_bin_lower", "target_fraction_bin_upper", "target_fraction_bin_center", "target_fraction_bin_count"]
        for coordinates, group in local.groupby([*keys, "metric"], dropna=False, sort=True):
            metric = str(coordinates[-1])
            values = pd.to_numeric(group["estimate"], errors="coerce")
            weights = (
                pd.to_numeric(group["n_observations"], errors="coerce")
                if weighting == "n_observations"
                else pd.Series(1.0, index=group.index)
            )
            supported = pd.Series(
                [
                    _status(record) != "unsupported"
                    for record in group.to_dict(orient="records")
                ],
                index=group.index,
            )
            valid = np.isfinite(values) & (weights > 0) & supported
            selected = group.loc[valid]
            selected_weights = weights.loc[valid].tolist()
            if metric in {"round_target_information_fraction", "eta_ir"}:
                numerator = _weighted(pd.to_numeric(selected["component_numerator"], errors="coerce"), selected_weights)
                denominator = _weighted(pd.to_numeric(selected["component_denominator"], errors="coerce"), selected_weights)
                estimate = numerator / denominator if math.isfinite(denominator) and denominator > 0 else math.nan
            else:
                estimate = _weighted(values.loc[valid].tolist(), selected_weights)
                numerator, denominator = (estimate, math.nan)
            null_draws: list[float] = []
            if metric == TARGET_CMI and not selected.empty:
                draw_count = min((len(value) for value in selected["null_draws"]), default=0)
                null_draws = [_weighted([value[draw] for value in selected["null_draws"]], selected_weights) for draw in range(draw_count)]
            null_mean = float(np.mean(null_draws)) if null_draws else math.nan
            bootstrap_values: list[float] = []
            for draw in range(bootstrap_resamples):
                draw_components = [value[draw] for value in selected["bootstrap_draws"]]
                draw_weights = [
                    float(component["n_observations"])
                    if weighting == "n_observations"
                    else 1.0
                    for component in draw_components
                ]
                aggregated = {
                    name: _weighted(
                        [component[name] for component in draw_components], draw_weights
                    )
                    for name in ("T", "H", "chi", "ir_numerator", "ir_denominator")
                }
                draw_value = _metric_value(metric, aggregated)[0]
                if math.isfinite(draw_value):
                    bootstrap_values.append(draw_value)
            coordinate_values = coordinates[:-1]
            output_coordinates = dict(zip(keys, coordinate_values, strict=True))
            rho_values = (
                sorted(selected["epistemic_persistence"].dropna().astype(str).unique())
                if "epistemic_persistence" in selected
                else []
            )
            semantic_values = (
                sorted(selected["target_semantics"].dropna().astype(str).unique())
                if "target_semantics" in selected
                else []
            )
            support_statuses = [
                _status(record)
                for record in selected.to_dict(orient="records")
            ]
            balanced_semantics_incomplete = (
                weighting == "balanced_cell"
                and "target_semantics" in marginalize
                and set(semantic_values) != {"truth", "false"}
            )
            rows.append({
                **output_coordinates,
                **{dimension: math.nan for dimension in marginalize},
                "metric": metric,
                "estimate": estimate,
                "units": str(selected["units"].iloc[0]) if not selected.empty else None,
                "ci_low": float(np.quantile(bootstrap_values, alpha)) if bootstrap_values else math.nan,
                "ci_high": float(np.quantile(bootstrap_values, 1.0 - alpha)) if bootstrap_values else math.nan,
                "bootstrap_sd": float(np.std(bootstrap_values, ddof=1)) if len(bootstrap_values) > 1 else math.nan,
                "bootstrap_resamples": len(bootstrap_values),
                "bootstrap_unit": "episode",
                "bootstrap_scope": "stratified_by_physical_cell",
                "aggregation_name": str(item["name"]),
                "aggregation_level": "+".join(marginalize) + "_marginalized_state_local",
                "aggregation_scope": "state_local_" + "+".join(marginalize) + "_marginalized",
                "aggregation_weight": weighting,
                "marginalized_dimensions": json.dumps(marginalize),
                "n_observations": int(
                    pd.to_numeric(selected["n_observations"], errors="coerce").sum()
                ),
                "state_occupancy": int(
                    pd.to_numeric(selected["n_observations"], errors="coerce").sum()
                ),
                "n_contributing_cells": int(selected["cell_id"].nunique()),
                "n_rho_contributing": int(selected["epistemic_persistence"].nunique()) if "epistemic_persistence" in selected else 0,
                "n_target_semantics_contributing": int(selected["target_semantics"].nunique()) if "target_semantics" in selected else 0,
                "rho_values_json": json.dumps(rho_values),
                "target_semantics_values_json": json.dumps(semantic_values),
                "weighted_dual_action_state_fraction": _weighted(pd.to_numeric(selected["round_dual_action_state_fraction"], errors="coerce"), selected_weights),
                "weighted_dual_action_event_fraction": _weighted(pd.to_numeric(selected["round_dual_action_event_fraction"], errors="coerce"), selected_weights),
                "dual_action_supported": bool(support_statuses) and all(value != "unsupported" for value in support_statuses),
                "support_status": "unsupported" if selected.empty or balanced_semantics_incomplete else "limited" if any(value != "adequate" for value in support_statuses) else "adequate",
                "null_mean": null_mean,
                "null_sd": float(np.std(null_draws, ddof=1)) if len(null_draws) > 1 else math.nan,
                "null_adjusted_estimate": estimate - null_mean if math.isfinite(estimate) and math.isfinite(null_mean) else math.nan,
                "permutation_p_value": ((1 + sum(value >= estimate for value in null_draws)) / (len(null_draws) + 1)) if null_draws and math.isfinite(estimate) else math.nan,
                "n_null_draws": len(null_draws),
                "component_numerator": numerator,
                "component_denominator": denominator,
                "descriptive_only": True,
                "analysis_semantics_version": ANALYSIS_SEMANTICS_VERSION,
                "analysis_hash": analysis_hash,
            })
    return pd.DataFrame(rows)


def _stability_rows(
    calculations: Mapping[str, _CellCalculation], cells: pd.DataFrame, config: Mapping[str, Any], *, seed: int
) -> pd.DataFrame:
    if not bool(config.get("enabled", False)):
        return pd.DataFrame()
    repetitions = int(config.get("repetitions", 100))
    null_permutations = int(config.get("null_permutations", 100))
    significance_level = float(config.get("significance_level", 0.05))
    sizes_raw = config.get("sample_sizes")
    controlled_cells = cells[cells["cell_id"].astype(str).isin(calculations)].copy()
    if "intervention_budget" not in controlled_cells:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for budget, frame in controlled_cells.groupby("intervention_budget", dropna=False, sort=True):
        group = [calculations[value] for value in sorted(frame["cell_id"].astype(str))]
        maximum = min((cell.n_episodes for cell in group), default=0)
        sizes = list(map(int, sizes_raw)) if sizes_raw is not None else list(range(5, maximum + 1, 5))
        if maximum and maximum not in sizes:
            sizes.append(maximum)
        event_lookup = config.get("_events", {})
        for size in sorted({value for value in sizes if 0 < value <= maximum}):
            estimates = {"T_pi": [], "T_pi_minus_null": [], "chi": []}
            significant = 0
            for repeat in range(repetitions):
                points: list[dict[str, float]] = []
                cell_nulls: list[tuple[float, ...]] = []
                for cell in group:
                    by_episode: dict[str, list[Any]] = {}
                    for event in event_lookup.get(cell.cell_id, ()): by_episode.setdefault(str(event.episode_id), []).append(event)
                    ids = sorted(by_episode)
                    rng = np.random.default_rng(_stable_seed(seed, "stability", budget, size, repeat, cell.cell_id))
                    chosen = rng.choice(ids, size=size, replace=False)
                    sample = [event for episode in chosen for event in by_episode[str(episode)]]
                    points.append(_components(sample))
                    cell_nulls.append(policy_resampling_null(
                        TARGET_CMI,
                        controlled_rows(sample),
                        permutations=null_permutations,
                        seed=_stable_seed(
                            seed, "stability-null", budget, size, repeat, cell.cell_id
                        ),
                    ))
                aggregate = _aggregate_raw_points(points)
                aggregate_null = [
                    float(np.mean([values[draw] for values in cell_nulls]))
                    for draw in range(null_permutations)
                ]
                null_value = float(np.mean(aggregate_null))
                estimates["T_pi"].append(aggregate["T"])
                estimates["T_pi_minus_null"].append(aggregate["T"] - null_value)
                estimates["chi"].append(aggregate["chi"])
                p_value = (
                    1 + sum(value >= aggregate["T"] for value in aggregate_null)
                ) / (null_permutations + 1)
                significant += int(p_value <= significance_level)
            for metric, values in estimates.items():
                finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
                rows.append({
                    "sample_size": size, "metric": metric, "intervention_budget": budget,
                    "mean_estimate": float(np.mean(finite)) if len(finite) else math.nan,
                    "sd_estimate": float(np.std(finite, ddof=1)) if len(finite) > 1 else math.nan,
                    "ci_width": float(np.quantile(finite, 0.975) - np.quantile(finite, 0.025)) if len(finite) else math.nan,
                    "sign_stability": float(np.mean(np.sign(finite) == np.sign(np.mean(finite)))) if len(finite) else math.nan,
                    "fraction_permutation_significant": significant / repetitions if metric == "T_pi_minus_null" else math.nan,
                    "repetitions": repetitions,
                    "n_null_draws": null_permutations,
                    "significance_level": significance_level,
                    "diagnostic_type": "empirical sample-size stability diagnostic",
                })
    return pd.DataFrame(rows)


def _aggregate_raw_points(points: Sequence[Mapping[str, float]]) -> dict[str, float]:
    return {name: float(np.mean([point[name] for point in points if math.isfinite(point[name])])) for name in ("T", "H", "chi", "ir_numerator", "ir_denominator")}


def _state_local_reconstruction(
    study: pd.DataFrame,
    state: pd.DataFrame,
    study_groupings: Sequence[Mapping[str, Any]],
    state_groupings: Sequence[Mapping[str, Any]],
    *,
    analysis_hash: str,
) -> pd.DataFrame:
    """Compare occupancy-reweighted local T with the independent whole-cell T.

    This is deliberately an audit table, not another estimator.  Only grouping
    definitions shared exactly by the whole-cell and state-local recipes can be
    compared without introducing an implicit marginalization.
    """

    if study.empty or state.empty:
        return pd.DataFrame()
    compatible = {
        (tuple(spec["group_by"]), tuple(spec["marginalize"])): spec
        for spec in study_groupings
    }
    rows: list[dict[str, Any]] = []
    for state_spec in state_groupings:
        signature = (
            tuple(state_spec["group_by"]),
            tuple(state_spec["marginalize"]),
        )
        study_spec = compatible.get(signature)
        if study_spec is None:
            continue
        local = state[
            (state["aggregation_name"] == state_spec["name"])
            & (state["metric"] == TARGET_CMI)
        ]
        whole = study[
            (study["aggregation_name"] == study_spec["name"])
            & (study["metric"] == TARGET_CMI)
        ]
        keys = list(state_spec["group_by"])
        grouped = local.groupby(keys, dropna=False, sort=True) if keys else [((), local)]
        for key_values, frame in grouped:
            key_tuple = (
                tuple(key_values) if isinstance(key_values, tuple) else (key_values,)
            )
            coordinates = dict(zip(keys, key_tuple, strict=True))
            match = whole
            for key, value in coordinates.items():
                match = match[match[key].isna()] if pd.isna(value) else match[match[key] == value]
            if len(match) != 1:
                continue
            reconstruction = _weighted(
                pd.to_numeric(frame["estimate"], errors="coerce"),
                pd.to_numeric(frame["n_observations"], errors="coerce"),
            )
            whole_value = _finite(match.iloc[0]["estimate"])
            difference = reconstruction - whole_value
            rows.append({
                **coordinates,
                "aggregation_name": state_spec["name"],
                "marginalized_dimensions": json.dumps(list(state_spec["marginalize"])),
                "state_local_reconstruction": reconstruction,
                "whole_cell_aggregate": whole_value,
                "difference": difference,
                "relative_difference": (
                    difference / whole_value
                    if math.isfinite(whole_value) and abs(whole_value) > 1e-12
                    else math.nan
                ),
                "n_state_bins": int(frame["target_fraction_bin_index"].nunique()),
                "n_observations": int(
                    pd.to_numeric(frame["n_observations"], errors="coerce").sum()
                ),
                "diagnostic_only": True,
                "analysis_semantics_version": ANALYSIS_SEMANTICS_VERSION,
                "analysis_hash": analysis_hash,
            })
    return pd.DataFrame(rows)


def derive_study_control_aggregates(
    events: Sequence[Any],
    cells: pd.DataFrame,
    recipe: Mapping[str, Any],
    resampling: Mapping[str, Any],
    analysis_hash: str,
) -> StudyAggregateOutputs:
    """Build configured study and state-local summaries from retained rounds."""

    config = recipe.get("derived_study_aggregates", {})
    if not isinstance(config, Mapping):
        raise ValueError("derived_study_aggregates must be a mapping")
    if not bool(config.get("enabled", False)):
        return StudyAggregateOutputs(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
    metrics = tuple(map(str, config.get("metrics", SUPPORTED_METRICS)))
    unknown = sorted(set(metrics) - set(SUPPORTED_METRICS))
    if unknown:
        raise ValueError("unsupported derived study metric(s): " + ", ".join(unknown))
    controlled_cells = cells.copy()
    if "target_semantics" not in controlled_cells:
        raise ValueError("derived study aggregation requires target_semantics")
    controlled_cells["target_semantics"] = controlled_cells["target_semantics"].map(_normalize_semantics)
    controlled_cells = controlled_cells[controlled_cells["target_semantics"].isin({"truth", "false"})]
    groupings = _validate_groupings(config, controlled_cells)
    by_event_cell: dict[str, list[Any]] = {}
    for event in events:
        by_event_cell.setdefault(str(event.cell_id), []).append(event)
    cell_coordinates = controlled_cells.set_index(controlled_cells["cell_id"].astype(str)).to_dict(orient="index")
    calculations = {
        cell_id: _cell_calculation(
            cell_id, by_event_cell[cell_id], coordinates,
            bootstrap_resamples=int(resampling["bootstrap_resamples"]),
            null_permutations=int(resampling["null_permutations"]),
            seed=int(resampling["seed"]),
        )
        for cell_id, coordinates in sorted(cell_coordinates.items())
        if cell_id in by_event_cell
    }
    study = pd.DataFrame(_study_rows(
        calculations, controlled_cells, groupings, metrics,
        confidence=float(resampling["confidence"]),
        bootstrap_resamples=int(resampling["bootstrap_resamples"]),
        null_permutations=int(resampling["null_permutations"]),
        analysis_hash=analysis_hash,
    ))
    state_config = config.get("state_local", {})
    state = pd.DataFrame()
    state_groupings: list[dict[str, Any]] = []
    if isinstance(state_config, Mapping) and bool(state_config.get("enabled", False)):
        bins = int(state_config.get("x_bins", recipe.get("state_local_x_bins", 8)))
        local = _state_local_calculations(
            [event for event in events if str(event.cell_id) in calculations],
            controlled_cells,
            bins=bins,
            bootstrap_resamples=int(resampling["bootstrap_resamples"]),
            null_permutations=int(resampling["null_permutations"]),
            seed=int(resampling["seed"]),
        )
        state_groupings = _validate_state_local_groupings(
            state_config, controlled_cells
        )
        state = _state_local_rows(
            local,
            state_groupings,
            confidence=float(resampling["confidence"]),
            bootstrap_resamples=int(resampling["bootstrap_resamples"]),
            analysis_hash=analysis_hash,
        )
    stability_config = config.get("sample_size_stability", {})
    stability = pd.DataFrame()
    if isinstance(stability_config, Mapping):
        stability_payload = dict(stability_config)
        stability_payload["_events"] = by_event_cell
        stability = _stability_rows(
            calculations, controlled_cells, stability_payload,
            seed=int(resampling["seed"]),
        )
    reconstruction = _state_local_reconstruction(
        study,
        state,
        groupings,
        state_groupings,
        analysis_hash=analysis_hash,
    )
    return StudyAggregateOutputs(study, state, stability, reconstruction)


__all__ = [
    "ANALYSIS_SEMANTICS_VERSION",
    "SUPPORTED_METRICS",
    "StudyAggregateOutputs",
    "derive_study_control_aggregates",
]
