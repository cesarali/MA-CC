"""Offline direct-counting analysis at the round-feedback clock."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mas_cc.analysis.estimators import (
    Estimate,
    conditional_mutual_information,
    mutual_information,
)

from ..imitation.controller import ADVOCATE_TARGET, NO_OP

MAIN_ESTIMATOR_VARIANT = "unsmoothed"

ROUND_INFORMATION_STATISTICS = (
    "round_sensing_mi",
    "round_population_actuation_cmi",
    "round_target_actuation_cmi",
    "round_truth_actuation_cmi",
    "round_order_actuation_cmi",
)
ROUND_DIAGNOSTIC_STATISTICS = (
    "round_controller_action_entropy",
    "round_controller_action_entropy_given_population",
    "round_population_information_fraction",
    "round_target_information_fraction",
    "round_dual_action_state_fraction",
    "round_dual_action_event_fraction",
    "round_single_action_slice_fraction",
    "round_conditioning_state_count",
    "round_singleton_fraction",
    "round_target_signed_actuation",
    "round_truth_signed_actuation",
    "round_order_signed_actuation",
    "round_sensor_mae",
    "round_sensor_mse",
)
ROUND_ANALYSIS_STATISTICS = (
    *ROUND_INFORMATION_STATISTICS,
    *ROUND_DIAGNOSTIC_STATISTICS,
)
MICRO_SLOT_STATISTICS = (
    "micro_slot_focal_actuation_cmi",
    "micro_slot_target_signed_response",
)


@dataclass(frozen=True, slots=True)
class RoundEvent:
    cell_id: str
    episode_id: str
    round_index: int
    event: Mapping[str, Any]

    @property
    def N_k(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.event["occupation_counts_before"])

    @property
    def N_k1(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.event["occupation_counts_after"])

    @property
    def Y_k(self) -> tuple[int, ...] | None:
        value = self.event.get("sensor_count_vector")
        return None if value is None else tuple(int(item) for item in value)

    @property
    def U_k(self) -> str | None:
        value = self.event.get("controller_action")
        return None if value is None else str(value)

    @property
    def p_k(self) -> float | None:
        value = self.event.get(
            "controller_advocate_probability",
            self.event.get("controller_advocacy_probability"),
        )
        return None if value is None else float(value)

    @property
    def target_before(self) -> int:
        return int(self.event["target_count_before"])

    @property
    def target_after(self) -> int:
        return int(self.event["target_count_after"])

    @property
    def truth_before(self) -> int:
        return int(self.event["truth_count_before"])

    @property
    def truth_after(self) -> int:
        return int(self.event["truth_count_after"])

    @property
    def order_before(self) -> int:
        return max(self.N_k)

    @property
    def order_after(self) -> int:
        return max(self.N_k1)


def adapt_round_record(
    record: Mapping[str, Any], *, cell_id: str = "run", episode_id: str | None = None
) -> RoundEvent:
    if record.get("record_type") not in {None, "imitation_round_feedback"}:
        raise ValueError("record is not an imitation_round_feedback row")
    required = {
        "round_index",
        "occupation_counts_before",
        "occupation_counts_after",
        "target_count_before",
        "target_count_after",
        "truth_count_before",
        "truth_count_after",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError("round record is missing: " + ", ".join(missing))
    before = tuple(int(value) for value in record["occupation_counts_before"])
    after = tuple(int(value) for value in record["occupation_counts_after"])
    if len(before) != len(after) or len(before) < 2:
        raise ValueError("round occupation vectors must have the same K >= 2")
    if sum(before) != sum(after):
        raise ValueError("round occupation vectors must conserve population size")
    return RoundEvent(
        cell_id=str(cell_id),
        episode_id=str(episode_id or record.get("episode_id", "episode")),
        round_index=int(record["round_index"]),
        event=dict(record),
    )


def _cell_id_for(path: Path) -> str:
    for parent in path.parents:
        if (parent / "overrides.json").is_file():
            return parent.name
    return "run"


def read_round_records(root: str | Path) -> list[RoundEvent]:
    source = Path(root)
    paths = [source] if source.is_file() else sorted(source.rglob("round_trajectory.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no round_trajectory.jsonl files under {source}")
    rows: list[RoundEvent] = []
    for path in paths:
        cell_id = _cell_id_for(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("record_type") != "imitation_round_feedback":
                continue
            rows.append(adapt_round_record(payload, cell_id=cell_id))
    if not rows:
        raise ValueError(f"round trajectory files under {source} contain no round records")
    return rows


def _estimate_for(name: str, rows: Sequence[RoundEvent]) -> Estimate:
    if name == "round_sensing_mi":
        return mutual_information([row.N_k for row in rows], [row.Y_k for row in rows])
    actions = [str(row.U_k) for row in rows]
    if name == "round_population_actuation_cmi":
        return conditional_mutual_information(
            actions, [row.N_k1 for row in rows], [row.N_k for row in rows]
        )
    if name == "round_target_actuation_cmi":
        return conditional_mutual_information(
            actions,
            [row.target_after for row in rows],
            [row.target_before for row in rows],
        )
    if name == "round_truth_actuation_cmi":
        return conditional_mutual_information(
            actions,
            [row.truth_after for row in rows],
            [row.truth_before for row in rows],
        )
    if name == "round_order_actuation_cmi":
        return conditional_mutual_information(
            actions,
            [row.order_after for row in rows],
            [row.order_before for row in rows],
        )
    raise ValueError(f"unknown round information statistic {name!r}")


def _entropy_bits(values: Sequence[Hashable]) -> float:
    if not values:
        return math.nan
    counts = Counter(values)
    total = len(values)
    return -sum(
        (count / total) * math.log2(count / total) for count in counts.values() if count
    )


def conditional_action_entropy_bits(
    actions: Sequence[str], states: Sequence[Hashable]
) -> float:
    if not actions:
        return math.nan
    grouped: dict[Hashable, list[str]] = defaultdict(list)
    for action, state in zip(actions, states, strict=True):
        grouped[state].append(action)
    return sum(
        len(group) / len(actions) * _entropy_bits(group) for group in grouped.values()
    )


def round_overlap_diagnostics(rows: Sequence[RoundEvent]) -> dict[str, Any]:
    controlled = [row for row in rows if row.U_k in {ADVOCATE_TARGET, NO_OP}]
    grouped: dict[tuple[int, ...], Counter[str]] = defaultdict(Counter)
    for row in controlled:
        grouped[row.N_k][str(row.U_k)] += 1
    dual = [
        counter
        for counter in grouped.values()
        if counter[ADVOCATE_TARGET] and counter[NO_OP]
    ]
    events_in_dual = sum(sum(counter.values()) for counter in dual)
    singleton_events = sum(
        sum(counter.values()) for counter in grouped.values() if sum(counter.values()) == 1
    )
    return {
        "round_dual_action_state_fraction": (
            math.nan if not grouped else len(dual) / len(grouped)
        ),
        "round_dual_action_event_fraction": (
            math.nan if not controlled else events_in_dual / len(controlled)
        ),
        "round_single_action_slice_fraction": (
            math.nan if not grouped else 1.0 - len(dual) / len(grouped)
        ),
        "round_conditioning_state_count": len(grouped),
        "round_singleton_fraction": (
            math.nan if not controlled else singleton_events / len(controlled)
        ),
    }


def _signed_response(
    rows: Sequence[RoundEvent],
    *,
    state: Callable[[RoundEvent], Hashable],
    delta: Callable[[RoundEvent], float],
) -> float:
    grouped: dict[Hashable, dict[str, list[float]]] = defaultdict(
        lambda: {ADVOCATE_TARGET: [], NO_OP: []}
    )
    for row in rows:
        if row.U_k in {ADVOCATE_TARGET, NO_OP}:
            grouped[state(row)][str(row.U_k)].append(delta(row))
    weighted = 0.0
    weight = 0
    for buckets in grouped.values():
        advocated = buckets[ADVOCATE_TARGET]
        no_op = buckets[NO_OP]
        if not advocated or not no_op:
            continue
        size = len(advocated) + len(no_op)
        weighted += size * (
            sum(advocated) / len(advocated) - sum(no_op) / len(no_op)
        )
        weight += size
    return math.nan if weight == 0 else weighted / weight


def _diagnostic_for(name: str, rows: Sequence[RoundEvent]) -> float:
    controlled = [row for row in rows if row.U_k in {ADVOCATE_TARGET, NO_OP}]
    actions = [str(row.U_k) for row in controlled]
    if name == "round_controller_action_entropy":
        return _entropy_bits(actions)
    if name == "round_controller_action_entropy_given_population":
        return conditional_action_entropy_bits(actions, [row.N_k for row in controlled])
    if name == "round_population_information_fraction":
        denominator = conditional_action_entropy_bits(
            actions, [row.N_k for row in controlled]
        )
        numerator = getattr(
            _estimate_for("round_population_actuation_cmi", controlled),
            MAIN_ESTIMATOR_VARIANT,
        )
        return math.nan if not math.isfinite(denominator) or denominator <= 1e-12 else numerator / denominator
    if name == "round_target_information_fraction":
        denominator = conditional_action_entropy_bits(
            actions, [row.target_before for row in controlled]
        )
        numerator = getattr(
            _estimate_for("round_target_actuation_cmi", controlled),
            MAIN_ESTIMATOR_VARIANT,
        )
        return math.nan if not math.isfinite(denominator) or denominator <= 1e-12 else numerator / denominator
    if name in {
        "round_dual_action_state_fraction",
        "round_dual_action_event_fraction",
        "round_single_action_slice_fraction",
        "round_conditioning_state_count",
        "round_singleton_fraction",
    }:
        return float(round_overlap_diagnostics(controlled)[name])
    if name == "round_target_signed_actuation":
        return _signed_response(
            controlled,
            state=lambda row: row.target_before,
            delta=lambda row: float(row.event["delta_m_ctrl"]),
        )
    if name == "round_truth_signed_actuation":
        return _signed_response(
            controlled,
            state=lambda row: row.truth_before,
            delta=lambda row: float(row.event["delta_m_truth"]),
        )
    if name == "round_order_signed_actuation":
        return _signed_response(
            controlled,
            state=lambda row: row.order_before,
            delta=lambda row: float(row.event["delta_m_order"]),
        )
    errors = [
        float(row.event["sensor_target_share"])
        - row.target_before / sum(row.N_k)
        for row in controlled
        if row.event.get("sensor_target_share") is not None
    ]
    if name == "round_sensor_mae":
        return math.nan if not errors else sum(abs(value) for value in errors) / len(errors)
    if name == "round_sensor_mse":
        return math.nan if not errors else sum(value * value for value in errors) / len(errors)
    raise ValueError(f"unknown round diagnostic {name!r}")


def _grouped(
    rows: Sequence[RoundEvent], key: Callable[[RoundEvent], Hashable]
) -> dict[Hashable, list[RoundEvent]]:
    result: dict[Hashable, list[RoundEvent]] = defaultdict(list)
    for row in rows:
        result[key(row)].append(row)
    return dict(result)


def bootstrap_episode_rows(
    rows: Sequence[RoundEvent], *, resamples: int, seed: int
) -> tuple[tuple[RoundEvent, ...], ...]:
    """Resample whole episode IDs, never individual round rows."""

    if resamples < 0:
        raise ValueError("resamples cannot be negative")
    by_episode = _grouped(rows, key=lambda row: row.episode_id)
    ids = tuple(by_episode)
    if not ids or resamples == 0:
        return ()
    rng = np.random.default_rng(seed)
    draws: list[tuple[RoundEvent, ...]] = []
    for _ in range(resamples):
        selected = rng.choice(ids, size=len(ids), replace=True)
        draws.append(
            tuple(row for episode_id in selected for row in by_episode[str(episode_id)])
        )
    return tuple(draws)


def _policy_resample(rows: Sequence[RoundEvent], rng: np.random.Generator) -> list[RoundEvent]:
    result = []
    for row in rows:
        if row.p_k is None:
            result.append(row)
            continue
        event = dict(row.event)
        event["controller_action"] = (
            ADVOCATE_TARGET if rng.random() < row.p_k else NO_OP
        )
        result.append(replace(row, event=event))
    return result


def policy_resampling_null(
    name: str,
    rows: Sequence[RoundEvent],
    *,
    permutations: int,
    seed: int,
) -> tuple[float, ...]:
    if name not in ROUND_INFORMATION_STATISTICS[1:]:
        raise ValueError("policy resampling is defined for round actuation statistics")
    values = []
    for index in range(permutations):
        perturbed = _policy_resample(rows, np.random.default_rng(seed + index))
        values.append(float(getattr(_estimate_for(name, perturbed), MAIN_ESTIMATOR_VARIANT)))
    return tuple(values)


def _support(rows: Sequence[RoundEvent]) -> dict[str, Any]:
    overlap = round_overlap_diagnostics(rows)
    counts = Counter(row.N_k for row in rows)
    return {
        "n_episodes": len({row.episode_id for row in rows}),
        "n_rounds": len(rows),
        "unique_population_states": len(counts),
        "unique_sensor_states": len({row.Y_k for row in rows if row.Y_k is not None}),
        "number_of_actions_observed": len({row.U_k for row in rows if row.U_k is not None}),
        "min_rounds_per_population_state": min(counts.values()) if counts else 0,
        "median_rounds_per_population_state": (
            math.nan if not counts else float(np.median(list(counts.values())))
        ),
        "max_rounds_per_population_state": max(counts.values()) if counts else 0,
        **overlap,
    }


def round_information_analysis(
    rows: Sequence[RoundEvent],
    *,
    statistics: Sequence[str] | None = None,
    bootstrap_resamples: int = 1000,
    null_permutations: int = 1000,
    confidence: float = 0.95,
    seed: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    names = tuple(ROUND_ANALYSIS_STATISTICS if statistics is None else statistics)
    unknown = sorted(set(names) - set(ROUND_ANALYSIS_STATISTICS))
    if unknown:
        raise ValueError("unknown round-feedback statistic(s): " + ", ".join(unknown))
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    alpha = (1.0 - confidence) / 2.0
    estimates: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []

    for name_index, name in enumerate(names):
        eligible = (
            [row for row in rows if row.Y_k is not None]
            if name == "round_sensing_mi"
            else [row for row in rows if row.U_k in {ADVOCATE_TARGET, NO_OP}]
        )
        # No-control cells have no fabricated U=NO_OP rows and therefore emit
        # no controller information/diagnostic estimate at all.
        if not eligible:
            continue
        if name in ROUND_INFORMATION_STATISTICS:
            estimate = _estimate_for(name, eligible)
            value = float(getattr(estimate, MAIN_ESTIMATOR_VARIANT))
            variants = {
                "jeffreys": estimate.jeffreys,
                "unsmoothed": estimate.unsmoothed,
                "miller_madow": estimate.miller_madow,
            }
        else:
            value = _diagnostic_for(name, eligible)
            variants = {
                "jeffreys": math.nan,
                "unsmoothed": value,
                "miller_madow": math.nan,
            }
        bootstrap_values = []
        for draw in bootstrap_episode_rows(
            eligible, resamples=bootstrap_resamples, seed=seed + name_index
        ):
            if name in ROUND_INFORMATION_STATISTICS:
                boot = float(getattr(_estimate_for(name, draw), MAIN_ESTIMATOR_VARIANT))
            else:
                boot = _diagnostic_for(name, draw)
            if math.isfinite(boot):
                bootstrap_values.append(boot)
        interval = (
            (math.nan, math.nan)
            if not bootstrap_values
            else (
                float(np.quantile(bootstrap_values, alpha)),
                float(np.quantile(bootstrap_values, 1.0 - alpha)),
            )
        )
        null_values: tuple[float, ...] = ()
        null_type = None
        if name in ROUND_INFORMATION_STATISTICS[1:]:
            null_values = policy_resampling_null(
                name,
                eligible,
                permutations=null_permutations,
                seed=seed + 100_000 * (name_index + 1),
            )
            null_type = "policy_conditional_randomization"
        elif name == "round_sensing_mi":
            permuted_values = []
            for permutation in range(null_permutations):
                rng = np.random.default_rng(seed + 100_000 * (name_index + 1) + permutation)
                sensors = [row.Y_k for row in eligible]
                order = rng.permutation(len(sensors))
                shuffled = [
                    replace(row, event={**dict(row.event), "sensor_count_vector": sensors[int(index)]})
                    for row, index in zip(eligible, order, strict=True)
                ]
                permuted_values.append(
                    float(getattr(_estimate_for(name, shuffled), MAIN_ESTIMATOR_VARIANT))
                )
            null_values = tuple(permuted_values)
            null_type = "sensor_permutation"
        finite_null = [item for item in null_values if math.isfinite(item)]
        for permutation, null_value in enumerate(null_values):
            null_rows.append(
                {
                    "statistic": name,
                    "permutation": permutation,
                    "null_type": null_type,
                    "estimate": null_value,
                }
            )
        support = _support(eligible)
        entropy_ceiling = math.nan
        entropy_bound_satisfied: bool | None = None
        if name == "round_population_actuation_cmi":
            entropy_ceiling = conditional_action_entropy_bits(
                [str(row.U_k) for row in eligible], [row.N_k for row in eligible]
            )
        elif name == "round_target_actuation_cmi":
            entropy_ceiling = conditional_action_entropy_bits(
                [str(row.U_k) for row in eligible],
                [row.target_before for row in eligible],
            )
        elif name == "round_truth_actuation_cmi":
            entropy_ceiling = conditional_action_entropy_bits(
                [str(row.U_k) for row in eligible],
                [row.truth_before for row in eligible],
            )
        elif name == "round_order_actuation_cmi":
            entropy_ceiling = conditional_action_entropy_bits(
                [str(row.U_k) for row in eligible],
                [row.order_before for row in eligible],
            )
        if math.isfinite(entropy_ceiling):
            entropy_bound_satisfied = bool(value <= entropy_ceiling + 1e-9)
        estimates.append(
            {
                "statistic": name,
                "estimate": value,
                "main_estimator_variant": MAIN_ESTIMATOR_VARIANT,
                "estimator_variant": "direct_counting",
                **variants,
                "bootstrap_ci_low": interval[0],
                "bootstrap_ci_high": interval[1],
                "null_type": null_type,
                "null_mean": (
                    math.nan if not finite_null else float(np.mean(finite_null))
                ),
                "units": "bits" if (
                    name in ROUND_INFORMATION_STATISTICS
                    or "entropy" in name
                ) else "dimensionless",
                "bootstrap_unit": "episode",
                "conditional_action_entropy_bits": entropy_ceiling,
                "entropy_bound_satisfied": entropy_bound_satisfied,
                **support,
            }
        )
    return estimates, null_rows


def _read_micro_events(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = [
        *root.rglob("trajectory.jsonl"),
        *root.rglob("micro_slot_trajectory.jsonl"),
    ]
    for path in sorted(paths):
        cell_id = _cell_id_for(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            event = payload.get("event", payload)
            if isinstance(event, Mapping) and "within_round_index" in event:
                rows.append({"cell_id": cell_id, **dict(event)})
    return rows


def _micro_signed(rows: Sequence[Mapping[str, Any]]) -> float:
    grouped: dict[Hashable, dict[bool, list[float]]] = defaultdict(
        lambda: {True: [], False: []}
    )
    for row in rows:
        state = tuple(int(row["occupation_counts_before"][option]) for option in row["possible_answers"])
        grouped[state][bool(row["controlled_slot"])].append(float(row["delta_m_ctrl"]))
    weighted = 0.0
    weight = 0
    for buckets in grouped.values():
        if not buckets[True] or not buckets[False]:
            continue
        size = len(buckets[True]) + len(buckets[False])
        weighted += size * (
            sum(buckets[True]) / len(buckets[True])
            - sum(buckets[False]) / len(buckets[False])
        )
        weight += size
    return math.nan if not weight else weighted / weight


def _micro_cmi(rows: Sequence[Mapping[str, Any]]) -> float:
    controls = [bool(row["controlled_slot"]) for row in rows]
    before = [
        (
            str(row["focal_opinion_before"]),
            tuple(
                int(row["occupation_counts_before"][option])
                for option in row["possible_answers"]
            ),
        )
        for row in rows
    ]
    after = [str(row["focal_opinion_after"]) for row in rows]
    return conditional_mutual_information(controls, after, before).unsmoothed


def _resample_micro_slots(
    rows: Sequence[Mapping[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["episode_id"]), int(row["round_index"]))].append(row)
    result: list[dict[str, Any]] = []
    for group in groups.values():
        budget = int(group[0]["intervention_budget"])
        selected = {
            int(index)
            for index in rng.choice(len(group), size=budget, replace=False)
        }
        result.extend(
            {**dict(row), "controlled_slot": index in selected}
            for index, row in enumerate(group)
        )
    return result


def micro_slot_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    null_permutations: int = 0,
    seed: int = 1,
) -> list[dict[str, Any]]:
    result = []
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("round_controller_action") == ADVOCATE_TARGET:
            grouped[str(row.get("cell_id", "run"))].append(row)
    for cell_id, group in grouped.items():
        null_cmi: list[float] = []
        null_signed: list[float] = []
        for permutation in range(null_permutations):
            perturbed = _resample_micro_slots(
                group, np.random.default_rng(seed + 10_000 * permutation)
            )
            null_cmi.append(_micro_cmi(perturbed))
            null_signed.append(_micro_signed(perturbed))
        result.extend(
            [
                {
                    "cell_id": cell_id,
                    "statistic": "micro_slot_focal_actuation_cmi",
                    "estimate": _micro_cmi(group),
                    "units": "bits",
                    "n_events": len(group),
                    "null_type": "exact_budget_within_round_slot_resampling",
                    "null_mean": (
                        math.nan if not null_cmi else float(np.nanmean(null_cmi))
                    ),
                },
                {
                    "cell_id": cell_id,
                    "statistic": "micro_slot_target_signed_response",
                    "estimate": _micro_signed(group),
                    "units": "order_parameter",
                    "n_events": len(group),
                    "null_type": "exact_budget_within_round_slot_resampling",
                    "null_mean": (
                        math.nan if not null_signed else float(np.nanmean(null_signed))
                    ),
                },
            ]
        )
    return result


def _episode_current_rows(
    rounds: Sequence[RoundEvent], micro: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    micro_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in micro:
        micro_groups[(str(row.get("cell_id", "run")), str(row["episode_id"]))].append(row)
    result = []
    for (cell_id, episode_id), group in _grouped(
        rounds, key=lambda row: (row.cell_id, row.episode_id)
    ).items():
        ordered = sorted(group, key=lambda row: row.round_index)
        micro_rows = micro_groups.get((str(cell_id), str(episode_id)), [])
        if micro_rows:
            increments = [int(row.get("truth_current_increment", 0)) for row in micro_rows]
            truth_current = sum(increments)
            truth_activity: int | None = sum(abs(value) for value in increments)
        else:
            truth_current = ordered[-1].truth_after - ordered[0].truth_before
            truth_activity = None
        result.append(
            {
                "cell_id": cell_id,
                "episode_id": episode_id,
                "rounds": len(ordered),
                "truth_current": truth_current,
                "truth_activity": truth_activity,
                "initial_truth_count": ordered[0].truth_before,
                "final_truth_count": ordered[-1].truth_after,
            }
        )
    return result


def _cell_summaries(
    rounds: Sequence[RoundEvent],
    episodes: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    confidence: float,
    seed: int,
) -> list[dict[str, Any]]:
    episode_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in episodes:
        episode_groups[str(row["cell_id"])].append(row)
    round_groups = _grouped(rounds, key=lambda row: row.cell_id)
    result = []
    for cell_id, group in round_groups.items():
        currents = np.asarray(
            [float(row["truth_current"]) for row in episode_groups[str(cell_id)]],
            dtype=float,
        )
        activities = [
            float(row["truth_activity"])
            for row in episode_groups[str(cell_id)]
            if row.get("truth_activity") is not None
        ]
        mean = float(currents.mean()) if len(currents) else math.nan
        variance = float(currents.var(ddof=1)) if len(currents) >= 2 else math.nan
        fano = (
            abs(mean) / variance
            if math.isfinite(variance) and variance > 0
            else math.nan
        )
        rng = np.random.default_rng(seed)
        bootstrap_means: list[float] = []
        bootstrap_fanos: list[float] = []
        if len(currents):
            for _ in range(bootstrap_resamples):
                drawn = rng.choice(currents, size=len(currents), replace=True)
                bootstrap_means.append(float(drawn.mean()))
                drawn_variance = (
                    float(drawn.var(ddof=1)) if len(drawn) >= 2 else math.nan
                )
                if math.isfinite(drawn_variance) and drawn_variance > 0:
                    bootstrap_fanos.append(abs(float(drawn.mean())) / drawn_variance)
        alpha = (1.0 - confidence) / 2.0
        result.append(
            {
                "cell_id": cell_id,
                "dynamics_mode": group[0].event.get("dynamics_mode"),
                "controller_enabled": bool(group[0].event.get("controller_enabled")),
                "episodes": len(currents),
                "rounds": len(group),
                "truth_current_mean": mean,
                "truth_current_variance": variance,
                "truth_current_fano": fano,
                "truth_current_mean_per_agent": mean / sum(group[0].N_k),
                "truth_current_mean_ci_low": (
                    math.nan
                    if not bootstrap_means
                    else float(np.quantile(bootstrap_means, alpha))
                ),
                "truth_current_mean_ci_high": (
                    math.nan
                    if not bootstrap_means
                    else float(np.quantile(bootstrap_means, 1.0 - alpha))
                ),
                "truth_current_fano_ci_low": (
                    math.nan
                    if not bootstrap_fanos
                    else float(np.quantile(bootstrap_fanos, alpha))
                ),
                "truth_current_fano_ci_high": (
                    math.nan
                    if not bootstrap_fanos
                    else float(np.quantile(bootstrap_fanos, 1.0 - alpha))
                ),
                "truth_activity_mean": (
                    math.nan if not activities else float(np.mean(activities))
                ),
                "truth_activity_variance": (
                    math.nan
                    if len(activities) < 2
                    else float(np.var(activities, ddof=1))
                ),
                "final_m_truth_mean": float(
                    np.mean([row.event["m_truth_after"] for row in group if row.round_index == max(
                        item.round_index for item in group if item.episode_id == row.episode_id
                    )])
                ),
            }
        )
    return result


def _write_markdown(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    lines = [
        "# Round feedback information estimates",
        "",
        "Direct-counting estimates use bits. Bootstrap resampling uses whole episodes; actuation nulls resample each action from its logged policy probability.",
        "",
        "| Cell | Statistic | Estimate | 95% bootstrap CI | Null mean |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['cell_id']}` | `{row['statistic']}` | {row['estimate']:.6g} | "
            f"[{row['bootstrap_ci_low']:.6g}, {row['bootstrap_ci_high']:.6g}] | "
            f"{row['null_mean']:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


ROUND_COMET_FIELDS = (
    "estimate",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "null_mean",
    "excess_over_null",
    "n_episodes",
    "n_rounds",
    "conditional_action_entropy_bits",
    "unique_population_states",
    "unique_sensor_states",
    "number_of_actions_observed",
    "round_dual_action_state_fraction",
    "round_dual_action_event_fraction",
    "round_single_action_slice_fraction",
    "round_conditioning_state_count",
    "round_singleton_fraction",
)
"""Per-cell round-feedback values that remain useful as Comet series."""


def _comet_metric_name(cell_id: Any, statistic: str, field: str) -> str:
    label = str(cell_id or "run").replace("/", "_").replace(" ", "_")
    return "/".join((label, statistic, field))


def round_analysis_comet_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Flatten round estimates and diagnostics into finite Comet scalars."""

    metrics: dict[str, float] = {}
    for row in rows:
        statistic = str(row.get("statistic") or "")
        if not statistic:
            continue
        derived = dict(row)
        estimate, null_mean = row.get("estimate"), row.get("null_mean")
        if (
            isinstance(estimate, (int, float))
            and not isinstance(estimate, bool)
            and isinstance(null_mean, (int, float))
            and not isinstance(null_mean, bool)
            and math.isfinite(float(estimate))
            and math.isfinite(float(null_mean))
        ):
            derived["excess_over_null"] = float(estimate) - float(null_mean)
        for field in ROUND_COMET_FIELDS:
            value = derived.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                continue
            metrics[_comet_metric_name(row.get("cell_id"), statistic, field)] = float(value)
        bound = row.get("entropy_bound_satisfied")
        if isinstance(bound, bool):
            metrics[
                _comet_metric_name(
                    row.get("cell_id"), statistic, "entropy_bound_satisfied"
                )
            ] = float(bound)
    return metrics


def _export_round_analysis_to_comet(
    rows: Sequence[Mapping[str, Any]],
    assets: Sequence[Path],
    *,
    enabled: bool,
    project_name: str,
    run_name: str,
    sink: Any | None = None,
    name_suffix: str | None = None,
) -> dict[str, Any]:
    """Publish one completed cell's metrics and reports to Comet."""

    if not enabled:
        return {
            "status": "disabled",
            "metrics": 0,
            "assets": 0,
            "url": None,
            "published_to": None,
        }
    borrowed = sink is not None
    if sink is None:
        from mas_cc.observability.recorder import CometMetricSink

        sink = CometMetricSink(True, project_name=project_name, run_name=run_name)
    metrics = round_analysis_comet_metrics(rows)
    uploaded_assets = 0
    try:
        sink.add_tags(("analysis", "information", "round-feedback"))
        if metrics:
            sink.log_metrics(metrics, 0)
        for asset in assets:
            path = Path(asset)
            if not path.is_file():
                continue
            asset_name = path.name
            if name_suffix:
                asset_name = f"{path.stem}__{name_suffix}{path.suffix}"
            sink.log_asset(path, name=asset_name)
            uploaded_assets += 1
        return {
            "status": sink.status,
            "metrics": len(metrics),
            "assets": uploaded_assets,
            "url": sink.url,
            "published_to": "master" if borrowed else "analysis_experiment",
        }
    finally:
        if not borrowed:
            sink.close()


def analyze_hidden_bench_imitation_round_feedback(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_resamples: int = 1000,
    null_permutations: int = 1000,
    confidence: float = 0.95,
    seed: int = 1,
    statistics: Sequence[str] | None = None,
    comet_export: bool = False,
    comet_project: str = "mas-cc",
    comet_run_name: str | None = None,
    comet_sink: Any | None = None,
    comet_name_suffix: str | None = None,
    artifact_profile: str = "full",
    resolved_config_hash: str | None = None,
) -> dict[str, Any]:
    rounds = read_round_records(run_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    estimates: list[dict[str, Any]] = []
    nulls: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    for cell_id, group in _grouped(rounds, key=lambda row: row.cell_id).items():
        cell_estimates, cell_nulls = round_information_analysis(
            group,
            statistics=statistics,
            bootstrap_resamples=bootstrap_resamples,
            null_permutations=null_permutations,
            confidence=confidence,
            seed=seed,
        )
        estimates.extend({"cell_id": cell_id, **row} for row in cell_estimates)
        nulls.extend({"cell_id": cell_id, **row} for row in cell_nulls)
        support.append({"cell_id": cell_id, **_support(group)})

    behavioral = []
    for cell_id, group in _grouped(rounds, key=lambda row: row.cell_id).items():
        behavioral.append(
            {
                "cell_id": cell_id,
                "episodes": len({row.episode_id for row in group}),
                "rounds": len(group),
                "mean_delta_m_ctrl": float(np.mean([row.event["delta_m_ctrl"] for row in group])),
                "mean_delta_m_truth": float(np.mean([row.event["delta_m_truth"] for row in group])),
                "mean_delta_m_order": float(np.mean([row.event["delta_m_order"] for row in group])),
                "mean_delta_H_vote": float(np.mean([row.event["delta_H_vote"] for row in group])),
            }
        )

    micro = _read_micro_events(Path(run_dir)) if Path(run_dir).is_dir() else []
    micro_rows = micro_slot_analysis(
        micro, null_permutations=null_permutations, seed=seed
    )
    episode_rows = _episode_current_rows(rounds, micro)
    cell_rows = _cell_summaries(
        rounds,
        episode_rows,
        bootstrap_resamples=bootstrap_resamples,
        confidence=confidence,
        seed=seed,
    )

    pd.DataFrame(estimates).to_csv(destination / "round_information_estimates.csv", index=False)
    _write_markdown(estimates, destination / "round_information_estimates.md")
    pd.DataFrame(nulls).to_csv(destination / "round_information_nulls.csv", index=False)
    pd.DataFrame(support).to_csv(destination / "round_support_diagnostics.csv", index=False)
    pd.DataFrame(behavioral).to_csv(destination / "round_behavioral_summary.csv", index=False)
    pd.DataFrame(micro_rows).to_csv(destination / "micro_slot_diagnostics.csv", index=False)
    pd.DataFrame(episode_rows).to_csv(destination / "episode_currents.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(destination / "cell_summaries.csv", index=False)
    summary = {
        "n_cells": len({row.cell_id for row in rounds}),
        "n_episodes": len({(row.cell_id, row.episode_id) for row in rounds}),
        "n_rounds": len(rounds),
        "n_micro_events": len(micro),
        "statistics": list(ROUND_ANALYSIS_STATISTICS if statistics is None else statistics),
        "bootstrap_unit": "episode",
        "bootstrap_resamples": bootstrap_resamples,
        "null_permutations": null_permutations,
        "main_estimator_variant": MAIN_ESTIMATOR_VARIANT,
        "artifact_profile": artifact_profile,
        "resolved_config_hash": resolved_config_hash,
    }
    summary["comet"] = _export_round_analysis_to_comet(
        estimates,
        (
            destination / "round_information_estimates.csv",
            destination / "round_information_estimates.md",
            destination / "round_support_diagnostics.csv",
            destination / "round_behavioral_summary.csv",
        ),
        enabled=comet_export,
        project_name=comet_project,
        run_name=comet_run_name or f"{Path(run_dir).name}/analysis",
        sink=comet_sink,
        name_suffix=comet_name_suffix,
    )
    (destination / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


__all__ = [
    "MICRO_SLOT_STATISTICS",
    "ROUND_ANALYSIS_STATISTICS",
    "ROUND_DIAGNOSTIC_STATISTICS",
    "ROUND_INFORMATION_STATISTICS",
    "RoundEvent",
    "adapt_round_record",
    "analyze_hidden_bench_imitation_round_feedback",
    "bootstrap_episode_rows",
    "conditional_action_entropy_bits",
    "micro_slot_analysis",
    "policy_resampling_null",
    "read_round_records",
    "round_analysis_comet_metrics",
    "round_information_analysis",
    "round_overlap_diagnostics",
]
