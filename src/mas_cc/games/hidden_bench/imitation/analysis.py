"""Offline behavioral and discrete-information analysis for imitation events.

The analysis consumes persisted ``trajectory.jsonl`` files.  It intentionally
does not maintain streaming estimator state: episodes are the bootstrap unit,
and temporal nulls have to see a complete episode before they can perturb it.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any, Callable, Hashable, Mapping, Sequence

import numpy as np
import pandas as pd

from mas_cc.analysis.estimators import (
    Estimate,
    conditional_mutual_information,
    mutual_information,
)

from .controller import ADVOCATE_TARGET, NO_OP
from .metrics import behavioral_transition_metrics, population_observables

MAIN_ESTIMATOR_VARIANT = "unsmoothed"
ACTION_ENTROPY_EPSILON_BITS = 1e-6
INFORMATION_STATISTICS = (
    "sensing_mi",
    "population_actuation_cmi",
    "target_actuation_cmi",
    "focal_actuation_cmi",
)


@dataclass(frozen=True, slots=True)
class ImitationEvent:
    """Canonical categorical encoding of ``N_t,Y_t,U_t,N_t1,Z_t,Xf_t``."""

    episode_id: str
    cell_id: str
    interaction_index: int
    options: tuple[str, ...]
    N_t: tuple[int, ...]
    N_t1: tuple[int, ...]
    Y_t: tuple[int, ...] | None
    U_t: str | None
    Z_t: int
    Z_t1: int
    Xf_t: str
    Xf_t1: str
    target: str
    event: Mapping[str, Any]
    overrides: Mapping[str, Any]


def _canonical_counts(
    counts: Mapping[str, Any], options: Sequence[str], *, field: str
) -> tuple[int, ...]:
    unknown = set(counts) - set(options)
    if unknown:
        raise ValueError(f"{field} contains options outside possible_answers: {sorted(unknown)}")
    return tuple(int(counts.get(option, 0)) for option in options)


def enrich_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Backfill v1 derived metrics when analyzing an older persisted event."""

    row = dict(event)
    options = tuple(str(item) for item in row["possible_answers"])
    target = str(
        row.get("analysis_target")
        or row.get("controller_target")
        or row["correct_answer"]
    )
    if target not in options:
        raise ValueError(f"analysis target {target!r} is outside possible_answers")
    before_votes = tuple(str(item) for item in row["population_state_before"])
    after_votes = tuple(str(item) for item in row["population_state_after"])
    before = population_observables(before_votes, options, str(row["correct_answer"]), target)
    after = population_observables(after_votes, options, str(row["correct_answer"]), target)
    before_fields = {
        "occupation_counts_before": before["occupation_counts"],
        "population_shares_before": before["population_shares"],
        "p_truth_before": before["p_truth"],
        "p_ctrl_before": before["p_ctrl"],
        "m_truth_before": before["m_truth"],
        "m_ctrl_before": before["m_ctrl"],
        "m_order_before": before["m_order"],
        "H_vote_before": before["H_vote"],
    }
    after_fields = {
        "occupation_counts_after": after["occupation_counts"],
        "population_shares_after": after["population_shares"],
        "p_truth": after["p_truth"],
        "p_ctrl": after["p_ctrl"],
        "m_truth": after["m_truth"],
        "m_ctrl": after["m_ctrl"],
        "m_order": after["m_order"],
        "H_vote": after["H_vote"],
    }
    for key, value in (*before_fields.items(), *after_fields.items()):
        if row.get(key) is None:
            row[key] = value
    row.setdefault("analysis_target", target)
    sensor_counts = row.get("sensor_count_vector")
    derived = behavioral_transition_metrics(
        before,
        after,
        focal_opinion_before=str(row["focal_opinion_before"]),
        focal_opinion_after=str(row["focal_opinion_after"]),
        target=target,
        controller_action=(
            None if row.get("controller_action") is None else str(row["controller_action"])
        ),
        sensor_count_vector=sensor_counts if isinstance(sensor_counts, Mapping) else None,
        sensor_sample_size=row.get("sensor_sample_size"),
    )
    for key, value in derived.items():
        if row.get(key) is None:
            row[key] = value
    return row


def adapt_event(
    event: Mapping[str, Any],
    *,
    episode_id: str | None = None,
    cell_id: str = "run",
    overrides: Mapping[str, Any] | None = None,
) -> ImitationEvent:
    """Encode mappings in the event's explicit canonical option order."""

    row = enrich_event(event)
    options = tuple(str(item) for item in row["possible_answers"])
    if len(options) < 2 or len(set(options)) != len(options):
        raise ValueError("possible_answers must contain at least two unique labels")
    before = _canonical_counts(row["occupation_counts_before"], options, field="N_t")
    after = _canonical_counts(row["occupation_counts_after"], options, field="N_t1")
    if sum(before) != int(row["N"]) or sum(after) != int(row["N"]):
        raise ValueError("occupation counts must sum to N before and after")
    action = None if row.get("controller_action") is None else str(row["controller_action"])
    sensor = None
    if action is not None:
        sensor = _canonical_counts(row["sensor_count_vector"], options, field="Y_t")
        if sum(sensor) != int(row["sensor_sample_size"]):
            raise ValueError("sensor count vector must sum to sensor_sample_size")
    target = str(row["analysis_target"])
    target_index = options.index(target)
    return ImitationEvent(
        episode_id=str(episode_id or row["episode_id"]),
        cell_id=str(cell_id),
        interaction_index=int(row["interaction_index"]),
        options=options,
        N_t=before,
        N_t1=after,
        Y_t=sensor,
        U_t=action,
        Z_t=before[target_index],
        Z_t1=after[target_index],
        Xf_t=str(row["focal_opinion_before"]),
        Xf_t1=str(row["focal_opinion_after"]),
        target=target,
        event=row,
        overrides=dict(overrides or {}),
    )


def binary_action_entropy_bits(actions: Sequence[str]) -> float | None:
    """Shannon entropy of observed controller actions, or NA without a controller."""

    values = [action for action in actions if action in {ADVOCATE_TARGET, NO_OP}]
    if not values:
        return None
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _mean(values: Sequence[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not finite else float(sum(finite) / len(finite))


def _behavioral_action_summary(events: Sequence[ImitationEvent]) -> dict[str, Any]:
    controlled = [event for event in events if event.U_t in {ADVOCATE_TARGET, NO_OP}]
    if not controlled:
        return {
            "controller_advocacy_rate": None,
            "controller_noop_rate": None,
            "controller_action_entropy_bits": None,
            "n_advocate": None,
            "n_noop": None,
            "controller_degenerate": None,
            "sensor_target_bias": None,
            "sensor_target_mae": None,
            "mean_delta_m_ctrl_advocate": None,
            "mean_delta_m_ctrl_noop": None,
            "mean_delta_m_truth_advocate": None,
            "mean_delta_m_truth_noop": None,
            "advocacy_delta_m_ctrl": None,
            "advocacy_delta_m_truth": None,
            "target_adoption_probability_advocate": None,
            "target_adoption_probability_noop": None,
            "target_adoption_lift": None,
        }
    actions = [event.U_t for event in controlled if event.U_t is not None]
    n_advocate = actions.count(ADVOCATE_TARGET)
    n_noop = actions.count(NO_OP)
    entropy = binary_action_entropy_bits(actions)

    def action_mean(action: str, field: str) -> float | None:
        return _mean([event.event.get(field) for event in controlled if event.U_t == action])

    def adoption_probability(action: str) -> float | None:
        eligible = [
            event
            for event in controlled
            if event.U_t == action and event.Xf_t != event.target
        ]
        return _mean([event.event["focal_adopted_target"] for event in eligible])

    ctrl_adv = action_mean(ADVOCATE_TARGET, "delta_m_ctrl")
    ctrl_noop = action_mean(NO_OP, "delta_m_ctrl")
    truth_adv = action_mean(ADVOCATE_TARGET, "delta_m_truth")
    truth_noop = action_mean(NO_OP, "delta_m_truth")
    adopt_adv = adoption_probability(ADVOCATE_TARGET)
    adopt_noop = adoption_probability(NO_OP)
    both_actions = n_advocate > 0 and n_noop > 0
    return {
        "controller_advocacy_rate": n_advocate / len(controlled),
        "controller_noop_rate": n_noop / len(controlled),
        "controller_action_entropy_bits": entropy,
        "n_advocate": n_advocate,
        "n_noop": n_noop,
        "controller_degenerate": (
            not both_actions or entropy is None or entropy <= ACTION_ENTROPY_EPSILON_BITS
        ),
        "sensor_target_bias": _mean(
            [event.event.get("sensor_target_error") for event in controlled]
        ),
        "sensor_target_mae": _mean(
            [event.event.get("sensor_target_abs_error") for event in controlled]
        ),
        "mean_delta_m_ctrl_advocate": ctrl_adv,
        "mean_delta_m_ctrl_noop": ctrl_noop,
        "mean_delta_m_truth_advocate": truth_adv,
        "mean_delta_m_truth_noop": truth_noop,
        "advocacy_delta_m_ctrl": (
            ctrl_adv - ctrl_noop if both_actions and ctrl_adv is not None and ctrl_noop is not None else None
        ),
        "advocacy_delta_m_truth": (
            truth_adv - truth_noop
            if both_actions and truth_adv is not None and truth_noop is not None
            else None
        ),
        "target_adoption_probability_advocate": adopt_adv,
        "target_adoption_probability_noop": adopt_noop,
        "target_adoption_lift": (
            adopt_adv - adopt_noop
            if both_actions and adopt_adv is not None and adopt_noop is not None
            else None
        ),
    }


def episode_summary(events: Sequence[ImitationEvent]) -> dict[str, Any]:
    """Summarize one episode; trajectory means include state 0 and all post states."""

    if not events:
        raise ValueError("episode_summary requires at least one event")
    ordered = sorted(events, key=lambda event: event.interaction_index)
    first = ordered[0].event
    last = ordered[-1].event

    def trajectory(field: str) -> list[float]:
        return [float(first[f"{field}_before"]), *(float(event.event[field]) for event in ordered)]

    m_ctrl = trajectory("m_ctrl")
    m_truth = trajectory("m_truth")
    m_order = trajectory("m_order")
    entropy = trajectory("H_vote")
    result = {
        "cell_id": ordered[0].cell_id,
        "episode_id": ordered[0].episode_id,
        "n_events": len(ordered),
        "initial_m_ctrl": m_ctrl[0],
        "final_m_ctrl": m_ctrl[-1],
        "delta_final_m_ctrl": m_ctrl[-1] - m_ctrl[0],
        "initial_m_truth": m_truth[0],
        "final_m_truth": m_truth[-1],
        "delta_final_m_truth": m_truth[-1] - m_truth[0],
        "mean_m_ctrl": _mean(m_ctrl),
        "mean_m_truth": _mean(m_truth),
        "mean_m_order": _mean(m_order),
        "mean_H_vote": _mean(entropy),
        "auc_m_ctrl": _mean(m_ctrl),
        "auc_m_truth": _mean(m_truth),
        "auc_convention": "equal_event_spacing_mean_including_initial_state",
        "initial_state": json.dumps(ordered[0].event["population_state_before"]),
        "final_state": json.dumps(last["population_state_after"]),
    }
    result.update(_behavioral_action_summary(ordered))
    return result


def cell_summary(events: Sequence[ImitationEvent]) -> dict[str, Any]:
    if not events:
        raise ValueError("cell_summary requires at least one event")
    episodes = [
        episode_summary(group)
        for _, group in _group_events(events, key=lambda event: event.episode_id).items()
    ]
    population_fields = (
        "initial_m_ctrl", "final_m_ctrl", "delta_final_m_ctrl",
        "initial_m_truth", "final_m_truth", "delta_final_m_truth",
        "mean_m_ctrl", "mean_m_truth", "mean_m_order", "mean_H_vote",
        "auc_m_ctrl", "auc_m_truth",
    )
    first = events[0]
    result: dict[str, Any] = {
        "cell_id": first.cell_id,
        "dynamics_mode": first.event.get("dynamics_mode"),
        "control_mechanism": first.overrides.get(
            "control.mechanism",
            "threshold_target" if first.U_t is not None else "none",
        ),
        "n_episodes": len(episodes),
        "n_events": len(events),
        "initial_state": episodes[0]["initial_state"],
        "all_episodes_share_initial_state": len({row["initial_state"] for row in episodes}) == 1,
        "auc_convention": "equal_event_spacing_mean_including_initial_state",
    }
    result.update({field: _mean([row[field] for row in episodes]) for field in population_fields})
    result.update(_behavioral_action_summary(events))
    return result


def bootstrap_episode_ids(
    episode_ids: Sequence[str], *, resamples: int, seed: int
) -> tuple[tuple[str, ...], ...]:
    """Return bootstrap draws over unique episodes (never individual rows)."""

    unique = tuple(dict.fromkeys(str(item) for item in episode_ids))
    if resamples < 0:
        raise ValueError("resamples cannot be negative")
    if not unique or resamples == 0:
        return ()
    rng = np.random.default_rng(seed)
    return tuple(
        tuple(str(item) for item in rng.choice(unique, size=len(unique), replace=True))
        for _ in range(resamples)
    )


def _group_events(
    events: Sequence[ImitationEvent], *, key: Callable[[ImitationEvent], Hashable]
) -> dict[Hashable, list[ImitationEvent]]:
    grouped: dict[Hashable, list[ImitationEvent]] = defaultdict(list)
    for event in events:
        grouped[key(event)].append(event)
    return dict(grouped)


def _estimate_for(name: str, events: Sequence[ImitationEvent]) -> Estimate:
    if name == "sensing_mi":
        return mutual_information([event.N_t for event in events], [event.Y_t for event in events])
    if name == "population_actuation_cmi":
        return conditional_mutual_information(
            [event.U_t for event in events], [event.N_t1 for event in events], [event.N_t for event in events]
        )
    if name == "target_actuation_cmi":
        return conditional_mutual_information(
            [event.U_t for event in events], [event.Z_t1 for event in events], [event.Z_t for event in events]
        )
    if name == "focal_actuation_cmi":
        return conditional_mutual_information(
            [event.U_t for event in events],
            [event.Xf_t1 for event in events],
            [(event.Xf_t, event.N_t) for event in events],
        )
    raise ValueError(f"unknown information statistic {name!r}")


def _conditioning_values(name: str, events: Sequence[ImitationEvent]) -> list[Hashable]:
    if name == "sensing_mi":
        return []
    if name == "population_actuation_cmi":
        return [event.N_t for event in events]
    if name == "target_actuation_cmi":
        return [event.Z_t for event in events]
    return [(event.Xf_t, event.N_t) for event in events]


def _support_diagnostics(name: str, events: Sequence[ImitationEvent]) -> dict[str, Any]:
    conditions = Counter(_conditioning_values(name, events))
    sizes = sorted(conditions.values())
    actions = [event.U_t for event in events if event.U_t is not None]
    entropy = binary_action_entropy_bits([str(action) for action in actions])
    return {
        "n_episodes": len({event.episode_id for event in events}),
        "n_events": len(events),
        "unique_N_t_states": len({event.N_t for event in events}),
        "unique_Y_t_states": len({event.Y_t for event in events if event.Y_t is not None}),
        "number_of_U_t_classes_observed": len(set(actions)),
        "H_U_bits": entropy,
        "occupied_conditioning_states": len(conditions),
        "min_events_per_conditioning_state": None if not sizes else sizes[0],
        "median_events_per_conditioning_state": None if not sizes else float(median(sizes)),
        "max_events_per_conditioning_state": None if not sizes else sizes[-1],
        "fraction_events_singleton_conditioning_states": (
            None if not sizes else sum(size == 1 for size in sizes) / len(events)
        ),
        "sparse_conditioning_table": (
            False if not sizes else median(sizes) < 5 or any(size == 1 for size in sizes)
        ),
        "controller_degenerate": (
            None
            if name == "sensing_mi"
            else len(set(actions)) < 2
            or entropy is None
            or entropy <= ACTION_ENTROPY_EPSILON_BITS
        ),
    }


def _resampled_events(
    by_episode: Mapping[str, Sequence[ImitationEvent]], draw: Sequence[str]
) -> list[ImitationEvent]:
    return [event for episode_id in draw for event in by_episode[episode_id]]


def _perturb_within_episode(
    events: Sequence[ImitationEvent], *, field: str, rng: np.random.Generator
) -> list[ImitationEvent]:
    result: list[ImitationEvent] = []
    for group in _group_events(events, key=lambda event: event.episode_id).values():
        ordered = sorted(group, key=lambda event: event.interaction_index)
        values = [getattr(event, field) for event in ordered]
        if len(values) > 1:
            permutation = rng.permutation(len(values))
            values = [values[int(index)] for index in permutation]
        result.extend(replace(event, **{field: value}) for event, value in zip(ordered, values, strict=True))
    return result


def information_analysis(
    events: Sequence[ImitationEvent],
    *,
    bootstrap_resamples: int = 1000,
    null_permutations: int = 1000,
    confidence: float = 0.95,
    seed: int = 1,
    statistics: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Estimate the four requested channels with episode bootstrap and temporal nulls."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    names = tuple(INFORMATION_STATISTICS if statistics is None else statistics)
    unknown = sorted(set(names) - set(INFORMATION_STATISTICS))
    if unknown:
        raise ValueError(
            "unknown HiddenBench imitation information statistic(s): "
            + ", ".join(unknown)
        )
    if not names:
        raise ValueError("at least one HiddenBench imitation information statistic is required")
    estimates: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    alpha = (1 - confidence) / 2

    for name_index, name in enumerate(names):
        eligible = [
            event
            for event in events
            if (event.Y_t is not None if name == "sensing_mi" else event.U_t is not None)
        ]
        support = _support_diagnostics(name, eligible)
        if not eligible:
            estimates.append({
                "statistic": name,
                "main_estimator_variant": MAIN_ESTIMATOR_VARIANT,
                "estimator_variant": "direct_counting",
                "estimate": math.nan,
                "jeffreys": math.nan,
                "unsmoothed": math.nan,
                "miller_madow": math.nan,
                "bootstrap_ci_low": math.nan,
                "bootstrap_ci_high": math.nan,
                "null_mean": math.nan,
                "null_ci_low": math.nan,
                "null_ci_high": math.nan,
                "scientifically_interpretable": False,
                **support,
            })
            continue
        estimate = _estimate_for(name, eligible)
        by_episode = _group_events(eligible, key=lambda event: event.episode_id)
        draws = bootstrap_episode_ids(
            list(by_episode), resamples=bootstrap_resamples, seed=seed + name_index
        )
        bootstrap_values = np.asarray(
            [
                getattr(_estimate_for(name, _resampled_events(by_episode, draw)), MAIN_ESTIMATOR_VARIANT)
                for draw in draws
            ],
            dtype=float,
        )
        bootstrap_values = bootstrap_values[np.isfinite(bootstrap_values)]
        bootstrap_interval = (
            (math.nan, math.nan)
            if not len(bootstrap_values)
            else (
                float(np.quantile(bootstrap_values, alpha)),
                float(np.quantile(bootstrap_values, 1 - alpha)),
            )
        )
        null_values: list[float] = []
        for permutation in range(null_permutations):
            rng = np.random.default_rng(seed + 10_000 * (name_index + 1) + permutation)
            perturbed = _perturb_within_episode(
                eligible,
                field="Y_t" if name == "sensing_mi" else "U_t",
                rng=rng,
            )
            null_estimate = _estimate_for(name, perturbed)
            null_value = getattr(null_estimate, MAIN_ESTIMATOR_VARIANT)
            null_values.append(null_value)
            null_rows.append({
                "statistic": name,
                "permutation": permutation,
                "null_type": (
                    "within_episode_sensor_permutation"
                    if name == "sensing_mi"
                    else "within_episode_controller_action_permutation"
                ),
                **asdict(null_estimate),
            })
        finite_null = np.asarray([value for value in null_values if math.isfinite(value)])
        null_interval = (
            (math.nan, math.nan)
            if not len(finite_null)
            else (
                float(np.quantile(finite_null, alpha)),
                float(np.quantile(finite_null, 1 - alpha)),
            )
        )
        main_value = getattr(estimate, MAIN_ESTIMATOR_VARIANT)
        degenerate = bool(support["controller_degenerate"])
        estimates.append({
            "statistic": name,
            "main_estimator_variant": MAIN_ESTIMATOR_VARIANT,
            "estimator_variant": "direct_counting",
            "estimate": main_value,
            **asdict(estimate),
            "bootstrap_ci_low": bootstrap_interval[0],
            "bootstrap_ci_high": bootstrap_interval[1],
            "null_mean": math.nan if not len(finite_null) else float(finite_null.mean()),
            "null_ci_low": null_interval[0],
            "null_ci_high": null_interval[1],
            "scientifically_interpretable": name == "sensing_mi" or not degenerate,
            **support,
        })
    return estimates, null_rows


def _cell_dir_for(path: Path) -> Path | None:
    for candidate in path.parents:
        if candidate.parent.name == "cells":
            return candidate
    return None


def read_imitation_events(run_dir: str | Path) -> list[ImitationEvent]:
    root = Path(run_dir)
    paths = [root] if root.is_file() else sorted(root.rglob("trajectory.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no trajectory.jsonl files under {root}")
    events: list[ImitationEvent] = []
    for path in paths:
        cell_dir = _cell_dir_for(path)
        cell_id = "run" if cell_dir is None else cell_dir.name
        overrides: Mapping[str, Any] = {}
        if cell_dir is not None and (cell_dir / "overrides.json").exists():
            overrides = json.loads((cell_dir / "overrides.json").read_text(encoding="utf-8")).get(
                "overrides", {}
            )
        episode_id = path.parent.name
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            event = payload.get("event")
            if not isinstance(event, Mapping):
                continue
            events.append(
                adapt_event(
                    event,
                    episode_id=episode_id,
                    cell_id=cell_id,
                    overrides=overrides,
                )
            )
    if not events:
        raise ValueError(f"trajectory files under {root} contain no imitation events")
    return events


def _cell_label(summary: Mapping[str, Any]) -> str | None:
    lookup = {
        ("reasoning", "none"): "A",
        ("reasoning", "threshold_target"): "B",
        ("classical", "none"): "C",
        ("classical", "threshold_target"): "D",
    }
    return lookup.get((summary.get("dynamics_mode"), summary.get("control_mechanism")))


def _trajectory_rows(events: Sequence[ImitationEvent]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shares: list[dict[str, Any]] = []
    observables: list[dict[str, Any]] = []
    for _, group in _group_events(events, key=lambda event: (event.cell_id, event.episode_id)).items():
        ordered = sorted(group, key=lambda event: event.interaction_index)
        states = [
            (
                0,
                ordered[0].event["population_shares_before"],
                {field: ordered[0].event[f"{field}_before"] for field in ("m_ctrl", "m_truth", "m_order", "H_vote")},
            ),
            *[
                (
                    event.interaction_index,
                    event.event["population_shares_after"],
                    {field: event.event[field] for field in ("m_ctrl", "m_truth", "m_order", "H_vote")},
                )
                for event in ordered
            ],
        ]
        for state_index, state_shares, state_observables in states:
            common = {
                "cell_id": ordered[0].cell_id,
                "episode_id": ordered[0].episode_id,
                "state_index": state_index,
                "dynamics_mode": ordered[0].event.get("dynamics_mode"),
                "control_mechanism": ordered[0].overrides.get(
                    "control.mechanism", "threshold_target" if ordered[0].U_t else "none"
                ),
            }
            shares.extend(
                {**common, "option": option, "share": state_shares[option]}
                for option in ordered[0].options
            )
            observables.append({**common, **state_observables})
    return shares, observables


def analyze_hidden_bench_imitation(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_resamples: int = 1000,
    null_permutations: int = 1000,
    confidence: float = 0.95,
    seed: int = 1,
    statistics: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write the first-pilot behavioral, response, support, MI, and null report."""

    events = read_imitation_events(run_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    event_rows = []
    for event in events:
        row = {
            "cell_id": event.cell_id,
            "episode_id": event.episode_id,
            "interaction_index": event.interaction_index,
            "dynamics_mode": event.event.get("dynamics_mode"),
            "control_mechanism": event.overrides.get(
                "control.mechanism", "threshold_target" if event.U_t else "none"
            ),
            "N_t": json.dumps(event.N_t),
            "N_t1": json.dumps(event.N_t1),
            "Y_t": None if event.Y_t is None else json.dumps(event.Y_t),
            "U_t": event.U_t,
            "Z_t": event.Z_t,
            "Z_t1": event.Z_t1,
            "Xf_t": event.Xf_t,
            "Xf_t1": event.Xf_t1,
        }
        for field in (
            "delta_m_ctrl", "delta_m_truth", "delta_m_order", "delta_H_vote",
            "focal_changed", "focal_adopted_target", "focal_left_target",
            "u_advocate", "sensor_target_share", "population_target_share",
            "sensor_target_error", "sensor_target_abs_error",
        ):
            row[field] = event.event.get(field)
        event_rows.append(row)
    pd.DataFrame(event_rows).to_csv(destination / "event_metrics.csv", index=False)

    episode_rows = [
        episode_summary(group)
        for group in _group_events(events, key=lambda event: (event.cell_id, event.episode_id)).values()
    ]
    pd.DataFrame(episode_rows).to_csv(destination / "episode_summaries.csv", index=False)

    cell_rows = [
        cell_summary(group)
        for group in _group_events(events, key=lambda event: event.cell_id).values()
    ]
    for row in cell_rows:
        row["scientific_cell"] = _cell_label(row)
    pd.DataFrame(cell_rows).to_csv(destination / "cell_summaries.csv", index=False)

    share_rows, observable_rows = _trajectory_rows(events)
    pd.DataFrame(share_rows).to_csv(destination / "option_share_trajectories.csv", index=False)
    pd.DataFrame(observable_rows).to_csv(
        destination / "order_parameter_trajectories.csv", index=False
    )

    information_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    for cell_id, group in _group_events(events, key=lambda event: event.cell_id).items():
        cell_estimates, cell_nulls = information_analysis(
            group,
            bootstrap_resamples=bootstrap_resamples,
            null_permutations=null_permutations,
            confidence=confidence,
            seed=seed,
            statistics=statistics,
        )
        cell_row = next(row for row in cell_rows if row["cell_id"] == cell_id)
        common = {
            "cell_id": cell_id,
            "scientific_cell": cell_row.get("scientific_cell"),
            "dynamics_mode": cell_row["dynamics_mode"],
            "control_mechanism": cell_row["control_mechanism"],
        }
        information_rows.extend({**common, **row} for row in cell_estimates)
        null_rows.extend({**common, **row} for row in cell_nulls)
    pd.DataFrame(information_rows).to_csv(destination / "information_estimates.csv", index=False)
    pd.DataFrame(null_rows).to_csv(destination / "information_nulls.csv", index=False)
    support_fields = [
        "cell_id", "scientific_cell", "statistic", "n_episodes", "n_events",
        "unique_N_t_states", "unique_Y_t_states", "number_of_U_t_classes_observed",
        "H_U_bits", "occupied_conditioning_states", "min_events_per_conditioning_state",
        "median_events_per_conditioning_state", "max_events_per_conditioning_state",
        "fraction_events_singleton_conditioning_states", "sparse_conditioning_table",
        "controller_degenerate", "scientifically_interpretable",
    ]
    pd.DataFrame(information_rows).reindex(columns=support_fields).to_csv(
        destination / "support_diagnostics.csv", index=False
    )

    by_label = {row.get("scientific_cell"): row for row in cell_rows}
    comparison_specs = (
        ("control_effect_within_reasoning", "B", "A"),
        ("control_effect_within_classical", "D", "C"),
        ("reasoning_effect_without_control", "A", "C"),
        ("reasoning_effect_under_feedback", "B", "D"),
    )
    contrast_rows = []
    contrast_metrics = (
        "final_m_ctrl", "delta_final_m_ctrl", "final_m_truth", "delta_final_m_truth",
        "mean_m_ctrl", "mean_m_truth", "auc_m_ctrl", "auc_m_truth",
    )
    for comparison, left, right in comparison_specs:
        if left not in by_label or right not in by_label:
            continue
        for metric in contrast_metrics:
            contrast_rows.append({
                "comparison": comparison,
                "left_cell": left,
                "right_cell": right,
                "metric": metric,
                "difference": by_label[left][metric] - by_label[right][metric],
            })
    pd.DataFrame(contrast_rows).to_csv(destination / "cell_contrasts.csv", index=False)

    initial_states = {row["initial_state"] for row in cell_rows}
    summary = {
        "n_cells": len(cell_rows),
        "n_episodes": len(episode_rows),
        "n_events": len(events),
        "scientific_cells": sorted(row["scientific_cell"] for row in cell_rows if row["scientific_cell"]),
        "matched_initial_state_across_cells": len(initial_states) == 1,
        "main_estimator_variant": MAIN_ESTIMATOR_VARIANT,
        "information_statistics": list(
            INFORMATION_STATISTICS if statistics is None else statistics
        ),
        "auc_convention": "equal_event_spacing_mean_including_initial_state",
        "output_dir": str(destination),
    }
    (destination / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


__all__ = [
    "ACTION_ENTROPY_EPSILON_BITS",
    "ImitationEvent",
    "INFORMATION_STATISTICS",
    "MAIN_ESTIMATOR_VARIANT",
    "adapt_event",
    "analyze_hidden_bench_imitation",
    "binary_action_entropy_bits",
    "bootstrap_episode_ids",
    "cell_summary",
    "enrich_event",
    "episode_summary",
    "information_analysis",
    "read_imitation_events",
]
