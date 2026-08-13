"""Offline behavioral and discrete-information analysis for imitation events.

The analysis consumes either persisted ``trajectory.jsonl`` files or versioned
compact scientific Parquet. It intentionally does not maintain streaming
estimator state: episodes are the bootstrap unit, and temporal nulls have to see
a complete episode before they can perturb it.
"""

from __future__ import annotations

import hashlib
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
ENTROPY_BOUND_TOLERANCE_BITS = 1e-9
"""Slack allowed on ``I(U;X'|S) <= H(U|S)`` before the bound counts as violated.

Both sides come from the same direct counts, so a real violation cannot happen;
only floating-point summation order can put the ratio a few ulps over one.  The
bound is *reported*, never enforced -- clipping it would hide exactly the
estimator pathology the check exists to surface.
"""

ORDER_PARAMETER_COUNT_FIELDS: Mapping[str, tuple[str, str]] = {
    "m_ctrl": ("Z_t", "Z_t1"),
    "m_truth": ("Mtruth_t", "Mtruth_t1"),
    "m_order": ("Morder_t", "Morder_t1"),
}
"""Integer headcount each order parameter is a strictly monotone function of.

``m_ctrl``, ``m_truth`` and ``m_order`` are floats, and a float is not a safe
contingency-table category.  For a cell's fixed ``N`` and ``K`` each is an
affine, strictly increasing function of one integer headcount -- the target
count, the correct-answer count, and the largest count respectively -- so
counting on the integer gives exactly the same partition of events, and
therefore exactly the same mutual information, with none of the float-equality
hazard.  ``m_ctrl`` reuses ``Z_t``: the target count *is* the ``m_ctrl``
coordinate, which makes ``m_ctrl_actuation_cmi`` and ``target_actuation_cmi``
the same estimate under two names.
"""

INFORMATION_STATISTICS = (
    "sensing_mi",
    "population_actuation_cmi",
    "target_actuation_cmi",
    "focal_actuation_cmi",
    "sensing_mi_m_ctrl",
    "sensing_mi_m_truth",
    "sensing_mi_m_order",
    "m_ctrl_actuation_cmi",
    "m_truth_actuation_cmi",
    "m_order_actuation_cmi",
)
CURRENT_STATISTICS = ("truth_current", "truth_current_fano")
"""Post-hoc trajectory statistics; unlike the channels above they have no action-shuffle null."""
INFORMATION_STATISTIC_DESCRIPTIONS: Mapping[str, str] = {
    "sensing_mi": (
        "Mutual information between the population's true opinion counts before the interaction "
        "(`N_t`) and the controller's noisy sample of them (`Y_t`). How much the controller's sensor "
        "actually tells it about the true population state."
    ),
    "population_actuation_cmi": (
        "Conditional mutual information between the controller's action (`U_t`) and the population's "
        "opinion counts after the interaction (`N_t1`), conditioned on the counts before (`N_t`). How "
        "much of the change in the whole population's opinion distribution the controller's action "
        "explains, beyond what the starting state already predicts."
    ),
    "target_actuation_cmi": (
        "Conditional mutual information between the controller's action (`U_t`) and whether the "
        "target option's count after the interaction (`Z_t1`) differs from before (`Z_t`), conditioned "
        "on `Z_t`. A narrower version of `population_actuation_cmi` focused only on the option the "
        "controller is trying to promote."
    ),
    "focal_actuation_cmi": (
        "Conditional mutual information between the controller's action (`U_t`) and the focal agent's "
        "opinion after the interaction (`Xf_t1`), conditioned on the focal agent's opinion before and "
        "the population counts before (`Xf_t, N_t`). Whether the controller's action changes what the "
        "one agent it is nudging actually ends up believing."
    ),
    "sensing_mi_m_ctrl": (
        "Mutual information between the target-alignment order parameter before the interaction "
        "(`m_ctrl`, encoded as the target option's headcount) and the controller's sensor sample "
        "(`Y_t`). How much the sensor tells the controller about the one macroscopic quantity it is "
        "trying to raise, rather than about the full count vector."
    ),
    "sensing_mi_m_truth": (
        "Mutual information between the truth-alignment order parameter before the interaction "
        "(`m_truth`, encoded as the correct answer's headcount) and the sensor sample (`Y_t`). How "
        "much the sensor reveals about how close the population is to the right answer."
    ),
    "sensing_mi_m_order": (
        "Mutual information between the consensus order parameter before the interaction "
        "(`m_order`, encoded as the largest option headcount) and the sensor sample (`Y_t`). How "
        "much the sensor reveals about how ordered the population is, irrespective of which option "
        "it has settled on."
    ),
    "m_ctrl_actuation_cmi": (
        "Conditional mutual information between the controller's action (`U_t`) and the "
        "target-alignment order parameter after the interaction, conditioned on its value before. "
        "The `population_actuation_cmi` question projected onto `m_ctrl`; identical by construction "
        "to `target_actuation_cmi`, since `m_ctrl` and the target headcount are one-to-one."
    ),
    "m_truth_actuation_cmi": (
        "Conditional mutual information between the controller's action (`U_t`) and the "
        "truth-alignment order parameter after the interaction, conditioned on its value before. "
        "Whether advocating moves the population toward or away from the correct answer, which "
        "differs from `m_ctrl` whenever the controller's target is not the correct answer."
    ),
    "m_order_actuation_cmi": (
        "Conditional mutual information between the controller's action (`U_t`) and the consensus "
        "order parameter after the interaction, conditioned on its value before. Whether advocating "
        "changes how ordered the population becomes, regardless of which option wins."
    ),
}


ACTUATION_CHANNEL_COUNT_FIELDS: Mapping[str, tuple[str, str]] = {
    "population": ("N_t", "N_t1"),
    **ORDER_PARAMETER_COUNT_FIELDS,
}
"""Conditioning variable of each actuation channel, before and after the event.

The order parameters reuse the headcount encoding above, so a controller
diagnostic conditions on exactly the same partition of events as the CMI it is
normalizing -- which is what makes the ratio auditable.
"""

CONDITIONING_LABELS: Mapping[str, str] = {
    "population": "N_t",
    "m_ctrl": "m_ctrl",
    "m_truth": "m_truth",
    "m_order": "m_order",
}

ACTUATION_CHANNEL_DELTA_FIELDS: Mapping[str, str] = {
    "m_ctrl": "delta_m_ctrl",
    "m_truth": "delta_m_truth",
    "m_order": "delta_m_order",
}
"""Signed per-event movement of each order parameter.

Signed responses use the float order parameter rather than the headcount it is
encoded as: the two differ by a fixed positive affine map, so the sign is the
same either way, and the float is the quantity the rest of the report plots.
"""

CONTROLLER_ENTROPY_STATISTICS = (
    "controller_action_entropy",
    "controller_action_entropy_given_population",
    "controller_action_entropy_given_m_ctrl",
    "controller_action_entropy_given_m_truth",
    "controller_action_entropy_given_m_order",
)
ACTUATION_INFORMATION_FRACTION_STATISTICS = (
    "population_actuation_information_fraction",
    "m_ctrl_actuation_information_fraction",
    "m_truth_actuation_information_fraction",
    "m_order_actuation_information_fraction",
)
SIGNED_ACTUATION_STATISTICS = (
    "m_ctrl_signed_actuation",
    "m_truth_signed_actuation",
    "m_order_signed_actuation",
)
ACTION_OVERLAP_STATISTICS = (
    "population_action_overlap",
    "m_ctrl_action_overlap",
    "m_truth_action_overlap",
    "m_order_action_overlap",
)
CONTROLLER_DIAGNOSTIC_STATISTICS = (
    *CONTROLLER_ENTROPY_STATISTICS,
    *ACTUATION_INFORMATION_FRACTION_STATISTICS,
    *SIGNED_ACTUATION_STATISTICS,
    *ACTION_OVERLAP_STATISTICS,
)
"""Diagnostics that make the actuation CMIs above readable.

`I(U;X'|S) <= H(U|S)` means a near-zero CMI has two very different causes: the
controller barely moves the population, or the controller has barely any action
entropy left once `S` is known.  These quantities separate them, and add the
sign the CMI cannot carry.  They are not new channels and not efficiencies.
"""

ACTION_OVERLAP_FIELDS = (
    "occupied_conditioning_states",
    "dual_action_conditioning_states",
    "fraction_conditioning_states_with_both_actions",
    "fraction_events_in_dual_action_conditioning_states",
)

CONTROLLER_DIAGNOSTIC_DESCRIPTIONS: Mapping[str, str] = {
    "controller_action_entropy": (
        "Shannon entropy of the controller's action (`H(U_t)`) over the whole cell. How variable "
        "the controller was overall: 0 bits means it always took the same action, 1 bit means it "
        "split evenly between `ADVOCATE_Z` and `NO_OP`. High values here do **not** imply the "
        "actions are comparable at a fixed population state."
    ),
    "controller_action_entropy_given_population": (
        "`H(U_t | N_t)` — how much uncertainty is left in the controller's action once the full "
        "occupation state before the interaction is known. This is the ceiling on "
        "`population_actuation_cmi`; near zero means a small CMI is structurally expected rather "
        "than evidence of a weak controller."
    ),
    "controller_action_entropy_given_m_ctrl": (
        "`H(U_t | m_ctrl,t)` — the same budget in the target-alignment conditioning space, the "
        "ceiling on `m_ctrl_actuation_cmi`."
    ),
    "controller_action_entropy_given_m_truth": (
        "`H(U_t | m_truth,t)` — the ceiling on `m_truth_actuation_cmi`. Equals the `m_ctrl` "
        "version whenever the controller's target is the correct answer, since the two "
        "conditioning encodings then coincide."
    ),
    "controller_action_entropy_given_m_order": (
        "`H(U_t | m_order,t)` — the ceiling on `m_order_actuation_cmi`. `m_order` is a coarse "
        "projection of `N_t`, so this typically stays well above `H(U_t | N_t)`."
    ),
    "population_actuation_information_fraction": (
        "`population_actuation_cmi / H(U_t | N_t)` — what fraction of the controller's remaining "
        "action information, after the current population state is fixed, is predictive of the "
        "next population state. A normalization diagnostic, not an efficiency."
    ),
    "m_ctrl_actuation_information_fraction": (
        "`m_ctrl_actuation_cmi / H(U_t | m_ctrl,t)` — the same ratio projected onto the "
        "target-alignment order parameter."
    ),
    "m_truth_actuation_information_fraction": (
        "`m_truth_actuation_cmi / H(U_t | m_truth,t)` — the same ratio projected onto the "
        "truth-alignment order parameter."
    ),
    "m_order_actuation_information_fraction": (
        "`m_order_actuation_cmi / H(U_t | m_order,t)` — the same ratio projected onto the "
        "consensus order parameter."
    ),
    "m_ctrl_signed_actuation": (
        "State-adjusted signed response of `m_ctrl`: within each conditioning state that saw "
        "**both** actions, the mean `delta_m_ctrl` under `ADVOCATE_Z` minus the mean under "
        "`NO_OP`, averaged over those states by event frequency. Positive means advocacy moves "
        "the population toward the controller's target; CMI alone cannot say which way."
    ),
    "m_truth_signed_actuation": (
        "The same state-adjusted contrast for `m_truth`. Under a wrong controller target this can "
        "have the opposite sign to `m_ctrl_signed_actuation` — controller success and epistemic "
        "success are then different things."
    ),
    "m_order_signed_actuation": (
        "The same state-adjusted contrast for `m_order`: whether advocacy raises or lowers "
        "consensus, irrespective of which option the population settles on. Not a measure of "
        "target success."
    ),
    "population_action_overlap": (
        "Action-overlap support in the `N_t` conditioning space. The headline number is the "
        "fraction of events living in states where both `ADVOCATE_Z` and `NO_OP` were observed — "
        "the only events that can contribute to a within-state comparison."
    ),
    "m_ctrl_action_overlap": "Action-overlap support in the `m_ctrl` conditioning space.",
    "m_truth_action_overlap": "Action-overlap support in the `m_truth` conditioning space.",
    "m_order_action_overlap": "Action-overlap support in the `m_order` conditioning space.",
}


def _is_sensing_statistic(name: str) -> bool:
    return name.startswith("sensing_mi")


def _order_parameter_of(name: str) -> str | None:
    """Return the order parameter a statistic projects onto, or ``None``."""

    for parameter in ORDER_PARAMETER_COUNT_FIELDS:
        if name in (f"sensing_mi_{parameter}", f"{parameter}_actuation_cmi"):
            return parameter
    return None


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
    Mtruth_t: int
    Mtruth_t1: int
    Morder_t: int
    Morder_t1: int
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
    correct = str(row["correct_answer"])
    if correct not in options:
        raise ValueError(f"correct_answer {correct!r} is outside possible_answers")
    correct_index = options.index(correct)
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
        Mtruth_t=before[correct_index],
        Mtruth_t1=after[correct_index],
        Morder_t=max(before),
        Morder_t1=max(after),
        Xf_t=str(row["focal_opinion_before"]),
        Xf_t1=str(row["focal_opinion_after"]),
        target=target,
        event=row,
        overrides=dict(overrides or {}),
    )


def control_mechanism_of(event: ImitationEvent) -> str:
    """Which controller produced this event.

    A grid cell records `control.mechanism` in its `overrides.json`, but a
    single-cell run has no overrides at all, so the event's own
    `controller_policy` is the next authority. The literal is only reached for
    events written before either field existed, when `threshold_target` was the
    one mechanism there was.
    """

    fallback = event.event.get("controller_policy")
    if fallback is None:
        fallback = "threshold_target" if event.U_t else "none"
    return str(event.overrides.get("control.mechanism", fallback))


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


def _controller_exposure_summary(events: Sequence[ImitationEvent]) -> dict[str, Any]:
    """Finite controller resource/exposure diagnostics, without energetic labels."""

    if not events:
        return {}
    controlled = _controlled_events(events)
    population_size = sum(events[0].N_t)
    advocated = [event for event in controlled if event.U_t == ADVOCATE_TARGET]
    exposed_focals = {
        str(event.event["focal_agent_id"])
        for event in advocated
        if event.event.get("focal_agent_id") is not None
    }
    return {
        "controller_decision_count": len(controlled),
        "controller_advocacy_count": len(advocated),
        "controller_noop_count": len(controlled) - len(advocated),
        "controller_decisions_per_agent": len(controlled) / population_size,
        "controller_advocacies_per_agent": len(advocated) / population_size,
        "unique_focal_agents_exposed_to_controller": len(exposed_focals),
        "fraction_population_ever_exposed_to_controller": (
            len(exposed_focals) / population_size
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
    truth_steps = [
        int(event.Xf_t1 == event.event["correct_answer"])
        - int(event.Xf_t == event.event["correct_answer"])
        for event in ordered
    ]
    truth_toward = sum(step == 1 for step in truth_steps)
    truth_away = sum(step == -1 for step in truth_steps)
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
        "truth_current": truth_toward - truth_away,
        "truth_switches_toward": truth_toward,
        "truth_switches_away": truth_away,
    }
    result.update(_behavioral_action_summary(ordered))
    result.update(_controller_exposure_summary(ordered))
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
        "control_mechanism": control_mechanism_of(first),
        "n_episodes": len(episodes),
        "n_events": len(events),
        "initial_state": episodes[0]["initial_state"],
        "all_episodes_share_initial_state": len({row["initial_state"] for row in episodes}) == 1,
        "auc_convention": "equal_event_spacing_mean_including_initial_state",
    }
    result.update({field: _mean([row[field] for row in episodes]) for field in population_fields})
    result.update(_behavioral_action_summary(events))
    # Exposure is a path property, so aggregate the episode diagnostics rather
    # than treating repeated agent labels across episodes as the same person.
    exposure_mean_fields = (
        "controller_decisions_per_agent",
        "controller_advocacies_per_agent",
        "unique_focal_agents_exposed_to_controller",
        "fraction_population_ever_exposed_to_controller",
    )
    for field in exposure_mean_fields:
        result[field] = _mean([row[field] for row in episodes])
    result["controller_decision_count"] = sum(
        int(row["controller_decision_count"]) for row in episodes
    )
    result["controller_advocacy_count"] = sum(
        int(row["controller_advocacy_count"]) for row in episodes
    )
    result["controller_noop_count"] = sum(
        int(row["controller_noop_count"]) for row in episodes
    )
    result.update(_truth_current_cell_fields(episodes, population_size=sum(first.N_t)))
    return result


def _truth_current_cell_fields(
    episodes: Sequence[Mapping[str, Any]], *, population_size: int
) -> dict[str, Any]:
    currents = np.asarray([float(row["truth_current"]) for row in episodes], dtype=float)
    mean = float(np.mean(currents)) if len(currents) else math.nan
    variance = float(np.var(currents, ddof=1)) if len(currents) >= 2 else math.nan
    fixed_horizon = len({int(row["n_events"]) for row in episodes}) <= 1
    zero_dispersion = bool(math.isfinite(variance) and variance == 0.0)
    undefined_reason = None
    if not fixed_horizon:
        fano = math.nan
        undefined_reason = "variable_horizon"
    elif not math.isfinite(variance):
        fano = math.nan
        undefined_reason = "fewer_than_two_episodes"
    elif variance == 0.0 and abs(mean) > 0.0:
        fano = math.inf
    elif variance == 0.0:
        fano = math.nan
        undefined_reason = "zero_mean_and_zero_dispersion"
    else:
        fano = abs(mean) / variance
    return {
        "truth_current_mean": mean,
        "truth_current_variance": variance,
        "truth_current_fano": fano,
        "truth_current_mean_per_agent": mean / population_size,
        "episodes": len(episodes),
        "fixed_horizon": fixed_horizon,
        "zero_dispersion": zero_dispersion,
        "truth_current_fano_undefined_reason": undefined_reason,
    }


def truth_current_analysis(
    events: Sequence[ImitationEvent],
    *,
    bootstrap_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 1,
    statistics: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Episode truth displacement and its across-episode precision ratio."""

    names = tuple(CURRENT_STATISTICS if statistics is None else statistics)
    unknown = sorted(set(names) - set(CURRENT_STATISTICS))
    if unknown:
        raise ValueError("unknown truth-current statistic(s): " + ", ".join(unknown))
    if not names:
        raise ValueError("at least one truth-current statistic is required")
    if bootstrap_resamples < 0:
        raise ValueError("bootstrap_resamples cannot be negative")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    episode_rows = [
        episode_summary(group)
        for group in _group_events(events, key=lambda event: event.episode_id).values()
    ]
    if not episode_rows:
        raise ValueError("truth_current_analysis requires at least one episode")
    result = _truth_current_cell_fields(
        episode_rows, population_size=sum(events[0].N_t)
    )
    result["statistics"] = list(names)
    result["bootstrap_resamples"] = bootstrap_resamples
    result["confidence"] = confidence
    result["bootstrap_unit"] = "episode"
    result["null_model"] = None

    rng = np.random.default_rng(seed)
    means: list[float] = []
    fanos: list[float] = []
    for _ in range(bootstrap_resamples):
        indices = rng.choice(len(episode_rows), size=len(episode_rows), replace=True)
        drawn = [episode_rows[int(index)] for index in indices]
        fields = _truth_current_cell_fields(drawn, population_size=sum(events[0].N_t))
        if math.isfinite(float(fields["truth_current_mean"])):
            means.append(float(fields["truth_current_mean"]))
        if math.isfinite(float(fields["truth_current_fano"])):
            fanos.append(float(fields["truth_current_fano"]))
    alpha = (1.0 - confidence) / 2.0
    result["truth_current_mean_ci_low"], result["truth_current_mean_ci_high"] = _quantiles(
        means, alpha
    )
    result["truth_current_fano_ci_low"], result["truth_current_fano_ci_high"] = _quantiles(
        fanos, alpha
    )
    result["truth_current_fano_bootstrap_valid_replicates"] = len(fanos)
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
    parameter = _order_parameter_of(name)
    if parameter is not None:
        before_field, after_field = ORDER_PARAMETER_COUNT_FIELDS[parameter]
        before = [getattr(event, before_field) for event in events]
        if _is_sensing_statistic(name):
            return mutual_information(before, [event.Y_t for event in events])
        return conditional_mutual_information(
            [event.U_t for event in events],
            [getattr(event, after_field) for event in events],
            before,
        )
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
    if _is_sensing_statistic(name):
        return []
    parameter = _order_parameter_of(name)
    if parameter is not None:
        before_field, _ = ORDER_PARAMETER_COUNT_FIELDS[parameter]
        return [getattr(event, before_field) for event in events]
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
            if _is_sensing_statistic(name)
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
    """Estimate the requested channels with episode bootstrap and temporal nulls."""

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
            if (
                event.Y_t is not None
                if _is_sensing_statistic(name)
                else event.U_t is not None
            )
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
                field="Y_t" if _is_sensing_statistic(name) else "U_t",
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
                    if _is_sensing_statistic(name)
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
            "scientifically_interpretable": _is_sensing_statistic(name) or not degenerate,
            **support,
        })
    return estimates, null_rows


@dataclass(frozen=True, slots=True)
class DiagnosticValue:
    """One controller diagnostic, with the parts a ratio was built from.

    ``numerator`` and ``denominator`` are carried so a reader can audit an
    information fraction instead of trusting it, and so a bootstrap replicate
    that lands on a zero denominator can say *why* it is ``NaN``.
    """

    value: float
    numerator: float | None = None
    denominator: float | None = None
    undefined_reason: str | None = None


def _entropy_bits(counts: Mapping[Hashable, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return math.nan
    return -sum(
        (count / total) * math.log2(count / total) for count in counts.values() if count
    )


def _states_by_action(
    actions: Sequence[str], states: Sequence[Hashable]
) -> dict[Hashable, Counter]:
    grouped: dict[Hashable, Counter] = defaultdict(Counter)
    for action, state in zip(actions, states, strict=True):
        grouped[state][action] += 1
    return dict(grouped)


def conditional_action_entropy_bits(
    actions: Sequence[str], states: Sequence[Hashable]
) -> float:
    """``H(U | state)`` in bits, by the direct counting the CMIs already use.

    This is the entropy budget for ``I(U; next_state | state)``: a controller
    with no remaining action uncertainty at a fixed state cannot carry
    information about where that state goes next, however strong its real
    effect on the population is.
    """

    if not actions:
        return math.nan
    total = len(actions)
    return float(
        sum(
            (sum(counter.values()) / total) * _entropy_bits(counter)
            for counter in _states_by_action(actions, states).values()
        )
    )


def action_overlap_diagnostics(
    actions: Sequence[str], states: Sequence[Hashable]
) -> dict[str, Any]:
    """How much data supports a within-state ``ADVOCATE_Z`` vs ``NO_OP`` contrast.

    A nonzero overall ``H(U)`` is not evidence of overlap: a hard threshold
    controller varies its action across the run while being deterministic in
    every individual state, so every event sits in a single-action slice and
    contributes nothing to a within-state comparison.
    """

    grouped = _states_by_action(actions, states)
    dual = [
        counter
        for counter in grouped.values()
        if counter[ADVOCATE_TARGET] > 0 and counter[NO_OP] > 0
    ]
    events_in_dual = sum(sum(counter.values()) for counter in dual)
    return {
        "occupied_conditioning_states": len(grouped),
        "dual_action_conditioning_states": len(dual),
        "fraction_conditioning_states_with_both_actions": (
            math.nan if not grouped else len(dual) / len(grouped)
        ),
        "fraction_events_in_dual_action_conditioning_states": (
            math.nan if not actions else events_in_dual / len(actions)
        ),
    }


def signed_actuation_response(
    actions: Sequence[str],
    states: Sequence[Hashable],
    deltas: Sequence[Any],
) -> DiagnosticValue:
    """State-adjusted ``ADVOCATE_Z`` minus ``NO_OP`` movement of an order parameter.

    Adjusted rather than a bare difference of means because the policy is a
    function of the current state: comparing all advocacy events against all
    no-op events would mostly measure which states each action is taken in.
    States seeing only one action are dropped, and the survivors are combined
    with empirical event-frequency weights over that overlap-supported subset.
    """

    grouped: dict[Hashable, dict[str, list[float]]] = defaultdict(
        lambda: {ADVOCATE_TARGET: [], NO_OP: []}
    )
    for action, state, delta in zip(actions, states, deltas, strict=True):
        if delta is None or not math.isfinite(float(delta)):
            continue
        grouped[state][action].append(float(delta))
    weighted, weight = 0.0, 0
    for buckets in grouped.values():
        advocate, no_op = buckets[ADVOCATE_TARGET], buckets[NO_OP]
        if not advocate or not no_op:
            continue
        size = len(advocate) + len(no_op)
        weighted += size * (sum(advocate) / len(advocate) - sum(no_op) / len(no_op))
        weight += size
    if weight == 0:
        return DiagnosticValue(
            math.nan, undefined_reason="no_conditioning_state_saw_both_actions"
        )
    return DiagnosticValue(weighted / weight)


def _diagnostic_family(name: str) -> str:
    if name in CONTROLLER_ENTROPY_STATISTICS:
        return "controller_entropy"
    if name in ACTUATION_INFORMATION_FRACTION_STATISTICS:
        return "information_fraction"
    if name in SIGNED_ACTUATION_STATISTICS:
        return "signed_actuation"
    if name in ACTION_OVERLAP_STATISTICS:
        return "action_overlap"
    raise ValueError(f"unknown HiddenBench imitation controller diagnostic {name!r}")


def _diagnostic_channel(name: str) -> str | None:
    """Which conditioning variable the diagnostic is measured against."""

    for channel in ACTUATION_CHANNEL_COUNT_FIELDS:
        if name.startswith(f"{channel}_") or name.endswith(f"_given_{channel}"):
            return channel
    return None


def _controlled_events(events: Sequence[ImitationEvent]) -> list[ImitationEvent]:
    return [event for event in events if event.U_t in {ADVOCATE_TARGET, NO_OP}]


def _channel_series(
    channel: str, events: Sequence[ImitationEvent]
) -> tuple[list[Hashable], list[Hashable]]:
    before_field, after_field = ACTUATION_CHANNEL_COUNT_FIELDS[channel]
    return (
        [getattr(event, before_field) for event in events],
        [getattr(event, after_field) for event in events],
    )


def _diagnostic_value(name: str, events: Sequence[ImitationEvent]) -> DiagnosticValue:
    """Compute one diagnostic; the single entry point bootstrap and null reuse."""

    controlled = _controlled_events(events)
    if not controlled:
        return DiagnosticValue(math.nan, undefined_reason="no_controlled_events")
    actions = [str(event.U_t) for event in controlled]
    if name == "controller_action_entropy":
        return DiagnosticValue(_entropy_bits(Counter(actions)))
    family = _diagnostic_family(name)
    channel = _diagnostic_channel(name)
    assert channel is not None  # every remaining diagnostic names its channel
    states, next_states = _channel_series(channel, controlled)
    if family == "controller_entropy":
        return DiagnosticValue(conditional_action_entropy_bits(actions, states))
    if family == "information_fraction":
        denominator = conditional_action_entropy_bits(actions, states)
        numerator = getattr(
            conditional_mutual_information(actions, next_states, states),
            MAIN_ESTIMATOR_VARIANT,
        )
        if not math.isfinite(denominator) or denominator <= ACTION_ENTROPY_EPSILON_BITS:
            # Undefined, not zero: the controller had no action freedom left to
            # spend, so the question the ratio asks does not arise.
            return DiagnosticValue(
                math.nan,
                numerator,
                denominator,
                "controller_deterministic_given_conditioning_state",
            )
        return DiagnosticValue(numerator / denominator, numerator, denominator)
    if family == "signed_actuation":
        deltas = [
            event.event.get(ACTUATION_CHANNEL_DELTA_FIELDS[channel]) for event in controlled
        ]
        return signed_actuation_response(actions, states, deltas)
    overlap = action_overlap_diagnostics(actions, states)
    return DiagnosticValue(overlap["fraction_events_in_dual_action_conditioning_states"])


def _quantiles(values: Sequence[float], alpha: float) -> tuple[float, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(finite):
        return (math.nan, math.nan)
    return (
        float(np.quantile(finite, alpha)),
        float(np.quantile(finite, 1 - alpha)),
    )


def controller_diagnostic_analysis(
    events: Sequence[ImitationEvent],
    *,
    bootstrap_resamples: int = 1000,
    null_permutations: int = 1000,
    confidence: float = 0.95,
    seed: int = 1,
    diagnostics: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Entropy, information-fraction, signed-response and overlap diagnostics.

    Whole episodes are the bootstrap unit, as for the CMIs themselves.  Ratios
    are recomputed from a resample's own numerator and denominator rather than
    assembled from separately resampled parts, so a replicate whose conditional
    entropy vanishes reports ``NaN`` and is excluded from the interval; the
    surviving count is kept on the row.

    Entropy and overlap rows carry no permutation null on purpose: they describe
    the controller's own policy, which a null that reshuffles that policy would
    simply destroy rather than test.
    """

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    names = tuple(
        CONTROLLER_DIAGNOSTIC_STATISTICS if diagnostics is None else diagnostics
    )
    unknown = sorted(set(names) - set(CONTROLLER_DIAGNOSTIC_STATISTICS))
    if unknown:
        raise ValueError(
            "unknown HiddenBench imitation controller diagnostic(s): " + ", ".join(unknown)
        )
    if not names:
        raise ValueError("at least one HiddenBench imitation controller diagnostic is required")
    alpha = (1 - confidence) / 2
    controlled = _controlled_events(events)
    by_episode = _group_events(controlled, key=lambda event: event.episode_id)
    actions = [str(event.U_t) for event in controlled]
    channel_entropy = {
        channel: conditional_action_entropy_bits(
            actions, _channel_series(channel, controlled)[0]
        )
        for channel in ACTUATION_CHANNEL_COUNT_FIELDS
        if controlled
    }
    overlap_by_channel = {
        channel: action_overlap_diagnostics(
            actions, _channel_series(channel, controlled)[0]
        )
        for channel in ACTUATION_CHANNEL_COUNT_FIELDS
        if controlled
    }

    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        family = _diagnostic_family(name)
        channel = _diagnostic_channel(name)
        main = _diagnostic_value(name, controlled)
        bootstrap_values: list[float] = []
        if controlled and family != "action_overlap":
            draws = bootstrap_episode_ids(
                list(by_episode), resamples=bootstrap_resamples, seed=seed + index
            )
            bootstrap_values = [
                _diagnostic_value(name, _resampled_events(by_episode, draw)).value
                for draw in draws
            ]
        null_values: list[float] = []
        if controlled and family in {"information_fraction", "signed_actuation"}:
            for permutation in range(null_permutations):
                rng = np.random.default_rng(
                    seed + 1_000_000 + 20_000 * (index + 1) + permutation
                )
                null_values.append(
                    _diagnostic_value(
                        name, _perturb_within_episode(controlled, field="U_t", rng=rng)
                    ).value
                )
        finite_null = [value for value in null_values if math.isfinite(value)]
        bootstrap_interval = _quantiles(bootstrap_values, alpha)
        null_interval = _quantiles(null_values, alpha)
        bound_satisfied: bool | None = None
        if (
            main.numerator is not None
            and main.denominator is not None
            and math.isfinite(main.numerator)
            and math.isfinite(main.denominator)
        ):
            bound_satisfied = bool(
                main.numerator <= main.denominator + ENTROPY_BOUND_TOLERANCE_BITS
            )
        entropy = None if channel is None else channel_entropy.get(channel)
        rows.append({
            "statistic": name,
            "family": family,
            "conditioning": None if channel is None else CONDITIONING_LABELS[channel],
            "main_estimator_variant": MAIN_ESTIMATOR_VARIANT,
            "value": main.value,
            "numerator_bits": math.nan if main.numerator is None else main.numerator,
            "denominator_bits": math.nan if main.denominator is None else main.denominator,
            "undefined_reason": main.undefined_reason,
            "bootstrap_ci_low": bootstrap_interval[0],
            "bootstrap_ci_high": bootstrap_interval[1],
            "bootstrap_resamples": len(bootstrap_values),
            "bootstrap_valid_replicates": sum(
                1 for value in bootstrap_values if math.isfinite(value)
            ),
            "null_mean": (
                math.nan if not finite_null else float(sum(finite_null) / len(finite_null))
            ),
            "null_ci_low": null_interval[0],
            "null_ci_high": null_interval[1],
            "null_permutations": len(null_values),
            "null_type": (
                None if not null_values else "within_episode_controller_action_permutation"
            ),
            "entropy_bound_satisfied": bound_satisfied,
            "controller_deterministic_given_conditioning_state": (
                None
                if entropy is None
                else bool(not math.isfinite(entropy) or entropy <= ACTION_ENTROPY_EPSILON_BITS)
            ),
            "n_episodes": len(by_episode),
            "n_events": len(controlled),
            **(
                {field: None for field in ACTION_OVERLAP_FIELDS}
                if channel is None or channel not in overlap_by_channel
                else overlap_by_channel[channel]
            ),
        })
    return rows


def _cell_dir_for(path: Path) -> Path | None:
    for candidate in path.parents:
        if candidate.parent.name == "cells":
            return candidate
    return None


def read_imitation_events(run_dir: str | Path) -> list[ImitationEvent]:
    root = Path(run_dir)
    compact_candidates = (
        [root]
        if root.is_file() and root.name.endswith(".parquet")
        else list(root.glob("scientific_events.parquet"))
        + list(root.glob("cells/*/scientific_events.parquet"))
        + list(root.rglob(".resume/*/scientific_events.parquet"))
    )
    if compact_candidates:
        from mas_cc.storage import iter_compact_imitation_events

        compact_events = []
        for event, episode_id, cell_id in iter_compact_imitation_events(root):
            mechanism = event.get("_compact_control_mechanism")
            compact_events.append(
                adapt_event(
                    event,
                    episode_id=episode_id,
                    cell_id=cell_id,
                    overrides=(
                        {} if mechanism is None else {"control.mechanism": mechanism}
                    ),
                )
            )
        if not compact_events:
            raise ValueError(f"compact scientific files under {root} contain no imitation events")
        return compact_events
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
                "control_mechanism": control_mechanism_of(ordered[0]),
            }
            shares.extend(
                {**common, "option": option, "share": state_shares[option]}
                for option in ordered[0].options
            )
            observables.append({**common, **state_observables})
    return shares, observables


def _format_bits(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.4f}" if math.isfinite(number) else "—"


def _format_ci(low: Any, high: Any) -> str:
    low_text, high_text = _format_bits(low), _format_bits(high)
    return "—" if low_text == "—" and high_text == "—" else f"[{low_text}, {high_text}]"


def _format_bool(value: Any) -> str:
    return "—" if value is None else ("yes" if value else "no")


def _format_number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return "—" if not math.isfinite(value) else f"{value:g}"
    return str(value)


def _format_percent(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number * 100:.1f}%" if math.isfinite(number) else "—"


def _format_reason(value: Any) -> str:
    return "—" if not value else f"`{value}`"


def _format_current(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isinf(number):
        return "+∞" if number > 0 else "−∞"
    return f"{number:.6g}" if math.isfinite(number) else "—"


def _write_truth_current_markdown(
    rows: Sequence[Mapping[str, Any]], destination: Path
) -> None:
    lines = [
        "# Truth-current estimates",
        "",
        "`truth_current` is switches toward the correct answer minus switches away. "
        "Because only the focal vote can change, it telescopes to final minus initial "
        "truth headcount. `truth_current_fano` is `abs(mean current) / sample variance` "
        "across equal-horizon episodes; it is a precision ratio, not a TUR claim.",
        "",
        "| Cell | Mean J | Sample variance | Fano / precision | Episodes | Fixed horizon "
        "| Zero dispersion | Mean 95% bootstrap CI | Fano 95% bootstrap CI | Undefined because |",
        "|---|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| `{cell}` | {mean} | {variance} | {fano} | {episodes} | {fixed} | {zero} "
            "| [{mean_low}, {mean_high}] | [{fano_low}, {fano_high}] | {reason} |".format(
                cell=row.get("cell_id", "run"),
                mean=_format_current(row.get("truth_current_mean")),
                variance=_format_current(row.get("truth_current_variance")),
                fano=_format_current(row.get("truth_current_fano")),
                episodes=row.get("episodes", "—"),
                fixed="yes" if row.get("fixed_horizon") else "no",
                zero="yes" if row.get("zero_dispersion") else "no",
                mean_low=_format_current(row.get("truth_current_mean_ci_low")),
                mean_high=_format_current(row.get("truth_current_mean_ci_high")),
                fano_low=_format_current(row.get("truth_current_fano_ci_low")),
                fano_high=_format_current(row.get("truth_current_fano_ci_high")),
                reason=_format_reason(row.get("truth_current_fano_undefined_reason")),
            )
        )
    destination.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _controller_diagnostic_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Render one cell's controller diagnostics as four small, purposeful tables.

    Split by family rather than dumped as one wide table because the four
    families answer different questions and share almost no columns: an entropy
    is in bits, a fraction is dimensionless and may be undefined, a signed
    response has a meaningful sign, and overlap is a support count.
    """

    if not rows:
        return []
    by_family: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row.get("family"), []).append(row)
    lines: list[str] = []

    entropies = by_family.get("controller_entropy", [])
    if entropies:
        lines.extend([
            "### Controller entropy / available action information",
            "",
            "How much action freedom the controller had left once the conditioning variable is "
            "known. Each conditional entropy is the ceiling on the actuation CMI measured in the "
            "same space, so a CMI far below its ceiling means a weak effect, and a ceiling near "
            "zero means the CMI could not have been large whatever the controller does.",
            "",
            "| Statistic | Bits | 95% bootstrap CI | Valid replicates | Deterministic given state |",
            "|---|---|---|---|---|",
        ])
        for row in entropies:
            lines.append(
                "| `{statistic}` | {value} | {ci} | {replicates} | {deterministic} |".format(
                    statistic=row.get("statistic"),
                    value=_format_bits(row.get("value")),
                    ci=_format_ci(row.get("bootstrap_ci_low"), row.get("bootstrap_ci_high")),
                    replicates=_format_number(row.get("bootstrap_valid_replicates")),
                    deterministic=_format_bool(
                        row.get("controller_deterministic_given_conditioning_state")
                    ),
                )
            )
        lines.append("")

    fractions = by_family.get("information_fraction", [])
    if fractions:
        lines.extend([
            "### Actuation information fractions",
            "",
            "Each actuation CMI divided by the conditional controller entropy in the same "
            "conditioning space. A **normalization diagnostic, not a thermodynamic efficiency**. "
            "The ratio is `—` (undefined, never 0) when the denominator vanishes; the numerator "
            "and denominator are printed so the ratio can be audited. Ratios are not clipped to "
            "`[0, 1]` — a value above 1 is an estimator pathology worth seeing, which "
            "**Bound holds** flags.",
            "",
            "| Statistic | CMI numerator (bits) | H(U \\| state) denominator (bits) | Ratio "
            "| 95% bootstrap CI | Valid replicates | Null mean | Bound holds | Undefined because |",
            "|---|---|---|---|---|---|---|---|---|",
        ])
        for row in fractions:
            lines.append(
                "| `{statistic}` | {numerator} | {denominator} | {value} | {ci} | {replicates} "
                "| {null} | {bound} | {reason} |".format(
                    statistic=row.get("statistic"),
                    numerator=_format_bits(row.get("numerator_bits")),
                    denominator=_format_bits(row.get("denominator_bits")),
                    value=_format_bits(row.get("value")),
                    ci=_format_ci(row.get("bootstrap_ci_low"), row.get("bootstrap_ci_high")),
                    replicates=_format_number(row.get("bootstrap_valid_replicates")),
                    null=_format_bits(row.get("null_mean")),
                    bound=_format_bool(row.get("entropy_bound_satisfied")),
                    reason=_format_reason(row.get("undefined_reason")),
                )
            )
        lines.append("")

    signed = by_family.get("signed_actuation", [])
    if signed:
        lines.extend([
            "### Signed behavioral actuation",
            "",
            "CMI is unsigned: it says the action predicts the transition, not which way the "
            "population moved. These are state-adjusted `ADVOCATE_Z` minus `NO_OP` contrasts, "
            "computed only within conditioning states that saw both actions and weighted by how "
            "many events those states hold. Positive means advocacy moves that order parameter "
            "up relative to doing nothing.",
            "",
            "| Statistic | Response | 95% bootstrap CI | Null mean | Dual-action states "
            "| Events in dual-action states | Undefined because |",
            "|---|---|---|---|---|---|---|",
        ])
        for row in signed:
            lines.append(
                "| `{statistic}` | {value} | {ci} | {null} | {dual} | {events} | {reason} |".format(
                    statistic=row.get("statistic"),
                    value=_format_bits(row.get("value")),
                    ci=_format_ci(row.get("bootstrap_ci_low"), row.get("bootstrap_ci_high")),
                    null=_format_bits(row.get("null_mean")),
                    dual=_format_number(row.get("dual_action_conditioning_states")),
                    events=_format_percent(
                        row.get("fraction_events_in_dual_action_conditioning_states")
                    ),
                    reason=_format_reason(row.get("undefined_reason")),
                )
            )
        lines.append("")

    overlap = by_family.get("action_overlap", [])
    if overlap:
        lines.extend([
            "### Action-overlap diagnostics",
            "",
            "What fraction of the data can actually support a within-state `ADVOCATE_Z` vs "
            "`NO_OP` comparison. **Events in dual-action states** is the quantity that matters: a "
            "nonzero overall `controller_action_entropy` is not evidence of overlap, because a "
            "hard threshold controller varies its action across the run while being deterministic "
            "in every individual state.",
            "",
            "| Conditioning | Occupied states | Dual-action states | States with both actions "
            "| Events in dual-action states |",
            "|---|---|---|---|---|",
        ])
        for row in overlap:
            lines.append(
                "| `{conditioning}` | {occupied} | {dual} | {state_fraction} | {event_fraction} |".format(
                    conditioning=row.get("conditioning"),
                    occupied=_format_number(row.get("occupied_conditioning_states")),
                    dual=_format_number(row.get("dual_action_conditioning_states")),
                    state_fraction=_format_percent(
                        row.get("fraction_conditioning_states_with_both_actions")
                    ),
                    event_fraction=_format_percent(
                        row.get("fraction_events_in_dual_action_conditioning_states")
                    ),
                )
            )
        lines.append("")
    return lines


def _write_information_estimates_markdown(
    rows: Sequence[Mapping[str, Any]],
    destination: Path,
    diagnostic_rows: Sequence[Mapping[str, Any]] = (),
) -> None:
    """Human-readable replacement for the wide, hard-to-scan information_estimates.csv.

    One table of headline numbers plus one table of support diagnostics per
    cell, instead of one ~25-column table where the two kinds of information
    are interleaved.
    """

    lines: list[str] = [
        "# Information-theoretic estimates",
        "",
        "Discrete mutual-information channels estimated from this run's imitation events, "
        "measured in bits. Higher means the controller's action (or sensor) carries more "
        "information about that piece of population/opinion state.",
        "",
        "## Statistics",
        "",
    ]
    reported = {row.get("statistic") for row in rows}
    for name in INFORMATION_STATISTICS:
        if name not in reported:
            continue
        description = INFORMATION_STATISTIC_DESCRIPTIONS.get(name, "")
        lines.append(f"- **`{name}`** — {description}")
    diagnostics_reported = {row.get("statistic") for row in diagnostic_rows}
    if diagnostics_reported:
        lines.extend([
            "",
            "## Controller diagnostics",
            "",
            "The channels above are unsigned and unnormalized, which makes a small CMI ambiguous: "
            "it can mean the controller barely moved the population, or that the controller had "
            "almost no action entropy left once the current state was known. The quantities below "
            "separate those two readings and add the direction the CMI cannot carry. They are "
            "diagnostics for interpreting the CMIs, not new channels and not efficiencies.",
            "",
        ])
        for name in CONTROLLER_DIAGNOSTIC_STATISTICS:
            if name not in diagnostics_reported:
                continue
            description = CONTROLLER_DIAGNOSTIC_DESCRIPTIONS.get(name, "")
            lines.append(f"- **`{name}`** — {description}")
    lines.extend(
        [
            "",
            "Each estimate's headline value uses the `unsmoothed` (direct-counting) plug-in "
            "entropy estimator; `jeffreys` and `miller-madow` are bias-corrected alternatives shown "
            "for sensitivity checking. **Bootstrap CI** resamples whole episodes (never individual "
            "events) into a 95% interval around the headline estimate. **Null mean/CI** comes from "
            "permuting the controller's action (or the sensor sample, for `sensing_mi`) independently "
            "within each episode — an estimate that does not clear the null band is not "
            "distinguishable from chance. **Interpretable** is `no` when the controller never varied "
            "its action in that cell (e.g. always advocated, or never did), which makes the "
            "conditional statistics undefined rather than merely small.",
            "",
        ]
    )
    by_cell: dict[tuple[Any, Any], list[Mapping[str, Any]]] = {}
    diagnostics_by_cell: dict[tuple[Any, Any], list[Mapping[str, Any]]] = {}
    for row in rows:
        by_cell.setdefault((row.get("cell_id"), row.get("scientific_cell")), []).append(row)
    for row in diagnostic_rows:
        diagnostics_by_cell.setdefault(
            (row.get("cell_id"), row.get("scientific_cell")), []
        ).append(row)
    # Both keyings, so a run that requested only diagnostics still gets its cell
    # sections rather than an empty report.
    for cell_id, scientific_cell in dict.fromkeys((*by_cell, *diagnostics_by_cell)):
        cell_rows = by_cell.get((cell_id, scientific_cell), [])
        cell_diagnostics = diagnostics_by_cell.get((cell_id, scientific_cell), [])
        label = f"Cell `{cell_id}`"
        if scientific_cell:
            label += f" (scientific cell {scientific_cell})"
        first = (cell_rows or cell_diagnostics)[0]
        lines.append(f"## {label}")
        lines.append("")
        lines.append(
            f"Dynamics: `{first.get('dynamics_mode')}` · Control: `{first.get('control_mechanism')}`"
        )
        lines.append("")
        if not cell_rows:
            lines.extend(_controller_diagnostic_lines(cell_diagnostics))
            continue
        lines.append(
            "| Statistic | Estimate (bits) | 95% bootstrap CI | Null mean (bits) | 95% null CI "
            "| unsmoothed / jeffreys / miller-madow | Interpretable | Episodes | Events |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for row in cell_rows:
            variants = " / ".join(
                _format_bits(row.get(name)) for name in ("unsmoothed", "jeffreys", "miller_madow")
            )
            lines.append(
                "| `{statistic}` | {estimate} | {bootstrap_ci} | {null_mean} | {null_ci} | "
                "{variants} | {interpretable} | {episodes} | {events} |".format(
                    statistic=row.get("statistic"),
                    estimate=_format_bits(row.get("estimate")),
                    bootstrap_ci=_format_ci(row.get("bootstrap_ci_low"), row.get("bootstrap_ci_high")),
                    null_mean=_format_bits(row.get("null_mean")),
                    null_ci=_format_ci(row.get("null_ci_low"), row.get("null_ci_high")),
                    variants=variants,
                    interpretable=_format_bool(row.get("scientifically_interpretable")),
                    episodes=_format_number(row.get("n_episodes")),
                    events=_format_number(row.get("n_events")),
                )
            )
        lines.append("")
        lines.append(
            "| Statistic | Conditioning states | Events/state (min / median / max) "
            "| Singleton states | Sparse table | Controller degenerate |"
        )
        lines.append("|---|---|---|---|---|---|")
        for row in cell_rows:
            singleton = row.get("fraction_events_singleton_conditioning_states")
            singleton_text = "—" if singleton is None else f"{float(singleton) * 100:.0f}%"
            events_per_state = "{} / {} / {}".format(
                _format_number(row.get("min_events_per_conditioning_state")),
                _format_number(row.get("median_events_per_conditioning_state")),
                _format_number(row.get("max_events_per_conditioning_state")),
            )
            lines.append(
                "| `{statistic}` | {states} | {events_per_state} | {singleton} | {sparse} "
                "| {degenerate} |".format(
                    statistic=row.get("statistic"),
                    states=_format_number(row.get("occupied_conditioning_states")),
                    events_per_state=events_per_state,
                    singleton=singleton_text,
                    sparse=_format_bool(row.get("sparse_conditioning_table")),
                    degenerate=_format_bool(row.get("controller_degenerate")),
                )
            )
        lines.append("")
        lines.extend(_controller_diagnostic_lines(cell_diagnostics))
    destination.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


INFORMATION_COMET_FIELDS = (
    "estimate",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "null_mean",
    "excess_over_null",
    "n_episodes",
    "n_events",
)
"""What each statistic contributes as *numbers* rather than as a report."""


CONTROLLER_DIAGNOSTIC_COMET_FIELDS = (
    "value",
    "numerator_bits",
    "denominator_bits",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "null_mean",
    "bootstrap_valid_replicates",
    "fraction_conditioning_states_with_both_actions",
    "fraction_events_in_dual_action_conditioning_states",
)
"""Numerator and denominator ride along so a ratio stays auditable on Comet."""


def _comet_metric_name(
    cell_id: Any, statistic: str, field: str, *, prefix: str | None = "information"
) -> str:
    label = str(cell_id or "run").replace("/", "_").replace(" ", "_")
    parts = (label, statistic, field) if prefix is None else (prefix, label, statistic, field)
    return "/".join(parts)


def controller_diagnostic_comet_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Flatten diagnostics into ``cell/statistic/field`` Comet scalars.

    The cell-first namespace keeps even the longest standard diagnostic below
    Comet's 100-character metric-name limit.  These remain off the
    ``information/`` prefix on purpose: they are not bits of channel capacity,
    and mixing a dimensionless ratio into the same namespace as the MI series
    invites plotting them on one axis.
    """

    metrics: dict[str, float] = {}
    for row in rows:
        statistic = str(row.get("statistic") or "")
        if not statistic:
            continue
        for field in (*CONTROLLER_DIAGNOSTIC_COMET_FIELDS, *ACTION_OVERLAP_FIELDS):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if math.isnan(float(value)):
                continue
            metrics[
                _comet_metric_name(
                    row.get("cell_id"), statistic, field, prefix=None
                )
            ] = float(value)
        for field in (
            "entropy_bound_satisfied",
            "controller_deterministic_given_conditioning_state",
        ):
            flag = row.get(field)
            if isinstance(flag, bool):
                metrics[
                    _comet_metric_name(
                        row.get("cell_id"), statistic, field, prefix=None
                    )
                ] = float(flag)
    return metrics


def information_comet_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Flatten the estimate table into loggable scalars.

    `excess_over_null` is derived here rather than left to the reader: an MI
    value on its own is not a result, because a permutation null is rarely
    zero. The quantity that answers "is there a channel" is estimate minus
    null mean, so it is the one that gets its own series.
    """

    metrics: dict[str, float] = {}
    for row in rows:
        statistic = str(row.get("statistic") or "")
        if not statistic:
            continue
        estimate, null_mean = row.get("estimate"), row.get("null_mean")
        derived = dict(row)
        if isinstance(estimate, (int, float)) and isinstance(null_mean, (int, float)):
            derived["excess_over_null"] = float(estimate) - float(null_mean)
        for field in INFORMATION_COMET_FIELDS:
            value = derived.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if math.isnan(float(value)):
                continue
            metrics[_comet_metric_name(row.get("cell_id"), statistic, field)] = float(value)
        interpretable = row.get("scientifically_interpretable")
        if isinstance(interpretable, bool):
            metrics[
                _comet_metric_name(row.get("cell_id"), statistic, "scientifically_interpretable")
            ] = float(interpretable)
    return metrics


def plot_information_estimates(
    rows: Sequence[Mapping[str, Any]], destination: Path
) -> list[Path]:
    """One figure per cell: each estimate with its CI, against its null mean.

    The null is drawn on the same axis on purpose. A bare bar chart of MI
    values invites reading height as evidence, when the only thing that counts
    as a channel is the gap between the estimate and the permutation null.
    """

    if not rows:
        return []
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for cell_id, group in _group_rows_by_cell(rows).items():
        usable = [row for row in group if isinstance(row.get("estimate"), (int, float))]
        if not usable:
            continue
        labels = [str(row.get("statistic")) for row in usable]
        estimates = [float(row["estimate"]) for row in usable]
        positions = list(range(len(usable)))
        lower = [
            max(0.0, value - float(row.get("bootstrap_ci_low") or value))
            for value, row in zip(estimates, usable)
        ]
        upper = [
            max(0.0, float(row.get("bootstrap_ci_high") or value) - value)
            for value, row in zip(estimates, usable)
        ]
        figure, axis = plt.subplots(figsize=(7, 4), dpi=120)
        axis.bar(positions, estimates, yerr=[lower, upper], capsize=4, alpha=0.75, label="estimate")
        nulls = [
            (index, float(row["null_mean"]))
            for index, row in enumerate(usable)
            if isinstance(row.get("null_mean"), (int, float))
        ]
        if nulls:
            axis.scatter(
                [index for index, _ in nulls],
                [value for _, value in nulls],
                marker="_", s=400, color="crimson", zorder=3, label="permutation null mean",
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=20, ha="right", fontsize="small")
        axis.set_ylabel("bits")
        axis.set_title(f"Information estimates — {cell_id or 'run'}")
        axis.grid(alpha=0.25, axis="y")
        axis.legend(frameon=False, fontsize="small")
        figure.tight_layout()
        label = str(cell_id or "run").replace("/", "_").replace(" ", "_")
        path = destination / f"information_estimates_{label}.png"
        figure.savefig(path)
        plt.close(figure)
        written.append(path)
    return written


def _group_rows_by_cell(
    rows: Sequence[Mapping[str, Any]],
) -> dict[Any, list[Mapping[str, Any]]]:
    grouped: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.get("cell_id"), []).append(row)
    return grouped


def _export_information_estimates_to_comet(
    rows: Sequence[Mapping[str, Any]],
    assets: Sequence[Path],
    images: Sequence[Path],
    *,
    enabled: bool,
    project_name: str,
    run_name: str,
    sink: Any | None = None,
    diagnostic_rows: Sequence[Mapping[str, Any]] = (),
    name_suffix: str | None = None,
) -> dict[str, Any]:
    """Publish the estimates as metrics, figures and assets, if requested.

    Metrics, not only the report. The markdown attachment was the whole export
    before, which meant the numbers this analysis exists to produce were
    downloadable but never plottable, comparable across runs, or visible on the
    dashboard at all.

    ``sink`` lets the caller supply an experiment that is already open — the
    run's master, under ``observability.comet.cell_reporting: master`` — so the
    report lands on the run it describes instead of a sibling experiment. A
    borrowed sink is never closed here: the master owns its own lifetime, and
    ending it early would silently drop everything logged afterwards.

    A local import: this module has no other reason to depend on the
    observability stack, and most runs never request a Comet export.
    """

    if not enabled:
        return {"status": "disabled", "metrics": 0, "images": 0, "assets": 0, "url": None}
    borrowed = sink is not None
    if sink is None:
        from mas_cc.observability.recorder import CometMetricSink

        sink = CometMetricSink(True, project_name=project_name, run_name=run_name)
    metrics = {
        **information_comet_metrics(rows),
        **controller_diagnostic_comet_metrics(diagnostic_rows),
    }
    uploaded_images = 0
    uploaded_assets = 0
    try:
        sink.add_tags(("analysis", "information"))
        if metrics:
            sink.log_metrics(metrics, 0)
        for image in images:
            if Path(image).is_file():
                image_name = Path(image).stem
                if name_suffix:
                    image_name = f"{image_name}__{name_suffix}"
                sink.log_image(image, name=image_name, step=0)
                uploaded_images += 1
        for asset in assets:
            if Path(asset).is_file():
                asset_name = Path(asset).name
                if name_suffix:
                    path = Path(asset)
                    asset_name = f"{path.stem}__{name_suffix}{path.suffix}"
                sink.log_asset(asset, name=asset_name)
                uploaded_assets += 1
        summary = {
            "status": sink.status,
            "metrics": len(metrics),
            "images": uploaded_images,
            "assets": uploaded_assets,
            "url": sink.url,
            "published_to": "master" if borrowed else "analysis_experiment",
        }
    finally:
        if not borrowed:
            sink.close()
    return summary


def analyze_hidden_bench_imitation(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_resamples: int = 1000,
    null_permutations: int = 1000,
    confidence: float = 0.95,
    seed: int = 1,
    statistics: Sequence[str] | None = None,
    diagnostics: Sequence[str] | None = None,
    current_statistics: Sequence[str] | None = None,
    comet_export: bool = False,
    comet_project: str = "mas-cc",
    comet_run_name: str | None = None,
    comet_sink: Any | None = None,
    comet_name_suffix: str | None = None,
    artifact_profile: str = "full",
    resolved_config_hash: str | None = None,
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
            "control_mechanism": control_mechanism_of(event),
            "N_t": json.dumps(event.N_t),
            "N_t1": json.dumps(event.N_t1),
            "Y_t": None if event.Y_t is None else json.dumps(event.Y_t),
            "U_t": event.U_t,
            "Z_t": event.Z_t,
            "Z_t1": event.Z_t1,
            "Mtruth_t": event.Mtruth_t,
            "Mtruth_t1": event.Mtruth_t1,
            "Morder_t": event.Morder_t,
            "Morder_t1": event.Morder_t1,
            "Xf_t": event.Xf_t,
            "Xf_t1": event.Xf_t1,
        }
        for field in (
            "delta_m_ctrl", "delta_m_truth", "delta_m_order", "delta_H_vote",
            "focal_changed", "focal_adopted_target", "focal_left_target",
            "u_advocate", "sensor_target_share", "population_target_share",
            "sensor_target_error", "sensor_target_abs_error",
            "controller_threshold", "controller_beta",
            "controller_advocacy_probability",
            "truth_current_increment", "truth_switch_toward", "truth_switch_away",
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

    requested_currents = tuple(
        CURRENT_STATISTICS if current_statistics is None else current_statistics
    )
    unknown_currents = sorted(set(requested_currents) - set(CURRENT_STATISTICS))
    if unknown_currents:
        raise ValueError(
            "unknown HiddenBench imitation current statistic(s): "
            + ", ".join(unknown_currents)
        )
    current_rows: list[dict[str, Any]] = []
    if requested_currents:
        for cell_id, group in _group_events(events, key=lambda event: event.cell_id).items():
            current_rows.append(
                {
                    "cell_id": cell_id,
                    **truth_current_analysis(
                        group,
                        bootstrap_resamples=bootstrap_resamples,
                        confidence=confidence,
                        seed=seed,
                        statistics=requested_currents,
                    ),
                }
            )
        pd.DataFrame(current_rows).to_csv(
            destination / "truth_current_estimates.csv", index=False
        )
        _write_truth_current_markdown(
            current_rows, destination / "truth_current_estimates.md"
        )

    share_rows, observable_rows = _trajectory_rows(events)
    pd.DataFrame(share_rows).to_csv(destination / "option_share_trajectories.csv", index=False)
    pd.DataFrame(observable_rows).to_csv(
        destination / "order_parameter_trajectories.csv", index=False
    )

    information_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for cell_id, group in _group_events(events, key=lambda event: event.cell_id).items():
        cell_row = next(row for row in cell_rows if row["cell_id"] == cell_id)
        common = {
            "cell_id": cell_id,
            "scientific_cell": cell_row.get("scientific_cell"),
            "dynamics_mode": cell_row["dynamics_mode"],
            "control_mechanism": cell_row["control_mechanism"],
        }
        if statistics is None or statistics:
            cell_estimates, cell_nulls = information_analysis(
                group,
                bootstrap_resamples=bootstrap_resamples,
                null_permutations=null_permutations,
                confidence=confidence,
                seed=seed,
                statistics=statistics,
            )
            information_rows.extend({**common, **row} for row in cell_estimates)
            null_rows.extend({**common, **row} for row in cell_nulls)
        if diagnostics is None or diagnostics:
            diagnostic_rows.extend(
                {**common, **row}
                for row in controller_diagnostic_analysis(
                    group,
                    bootstrap_resamples=bootstrap_resamples,
                    null_permutations=null_permutations,
                    confidence=confidence,
                    seed=seed,
                    diagnostics=diagnostics,
                )
            )
    _write_information_estimates_markdown(
        information_rows, destination / "information_estimates.md", diagnostic_rows
    )
    pd.DataFrame(diagnostic_rows).to_csv(
        destination / "controller_diagnostics.csv", index=False
    )
    # The markdown is for reading; this is for everything else. Without it the
    # numbers this analysis exists to produce are only recoverable by parsing a
    # table out of prose, which is why the Comet export could never send them.
    pd.DataFrame(information_rows).to_csv(destination / "information_estimates.csv", index=False)
    pd.DataFrame(null_rows).to_csv(destination / "information_nulls.csv", index=False)
    information_plots = plot_information_estimates(information_rows, destination / "plots")
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
    analysis_settings = {
        "bootstrap_resamples": bootstrap_resamples,
        "null_permutations": null_permutations,
        "confidence": confidence,
        "seed": seed,
        "statistics": list(INFORMATION_STATISTICS if statistics is None else statistics),
        "diagnostics": list(
            CONTROLLER_DIAGNOSTIC_STATISTICS if diagnostics is None else diagnostics
        ),
        "current_statistics": list(requested_currents),
    }
    summary = {
        "n_cells": len(cell_rows),
        "n_episodes": len(episode_rows),
        "n_events": len(events),
        "scientific_cells": sorted(row["scientific_cell"] for row in cell_rows if row["scientific_cell"]),
        "matched_initial_state_across_cells": len(initial_states) == 1,
        "main_estimator_variant": MAIN_ESTIMATOR_VARIANT,
        "information_statistics": analysis_settings["statistics"],
        "controller_diagnostics": analysis_settings["diagnostics"],
        "current_statistics": analysis_settings["current_statistics"],
        "bootstrap_resamples": bootstrap_resamples,
        "null_permutations": null_permutations,
        "confidence": confidence,
        "seed": seed,
        "analysis_config_hash": hashlib.sha256(
            json.dumps(
                analysis_settings, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "analysis_code_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifact_profile": artifact_profile,
        "resolved_config_hash": resolved_config_hash,
        "auc_convention": "equal_event_spacing_mean_including_initial_state",
        "output_dir": str(destination),
    }
    (destination / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["information_plots"] = [str(path) for path in information_plots]
    export_assets = [
        destination / "information_estimates.md",
        destination / "information_estimates.csv",
        destination / "support_diagnostics.csv",
        destination / "controller_diagnostics.csv",
    ]
    if requested_currents:
        export_assets.extend(
            (
                destination / "truth_current_estimates.md",
                destination / "truth_current_estimates.csv",
            )
        )
    summary["comet"] = _export_information_estimates_to_comet(
        information_rows,
        export_assets,
        information_plots,
        enabled=comet_export,
        project_name=comet_project,
        run_name=comet_run_name or destination.name,
        sink=comet_sink,
        diagnostic_rows=diagnostic_rows,
        name_suffix=comet_name_suffix,
    )
    (destination / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if artifact_profile == "results_only":
        for name in (
            "event_metrics.csv",
            "episode_summaries.csv",
            "cell_summaries.csv",
            "option_share_trajectories.csv",
            "order_parameter_trajectories.csv",
            "information_nulls.csv",
            "controller_diagnostics.csv",
        ):
            path = destination / name
            if path.is_file():
                path.unlink()
    return summary


__all__ = [
    "ACTION_ENTROPY_EPSILON_BITS",
    "ACTION_OVERLAP_STATISTICS",
    "ACTUATION_CHANNEL_COUNT_FIELDS",
    "ACTUATION_INFORMATION_FRACTION_STATISTICS",
    "CONTROLLER_DIAGNOSTIC_STATISTICS",
    "CONTROLLER_ENTROPY_STATISTICS",
    "DiagnosticValue",
    "ENTROPY_BOUND_TOLERANCE_BITS",
    "ImitationEvent",
    "CURRENT_STATISTICS",
    "INFORMATION_STATISTICS",
    "MAIN_ESTIMATOR_VARIANT",
    "ORDER_PARAMETER_COUNT_FIELDS",
    "SIGNED_ACTUATION_STATISTICS",
    "action_overlap_diagnostics",
    "adapt_event",
    "analyze_hidden_bench_imitation",
    "binary_action_entropy_bits",
    "bootstrap_episode_ids",
    "cell_summary",
    "conditional_action_entropy_bits",
    "controller_diagnostic_analysis",
    "controller_diagnostic_comet_metrics",
    "enrich_event",
    "episode_summary",
    "information_analysis",
    "truth_current_analysis",
    "read_imitation_events",
    "signed_actuation_response",
]
