"""Population order parameters for HiddenBench imitation trajectories."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

from mas_cc.metrics import ActionSharePerOption, AgentCurrentValue, DominantValueShare, RoundView
from mas_cc.metrics.base import MetricKey, StreamingMetric

from ..metrics_common import DisclosureReach, UnsharedDisclosureRate
from .state import ImitationGameState


def population_observables(
    votes: Sequence[str],
    options: Sequence[str],
    correct_answer: str,
    controller_target: str | None = None,
) -> dict[str, Any]:
    if len(options) < 2 or not votes:
        raise ValueError("observables need at least two options and one vote")
    if any(vote not in options for vote in votes):
        raise ValueError("every vote must be in the option alphabet")
    counts = Counter(votes)
    n = len(votes)
    k = len(options)
    shares = {option: counts.get(option, 0) / n for option in options}

    def aligned(target: str) -> float:
        return (k * shares[target] - 1) / (k - 1)

    entropy = -sum(p * math.log(p) for p in shares.values() if p > 0) / math.log(k)
    return {
        "occupation_counts": {option: counts.get(option, 0) for option in options},
        "population_shares": shares,
        "p_truth": shares[correct_answer],
        "p_ctrl": None if controller_target is None else shares[controller_target],
        "m_truth": aligned(correct_answer),
        "m_ctrl": None if controller_target is None else aligned(controller_target),
        "m_order": (k * max(shares.values()) - 1) / (k - 1),
        "H_vote": entropy,
    }


def behavioral_transition_metrics(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    focal_opinion_before: str,
    focal_opinion_after: str,
    target: str,
    controller_action: str | None,
    sensor_count_vector: Mapping[str, int] | None = None,
    sensor_sample_size: int | None = None,
) -> dict[str, Any]:
    """Derive one event's response and sensor diagnostics from pre/post state.

    ``target`` is the scientific target used for target-adoption and ``m_ctrl``
    diagnostics.  It may exist in an uncontrolled comparator even though the
    controller-only fields deliberately remain missing there.
    """

    controlled = controller_action is not None
    population_target_share = float(before["population_shares"][target])
    sensor_target_share: float | None = None
    if controlled and sensor_sample_size:
        sensor_target_share = float((sensor_count_vector or {}).get(target, 0)) / int(
            sensor_sample_size
        )
    sensor_error = (
        None
        if sensor_target_share is None
        else sensor_target_share - population_target_share
    )
    return {
        "delta_m_ctrl": float(after["m_ctrl"] - before["m_ctrl"]),
        "delta_m_truth": float(after["m_truth"] - before["m_truth"]),
        "delta_m_order": float(after["m_order"] - before["m_order"]),
        "delta_H_vote": float(after["H_vote"] - before["H_vote"]),
        "focal_changed": int(focal_opinion_after != focal_opinion_before),
        "focal_adopted_target": int(
            focal_opinion_before != target and focal_opinion_after == target
        ),
        "focal_left_target": int(
            focal_opinion_before == target and focal_opinion_after != target
        ),
        "u_advocate": None if not controlled else int(controller_action == "ADVOCATE_Z"),
        "sensor_target_share": sensor_target_share,
        "population_target_share": (
            population_target_share if controlled else None
        ),
        "sensor_target_error": sensor_error,
        "sensor_target_abs_error": None if sensor_error is None else abs(sensor_error),
    }


def _summary(view: RoundView) -> Mapping[str, Any]:
    return view.recent_history[-1] if view.recent_history else {}


class _SummaryMetric(StreamingMetric):
    def __init__(self, source: str, *, name: str | None = None) -> None:
        super().__init__(name or source, scope="population")
        self.source = source

    def compute_round(self, view: RoundView) -> Mapping[MetricKey, Any]:
        return {None: _summary(view).get(self.source)}


def to_round_view(state: ImitationGameState) -> RoundView:
    return RoundView(
        agent_values={agent.agent_id: agent.committed_action for agent in state.agents},
        options=state.possible_answers,
        recent_history=state.evaluator_history,
    )


METRICS = [
    ActionSharePerOption(),
    AgentCurrentValue(name="agent_current_action"),
    DominantValueShare(name="dominant_action_share"),
    _SummaryMetric("m_truth"),
    _SummaryMetric("m_ctrl"),
    _SummaryMetric("m_order"),
    _SummaryMetric("H_vote", name="normalized_vote_entropy"),
    UnsharedDisclosureRate(),
    DisclosureReach(),
]
