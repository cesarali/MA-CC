"""Streaming metrics for relational reasoning trajectories.

Two families sit side by side here, and keeping them apart is the point:

* **vote** observables (``m_truth``, ``m_ctrl``, ``m_order``, vote entropy) come
  from the shared HiddenBench implementation - they only need a vote vector and
  an option alphabet, so reimplementing them would just be a second thing to
  keep in step;
* **knowledge** observables (supporting-fact coverage, full-proof share, fact
  exposure counts) are new, and are computed from ``K_i`` alone.

Version 1 deliberately stops at counting.  No mutual information, no transfer
entropy, no estimator of any kind: the job here is to log enough exact state
that those can be computed later from the trajectory without rerunning
anything.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mas_cc.metrics import ActionSharePerOption, AgentCurrentValue, DominantValueShare, RoundView
from mas_cc.metrics.base import MetricKey, StreamingMetric

from .state import RelationalGameState


def supporting_fact_coverage(
    known_fact_ids: Sequence[str], supporting_fact_ids: Sequence[str]
) -> float:
    """``|K_i n S| / |S|`` - how much of the required proof this agent holds."""

    supporting = set(supporting_fact_ids)
    if not supporting:
        return 1.0
    return len(set(known_fact_ids) & supporting) / len(supporting)


def knowledge_observables(
    agents: Sequence[Any], supporting_fact_ids: Sequence[str]
) -> dict[str, Any]:
    """Population-level knowledge state, from ``K_1..K_N`` alone.

    ``full_proof_agent_share`` is the fraction of agents that could, in
    principle, derive the answer alone.  On a ``no_single_agent_solution`` task
    it starts at exactly zero, so any positive value is a direct measurement of
    how much evidence language has actually moved.
    """

    supporting = set(supporting_fact_ids)
    coverages = [
        supporting_fact_coverage(
            tuple(str(item) for item in agent.attributes.get("known_fact_ids", ())),
            supporting_fact_ids,
        )
        for agent in agents
    ]
    total_known = sum(
        len(tuple(agent.attributes.get("known_fact_ids", ()))) for agent in agents
    )
    return {
        "mean_supporting_fact_coverage": (
            sum(coverages) / len(coverages) if coverages else 0.0
        ),
        "full_proof_agent_share": (
            sum(1 for value in coverages if value >= 1.0) / len(coverages)
            if coverages
            else 0.0
        ),
        "supporting_fact_reach": [
            sum(
                1
                for agent in agents
                if fact_id in set(agent.attributes.get("known_fact_ids", ()))
            )
            for fact_id in sorted(supporting)
        ],
        "mean_known_fact_count": total_known / len(agents) if agents else 0.0,
    }


def knowledge_strata(
    agents: Sequence[Any],
    supporting_fact_ids: Sequence[str],
    correct_answer: str,
    votes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Population split by *how much of the proof* each agent holds.

    For a task with ``|S|`` supporting facts this returns, for every
    ``k = 0..|S|``:

    ``knowledge_share_k{k}``
        the fraction of the population currently knowing exactly ``k`` of them;

    ``truth_share_k{k}``
        the fraction **of that stratum** currently voting the correct answer,
        or ``None`` when the stratum is empty.

    The conditional form is the point.  A population-wide ``p_truth`` cannot
    distinguish "more agents hold the proof" from "agents who hold the proof
    use it", and the r-scan is precisely a sweep over the first while asking
    about the second.  ``None`` rather than ``0.0`` for an empty stratum keeps
    "nobody is here" separable from "everybody here is wrong".
    """

    supporting = set(supporting_fact_ids)
    depth = len(supporting)
    counts = [0] * (depth + 1)
    correct = [0] * (depth + 1)
    for agent in agents:
        known = set(str(item) for item in agent.attributes.get("known_fact_ids", ()))
        k = len(known & supporting)
        counts[k] += 1
        vote = (
            agent.attributes.get("committed_action")
            if votes is None
            else votes.get(str(agent.agent_id))
        )
        if vote is not None and str(vote) == correct_answer:
            correct[k] += 1
    total = len(agents)
    result: dict[str, Any] = {}
    for k in range(depth + 1):
        result[f"knowledge_share_k{k}"] = counts[k] / total if total else 0.0
        result[f"truth_share_k{k}"] = (
            correct[k] / counts[k] if counts[k] else None
        )
    result["knowledge_stratum_counts"] = counts
    result["truth_counts_by_stratum"] = correct
    return result


def _summary(view: RoundView) -> Mapping[str, Any]:
    return view.recent_history[-1] if view.recent_history else {}


class _SummaryMetric(StreamingMetric):
    """One scalar read off the latest persisted event row."""

    def __init__(self, source: str, *, name: str | None = None) -> None:
        super().__init__(name or source, scope="population")
        self.source = source

    def compute_round(self, view: RoundView) -> Mapping[MetricKey, Any]:
        return {None: _summary(view).get(self.source)}


def to_round_view(state: RelationalGameState) -> RoundView:
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
    _SummaryMetric("delta_m_truth"),
    _SummaryMetric("delta_m_ctrl"),
    _SummaryMetric("delta_m_order"),
    _SummaryMetric("focal_changed"),
    _SummaryMetric("focal_adopted_target"),
    _SummaryMetric("truth_vote_share"),
    # The epistemic side of the same trajectory (§15).
    _SummaryMetric("mean_supporting_fact_coverage"),
    _SummaryMetric("full_proof_agent_share"),
    _SummaryMetric("peer_fact_exposures"),
    _SummaryMetric("controller_fact_exposures"),
    _SummaryMetric("new_peer_facts"),
    _SummaryMetric("new_controller_facts"),
]


__all__ = [
    "METRICS",
    "knowledge_observables",
    "knowledge_strata",
    "supporting_fact_coverage",
    "to_round_view",
]
