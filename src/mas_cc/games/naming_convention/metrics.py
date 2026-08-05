"""Naming-convention metrics: a thin adapter onto the shared generic library.

`committed_action` is the per-agent value this game contributes to a
`RoundView`; everything else is picked off the shelf from `mas_cc.metrics`,
plus two rolling-window metrics defined here because they're specific to
this game's pairwise-interaction shape (`RoundView.recent_history`), not a
generic cross-game concept yet.
"""

from __future__ import annotations

from typing import Any, Mapping

from mas_cc.core import AgentId
from mas_cc.metrics import (
    AgentCurrentValue,
    DominantValueShare,
    FirstConsensusTime,
    RoundView,
    StreamingMetric,
    ValueShare,
)

from .records import ConventionGameState


def _current_action(agent) -> str | None:
    """An agent's most recently played action, or None if it hasn't played yet.

    `attributes["committed_action"]` is initialized to None and never updated
    by the game (see game.py); the private history is the actual record of
    what an agent last played.
    """

    history = agent.private_history
    return history[-1].own_action if history else None


def to_round_view(state: ConventionGameState) -> RoundView:
    return RoundView(
        agent_values={agent.agent_id: _current_action(agent) for agent in state.agents},
        recent_history=state.evaluator_history,
    )


class RollingCoordinationRate(StreamingMetric):
    """Success rate over the most recent ``window`` interactions (not the full history)."""

    def __init__(self, window: int, *, name: str = "rolling_coordination_rate") -> None:
        super().__init__(name, scope="population")
        if window < 1:
            raise ValueError("window must be positive")
        self.window = window

    def compute_round(self, view: RoundView) -> Mapping[AgentId | None, Any]:
        recent = view.recent_history[-self.window :]
        if not recent:
            return {None: 0.0}
        return {None: sum(1 for entry in recent if entry["success"]) / len(recent)}


class RollingActionShare(StreamingMetric):
    """Share of ``action`` among every action played in the most recent ``window`` interactions.

    Denominator is played *actions*, not interactions - two per interaction,
    matching the legacy `mas-cc game run` trajectory's `rolling_share_*` math.
    """

    def __init__(self, action: str, window: int, *, name: str | None = None) -> None:
        super().__init__(name or f"rolling_action_share_{action.lower()}", scope="population")
        if window < 1:
            raise ValueError("window must be positive")
        self.action = action
        self.window = window

    def compute_round(self, view: RoundView) -> Mapping[AgentId | None, Any]:
        recent = view.recent_history[-self.window :]
        played = [value for entry in recent for value in entry["actions"]]
        if not played:
            return {None: 0.0}
        return {None: played.count(self.action) / len(played)}


def build_metrics(actions: tuple[str, ...] = ("Q", "M"), population_size: int = 4) -> list:
    return [
        *(ValueShare(action, name=f"population_action_share_{action.lower()}") for action in actions),
        AgentCurrentValue(name="agent_current_action"),
        DominantValueShare(name="dominant_action_share"),
        FirstConsensusTime(threshold=0.95, name="first_consensus_time"),
        RollingCoordinationRate(window=population_size),
        *(RollingActionShare(action, window=population_size) for action in actions),
    ]


METRICS = build_metrics()
