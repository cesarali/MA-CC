"""Metrics recorded for a synthetic Bernoulli episode.

Nothing bespoke: the point of a synthetic game is that it goes through the same
machinery as a real one, so it declares off-the-shelf metrics against the same
`RoundView` adapter shape every other game uses.

`agent_current_action` matters more than it looks. It is what writes the
per-agent action series into `metrics/streaming.csv`, and that file is what the
mutual-information estimate is later computed from - including after the run has
been pulled back off the cluster. If the recorder writes it wrongly, the
estimate misses the closed form, which is the failure we want to catch.
"""

from __future__ import annotations

from mas_cc.games.protocols import GameState
from mas_cc.metrics import (
    ActionSharePerOption,
    AgentCurrentValue,
    DominantValueShare,
    RoundView,
)

ACTION_METRIC_NAME = "agent_current_action"
"""The metric whose rows carry the recorded action series; see `analysis.py`."""


def to_round_view(state: GameState) -> RoundView:
    return RoundView(
        agent_values={
            agent.agent_id: agent.attributes.get("committed_action") for agent in state.agents
        },
        options=tuple(state.data.get("action_pool", ())),
        recent_history=tuple(state.data.get("round_history", ())),
    )


METRICS = [
    AgentCurrentValue(name=ACTION_METRIC_NAME),
    ActionSharePerOption(),
    DominantValueShare(name="dominant_action_share"),
]
