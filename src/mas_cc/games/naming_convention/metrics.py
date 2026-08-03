"""Naming-convention metrics: a thin adapter onto the shared generic library.

`committed_action` is the per-agent value this game contributes to a
`RoundView`; everything else is picked off the shelf from `mas_cc.metrics`.
"""

from __future__ import annotations

from mas_cc.metrics import (
    AgentCurrentValue,
    DominantValueShare,
    FirstConsensusTime,
    RoundView,
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
    return RoundView(agent_values={agent.agent_id: _current_action(agent) for agent in state.agents})


def build_metrics(actions: tuple[str, ...] = ("Q", "M")) -> list:
    return [
        *(ValueShare(action, name=f"population_action_share_{action.lower()}") for action in actions),
        AgentCurrentValue(name="agent_current_action"),
        DominantValueShare(name="dominant_action_share"),
        FirstConsensusTime(threshold=0.95, name="first_consensus_time"),
    ]


METRICS = build_metrics()
