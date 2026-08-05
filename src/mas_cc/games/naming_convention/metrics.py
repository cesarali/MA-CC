"""Naming-convention metrics: a thin adapter onto the shared generic library.

`committed_action` is the per-agent value this game contributes to a
`RoundView`; everything else is picked off the shelf from `mas_cc.metrics`,
including the three rolling-window metrics that used to be defined here. They
moved to `mas_cc.metrics.rolling` once a second game populated
`RoundView.recent_history` in the same shape, and are re-exported below so
existing imports of them from this module keep working.

Two views of the same question live here, and they are deliberately different:

- `population_action_share_per_option` is a *standing* statistic - where the
  population currently stands, one value per option, summing to 1.
- `rolling_action_share_per_option` is a *flow* statistic - what was actually
  played in the recent window, which moves first when a convention starts to
  tip and is therefore the leading indicator of the other.
"""

from __future__ import annotations

from mas_cc.metrics import (
    ActionSharePerOption,
    AgentCurrentValue,
    ConsensusFlipBySuccessRate,
    DominantValueShare,
    FirstConsensusTime,
    RollingActionSharePerOption,
    RollingCoordinationRate,
    RoundView,
)

from .records import ConventionGameState

__all__ = [
    # Re-exported for the modules that imported them from here before they
    # became cross-game metrics; they are defined in `mas_cc.metrics.rolling`.
    "ConsensusFlipBySuccessRate",
    "METRICS",
    "RollingActionSharePerOption",
    "RollingCoordinationRate",
    "build_metrics",
    "to_round_view",
]


def to_round_view(state: ConventionGameState) -> RoundView:
    return RoundView(
        agent_values={agent.agent_id: agent.committed_action for agent in state.agents},
        options=tuple(state.data.get("action_pool", ())),
        recent_history=state.evaluator_history,
    )


def build_metrics(population_size: int = 4) -> list:
    """The metric set recorded for every naming-convention episode.

    No action list is needed: the option set comes from the game state through
    `RoundView.options`, so the same metric objects work for a Q/M run and a
    ten-word run without being rebuilt per config.
    """

    return [
        ActionSharePerOption(),
        AgentCurrentValue(name="agent_current_action"),
        DominantValueShare(name="dominant_action_share"),
        FirstConsensusTime(threshold=0.95),
        RollingCoordinationRate(window=population_size),
        RollingActionSharePerOption(window=population_size),
        # The paper's 3N rolling-success criterion, and the word that won it.
        ConsensusFlipBySuccessRate(window=3 * population_size, report="interaction"),
        ConsensusFlipBySuccessRate(window=3 * population_size, report="action"),
    ]


METRICS = build_metrics()
