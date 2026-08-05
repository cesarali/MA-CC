"""The metric set every synthetic game declares.

All three games have the same recorded shape - a whole-population round with a
per-agent committed action and a unanimity indicator - so they share one
adapter and one metric list rather than three near-copies. Each game's
``metrics.py`` is then a two-line re-export, which is what keeps
`game_metrics` discovery working per package without duplicating the wiring.

Two notes carried over from the Bernoulli game and true of all three:

**A round is a whole-population interaction, not a pair.** One round
contributes N actions rather than 2, so the rolling windows are measured in
rounds. The naming-convention convention of "3N pair interactions" would be 3
rounds here.

**"Success" means unanimity.** These agents have no partner to match and no
payoff, so the per-round binary event is whether the whole population said the
same word.
"""

from __future__ import annotations

from mas_cc.games.protocols import GameState
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

ACTION_METRIC_NAME = "agent_current_action"
"""The metric whose rows carry the recorded action series; see `analysis.py`."""

DEFAULT_WINDOW = 25
"""Rounds in the trailing window, for whole-population rounds.

Long enough that the rate is readable rather than a sawtooth, short enough to
respond within a few hundred rounds. Not derived from the population size,
because unlike the pairwise games a round here already involves everyone.
"""


def to_round_view(state: GameState) -> RoundView:
    return RoundView(
        agent_values={
            agent.agent_id: agent.attributes.get("committed_action") for agent in state.agents
        },
        options=tuple(state.data.get("action_pool", ())),
        recent_history=tuple(state.data.get("round_history", ())),
    )


def build_metrics(window: int = DEFAULT_WINDOW) -> list:
    """The metric set recorded for every synthetic episode."""

    return [
        # Standing: where the population currently is.
        AgentCurrentValue(name=ACTION_METRIC_NAME),
        ActionSharePerOption(),
        DominantValueShare(name="dominant_action_share"),
        # Flow: what was actually played in the recent window.
        RollingCoordinationRate(window=window),
        RollingActionSharePerOption(window=window),
        # Final: the two consensus criteria, which measure different things and
        # routinely disagree here - see the docs on why that is correct.
        FirstConsensusTime(threshold=0.95),
        ConsensusFlipBySuccessRate(window=window, report="interaction"),
        ConsensusFlipBySuccessRate(window=window, report="action"),
    ]
