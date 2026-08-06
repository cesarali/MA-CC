"""`hidden_bench_naming` metrics.

Same shelf and the same HiddenBench quantities as the vanilla game, so numbers
from the two are directly comparable, plus `disclosure_reach` - how far each
hidden fact actually diffused through the interaction graph, which only means
anything when messages are private.

The rolling metrics from `mas_cc.metrics.rolling` apply here and not to the
vanilla game: this game has one pairwise success event per round, which is the
shape they were written against.
"""

from __future__ import annotations

from mas_cc.metrics import (
    ActionSharePerOption,
    AgentCurrentValue,
    DominantValueShare,
    FirstConsensusTime,
    RollingActionSharePerOption,
    RollingCoordinationRate,
    RoundView,
)

from ..metrics_common import (
    AccuracyAverage,
    AccuracyMajority,
    DecoyShare,
    DisclosureReach,
    FinalAccuracy,
    FinalDisclosureRate,
    FirstCommitmentAccuracy,
    NamingImprovement,
    UnsharedDisclosureRate,
)
from ..records import HiddenBenchGameState

__all__ = ["METRICS", "build_metrics", "to_round_view"]


def to_round_view(state: HiddenBenchGameState) -> RoundView:
    return RoundView(
        agent_values={agent.agent_id: agent.committed_action for agent in state.agents},
        options=state.possible_answers,
        recent_history=state.evaluator_history,
    )


def build_metrics(population_size: int = 4) -> list:
    """Convention formation and truth-finding, reported side by side.

    `dominant_action_share` answers "has a convention formed"; `accuracy_average`
    answers "is it the right one". Under `payoff.mode: coordination` only the
    first is paid, so the two coming apart is a result rather than a bug - which
    is why both are always recorded regardless of the configured payoff mode.
    """

    return [
        ActionSharePerOption(),
        AgentCurrentValue(name="agent_current_action"),
        DominantValueShare(name="dominant_action_share"),
        AccuracyAverage(),
        AccuracyMajority(),
        DecoyShare(),
        UnsharedDisclosureRate(),
        DisclosureReach(),
        RollingCoordinationRate(window=population_size),
        RollingActionSharePerOption(window=population_size),
        FirstConsensusTime(threshold=0.95),
        # Deliberately *not* YPre/YPost: this game has no plenary vote, so those
        # estimators have nothing to measure here. See FirstCommitmentAccuracy.
        FirstCommitmentAccuracy(),
        FinalAccuracy(),
        NamingImprovement(),
        FinalDisclosureRate(),
    ]


METRICS = build_metrics()
