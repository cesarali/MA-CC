"""`hidden_bench_vanilla` metrics: the generic shelf plus the paper's quantities.

`to_round_view` is the whole adapter - each agent's current vote becomes its
`RoundView` value, the task's option set becomes `options`, and the game's own
round summaries become `recent_history`, which is where `correct_answer`, the
decoy and the phase reach the HiddenBench-specific metrics.
"""

from __future__ import annotations

from mas_cc.metrics import (
    ActionSharePerOption,
    AgentCurrentValue,
    DominantValueShare,
    FirstConsensusTime,
    RoundView,
)

from ..metrics_common import (
    AccuracyAverage,
    AccuracyMajority,
    DecoyShare,
    FinalDisclosureRate,
    Improvement,
    UnsharedDisclosureRate,
    YPost,
    YPre,
)
from ..records import HiddenBenchGameState

__all__ = ["METRICS", "build_metrics", "to_round_view"]


def to_round_view(state: HiddenBenchGameState) -> RoundView:
    return RoundView(
        agent_values={agent.agent_id: agent.committed_action for agent in state.agents},
        options=state.possible_answers,
        recent_history=state.evaluator_history,
    )


def build_metrics() -> list:
    """Every metric a vanilla episode records.

    The four shelf metrics need no adaptation because the action alphabet is the
    option set (`game_family: choice`). `first_consensus_time_by_action_share`
    is meaningful but blunt here - a vanilla episode has only two rounds that
    move any vote - and is kept for cross-game comparability with the naming
    variant rather than for its own sake.
    """

    return [
        ActionSharePerOption(),
        AgentCurrentValue(name="agent_current_action"),
        DominantValueShare(name="dominant_action_share"),
        AccuracyAverage(),
        AccuracyMajority(),
        DecoyShare(),
        UnsharedDisclosureRate(),
        FirstConsensusTime(threshold=0.95),
        YPre(),
        YPost(),
        Improvement(),
        FinalDisclosureRate(),
    ]


METRICS = build_metrics()
