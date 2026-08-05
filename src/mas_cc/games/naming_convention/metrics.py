"""Naming-convention metrics: a thin adapter onto the shared generic library.

`committed_action` is the per-agent value this game contributes to a
`RoundView`; everything else is picked off the shelf from `mas_cc.metrics`,
plus two rolling-window metrics defined here because they're specific to
this game's pairwise-interaction shape (`RoundView.recent_history`), not a
generic cross-game concept yet.

Two views of the same question live here, and they are deliberately different:

- `population_action_share_per_option` is a *standing* statistic - where the
  population currently stands, one value per option, summing to 1.
- `rolling_action_share_per_option` is a *flow* statistic - what was actually
  played in the recent window, which moves first when a convention starts to
  tip and is therefore the leading indicator of the other.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from mas_cc.metrics import (
    ActionSharePerOption,
    AgentCurrentValue,
    DominantValueShare,
    FinalMetric,
    FirstConsensusTime,
    MetricKey,
    RoundView,
    StreamingMetric,
)
from mas_cc.metrics.interactions import InteractionOutcome, consensus_flip

from .records import ConventionGameState


def to_round_view(state: ConventionGameState) -> RoundView:
    return RoundView(
        agent_values={agent.agent_id: agent.committed_action for agent in state.agents},
        options=tuple(state.data.get("action_pool", ())),
        recent_history=state.evaluator_history,
    )


class RollingCoordinationRate(StreamingMetric):
    """Success rate over the most recent ``window`` interactions (not the full history)."""

    def __init__(self, window: int, *, name: str = "rolling_coordination_rate") -> None:
        super().__init__(name, scope="population")
        if window < 1:
            raise ValueError("window must be positive")
        self.window = window

    def compute_round(self, view: RoundView) -> Mapping[MetricKey, Any]:
        recent = view.recent_history[-self.window :]
        if not recent:
            return {None: 0.0}
        return {None: sum(1 for entry in recent if entry["success"]) / len(recent)}


class RollingActionSharePerOption(StreamingMetric):
    """Share of each option among every action played in the most recent ``window`` interactions.

    Denominator is played *actions*, not interactions - two per interaction -
    so this counts what the population is currently doing rather than where it
    currently stands, which is the flow counterpart to
    `ActionSharePerOption`.
    """

    requires_game_family = "choice"

    def __init__(self, window: int, *, name: str = "rolling_action_share_per_option") -> None:
        super().__init__(name, scope="option")
        if window < 1:
            raise ValueError("window must be positive")
        self.window = window

    def compute_round(self, view: RoundView) -> Mapping[MetricKey, Any]:
        recent = view.recent_history[-self.window :]
        played = Counter(value for entry in recent for value in entry["actions"])
        options = view.options or tuple(sorted(played))
        total = sum(played.values())
        if not total:
            return {option: 0.0 for option in options}
        return {option: played[option] / total for option in options}


class ConsensusFlipBySuccessRate(FinalMetric):
    """The Ashery spec's §7 consensus criterion, over a rolling window of interactions.

    A flip is declared at the first interaction where at least ``threshold`` of
    the most recent ``window`` pair interactions succeeded - with ``window`` =
    3N, the paper's rule. Deliberately *not* the non-overlapping binning used to
    draw a trajectory: same per-interaction success indicator, different window.

    Success rate alone says the population agreed but not on what, so the
    winning word is read from the same window and reported alongside. One
    instance emits one of the two, selected by ``report``, because a final
    metric writes a single value per name.
    """

    def __init__(
        self,
        *,
        window: int,
        report: str,
        threshold: float = 0.95,
        exclude_committed_outputs: bool = False,
        name: str | None = None,
    ) -> None:
        if report not in {"interaction", "action"}:
            raise ValueError("report must be 'interaction' or 'action'")
        default = (
            "first_consensus_time_by_success_rate"
            if report == "interaction"
            else "consensus_action_by_success_rate"
        )
        super().__init__(name or default, scope="population")
        self.window = window
        self.report = report
        self.threshold = threshold
        self.exclude_committed_outputs = exclude_committed_outputs

    def compute_final(self, views: tuple[RoundView, ...]) -> Any:
        if not views:
            return None
        records = [
            InteractionOutcome.from_evaluator_entry(entry)
            for entry in views[-1].recent_history
        ]
        flip = consensus_flip(
            records,
            window=self.window,
            threshold=self.threshold,
            exclude_committed_outputs=self.exclude_committed_outputs,
        )
        if flip is None:
            return None
        interaction_index, winner = flip
        return interaction_index if self.report == "interaction" else winner


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
