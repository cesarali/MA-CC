"""Trailing-window metrics over a game's per-round interaction records.

These were written inside ``games/naming_convention/metrics.py``, where their
docstring said they were "specific to this game's pairwise-interaction shape
[...] not a generic cross-game concept yet". A second game now populates
``RoundView.recent_history`` in the same shape, so *yet* has arrived and they
live here - unchanged in behaviour, and still re-exported from the naming
convention module so existing imports keep working.

They read ``RoundView.recent_history``: one mapping per past round, oldest
first, carrying at least ``actions`` and ``success``. Nothing here assumes two
participants per interaction; ``InteractionOutcome`` is defined for any number,
so a game where the whole population acts each round works the same way.

Relation to the binned tables in ``interactions.py``: same per-interaction
indicators, different window. These slide a trailing window and emit once per
round; those bin non-overlapping population rounds and emit once per bin. The
two must not be conflated - see the Ashery spec's §7.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from mas_cc.metrics.base import FinalMetric, MetricKey, StreamingMetric
from mas_cc.metrics.generic import RoundView
from mas_cc.metrics.interactions import InteractionOutcome, consensus_flip


class RollingCoordinationRate(StreamingMetric):
    """Success rate over the most recent ``window`` interactions (not the full history).

    "Success" is whatever the game recorded as such; for a pairwise convention
    game that is the two agents matching, for a whole-population game it is
    unanimity. Either way this is the *flow* of coordination, so it moves as
    soon as behaviour changes rather than after the cumulative average catches
    up.
    """

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

    Denominator is played *actions*, not interactions, so this counts what the
    population is currently doing rather than where it currently stands - the
    flow counterpart to `ActionSharePerOption`.
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
    the most recent ``window`` interactions succeeded - with ``window`` = 3N,
    the paper's rule. Deliberately *not* the non-overlapping binning used to
    draw a trajectory: same per-interaction success indicator, different window.

    Success rate alone says the population agreed but not on what, so the
    winning word is read from the same window and reported alongside. One
    instance emits one of the two, selected by ``report``, because a final
    metric writes a single value per name.

    Returns ``None`` when the criterion never holds. That is a real answer, not
    a missing one: a population that never coordinates has no flip time, and a
    metric that invented one would be worse than a blank.
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


__all__ = [
    "ConsensusFlipBySuccessRate",
    "RollingActionSharePerOption",
    "RollingCoordinationRate",
]
