"""Cross-game metrics written once against a generic per-round view.

Most of the games in scope reduce to the same shape: each round, each agent
holds a current value (a discrete choice for a voting/convention game, a
number for a summing game). Metrics here are written against that shape
(``RoundView``) so a new game only needs a small adapter function, not new
metric classes, to reuse population-share, consensus, and error metrics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

from mas_cc.core import AgentId
from mas_cc.metrics.base import FinalMetric, MetricKey, StreamingMetric


@dataclass(frozen=True, slots=True)
class RoundView:
    """One round's per-agent state, in a game-neutral shape.

    ``options`` is the game-declared choice set (empty for games that aren't
    a choice family). Metrics use it so every option gets a row every round,
    including options nobody has picked yet - otherwise the curves would start
    at ragged rounds and a share of 0 would be indistinguishable from a
    missing measurement.

    ``recent_history`` is optional and empty by default: a game whose
    outcome is a discrete pairwise event (this round's participants, what
    they each played, whether it counted as a success) may populate it with
    one dict per past round/interaction, oldest first, so a rolling-window
    metric can slice the tail it needs without every game having to define
    its own equivalent. A game with no such per-round event concept just
    leaves it empty; nothing about ``agent_values``/``agent_targets`` depends
    on it.
    """

    agent_values: Mapping[AgentId, Any]
    agent_targets: Mapping[AgentId, Any] | None = None
    options: tuple[str, ...] = ()
    recent_history: tuple[Mapping[str, Any], ...] = ()


class ActionSharePerOption(StreamingMetric):
    """Population composition: what fraction of the population stands on each option.

    For every option, the count of agents whose *most recent* choice is that
    option, divided by the number of agents that have chosen at all. Agents
    that have not yet played are excluded from both sides, so the values sum
    to 1 from the first round onward rather than creeping up as the population
    warms up.

    This is a *standing* quantity, not a flow: an agent keeps contributing its
    last choice on every round it isn't selected. That is what makes the curve
    a readable population statistic in a game where only two agents act per
    round - counting just the current round's choices would give a
    0/0.5/1 sawtooth carrying almost no information.
    """

    requires_game_family = "choice"

    def __init__(self, *, name: str = "population_action_share_per_option") -> None:
        super().__init__(name, scope="option")

    def compute_round(self, view: RoundView) -> Mapping[MetricKey, Any]:
        chosen = Counter(value for value in view.agent_values.values() if value is not None)
        options = view.options or tuple(sorted(chosen))
        total = sum(chosen.values())
        if not total:
            return {option: 0.0 for option in options}
        return {option: chosen[option] / total for option in options}


class AgentCurrentValue(StreamingMetric):
    """Per-agent passthrough of its current value this round."""

    def __init__(self, *, name: str = "agent_current_value") -> None:
        super().__init__(name, scope="agent")

    def compute_round(self, view: RoundView) -> Mapping[AgentId | None, Any]:
        return dict(view.agent_values)


class DominantValueShare(StreamingMetric):
    """Population share of the single most common value this round."""

    def __init__(self, *, name: str = "dominant_value_share") -> None:
        super().__init__(name, scope="population")

    def compute_round(self, view: RoundView) -> Mapping[AgentId | None, Any]:
        known = [v for v in view.agent_values.values() if v is not None]
        if not known:
            return {None: 0.0}
        _, count = Counter(known).most_common(1)[0]
        return {None: count / len(known)}


class FirstConsensusTime(FinalMetric):
    """First round index at which the dominant value's share reaches ``threshold``.

    Consensus *by standing action share*: it asks whether enough agents are
    currently holding the same value. That is a different question from "are
    recent interactions succeeding", which is what the Ashery spec's §7
    criterion measures (see
    ``games/naming_convention/metrics.py::ConsensusFlipBySuccessRate``). The two
    normally agree eventually but not on the same round, so both carry the
    quantity they measure in their metric name.

    Returns ``None`` if consensus was never reached.
    """

    def __init__(
        self, threshold: float = 0.95, *, name: str = "first_consensus_time_by_action_share"
    ) -> None:
        super().__init__(name, scope="population")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold

    def compute_final(self, views: tuple[RoundView, ...]) -> Any:
        for round_index, view in enumerate(views, start=1):
            known = [v for v in view.agent_values.values() if v is not None]
            if not known:
                continue
            _, count = Counter(known).most_common(1)[0]
            if count / len(known) >= self.threshold:
                return round_index
        return None


class AgentAbsoluteError(StreamingMetric):
    """Per-agent |value - target| for numeric games with a per-agent target."""

    def __init__(self, *, name: str = "agent_absolute_error") -> None:
        super().__init__(name, scope="agent")

    def compute_round(self, view: RoundView) -> Mapping[AgentId | None, Any]:
        if view.agent_targets is None:
            raise ValueError("AgentAbsoluteError requires RoundView.agent_targets")
        return {
            agent_id: abs(value - view.agent_targets[agent_id])
            for agent_id, value in view.agent_values.items()
            if value is not None and view.agent_targets.get(agent_id) is not None
        }


class MeanAbsoluteError(StreamingMetric):
    """Population mean of |value - target| for numeric games."""

    def __init__(self, *, name: str = "mean_absolute_error") -> None:
        super().__init__(name, scope="population")

    def compute_round(self, view: RoundView) -> Mapping[AgentId | None, Any]:
        if view.agent_targets is None:
            raise ValueError("MeanAbsoluteError requires RoundView.agent_targets")
        errors = [
            abs(value - view.agent_targets[agent_id])
            for agent_id, value in view.agent_values.items()
            if value is not None and view.agent_targets.get(agent_id) is not None
        ]
        return {None: 0.0 if not errors else sum(errors) / len(errors)}
