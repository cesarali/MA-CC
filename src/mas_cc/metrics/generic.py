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
from mas_cc.metrics.base import FinalMetric, StreamingMetric


@dataclass(frozen=True, slots=True)
class RoundView:
    """One round's per-agent state, in a game-neutral shape."""

    agent_values: Mapping[AgentId, Any]
    agent_targets: Mapping[AgentId, Any] | None = None


class ValueShare(StreamingMetric):
    """Population share of agents currently holding ``value``. Unset (None) agents are excluded."""

    def __init__(self, value: Any, *, name: str | None = None) -> None:
        super().__init__(name or f"value_share_{value}", scope="population")
        self.value = value

    def compute_round(self, view: RoundView) -> Mapping[AgentId | None, Any]:
        known = [v for v in view.agent_values.values() if v is not None]
        share = 0.0 if not known else sum(1 for v in known if v == self.value) / len(known)
        return {None: share}


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

    Returns ``None`` if consensus was never reached.
    """

    def __init__(self, threshold: float = 0.95, *, name: str = "first_consensus_time") -> None:
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
