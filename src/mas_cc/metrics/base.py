"""Minimal metric contracts: a value per round (streaming) or per episode (final).

Deliberately small.  A metric is either computed once per round from the
current game state, or once at episode end from the accumulated round
history.  There is no registry, no versioned extractor, and no separate
recording-plan object: a game just lists its metric instances in
``games/<game>/metrics.py`` and the recorder calls them directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from mas_cc.core import AgentId


class Metric(ABC):
    """Shared identity for a computed scientific quantity."""

    name: str
    scope: str  # "agent" or "population"

    def __init__(self, name: str, *, scope: str) -> None:
        if scope not in {"agent", "population"}:
            raise ValueError("Metric.scope must be 'agent' or 'population'")
        self.name = name
        self.scope = scope


class StreamingMetric(Metric, ABC):
    """Computed once per round from the current game state."""

    @abstractmethod
    def compute_round(self, view: Any) -> Mapping[AgentId | None, Any]:
        """Return one value per agent (scope='agent') or one value keyed by
        ``None`` (scope='population'). Values may be numeric or categorical;
        the sink writes them as-is."""


class FinalMetric(Metric, ABC):
    """Computed once at episode end from the accumulated round history."""

    @abstractmethod
    def compute_final(self, views: tuple[Any, ...]) -> Any: ...
