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


SCOPES = ("agent", "population", "option")
"""What the keys of a metric's per-round mapping mean.

``agent``       one value per agent, keyed by ``AgentId``.
``population``  a single value for the whole population, keyed by ``None``.
``option``      one value per choice option, keyed by the option label - a
                family of curves sharing one metric name, which is how a
                per-option share stays a *single* metric instead of degenerating
                into one metric per option.
"""

MetricKey = AgentId | str | None


class Metric(ABC):
    """Shared identity for a computed scientific quantity."""

    name: str
    scope: str
    requires_game_family: str | None = None
    """Game family this metric is only meaningful for (see ``GameSpec.game_family``).

    ``None`` means the metric applies to any game. Otherwise the wiring that
    attaches metrics to a game rejects the mismatch up front, rather than
    letting a share-of-options metric run against a game that has no options.
    """

    def __init__(self, name: str, *, scope: str) -> None:
        if scope not in SCOPES:
            raise ValueError(f"Metric.scope must be one of {SCOPES}, got {scope!r}")
        self.name = name
        self.scope = scope


class StreamingMetric(Metric, ABC):
    """Computed once per round from the current game state."""

    @abstractmethod
    def compute_round(self, view: Any) -> Mapping[MetricKey, Any]:
        """Return one value per key, where the key's meaning follows ``scope``:
        an ``AgentId`` for 'agent', ``None`` for 'population', an option label
        for 'option'. Values may be numeric or categorical; the sink writes
        them as-is."""


class FinalMetric(Metric, ABC):
    """Computed once at episode end from the accumulated round history."""

    @abstractmethod
    def compute_final(self, views: tuple[Any, ...]) -> Any: ...
