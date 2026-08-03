"""Composable, cross-game scientific metrics (mas_cc Phase 8.0, trimmed design)."""

from .base import FinalMetric, Metric, StreamingMetric
from .generic import (
    AgentAbsoluteError,
    AgentCurrentValue,
    DominantValueShare,
    FirstConsensusTime,
    MeanAbsoluteError,
    RoundView,
    ValueShare,
)
from .plotting import plot_streaming_metrics

__all__ = [
    "AgentAbsoluteError",
    "AgentCurrentValue",
    "DominantValueShare",
    "FinalMetric",
    "FirstConsensusTime",
    "MeanAbsoluteError",
    "Metric",
    "RoundView",
    "StreamingMetric",
    "ValueShare",
    "plot_streaming_metrics",
]
