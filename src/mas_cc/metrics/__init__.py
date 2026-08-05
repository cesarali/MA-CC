"""Composable, cross-game scientific metrics (mas_cc Phase 8.0, trimmed design)."""

from .base import SCOPES, FinalMetric, Metric, MetricKey, StreamingMetric
from .generic import (
    ActionSharePerOption,
    AgentAbsoluteError,
    AgentCurrentValue,
    DominantValueShare,
    FirstConsensusTime,
    MeanAbsoluteError,
    RoundView,
)
from .plotting import plot_streaming_metrics
from .rolling import (
    ConsensusFlipBySuccessRate,
    RollingActionSharePerOption,
    RollingCoordinationRate,
)

__all__ = [
    "SCOPES",
    "ActionSharePerOption",
    "AgentAbsoluteError",
    "AgentCurrentValue",
    "ConsensusFlipBySuccessRate",
    "DominantValueShare",
    "FinalMetric",
    "FirstConsensusTime",
    "MeanAbsoluteError",
    "Metric",
    "MetricKey",
    "RollingActionSharePerOption",
    "RollingCoordinationRate",
    "RoundView",
    "StreamingMetric",
    "plot_streaming_metrics",
]
