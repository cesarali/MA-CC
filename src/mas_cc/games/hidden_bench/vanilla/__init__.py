"""The paper's own HiddenBench protocol (brief §5)."""

from .game import HiddenBenchVanillaGame
from .metrics import METRICS, build_metrics, to_round_view

__all__ = ["METRICS", "HiddenBenchVanillaGame", "build_metrics", "to_round_view"]
