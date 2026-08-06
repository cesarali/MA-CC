"""Hidden Profile information under naming-game dyadic interaction (brief §6)."""

from .game import HiddenBenchNamingGame
from .metrics import METRICS, build_metrics, to_round_view

__all__ = ["METRICS", "HiddenBenchNamingGame", "build_metrics", "to_round_view"]
