"""HiddenBench imitation game: matched reasoning and classical opinion dynamics."""

from .game import HiddenBenchImitationGame
from .analysis import analyze_hidden_bench_imitation
from .runtime import (
    ImitationGameResult,
    ImitationInteractionRecord,
    run_hidden_bench_imitation_game,
    run_hidden_bench_imitation_game_sync,
)

__all__ = [
    "HiddenBenchImitationGame",
    "analyze_hidden_bench_imitation",
    "ImitationGameResult",
    "ImitationInteractionRecord",
    "run_hidden_bench_imitation_game",
    "run_hidden_bench_imitation_game_sync",
]
