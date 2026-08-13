"""Round-level budgeted feedback variant of HiddenBench imitation."""

from .analysis import analyze_hidden_bench_imitation_round_feedback
from .controller import RoundSoftTargetBudgetedControl
from .game import HiddenBenchImitationRoundFeedbackGame
from .runtime import (
    RoundFeedbackGameResult,
    run_hidden_bench_imitation_round_feedback_game,
    run_hidden_bench_imitation_round_feedback_game_sync,
)
from .state import RoundFeedbackRecord, RoundFeedbackRules

__all__ = [
    "HiddenBenchImitationRoundFeedbackGame",
    "RoundFeedbackGameResult",
    "RoundFeedbackRecord",
    "RoundFeedbackRules",
    "RoundSoftTargetBudgetedControl",
    "analyze_hidden_bench_imitation_round_feedback",
    "run_hidden_bench_imitation_round_feedback_game",
    "run_hidden_bench_imitation_round_feedback_game_sync",
]
