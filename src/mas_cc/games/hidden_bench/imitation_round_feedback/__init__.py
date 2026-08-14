"""Round-level budgeted feedback variant of HiddenBench imitation."""

from .analysis import analyze_hidden_bench_imitation_round_feedback
from .controller import RoundSoftTargetBudgetedControl
from .game import HiddenBenchImitationRoundFeedbackGame
from .prompts import (
    SOCIAL_ENVIRONMENT,
    build_public_ballot_update_prompt,
    hidden_bench_public_ballot_prompt,
    parse_public_ballot_update,
    render_control_reason,
    render_social_source,
)
from .runtime import (
    RoundFeedbackGameResult,
    build_social_sources,
    run_hidden_bench_imitation_round_feedback_game,
    run_hidden_bench_imitation_round_feedback_game_sync,
)
from .state import RoundFeedbackRecord, RoundFeedbackRules, get_public_reason

__all__ = [
    "SOCIAL_ENVIRONMENT",
    "HiddenBenchImitationRoundFeedbackGame",
    "RoundFeedbackGameResult",
    "RoundFeedbackRecord",
    "RoundFeedbackRules",
    "RoundSoftTargetBudgetedControl",
    "analyze_hidden_bench_imitation_round_feedback",
    "build_public_ballot_update_prompt",
    "build_social_sources",
    "get_public_reason",
    "hidden_bench_public_ballot_prompt",
    "parse_public_ballot_update",
    "render_control_reason",
    "render_social_source",
    "run_hidden_bench_imitation_round_feedback_game",
    "run_hidden_bench_imitation_round_feedback_game_sync",
]
