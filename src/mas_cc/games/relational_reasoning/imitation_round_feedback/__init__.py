"""Relational reasoning imitation with one controller decision per round."""

from .controller import (
    MESSAGE_MODES,
    RECOMMENDATION_ONLY,
    RECOMMENDATION_PLUS_FACT,
    SCHEDULE_ALWAYS,
    SCHEDULE_NEVER,
    SCHEDULE_SOFT,
    RelationalRoundBudgetedControl,
    create_relational_round_budgeted_control,
)
from .game import RelationalImitationRoundFeedbackGame
from .metrics import knowledge_observables, supporting_fact_coverage
from .prompts import (
    SOCIAL_ENVIRONMENT,
    agent_label,
    build_relational_ballot_prompt,
    control_label,
    parse_relational_ballot,
    relational_public_ballot_prompt,
    render_control_reason,
    render_social_source,
)
from .runtime import (
    CONTROL_SOURCE_ID,
    RelationalGameResult,
    build_social_sources,
    run_relational_imitation_round_feedback_game,
    run_relational_imitation_round_feedback_game_sync,
    sample_controlled_positions,
)
from .state import (
    GAME_TYPE,
    RelationalAgentState,
    RelationalGameState,
    RelationalRoundRecord,
    RelationalRules,
    RelationalTransition,
)

__all__ = [
    "CONTROL_SOURCE_ID",
    "GAME_TYPE",
    "MESSAGE_MODES",
    "RECOMMENDATION_ONLY",
    "RECOMMENDATION_PLUS_FACT",
    "SCHEDULE_ALWAYS",
    "SCHEDULE_NEVER",
    "SCHEDULE_SOFT",
    "SOCIAL_ENVIRONMENT",
    "RelationalAgentState",
    "RelationalGameResult",
    "RelationalGameState",
    "RelationalImitationRoundFeedbackGame",
    "RelationalRoundBudgetedControl",
    "RelationalRoundRecord",
    "RelationalRules",
    "RelationalTransition",
    "agent_label",
    "build_relational_ballot_prompt",
    "build_social_sources",
    "control_label",
    "create_relational_round_budgeted_control",
    "knowledge_observables",
    "parse_relational_ballot",
    "relational_public_ballot_prompt",
    "render_control_reason",
    "render_social_source",
    "run_relational_imitation_round_feedback_game",
    "run_relational_imitation_round_feedback_game_sync",
    "sample_controlled_positions",
    "supporting_fact_coverage",
]
