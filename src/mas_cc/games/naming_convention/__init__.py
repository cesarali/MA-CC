"""Ashery–Aiello–Baronchelli repeated naming-convention game."""

from .game import NamingConventionGame, NamingConventionGameSpec
from .parsing import parse_convention_response
from .prompts import (
    DescriptionBlock,
    NamingConventionFullPrompt,
    PresentedActionsBlock,
    RulesBlock,
    VisibleMemoryBlock,
    VisibleScoreBlock,
    bind_naming_convention_prompt,
    naming_convention_prompt,
)
from .records import (
    ConventionAgentState,
    ConventionDecisionOutcome,
    ConventionDecisionRequest,
    ConventionGameResult,
    ConventionGameState,
    ConventionInteractionRecord,
    ConventionTransition,
    InvalidConventionResponse,
    ParsedConventionResponse,
    PrivateMemoryEntry,
)
from .runtime import run_naming_convention_game, run_naming_convention_game_sync

__all__ = [
    "ConventionAgentState",
    "ConventionDecisionOutcome",
    "ConventionDecisionRequest",
    "ConventionGameResult",
    "ConventionGameState",
    "ConventionInteractionRecord",
    "ConventionTransition",
    "DescriptionBlock",
    "InvalidConventionResponse",
    "NamingConventionGame",
    "NamingConventionGameSpec",
    "NamingConventionFullPrompt",
    "ParsedConventionResponse",
    "PrivateMemoryEntry",
    "PresentedActionsBlock",
    "RulesBlock",
    "VisibleMemoryBlock",
    "VisibleScoreBlock",
    "bind_naming_convention_prompt",
    "naming_convention_prompt",
    "parse_convention_response",
    "run_naming_convention_game",
    "run_naming_convention_game_sync",
]
