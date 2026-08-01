"""Prompt adaptation of the social-conventions supplementary example."""

from __future__ import annotations

from mas_cc.core import MessageRole

from ..blocks import PromptBlock
from ..context import PromptContext, _thaw
from ..contracts import ResponseContract
from ..registry import PromptDefinition
from ..versions import PromptVersion


def _context(context: PromptContext, _: ResponseContract) -> str:
    return context.task_description


def _rules(context: PromptContext, _: ResponseContract) -> str:
    return "\n".join(
        f"{index}. {rule}" for index, rule in enumerate(context.game_rules, start=1)
    )


def _memory(context: PromptContext, _: ResponseContract) -> str:
    player = str(context.private_state.get("player", "Player 1"))
    lines = [
        "The objective of each Player is to maximize their own accumulated point "
        "tally, conditional on the behavior of the other player.",
        "This is the history of choices in past rounds:",
    ]
    if not context.recent_memory:
        lines.append("No past rounds are available in memory.")
    for entry in context.recent_memory:
        values = _thaw(entry)
        lines.append(
            "{"
            f"'round': {values.get('round')}, "
            f"'{player}': '{values.get('own_action')}', "
            f"'Other Player': '{values.get('other_action')}', "
            f"'payoff': {values.get('payoff')}"
            "}"
        )
    return "\n".join(lines)


def _round_state(context: PromptContext, _: ResponseContract) -> str:
    player = str(context.private_state.get("player", "Player 1"))
    round_number = context.current_interaction.get("round")
    score = context.private_state.get("score")
    return (
        f"It is now round {round_number}. The current score of {player} is {score}. "
        f"Answer saying which value {player} should pick. Please think step by step "
        "before making a decision. Remember, examining history explicitly is important."
    )


def _output(_: PromptContext, contract: ResponseContract) -> str:
    return contract.instruction()


def _decision(context: PromptContext, _: ResponseContract) -> str:
    return context.decision_instruction


def prompt_definition() -> PromptDefinition:
    blocks = (
        PromptBlock("partnership_context", "Partnership context", MessageRole.SYSTEM, _context),
        PromptBlock("payoff_rules", "Payoff rules", MessageRole.SYSTEM, _rules),
        PromptBlock("bounded_memory", "Bounded interaction memory", MessageRole.SYSTEM, _memory),
        PromptBlock("round_state", "Current round and score", MessageRole.SYSTEM, _round_state),
        PromptBlock("output_contract", "Answer-first output contract", MessageRole.SYSTEM, _output),
        PromptBlock("decision_request", "Decision request", MessageRole.USER, _decision),
    )
    return PromptDefinition(
        prompt_version=PromptVersion("social_conventions_paper", 1),
        blocks=blocks,
        required_blocks=tuple(block.name for block in blocks),
    )
