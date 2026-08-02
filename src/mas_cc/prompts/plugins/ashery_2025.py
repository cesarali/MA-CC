"""Versioned Ashery–Aiello–Baronchelli convention-game prompt."""

from __future__ import annotations

import json

from mas_cc.core import MessageRole

from ..compatibility import LegacyPromptBlock as PromptBlock
from ..context import PromptContext, _thaw
from ..contracts import ResponseContract
from ..registry import PromptDefinition
from ..versions import PromptVersion


def _fixed_game(context: PromptContext, _: ResponseContract) -> str:
    actions = list(context.private_state.get("presented_actions", ()))
    lines = [context.task_description]
    lines.extend(context.game_rules)
    lines.append(
        "The available values, in the order presented for this decision, are: "
        + json.dumps(actions, ensure_ascii=False)
        + "."
    )
    lines.append(
        "Your objective is to maximize Player 1's accumulated points conditional "
        "on Player 2's behavior."
    )
    return "\n".join(lines)


def _bounded_memory(context: PromptContext, _: ResponseContract) -> str:
    if not context.recent_memory:
        return "Player 1 has no past rounds available in memory."
    lines = ["Player 1's available history of choices in past rounds is:"]
    for local_round, raw in enumerate(context.recent_memory, start=1):
        entry = _thaw(raw)
        lines.append(
            json.dumps(
                {
                    "round": local_round,
                    "Player 1": entry["own_action"],
                    "Player 2": entry["partner_action"],
                    "payoff": entry["payoff"],
                },
                ensure_ascii=False,
                sort_keys=False,
            )
        )
    return "\n".join(lines)


def _local_state(context: PromptContext, _: ResponseContract) -> str:
    return (
        f"It is now local round {context.current_interaction['local_round']}. "
        f"Player 1's score over the visible memory window is "
        f"{context.private_state['visible_score']}."
    )


def _output(context: PromptContext, _: ResponseContract) -> str:
    actions = " or ".join(str(item) for item in context.private_state["presented_actions"])
    return (
        "Put the decision before the explanation. Return only an object in this "
        "answer-first form: "
        "{'value': '<ACTION>'; 'reason': '<YOUR REASON>'}. "
        f"The value must be exactly {actions}. Do not add text outside the object."
    )


def _decision(context: PromptContext, _: ResponseContract) -> str:
    return context.decision_instruction


def prompt_definition() -> PromptDefinition:
    blocks = (
        PromptBlock("fixed_game", "Fixed game", MessageRole.SYSTEM, _fixed_game),
        PromptBlock("bounded_memory", "Dynamic bounded memory", MessageRole.SYSTEM, _bounded_memory),
        PromptBlock("local_state", "Local round and visible score", MessageRole.SYSTEM, _local_state),
        PromptBlock("output_instruction", "Answer-first output instruction", MessageRole.SYSTEM, _output),
        PromptBlock("decision_request", "Decision request", MessageRole.USER, _decision),
    )
    return PromptDefinition(
        prompt_version=PromptVersion("ashery_2025_v1", 1),
        blocks=blocks,
        required_blocks=tuple(block.name for block in blocks),
    )
