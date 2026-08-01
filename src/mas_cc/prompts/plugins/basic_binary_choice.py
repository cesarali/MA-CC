"""Version 1 of the inspectable basic binary-choice prompt family."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from mas_cc.core import MessageRole

from ..blocks import PromptBlock
from ..context import PromptContext, _thaw
from ..contracts import ResponseContract
from ..registry import PromptDefinition
from ..versions import PromptVersion


def _label(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _mapping_lines(values: Mapping[str, Any]) -> str:
    if not values:
        return "- None provided."
    return "\n".join(
        f"- {_label(str(key))}: {json.dumps(_thaw(value), sort_keys=True, ensure_ascii=False)}"
        for key, value in values.items()
    )


def _task(context: PromptContext, _: ResponseContract) -> str:
    return context.task_description


def _rules(context: PromptContext, _: ResponseContract) -> str:
    return "\n".join(f"{index}. {rule}" for index, rule in enumerate(context.game_rules, start=1))


def _private_state(context: PromptContext, _: ResponseContract) -> str:
    return _mapping_lines(context.private_state)


def _recent_memory(context: PromptContext, _: ResponseContract) -> str:
    if not context.recent_memory:
        return "No previous interactions are available."
    return "\n".join(
        f"- Interaction {index}: {json.dumps(_thaw(entry), sort_keys=True, ensure_ascii=False)}"
        for index, entry in enumerate(context.recent_memory, start=1)
    )


def _current_interaction(context: PromptContext, _: ResponseContract) -> str:
    return _mapping_lines(context.current_interaction)


def _decision(context: PromptContext, _: ResponseContract) -> str:
    return context.decision_instruction


def _output(_: PromptContext, contract: ResponseContract) -> str:
    return contract.instruction()


def prompt_definition() -> PromptDefinition:
    """Return a new immutable definition for ``basic_binary_choice@1``."""

    blocks = (
        PromptBlock("task_description", "Task description", MessageRole.SYSTEM, _task),
        PromptBlock("game_rules", "Game rules", MessageRole.SYSTEM, _rules),
        PromptBlock("private_state", "Private information", MessageRole.USER, _private_state),
        PromptBlock("recent_memory", "Recent memory", MessageRole.USER, _recent_memory),
        PromptBlock(
            "current_interaction",
            "Current interaction",
            MessageRole.USER,
            _current_interaction,
        ),
        PromptBlock(
            "decision_instruction", "Decision instruction", MessageRole.USER, _decision
        ),
        PromptBlock("output_contract", "Output contract", MessageRole.USER, _output),
    )
    return PromptDefinition(
        prompt_version=PromptVersion("basic_binary_choice", 1),
        blocks=blocks,
        required_blocks=("task_description", "decision_instruction", "output_contract"),
    )
