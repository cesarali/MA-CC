"""Concrete prompt values owned by the toy coordination game."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mas_cc.llm_runtime.messages import MessageRole
from mas_cc.llm_runtime.validation import ValidationIssue
from mas_cc.llm_runtime.prompts import UNBOUND, FullPrompt, PromptBlock, ResponseContract, Unbound
from mas_cc.llm_runtime.prompts._values import thaw


def _label(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _mapping_lines(values: Mapping[str, Any]) -> str:
    if not values:
        return "- None provided."
    return "\n".join(
        f"- {_label(str(key))}: {json.dumps(thaw(value), sort_keys=True, ensure_ascii=False)}"
        for key, value in values.items()
    )


@dataclass(frozen=True, slots=True)
class ToyDescriptionBlock(PromptBlock[str]):
    name: str = field(init=False, default="description")
    title: str = field(init=False, default="Task description")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: str | Unbound = (
        "Coordinate with the other selected agent by choosing the same action."
    )
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="fixed")

    def value_issues(self, value: str) -> tuple[ValidationIssue, ...]:
        return () if isinstance(value, str) and value.strip() else (
            ValidationIssue("prompt.blocks.description.value", "must be non-empty text", value),
        )

    def render(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class ToyRulesBlock(PromptBlock[tuple[str, ...]]):
    name: str = field(init=False, default="rules")
    title: str = field(init=False, default="Game rules")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: tuple[str, ...] | Unbound = UNBOUND
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="dynamic")

    def value_issues(self, value: tuple[str, ...]) -> tuple[ValidationIssue, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            return (ValidationIssue("prompt.blocks.rules.value", "must be a sequence of strings", value),)
        if any(not isinstance(item, str) or not item.strip() for item in value):
            return (ValidationIssue("prompt.blocks.rules.value", "must contain non-empty strings", value),)
        return ()

    def render(self) -> str:
        return "\n".join(f"{index}. {rule}" for index, rule in enumerate(self.value, 1))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ToyMappingBlock(PromptBlock[Mapping[str, Any]]):
    def value_issues(self, value: Mapping[str, Any]) -> tuple[ValidationIssue, ...]:
        return () if isinstance(value, Mapping) else (
            ValidationIssue(f"prompt.blocks.{self.name}.value", "must be a mapping", value),
        )

    def render(self) -> str:
        return _mapping_lines(self.value)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ToyMemoryBlock(PromptBlock[tuple[Mapping[str, Any], ...]]):
    name: str = field(init=False, default="visible_memory")
    title: str = field(init=False, default="Recent memory")
    role: MessageRole = field(init=False, default=MessageRole.USER)
    value: tuple[Mapping[str, Any], ...] | Unbound = UNBOUND
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="dynamic")

    def value_issues(self, value: tuple[Mapping[str, Any], ...]) -> tuple[ValidationIssue, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            return (ValidationIssue("prompt.blocks.visible_memory.value", "must be a sequence", value),)
        if any(not isinstance(item, Mapping) for item in value):
            return (ValidationIssue("prompt.blocks.visible_memory.value", "items must be mappings", value),)
        return ()

    def render(self) -> str:
        values = self.value  # type: ignore[assignment]
        if not values:
            return "No previous interactions are available."
        return "\n".join(
            f"- Interaction {index}: {json.dumps(thaw(entry), sort_keys=True, ensure_ascii=False)}"
            for index, entry in enumerate(values, 1)
        )


class ToyCoordinationFullPrompt(FullPrompt):
    def concrete_prompt_type(self) -> str:
        return "toy_coordination_decision"


def toy_coordination_prompt(*, horizon: int = 1) -> ToyCoordinationFullPrompt:
    return ToyCoordinationFullPrompt(
        family="toy_coordination_decision",
        version=1,
        blocks=(
            ToyDescriptionBlock(),
            ToyRulesBlock(),
            ToyMappingBlock(
                "private_state", "Private information", MessageRole.USER, UNBOUND,
                sensitive=True,
            ),
            ToyMemoryBlock(),
            ToyMappingBlock(
                "current_interaction", "Current interaction", MessageRole.USER, UNBOUND
            ),
        ),
        response_contract=ResponseContract(
            "choice_only",
            ("A", "B"),
            {"decision_instruction": "Choose your action for this interaction now."},
        ),
        message_mode="per_block",
    )


def bind_toy_prompt(
    *,
    horizon: int,
    agent_id: str,
    score: float,
    memory: tuple[Mapping[str, Any], ...],
    interaction: Mapping[str, Any],
) -> ToyCoordinationFullPrompt:
    rules = (
        "Choose exactly one action: A or B.",
        "Both selected agents decide without seeing the other's current action.",
        "Each selected agent receives 1 point if the two actions match and 0 otherwise.",
        f"The game ends after {horizon} interactions.",
    )
    return toy_coordination_prompt(horizon=horizon).bind(
        rules=rules,
        private_state={
            "agent_id": agent_id,
            "score": score,
            "available_actions": ["A", "B"],
        },
        visible_memory=memory,
        current_interaction=interaction,
    )
