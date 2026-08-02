"""Paper-faithful concrete prompt values for the Naming Convention game."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mas_cc.core import Message, MessageRole, ValidationIssue, ValidationResult
from mas_cc.prompts import UNBOUND, FullPrompt, PromptBlock, ResponseContract, Unbound
from mas_cc.prompts._values import thaw


@dataclass(frozen=True, slots=True)
class DescriptionBlock(PromptBlock[str]):
    name: str = field(init=False, default="description")
    title: str = field(init=False, default="Game description")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: str | Unbound = (
        "Player 1 is playing a repeated two-player partnership game with Player 2. "
        "In each round both players choose simultaneously."
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
class RulesBlock(PromptBlock[tuple[str, ...]]):
    name: str = field(init=False, default="rules")
    title: str = field(init=False, default="Game rules")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: tuple[str, ...] | Unbound = (
        "If both players choose the same value, both receive +100 points.",
        "If the players choose different values, both receive -50 points.",
        "Player 1 cannot see Player 2's current choice before deciding.",
    )
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="fixed")

    def value_issues(self, value: tuple[str, ...]) -> tuple[ValidationIssue, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            return (ValidationIssue("prompt.blocks.rules.value", "must be a sequence", value),)
        if any(not isinstance(item, str) or not item.strip() for item in value):
            return (ValidationIssue("prompt.blocks.rules.value", "must contain non-empty text", value),)
        return ()

    def render(self) -> str:
        return "\n".join(self.value)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class PresentedActionsBlock(PromptBlock[tuple[str, ...]]):
    name: str = field(init=False, default="presented_actions")
    title: str = field(init=False, default="Presented actions")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: tuple[str, ...] | Unbound = UNBOUND
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="dynamic")

    def value_issues(self, value: tuple[str, ...]) -> tuple[ValidationIssue, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            return (ValidationIssue("prompt.blocks.presented_actions.value", "must be a sequence", value),)
        if len(value) < 2 or len(set(value)) != len(value) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            return (ValidationIssue("prompt.blocks.presented_actions.value", "must contain unique action labels", value),)
        return ()

    def render(self) -> str:
        actions = json.dumps(list(self.value), ensure_ascii=False)  # type: ignore[arg-type]
        return (
            "The available values, in the order presented for this decision, are: "
            f"{actions}.\n"
            "Your objective is to maximize Player 1's accumulated points conditional "
            "on Player 2's behavior."
        )


@dataclass(frozen=True, slots=True)
class VisibleMemoryBlock(PromptBlock[tuple[Mapping[str, Any], ...]]):
    name: str = field(init=False, default="visible_memory")
    title: str = field(init=False, default="Visible bounded memory")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: tuple[Mapping[str, Any], ...] | Unbound = UNBOUND
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="dynamic")
    sensitive: bool = field(init=False, default=True)

    def value_issues(self, value: tuple[Mapping[str, Any], ...]) -> tuple[ValidationIssue, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            return (ValidationIssue("prompt.blocks.visible_memory.value", "must be a sequence", value),)
        if any(not isinstance(item, Mapping) for item in value):
            return (ValidationIssue("prompt.blocks.visible_memory.value", "items must be mappings", value),)
        required = {"own_action", "partner_action", "payoff"}
        for index, item in enumerate(value):
            missing = required - set(item)
            if missing:
                return (
                    ValidationIssue(
                        f"prompt.blocks.visible_memory.value[{index}]",
                        f"must contain {', '.join(sorted(required))}",
                        thaw(item),
                    ),
                )
            for action_key in ("own_action", "partner_action"):
                action = item[action_key]
                if not isinstance(action, str) or not action:
                    return (
                        ValidationIssue(
                            f"prompt.blocks.visible_memory.value[{index}].{action_key}",
                            "must be a non-empty action label",
                            action,
                        ),
                    )
            payoff = item["payoff"]
            if isinstance(payoff, bool) or not isinstance(payoff, (int, float)):
                return (
                    ValidationIssue(
                        f"prompt.blocks.visible_memory.value[{index}].payoff",
                        "must be numeric",
                        payoff,
                    ),
                )
        return ()

    def render(self) -> str:
        memory = self.value  # type: ignore[assignment]
        if not memory:
            return "Player 1 has no past rounds available in memory."
        lines = ["Player 1's available history of choices in past rounds is:"]
        for local_round, raw in enumerate(memory, start=1):
            entry = thaw(raw)
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


@dataclass(frozen=True, slots=True)
class VisibleScoreBlock(PromptBlock[Mapping[str, int]]):
    name: str = field(init=False, default="visible_score")
    title: str = field(init=False, default="Visible score and local round")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: Mapping[str, int] | Unbound = UNBOUND
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="dynamic")

    def value_issues(self, value: Mapping[str, int]) -> tuple[ValidationIssue, ...]:
        if not isinstance(value, Mapping):
            return (ValidationIssue("prompt.blocks.visible_score.value", "must be a mapping", value),)
        if set(value) != {"score", "local_round"}:
            return (ValidationIssue("prompt.blocks.visible_score.value", "must contain score and local_round", value),)
        if any(isinstance(value[key], bool) or not isinstance(value[key], int) for key in value):
            return (ValidationIssue("prompt.blocks.visible_score.value", "values must be integers", value),)
        if value["local_round"] < 1:
            return (ValidationIssue("prompt.blocks.visible_score.value.local_round", "must be positive", value["local_round"]),)
        return ()

    def render(self) -> str:
        value = self.value  # type: ignore[assignment]
        return (
            f"It is now local round {value['local_round']}. "
            "Player 1's score over the visible memory window is "
            f"{value['score']}."
        )


@dataclass(frozen=True, slots=True)
class NamingConventionResponseContract(ResponseContract):
    type: str = field(init=False, default="paper_choice_reason")
    allowed_values: tuple[str, ...] = ("Q", "M")

    def instruction_messages(self) -> tuple[tuple[MessageRole, str], ...]:
        actions = " or ".join(self.allowed_values)
        output = (
            "Put the decision before the explanation. Return only an object in this "
            "answer-first form: {'value': '<ACTION>'; 'reason': '<YOUR REASON>'}. "
            f"The value must be exactly {actions}. Do not add text outside the object."
        )
        return (
            (MessageRole.SYSTEM, output),
            (MessageRole.USER, "Answer saying which action Player 1 should play."),
        )


class NamingConventionFullPrompt(FullPrompt):
    def concrete_prompt_type(self) -> str:
        return "naming_convention_decision"

    def validate(self) -> ValidationResult:
        result = super().validate()
        presented = self.block("presented_actions").value
        if presented is UNBOUND:
            return result
        if set(presented) != set(self.response_contract.allowed_values):
            return ValidationResult(
                (*result.issues, ValidationIssue(
                    "prompt.blocks.presented_actions.value",
                    "must contain exactly the response-contract allowed values",
                    thaw(presented),
                ))
            )
        return result

    def _merge_messages(self, messages: tuple[Message, ...]) -> tuple[Message, ...]:
        """Preserve the Version 2 wire text while exposing five separate blocks."""

        if len(messages) != 7 or messages[-1].role != MessageRole.USER:
            return super()._merge_messages(messages)
        presented = tuple(self.block("presented_actions").value)  # type: ignore[arg-type]
        output_instruction = (
            "Put the decision before the explanation. Return only an object in this "
            "answer-first form: {'value': '<ACTION>'; 'reason': '<YOUR REASON>'}. "
            f"The value must be exactly {' or '.join(presented)}. "
            "Do not add text outside the object."
        )
        system = (
            messages[0].content
            + "\n"
            + messages[1].content
            + "\n"
            + messages[2].content
            + "\n\n"
            + messages[3].content
            + "\n\n"
            + messages[4].content
            + "\n\n"
            + output_instruction
        )
        return (
            Message(
                MessageRole.SYSTEM,
                system,
                metadata={
                    "prompt_family": self.family,
                    "prompt_version": self.version,
                    "blocks": [block.name for block in self.blocks],
                    "response_contract_instruction": True,
                    "message_order": 1,
                },
            ),
            Message(
                MessageRole.USER,
                messages[-1].content,
                metadata={
                    "prompt_family": self.family,
                    "prompt_version": self.version,
                    "response_contract": self.response_contract.type,
                    "message_order": 2,
                },
            ),
        )


def naming_convention_prompt(
    actions: tuple[str, ...] = ("Q", "M"),
) -> NamingConventionFullPrompt:
    return NamingConventionFullPrompt(
        family="naming_convention_decision",
        version=1,
        blocks=(
            DescriptionBlock(),
            RulesBlock(),
            PresentedActionsBlock(),
            VisibleMemoryBlock(),
            VisibleScoreBlock(),
        ),
        response_contract=NamingConventionResponseContract(allowed_values=actions),
        message_mode="merge_consecutive_roles",
        block_separator="\n\n",
    )


def bind_naming_convention_prompt(
    *,
    presented_actions: tuple[str, ...],
    visible_memory: tuple[Mapping[str, Any], ...],
    visible_score: int,
    local_round: int,
    allowed_actions: tuple[str, ...] = ("Q", "M"),
) -> NamingConventionFullPrompt:
    return naming_convention_prompt(tuple(allowed_actions)).bind(
        presented_actions=tuple(presented_actions),
        visible_memory=tuple(visible_memory),
        visible_score={"score": visible_score, "local_round": local_round},
    )
