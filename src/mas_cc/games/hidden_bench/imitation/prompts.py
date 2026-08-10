"""Reasoning-only prompts for one-focal HiddenBench imitation updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mas_cc.llm_runtime.messages import MessageRole
from mas_cc.llm_runtime.prompts import UNBOUND, FullPrompt, PromptBlock, Unbound
from mas_cc.llm_runtime.validation import ValidationIssue

from ..vanilla.prompts import (
    RESPONSE_STYLE,
    DiscussionContract,
    InformationBlock,
    ResponseStyleBlock,
    ScenarioBlock,
    VoteContract,
)


@dataclass(frozen=True, slots=True)
class PrivateHistoryBlock(PromptBlock[tuple[Mapping[str, Any], ...]]):
    name: str = field(init=False, default="private_history")
    title: str = field(init=False, default="Your private interaction history")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: tuple[Mapping[str, Any], ...] | Unbound = UNBOUND
    required: bool = field(init=False, default=False)
    binding: str = field(init=False, default="dynamic")
    sensitive: bool = field(init=False, default=True)

    def value_issues(self, value: tuple[Mapping[str, Any], ...]) -> tuple[ValidationIssue, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            return (ValidationIssue("prompt.blocks.private_history.value", "must be a sequence"),)
        return ()

    def render(self) -> str:
        lines = ["Earlier private interactions you participated in:"]
        for item in self.value:  # type: ignore[union-attr]
            lines.append(
                f"- Event {item.get('event')}: partner/controller said "
                f"{item.get('received_message', '(nothing)')}; you committed "
                f"{item.get('own_vote_after', item.get('own_vote_before'))}."
            )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class InteractionBlock(PromptBlock[str]):
    name: str = field(init=False, default="interaction")
    title: str = field(init=False, default="Current interaction")
    role: MessageRole = field(init=False, default=MessageRole.USER)
    value: str | Unbound = UNBOUND
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="dynamic")
    sensitive: bool = field(init=False, default=True)

    def value_issues(self, value: str) -> tuple[ValidationIssue, ...]:
        if not isinstance(value, str) or not value.strip():
            return (ValidationIssue("prompt.blocks.interaction.value", "must be non-empty"),)
        return ()

    def render(self) -> str:
        return str(self.value)


class ImitationInitialPrompt(FullPrompt):
    def concrete_prompt_type(self) -> str:
        return "hidden_bench_imitation_initial"


class ImitationMessagePrompt(FullPrompt):
    def concrete_prompt_type(self) -> str:
        return "hidden_bench_imitation_message"


class ImitationUpdatePrompt(FullPrompt):
    def concrete_prompt_type(self) -> str:
        return "hidden_bench_imitation_update"


def hidden_bench_imitation_initial_prompt(
    possible_answers: Sequence[str] = ("A", "B", "C"),
) -> ImitationInitialPrompt:
    return ImitationInitialPrompt(
        "hidden_bench_imitation_initial",
        1,
        (ScenarioBlock(), InformationBlock(), ResponseStyleBlock(), InteractionBlock()),
        VoteContract(allowed_values=tuple(possible_answers)),
    )


def hidden_bench_imitation_message_prompt() -> ImitationMessagePrompt:
    return ImitationMessagePrompt(
        "hidden_bench_imitation_message",
        1,
        (
            ScenarioBlock(),
            InformationBlock(),
            ResponseStyleBlock(),
            PrivateHistoryBlock(),
            InteractionBlock(),
        ),
        DiscussionContract(),
    )


def hidden_bench_imitation_update_prompt(
    possible_answers: Sequence[str] = ("A", "B", "C"),
) -> ImitationUpdatePrompt:
    return ImitationUpdatePrompt(
        "hidden_bench_imitation_update",
        1,
        (
            ScenarioBlock(),
            InformationBlock(),
            ResponseStyleBlock(),
            PrivateHistoryBlock(),
            InteractionBlock(),
        ),
        VoteContract(allowed_values=tuple(possible_answers)),
    )


def _base(prompt: FullPrompt, *, scenario: str, information: Sequence[str]) -> FullPrompt:
    return prompt.bind(
        scenario=scenario,
        information=tuple(information),
        response_style=RESPONSE_STYLE,
    )


def bind_initial_prompt(
    *, scenario: str, information: Sequence[str], possible_answers: Sequence[str]
) -> ImitationInitialPrompt:
    return _base(
        hidden_bench_imitation_initial_prompt(possible_answers),
        scenario=scenario,
        information=information,
    ).bind(
        interaction="Using only the information you received, choose your current option now."
    )  # type: ignore[return-value]


def bind_message_prompt(
    *,
    scenario: str,
    information: Sequence[str],
    history: Sequence[Mapping[str, Any]],
    dialogue: Sequence[Mapping[str, Any]],
    allow_relay: bool,
) -> ImitationMessagePrompt:
    relay = (
        "You may relay information learned in earlier interactions."
        if allow_relay
        else "Only state information that was given to you directly."
    )
    if dialogue:
        prior = "\n".join(
            f"{'You' if item['is_self'] else 'The other participant'}: {item['message']}"
            for item in dialogue
        )
        interaction = f"This is a private exchange. {relay}\nSo far:\n{prior}\nRespond now."
    else:
        interaction = f"This is a private exchange with one participant. {relay} Speak now."
    prompt = _base(
        hidden_bench_imitation_message_prompt(),
        scenario=scenario,
        information=information,
    ).bind(interaction=interaction)
    if history:
        prompt = prompt.bind(private_history=tuple(history))
    return prompt  # type: ignore[return-value]


def bind_update_prompt(
    *,
    scenario: str,
    information: Sequence[str],
    history: Sequence[Mapping[str, Any]],
    possible_answers: Sequence[str],
    current_vote: str,
    dialogue: Sequence[Mapping[str, Any]],
    controller_message: str | None = None,
) -> ImitationUpdatePrompt:
    if controller_message is not None:
        current = f"External controller message: {controller_message}"
    else:
        lines = [
            f"{'You' if item['is_self'] else 'The other participant'}: {item['message']}"
            for item in dialogue
        ]
        current = "Private exchange:\n" + "\n".join(lines)
    instruction = (
        f"Your current committed option is {current_vote}.\n{current}\n"
        "After considering this interaction, commit your option for this event."
    )
    prompt = _base(
        hidden_bench_imitation_update_prompt(possible_answers),
        scenario=scenario,
        information=information,
    ).bind(interaction=instruction)
    if history:
        prompt = prompt.bind(private_history=tuple(history))
    return prompt  # type: ignore[return-value]
