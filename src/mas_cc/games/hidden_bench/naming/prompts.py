"""Dyadic prompts for `hidden_bench_naming` (brief §6).

The *information* half is the paper's, unchanged: the same scenario block, the
same shuffled-facts block with the same "randomly shuffle" preamble, reused
directly from `vanilla/prompts.py` so a fact is worded identically whichever
game an agent meets it in. Only the *interaction* half is replaced - one partner
instead of a room, private memory instead of a global transcript.

Deliberately not paper-verbatim, and it cannot be: the paper has no dyadic
condition. What is kept verbatim is kept so the two games differ in exactly one
thing (the interaction structure) and nothing else.
"""

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
    render_extra,
)

PAIRING_RULE = (
    "You are speaking privately with one other participant. Nobody else can hear this "
    "conversation, and you cannot see any conversation you were not part of."
)

RELAY_ALLOWED = (
    "You may pass on anything you have learned from earlier conversations, whether or not it "
    "was given to you directly."
)

RELAY_FORBIDDEN = (
    "You may only state information that was given to you directly. Do not pass on anything you "
    "learned from another participant."
)

COORDINATION_GOAL = (
    "At the end of this conversation you and your partner will each privately choose one option. "
    "You are rewarded when the two of you choose the same option."
)

CORRECTNESS_GOAL = (
    "At the end of this conversation you and your partner will each privately choose one option. "
    "You are rewarded when you choose the correct option."
)

BOTH_GOAL = (
    "At the end of this conversation you and your partner will each privately choose one option. "
    "You are rewarded when the two of you choose the same option, and rewarded further when that "
    "option is the correct one."
)

_GOALS = {
    "coordination": COORDINATION_GOAL,
    "correctness": CORRECTNESS_GOAL,
    "coordination_plus_correctness": BOTH_GOAL,
}


def interaction_rules(*, payoff_mode: str, allow_relay: bool) -> str:
    """The dyadic framing, stated once per prompt.

    The payoff mode is *told* to the agent rather than only being applied
    afterwards, because an agent that is not told what it is being paid for
    cannot be said to be optimizing it - and the whole point of separating
    `coordination` from `correctness` is to see two different behaviours, not
    two different scorings of the same behaviour.
    """

    return "\n".join(
        (PAIRING_RULE, _GOALS[payoff_mode], RELAY_ALLOWED if allow_relay else RELAY_FORBIDDEN)
    )


@dataclass(frozen=True, slots=True)
class DyadicTurnBlock(PromptBlock[str]):
    """This agent's view of the current private conversation."""

    name: str = field(init=False, default="conversation")
    title: str = field(init=False, default="Private conversation")
    role: MessageRole = field(init=False, default=MessageRole.USER)
    value: str | Unbound = UNBOUND
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="dynamic")
    sensitive: bool = field(init=False, default=True)

    def value_issues(self, value: str) -> tuple[ValidationIssue, ...]:
        if not isinstance(value, str) or not value.strip():
            return (
                ValidationIssue("prompt.blocks.conversation.value", "must be non-empty text", value),
            )
        return ()

    def render(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class PastEncountersBlock(PromptBlock[tuple[Mapping[str, Any], ...]]):
    """This agent's own bounded memory of who it met and what happened.

    `sensitive=True` and `required=False`: an agent that has met nobody yet has
    an empty, *distinct* state, and the block is simply omitted rather than
    rendering an "(empty)" placeholder that the model would try to interpret.
    """

    name: str = field(init=False, default="past_encounters")
    title: str = field(init=False, default="Your earlier conversations")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: tuple[Mapping[str, Any], ...] | Unbound = UNBOUND
    required: bool = field(init=False, default=False)
    binding: str = field(init=False, default="dynamic")
    sensitive: bool = field(init=False, default=True)

    def value_issues(self, value: tuple[Mapping[str, Any], ...]) -> tuple[ValidationIssue, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            return (
                ValidationIssue("prompt.blocks.past_encounters.value", "must be a sequence", value),
            )
        return ()

    def render(self) -> str:
        lines = ["What you learned in earlier conversations:"]
        for entry in self.value:  # type: ignore[union-attr]
            heard = entry.get("partner_said") or ()
            said = "; ".join(str(item) for item in heard) if heard else "(nothing)"
            lines.append(
                f"- Round {entry.get('round')}: your partner said: {said} "
                f"| you chose {entry.get('own_choice')}, they chose {entry.get('partner_choice')} "
                f"| reward {entry.get('payoff')}"
            )
        return "\n".join(lines)


class HiddenBenchDyadicPrompt(FullPrompt):
    def concrete_prompt_type(self) -> str:
        return "hidden_bench_naming_message"


class HiddenBenchCommitPrompt(FullPrompt):
    def concrete_prompt_type(self) -> str:
        return "hidden_bench_naming_commit"


def hidden_bench_naming_message_prompt() -> HiddenBenchDyadicPrompt:
    return HiddenBenchDyadicPrompt(
        "hidden_bench_naming_message",
        1,
        (ScenarioBlock(), InformationBlock(), ResponseStyleBlock(), PastEncountersBlock(), DyadicTurnBlock()),
        DiscussionContract(),
    )


def hidden_bench_naming_commit_prompt(
    possible_answers: Sequence[str] = ("A", "B", "C"),
) -> HiddenBenchCommitPrompt:
    return HiddenBenchCommitPrompt(
        "hidden_bench_naming_commit",
        1,
        (ScenarioBlock(), InformationBlock(), ResponseStyleBlock(), PastEncountersBlock(), DyadicTurnBlock()),
        VoteContract(allowed_values=tuple(possible_answers)),
    )


def _style(*, payoff_mode: str, allow_relay: bool, extra: str | None) -> str:
    return (
        f"{RESPONSE_STYLE}{render_extra(extra)}\n\n"
        f"{interaction_rules(payoff_mode=payoff_mode, allow_relay=allow_relay)}"
    )


def _conversation(dialogue: Sequence[Mapping[str, Any]], *, closing: str) -> str:
    if not dialogue:
        body = "This conversation has just started; you speak first."
    else:
        body = "\n".join(
            f"{'You' if item['is_self'] else 'Your partner'}: {item['message']}" for item in dialogue
        )
        body = f"So far in this conversation:\n{body}"
    return f"{body}\n\n{closing}"


def bind_message_prompt(
    *,
    scenario: str,
    information: Sequence[str],
    past_encounters: Sequence[Mapping[str, Any]],
    dialogue: Sequence[Mapping[str, Any]],
    payoff_mode: str,
    allow_relay: bool,
    extra: str | None,
) -> HiddenBenchDyadicPrompt:
    prompt = hidden_bench_naming_message_prompt().bind(
        scenario=scenario,
        information=tuple(information),
        response_style=_style(payoff_mode=payoff_mode, allow_relay=allow_relay, extra=extra),
        conversation=_conversation(dialogue, closing="It's your turn to speak to your partner."),
    )
    if past_encounters:
        prompt = prompt.bind(past_encounters=tuple(past_encounters))
    return prompt


def bind_commit_prompt(
    *,
    scenario: str,
    information: Sequence[str],
    past_encounters: Sequence[Mapping[str, Any]],
    dialogue: Sequence[Mapping[str, Any]],
    possible_answers: Sequence[str],
    payoff_mode: str,
    allow_relay: bool,
    extra: str | None,
) -> HiddenBenchCommitPrompt:
    prompt = hidden_bench_naming_commit_prompt(possible_answers).bind(
        scenario=scenario,
        information=tuple(information),
        response_style=_style(payoff_mode=payoff_mode, allow_relay=allow_relay, extra=extra),
        conversation=_conversation(
            dialogue, closing="The conversation is over. Choose your option now."
        ),
    )
    if past_encounters:
        prompt = prompt.bind(past_encounters=tuple(past_encounters))
    return prompt
