"""Reasoning-only prompts for one-focal HiddenBench imitation updates.

**Two text versions ship side by side.** `v1` is the wording every result
recorded before 2026-08-11 was produced under and is frozen byte-for-byte; `v2`
is the revision specified in
`docs/tdd/misselaneous/11082026_prompt_modifications_v2.md`, selected with
`game.options.prompt_version: 2`.

`v2` is not a bug fix, it is an intervention. Three of its four changes push
directly on how much information an agent volunteers, which is the quantity the
study measures:

- `response_style` drops the two-sentence cap and asks for one not-yet-mentioned
  fact per turn (§1.1). This is close to the paper's "Share All Information"
  condition, which moved GPT-4.1 from 0.233 to 0.467;
- `private_history` shows the agent what *it* said, not only what it heard
  (§1.3) - without which "mention something you have not mentioned yet" is an
  instruction the agent has no way to follow;
- `interaction` stops saying "relay information learned in earlier
  interactions", which pointed the agent at facts already in circulation (§1.4).

The fourth change is the "Informing Asymmetry" hint (§1.2), gated separately
behind `inform_asymmetry` because it is a second, independently reportable
condition (0.367 vs 0.233 in the paper). It says only that information is
*uneven*; it never marks which facts are unique, because an agent that can
recover its own private facts is no longer doing a hidden-profile task.

`v2` also closes the second of the two "controller" leaks (§2.1): `v1` renders
every history line as `partner/controller said`, which tells the agent an
external controller exists and which turns it spoke on. `v2` renders every
source as "the other participant". The first leak lives in the controller's own
message text, see `controller.py`.
"""

from __future__ import annotations

import re
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

PROMPT_VERSIONS = (1, 2)
SCENARIO_VARIANTS = (1, 2)

RESPONSE_STYLE_V2 = (
    "Keep your response short — two to four sentences. Each time you speak,\n"
    "mention at least one specific piece of information from your list that you\n"
    "have not already mentioned in an earlier message, and end by saying which\n"
    "option you are voting for."
)
"""§1.1. The doc writes "which location"; the corpus also contains tasks whose
options are companies, people and routes, so the noun is generalised to
"option". Nothing else in the paragraph is changed."""

ASYMMETRY_NOTICE = (
    "Other participants have received their own information. Some of what you\n"
    "know may be known only to you, and some of what they know may be unknown\n"
    "to you."
)
"""§1.2. Deliberately the *strongest* hint that does not identify which facts
are the private ones."""

RELAY_ALLOWED_V1 = "You may relay information learned in earlier interactions."
RELAY_FORBIDDEN_V1 = "Only state information that was given to you directly."

DISCLOSE_ALLOWED_V2 = (
    "Tell them what you know, including anything you have not yet told anyone, "
    "and say how you are voting."
)
DISCLOSE_FORBIDDEN_V2 = (
    "Tell them what you know, including anything you have not yet told anyone, "
    "but only state information that was given to you directly, and say how you "
    "are voting."
)

_COORDINATION_BULLET = re.compile(
    r"^- If (?:all other participants|everyone|all participants)\b.*?"
    r"(?:additional \$|total of \$|each earn|earn a bonus)[^\n]*\n?",
    re.MULTILINE,
)
_COORDINATION_SENTENCE = re.compile(
    r"This means that coordinating with others is critical to maximize your rewards\.\s*"
)


def strip_coordination_bonus(description: str) -> tuple[str, bool]:
    """`scenario@2` (§1.5): the same scenario minus the conform-with-others payoff.

    The coordination bonus is canonical to the paper's task, so it stays in
    `scenario@1`. It is also an explicit instruction to agree with everyone
    else, sitting in the system prompt and pulling against every §1 change - in
    other words it is this study's imitation coupling strength written in
    English, and exposing it as a variant makes it a semantic analogue of a
    classical conformity parameter.

    Only 7 of the 65 corpus tasks carry the clause, so a task without it is not
    an error: the caller gets `False` back and records that the variant was a
    no-op for this task rather than silently claiming a manipulation that never
    happened.
    """

    stripped = _COORDINATION_SENTENCE.sub("", _COORDINATION_BULLET.sub("", description))
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()
    return stripped, stripped != description.strip()


def scenario_for_variant(description: str, variant: int) -> tuple[str, bool]:
    if variant not in SCENARIO_VARIANTS:
        raise ValueError(f"scenario_variant must be one of {list(SCENARIO_VARIANTS)}")
    if variant == 1:
        return description, False
    return strip_coordination_bonus(description)


@dataclass(frozen=True, slots=True)
class PromptStyle:
    """The prompt-text knobs, carried as one value so they cannot drift apart."""

    version: int = 1
    inform_asymmetry: bool = False

    def __post_init__(self) -> None:
        if self.version not in PROMPT_VERSIONS:
            raise ValueError(f"prompt_version must be one of {list(PROMPT_VERSIONS)}")
        if self.inform_asymmetry and self.version == 1:
            # v1 is frozen: bolting the notice onto it would produce a prompt
            # that is neither the recorded v1 nor the reportable v2.
            raise ValueError("inform_asymmetry requires prompt_version 2")

    @property
    def message_response_style(self) -> str:
        return RESPONSE_STYLE if self.version == 1 else RESPONSE_STYLE_V2

    @property
    def vote_response_style(self) -> str:
        """Always the terse v1 line, at both prompt versions.

        `RESPONSE_STYLE_V2` is a *discussion* instruction: "each time you
        speak", "mention a fact you have not already mentioned", "end by saying
        which option you are voting for". A vote turn is not a speaking turn -
        it returns a JSON object whose vote field already states the opinion.

        Two things go wrong if it is applied here anyway, and both did in
        run `...-control-10-v2-20260840`:

        - the instruction inflates the `rationale` field until the response is
          truncated at `max_output_tokens` with no closing brace, which the
          contract correctly rejects, killing the episode;
        - more quietly, it conditions the measurement instrument. The vanilla
          game already refuses to do this - see `bind_vote_prompt`, which binds
          the plain `RESPONSE_STYLE` with no `%extra%` for exactly this reason.
        """

        return RESPONSE_STYLE

    def to_dict(self) -> dict[str, Any]:
        return {"prompt_version": self.version, "inform_asymmetry": self.inform_asymmetry}


DEFAULT_STYLE = PromptStyle()


@dataclass(frozen=True, slots=True)
class AsymmetryInformationBlock(InformationBlock):
    """`information` plus the §1.2 notice that information is unevenly held."""

    def render(self) -> str:
        return f"{InformationBlock.render(self)}\n{ASYMMETRY_NOTICE}"


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
            lines.append(self._line(item))
        return "\n".join(lines)

    def _line(self, item: Mapping[str, Any]) -> str:
        vote = item.get("own_vote_after", item.get("own_vote_before"))
        received = item.get("received_message", "(nothing)")
        if self.version == 1:
            return f"- Event {item.get('event')}: partner/controller said {received}; you committed {vote}."
        # v2: no source label that distinguishes a controller from a peer
        # (§2.1), and the agent's own message is shown back to it (§1.3).
        own = item.get("own_message")
        reply = "" if not own else f" you replied {own};"
        return f"- Event {item.get('event')}: the other participant said {received};{reply} you committed {vote}."


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


def _information_block(style: PromptStyle) -> InformationBlock:
    return AsymmetryInformationBlock() if style.inform_asymmetry else InformationBlock()


def hidden_bench_imitation_initial_prompt(
    possible_answers: Sequence[str] = ("A", "B", "C"),
    style: PromptStyle = DEFAULT_STYLE,
) -> ImitationInitialPrompt:
    return ImitationInitialPrompt(
        "hidden_bench_imitation_initial",
        style.version,
        (ScenarioBlock(), _information_block(style), ResponseStyleBlock(), InteractionBlock()),
        VoteContract(allowed_values=tuple(possible_answers)),
    )


def hidden_bench_imitation_message_prompt(
    style: PromptStyle = DEFAULT_STYLE,
) -> ImitationMessagePrompt:
    return ImitationMessagePrompt(
        "hidden_bench_imitation_message",
        style.version,
        (
            ScenarioBlock(),
            _information_block(style),
            ResponseStyleBlock(),
            PrivateHistoryBlock(version=style.version),
            InteractionBlock(),
        ),
        DiscussionContract(),
    )


def hidden_bench_imitation_update_prompt(
    possible_answers: Sequence[str] = ("A", "B", "C"),
    style: PromptStyle = DEFAULT_STYLE,
) -> ImitationUpdatePrompt:
    return ImitationUpdatePrompt(
        "hidden_bench_imitation_update",
        style.version,
        (
            ScenarioBlock(),
            _information_block(style),
            ResponseStyleBlock(),
            PrivateHistoryBlock(version=style.version),
            InteractionBlock(),
        ),
        VoteContract(allowed_values=tuple(possible_answers)),
    )


def _base(
    prompt: FullPrompt,
    *,
    scenario: str,
    information: Sequence[str],
    response_style: str,
) -> FullPrompt:
    return prompt.bind(
        scenario=scenario,
        information=tuple(information),
        response_style=response_style,
    )


def bind_initial_prompt(
    *,
    scenario: str,
    information: Sequence[str],
    possible_answers: Sequence[str],
    style: PromptStyle = DEFAULT_STYLE,
) -> ImitationInitialPrompt:
    return _base(
        hidden_bench_imitation_initial_prompt(possible_answers, style),
        scenario=scenario,
        information=information,
        response_style=style.vote_response_style,
    ).bind(
        interaction="Using only the information you received, choose your current option now."
    )  # type: ignore[return-value]


def message_interaction_text(
    *,
    style: PromptStyle,
    allow_relay: bool,
    dialogue: Sequence[Mapping[str, Any]],
) -> str:
    if style.version == 1:
        clause = RELAY_ALLOWED_V1 if allow_relay else RELAY_FORBIDDEN_V1
    else:
        clause = DISCLOSE_ALLOWED_V2 if allow_relay else DISCLOSE_FORBIDDEN_V2
    if dialogue:
        prior = "\n".join(
            f"{'You' if item['is_self'] else 'The other participant'}: {item['message']}"
            for item in dialogue
        )
        return f"This is a private exchange. {clause}\nSo far:\n{prior}\nRespond now."
    return f"This is a private exchange with one participant. {clause} Speak now."


def bind_message_prompt(
    *,
    scenario: str,
    information: Sequence[str],
    history: Sequence[Mapping[str, Any]],
    dialogue: Sequence[Mapping[str, Any]],
    allow_relay: bool,
    style: PromptStyle = DEFAULT_STYLE,
) -> ImitationMessagePrompt:
    prompt = _base(
        hidden_bench_imitation_message_prompt(style),
        scenario=scenario,
        information=information,
        response_style=style.message_response_style,
    ).bind(
        interaction=message_interaction_text(
            style=style, allow_relay=allow_relay, dialogue=dialogue
        )
    )
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
    influence_inputs: Sequence[Mapping[str, Any]] | None = None,
    style: PromptStyle = DEFAULT_STYLE,
) -> ImitationUpdatePrompt:
    if influence_inputs is not None:
        # q>1 is a group interaction with a fixed number of ordered influence
        # slots.  Controller and peer inputs use the same label deliberately:
        # the controller acts like a peer locally while remaining external to
        # the voting population.  Slot order is the scheduler's logged order.
        lines = [
            f"The other participant in slot {int(item['slot']) + 1}: "
            f"{item.get('message') or '(no message)'}"
            for item in influence_inputs
        ]
        own_messages = [str(item["message"]) for item in dialogue if item.get("is_self")]
        own = ""
        if own_messages:
            own = "\nYour messages in these exchanges:\n" + "\n".join(
                f"You: {message}" for message in own_messages
            )
        current = "Private group exchange:\n" + "\n".join(lines) + own
    elif controller_message is not None:
        # v1 labels the source; v2 must not, because "External controller
        # message:" is leak 1 of §2.1 and is by itself enough for the agent to
        # discount the turn as an outside voice.
        current = (
            f"External controller message: {controller_message}"
            if style.version == 1
            else f"Private exchange:\nThe other participant: {controller_message}"
        )
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
        hidden_bench_imitation_update_prompt(possible_answers, style),
        scenario=scenario,
        information=information,
        response_style=style.vote_response_style,
    ).bind(interaction=instruction)
    if history:
        prompt = prompt.bind(private_history=tuple(history))
    return prompt  # type: ignore[return-value]
