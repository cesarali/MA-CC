"""The prompt a synthetic agent actually reads.

There is one design decision in this file worth stating outright: the agent's
whole decision input is rendered *into the prompt*, and the synthetic agent
recovers it by reading the compiled prompt back - it is never handed the
observation through a side channel.

That is deliberate and it is the point. If the prompt fails to carry the
conditioning information, or carries a stale round, the agent decodes the wrong
thing and the measured mutual information misses its closed-form value. A
side-channel agent would *exercise* prompt construction; this one *checks* it.

The machine-readable payload is a single marked line rather than free prose so
the decoding stays exact. The surrounding human-readable text is not decoration
either - it is what makes a rendered synthetic prompt legible next to a real
one in `prompts/`, which is how you tell at a glance that the two pipelines are
the same pipeline.

One prompt family serves every synthetic game. What changes between them is the
decoding rule stated in the `protocol` block and the payload in the
`observation` block - not the prompt's shape - so each game contributes a
binder (`bind_bernoulli_prompt` and its future siblings) rather than a family
of its own.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping

from mas_cc.llm_runtime.messages import MessageRole
from mas_cc.llm_runtime.prompts import UNBOUND, FullPrompt, PromptBlock, ResponseContract, Unbound
from mas_cc.llm_runtime.prompts._values import thaw
from mas_cc.llm_runtime.validation import ValidationIssue

OBSERVATION_MARKER = "SYNTHETIC-OBSERVATION-V1"
"""The token the synthetic agent looks for in the compiled prompt.

Versioned because it is a wire format between two files that must not drift:
change what the observation payload means and this marker changes with it, so
an old agent meets an unfamiliar marker instead of silently misreading a new
payload as an old one.
"""

PROMPT_FAMILY = "synthetic_agent_decision"
PROMPT_VERSION = 1


def render_observation(payload: Mapping[str, Any]) -> str:
    """The one canonical way an observation becomes prompt text."""

    return f"{OBSERVATION_MARKER} " + json.dumps(
        thaw(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


@dataclass(frozen=True, slots=True)
class SyntheticTaskBlock(PromptBlock[str]):
    name: str = field(init=False, default="task")
    title: str = field(init=False, default="Task")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: str | Unbound = (
        "You are one agent in a calibration game with analytically known statistics. "
        "Each round you are given a signal and a private instruction, and you report "
        "exactly one action. There is nothing to optimise and no partner to outguess: "
        "your only job is to follow the decoding rule exactly."
    )
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="fixed")

    def value_issues(self, value: str) -> tuple[ValidationIssue, ...]:
        if isinstance(value, str) and value.strip():
            return ()
        return (ValidationIssue("prompt.blocks.task.value", "must be non-empty text", value),)

    def render(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class SyntheticProtocolBlock(PromptBlock[tuple[str, ...]]):
    name: str = field(init=False, default="protocol")
    title: str = field(init=False, default="Decoding rule")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: tuple[str, ...] | Unbound = UNBOUND
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="dynamic")

    def value_issues(self, value: tuple[str, ...]) -> tuple[ValidationIssue, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            return (
                ValidationIssue(
                    "prompt.blocks.protocol.value", "must be a sequence of strings", value
                ),
            )
        if any(not isinstance(item, str) or not item.strip() for item in value):
            return (
                ValidationIssue(
                    "prompt.blocks.protocol.value", "must contain non-empty strings", value
                ),
            )
        return ()

    def render(self) -> str:
        return "\n".join(f"{index}. {rule}" for index, rule in enumerate(self.value, 1))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class SyntheticObservationBlock(PromptBlock[Mapping[str, Any]]):
    """This round's decision input, as the marked machine-readable line."""

    name: str = field(init=False, default="observation")
    title: str = field(init=False, default="This round")
    role: MessageRole = field(init=False, default=MessageRole.USER)
    value: Mapping[str, Any] | Unbound = UNBOUND
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="dynamic")

    def value_issues(self, value: Mapping[str, Any]) -> tuple[ValidationIssue, ...]:
        if not isinstance(value, Mapping):
            return (
                ValidationIssue("prompt.blocks.observation.value", "must be a mapping", value),
            )
        missing = {"policy", "round"} - set(value)
        if missing:
            return (
                ValidationIssue(
                    "prompt.blocks.observation.value",
                    f"must declare {sorted(missing)}",
                    value,
                ),
            )
        return ()

    def render(self) -> str:
        return render_observation(self.value)  # type: ignore[arg-type]


class SyntheticAgentFullPrompt(FullPrompt):
    def concrete_prompt_type(self) -> str:
        return PROMPT_FAMILY


DEFAULT_ACTIONS = ("Q", "M")
"""The alphabet the registered definition declares, matching the real naming game."""


def synthetic_agent_decision_prompt() -> SyntheticAgentFullPrompt:
    """The zero-argument registry entry for this family.

    The registry holds prompt *definitions* - block order, roles, contract
    shape - which is what config validation and prompt export need. The bound
    per-round prompt a game actually sends comes from a binder below, with the
    action alphabet the resolved config asked for.
    """

    return synthetic_agent_prompt(actions=DEFAULT_ACTIONS)


def synthetic_agent_prompt(*, actions: tuple[str, ...]) -> SyntheticAgentFullPrompt:
    return SyntheticAgentFullPrompt(
        family=PROMPT_FAMILY,
        version=PROMPT_VERSION,
        blocks=(
            SyntheticTaskBlock(),
            SyntheticProtocolBlock(),
            SyntheticObservationBlock(),
        ),
        response_contract=ResponseContract(
            "choice_only",
            actions,
            {"decision_instruction": "Apply the decoding rule to this round's line and report your action."},
        ),
        message_mode="merge_consecutive_roles",
    )


def bind_bernoulli_prompt(
    *, actions: tuple[str, ...], observation: Mapping[str, Any]
) -> SyntheticAgentFullPrompt:
    """Bind the Game 1 decoding rule and one round's observation."""

    protocol = (
        f"Your action is one of: {', '.join(actions)}.",
        f"The line below starts with {OBSERVATION_MARKER} and is a JSON object.",
        "Read its 'signal' field. If 'flip' is false, report the signal unchanged.",
        "If 'flip' is true, report the other action instead.",
        "Report nothing else.",
    )
    return synthetic_agent_prompt(actions=actions).bind(
        protocol=protocol, observation=observation
    )


def bind_markov_prompt(
    *, actions: tuple[str, ...], observation: Mapping[str, Any]
) -> SyntheticAgentFullPrompt:
    """Bind the Game 2 / Game 3 decoding rule and one round's observation.

    The payload carries the partner's action *and* the realized kernel map, so
    the agent does the coupling lookup itself. That is the point: if the game
    renders the wrong partner, or last round's action instead of this one's,
    the agent decodes something legal but wrong and the measured transfer
    entropy misses its exact value - including the structural zeros, which is
    precisely the direction-and-alignment bug this game exists to catch.

    ``forced`` appears only in the controlled game, and overrides everything: a
    controlled agent reports the control's action regardless of what it saw.
    """

    protocol = [
        f"Your action is one of: {', '.join(actions)}.",
        f"The line below starts with {OBSERVATION_MARKER} and is a JSON object.",
    ]
    if "forced" in observation:
        protocol.append(
            "If its 'forced' field is not null, report that action and stop; "
            "ignore every other field."
        )
    protocol.extend(
        [
            "Read 'observed' - the action the agent you follow played last round.",
            "Look that action up in the 'rule' object to get your target action.",
            "If 'flip' is false, report the target. If 'flip' is true, report the other action.",
            "Report nothing else.",
        ]
    )
    return synthetic_agent_prompt(actions=actions).bind(
        protocol=tuple(protocol), observation=observation
    )
