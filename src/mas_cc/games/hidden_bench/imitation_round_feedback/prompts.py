"""One-call public-ballot prompts for the round-feedback reasoning game.

**One focal update = one LLM call = one actual vote + one public reason.**

The parent `hidden_bench_imitation` game reasons in two steps: peers generate
private dialogue, then the focal agent votes in a separate call. That split
gives an agent a place to say one thing and vote another, and it costs
`2 q m + 1` provider calls per microscopic update. This family removes the
split. The focal agent sees the current public ballots `(vote, reason)` of its
`q` social sources and returns a single JSON object whose `vote` *is* its new
committed action and whose `reason` *is* the public text later focal agents
will read.

There is exactly one vote variable and exactly one reason variable per agent.
Nothing in this module may introduce a "spoken" vote distinct from the
committed one.

Two things are deliberately fixed rather than swept:

- `SOCIAL_ENVIRONMENT` (the Strategic-Uncertainty block) is bound as a *fixed*
  block, so it is byte-identical across tasks, models, targets, and controlled
  and uncontrolled updates, and its text is pinned by the prompt definition
  hash rather than by convention;
- `render_control_reason` is the single controller public-reason renderer. The
  controller occupies one ordinary social slot as a `(vote, reason)` pair and
  its reason is a deterministic, fact-free function of the target.

`vote_visibility` is the one rendering axis. `public` (implemented) shows a
source's vote and reason; `hidden` (reserved) shows only its reason. The
prompt shape, the provider-call count, and the output schema are identical in
both modes - only `render_social_source` branches.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import re
from typing import Any

from mas_cc.llm_runtime.messages import MessageRole
from mas_cc.llm_runtime.prompts import (
    UNBOUND,
    FullPrompt,
    PromptBlock,
    ResponseContract,
    Unbound,
)
from mas_cc.llm_runtime.validation import ValidationIssue, ValidationResult

from ..vanilla.prompts import extract_json_object, normalize_vote

PROMPT_FAMILY = "hidden_bench_public_ballot"
PROMPT_VERSION = 1

VOTE_VISIBILITIES = ("public", "hidden")
"""Every rendering mode the source renderer knows about."""

IMPLEMENTED_VOTE_VISIBILITIES = ("public",)
"""What a run may actually be configured with today.  `hidden` is reserved: the
renderer supports it, but the surrounding prompt text still says the vote
becomes public, so enabling it is a text decision that has not been made."""

SOCIAL_ENVIRONMENT = (
    "Participant identities persist across interactions.\n"
    "\n"
    "Different participants may possess different information and may also have\n"
    "objectives that differ from yours. Some recommendations may therefore be\n"
    "strategic. You do not know which participants, if any, have different\n"
    "objectives."
)
"""The fixed Strategic-Uncertainty environment, verbatim in every prompt.

It never says the controller is strategic and never attaches reputation to a
specific source: it is a property of the population, so a controlled and an
uncontrolled update are indistinguishable from this block alone."""

DECISION_BASIS_SOCIAL = (
    "Make your own decision using:\n"
    "- your private information;\n"
    "- your current position;\n"
    "- the current public positions and reasons of the participants shown below.\n"
    "\n"
    "Your final vote and your reason will become your public position and may be\n"
    "shown to other participants in later interactions."
)

DECISION_BASIS_INITIAL = (
    "Make your own decision using your private information alone. No other\n"
    "participant has stated a position yet.\n"
    "\n"
    "Your final vote and your reason will become your public position and may be\n"
    "shown to other participants in later interactions."
)

DECISION_INSTRUCTION = (
    "DECISION\n"
    "\n"
    "Decide what you currently believe is the best option.\n"
    "\n"
    "Your reason should briefly explain why you are making this choice. You may use\n"
    "your private information in your reason if you think it is relevant, but you\n"
    "are not required to reveal all of your private information.\n"
    "\n"
    "Do not claim factual information that has not been provided to you through\n"
    "your private information or the public information shown above.\n"
    "\n"
    "Keep your reason to at most three sentences."
)
"""The agent is never *required* to disclose.  A prompt that forced its private
facts into the reason would delete the hidden-profile problem this game exists
to measure: useful private information must enter the public state only if an
agent chooses to put it there."""

NO_PREVIOUS_REASON = "No previous public reason."

_POPULATION_BOOKKEEPING = re.compile(
    r"(?:\n\s*)?There are exactly \d+ agents in this experiment\.\s*"
    r"Private information derives from \d+ latent evidence types\.\s*"
    r"Several agents may carry different realizations or components of the same type\.\s*"
    r"Ignore any historical participant count left in the scenario wording\.?",
    re.IGNORECASE,
)


def sanitize_scenario(scenario: str) -> str:
    """Remove HiddenBench's obsolete study/chat wrapper, not task facts.

    The corpus remains immutable.  This renderer strips only benchmark framing,
    population-allocation bookkeeping, discussion timing, and reward protocol
    from the copy placed in an atomic public-ballot prompt.
    """

    text = _POPULATION_BOOKKEEPING.sub("", str(scenario))
    text = re.sub(
        r"\bYou are participating in a study, acting as\b",
        "You are",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bYou are participating in a study and are\b",
        "You are",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bthe other three (community leaders|participants|investigators)\b",
        r"the other \1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Your Task:\s*\n\s*"
        r"you will discuss with (?:three )?other participants, who are also acting as ([^,]+), "
        r"to decide ([^.]+)\.",
        lambda match: (
            f"Your task is to decide {match.group(2)} together with the other "
            f"{match.group(1)}."
        ),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Your Task:\s*\n\s*"
        r"you will discuss the proposals with three other participants, who are also "
        r"acting as ([^,]+), to decide ([^.]+)\.",
        lambda match: (
            f"Your task is to decide {match.group(2)} together with the other "
            f"{match.group(1)}."
        ),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"Your Task:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\bYou have (?:three|four) (options|suspects|candidates):",
        r"The available \1 are:",
        text,
        flags=re.IGNORECASE,
    )

    # In the original tasks this marker starts a terminal benchmark-payoff and
    # chat-timing section.  Keep the immediately preceding decision fact.
    text = re.sub(
        r"\s+After the discussion:\s*\n(?:.|\n)*\Z", "", text, flags=re.IGNORECASE
    )
    procedural_lines = (
        r"If you (?:choose|identify) the correct .+?(?:earn|receive).+",
        r"If (?:all other participants|everyone).+?(?:earn|receive).+",
        r"This means that coordinating with others is critical to maximize your rewards\..*",
        r"The chat will at most take .+",
        r"However, the exact time when the chat will end is unknown\..*",
        r"You don't know when the discussion will end, so decide carefully\..*",
    )
    for pattern in procedural_lines:
        text = re.sub(rf"^\s*-?\s*{pattern}\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

MAX_REASON_CHARACTERS = 600
"""A generous cap on "two to three sentences".  It exists to reject a runaway
essay - which truncates at `max_output_tokens` and breaks the JSON anyway - not
to police wording, so a genuine three-sentence reason never hits it."""


def agent_label(agent_id: Any) -> str:
    """`agent-006` -> `Agent 7`: an ordinary 1-based participant label.

    The corpus scenarios address "the other participants" by number, so the
    social block has to speak the same language.  An id that does not end in
    digits is passed through unchanged rather than mangled.
    """

    text = str(agent_id)
    digits = text.rsplit("-", 1)[-1]
    if not digits.isdigit():
        return text
    return f"Agent {int(digits) + 1}"


def control_label(population_size: int) -> str:
    """The controller's persistent ordinary-looking identity.

    One label for the whole episode, one past the highest ordinary participant,
    so the controller is a participant that keeps its identity across updates
    rather than a voice that borrows a different agent's name each time.
    """

    return f"Agent {int(population_size) + 1}"


def render_control_reason(target: str) -> str:
    """The controller's public reason: deterministic in `Z`, and fact-free.

    It cites no task evidence, fabricates nothing, and never names itself as a
    controller, an experiment, or anything external.  The `(vote, reason)` pair
    *is* the whole intervention - nothing else is appended to the prompt.
    """

    return (
        f"Based on the discussion so far, {target} still looks like the "
        f"strongest option to me and deserves more weight."
    )


def render_social_source(
    source: Mapping[str, Any], *, vote_visibility: str = "public"
) -> str:
    """One visible social source, rendered identically for peers and control."""

    if vote_visibility not in VOTE_VISIBILITIES:
        raise ValueError(f"vote_visibility must be one of {list(VOTE_VISIBILITIES)}")
    label = str(source["label"])
    reason = str(source.get("reason") or NO_PREVIOUS_REASON)
    if vote_visibility == "hidden":
        return f"{label}\nReason: {reason}"
    return f"{label}\nVote: {source['vote']}\nReason: {reason}"


def render_social_sources(
    sources: Sequence[Mapping[str, Any]], *, vote_visibility: str = "public"
) -> tuple[str, ...]:
    return tuple(
        render_social_source(source, vote_visibility=vote_visibility)
        for source in sources
    )


# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------


def _text_issues(name: str, value: Any) -> tuple[ValidationIssue, ...]:
    if not isinstance(value, str) or not value.strip():
        return (ValidationIssue(f"prompt.blocks.{name}.value", "must be non-empty text"),)
    return ()


@dataclass(frozen=True, slots=True)
class IdentityBlock(PromptBlock[str]):
    """Who this agent is, in the same label space as the social sources."""

    name: str = field(init=False, default="identity")
    title: str = field(init=False, default="Focal identity")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: str | Unbound = UNBOUND
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="dynamic")

    def value_issues(self, value: str) -> tuple[ValidationIssue, ...]:
        return _text_issues("identity", value)

    def render(self) -> str:
        return (
            f"You are {self.value} participating in a group decision.\n"
            "\n"
            "Your goal is to identify the correct option."
        )


@dataclass(frozen=True, slots=True)
class SocialEnvironmentBlock(PromptBlock[str]):
    """The Strategic-Uncertainty text, as a *fixed* block.

    `binding="fixed"` is the invariance guarantee: the value cannot be rebound
    per cell, and it is part of the prompt definition hash, so a run whose
    social environment differed from another run's would have a different
    definition hash rather than a silently different prompt.
    """

    name: str = field(init=False, default="social_environment")
    title: str = field(init=False, default="Social environment")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: str | Unbound = SOCIAL_ENVIRONMENT
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="fixed")

    def value_issues(self, value: str) -> tuple[ValidationIssue, ...]:
        return _text_issues("social_environment", value)

    def render(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class DecisionBasisBlock(PromptBlock[str]):
    """What the agent is told to decide from, and that its ballot goes public."""

    name: str = field(init=False, default="decision_basis")
    title: str = field(init=False, default="Decision basis")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: str | Unbound = UNBOUND
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="dynamic")

    def value_issues(self, value: str) -> tuple[ValidationIssue, ...]:
        return _text_issues("decision_basis", value)

    def render(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class TaskBlock(PromptBlock[Mapping[str, Any]]):
    """The scenario plus the option alphabet the vote must resolve into."""

    name: str = field(init=False, default="task")
    title: str = field(init=False, default="Task and options")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: Mapping[str, Any] | Unbound = UNBOUND
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="dynamic")

    def value_issues(self, value: Mapping[str, Any]) -> tuple[ValidationIssue, ...]:
        if not isinstance(value, Mapping) or {"scenario", "options"} - set(value):
            return (
                ValidationIssue("prompt.blocks.task.value", "must contain scenario and options"),
            )
        options = value["options"]
        if isinstance(options, (str, bytes)) or not isinstance(options, Sequence) or not options:
            return (
                ValidationIssue("prompt.blocks.task.value.options", "must be a non-empty sequence"),
            )
        return _text_issues("task", value["scenario"])

    def render(self) -> str:
        options = "\n".join(f"- {option}" for option in self.value["options"])  # type: ignore[index]
        scenario = str(self.value["scenario"])  # type: ignore[index]
        if all(str(option) in scenario for option in self.value["options"]):  # type: ignore[index]
            return f"TASK\n\n{scenario}"
        return (
            "TASK\n"
            "\n"
            f"{scenario}\n"
            "\n"
            "The available options are:\n"
            "\n"
            f"{options}\n"
            "\n"
            "Exactly one of these options is correct."
        )


@dataclass(frozen=True, slots=True)
class PrivateInformationBlock(PromptBlock[tuple[str, ...]]):
    """`E_i` - this agent's own evidence, and nobody else's business.

    `sensitive=True` keeps it out of shared audit records, exactly as in the
    parent game: an audit log must not become the channel that leaks the
    information the experiment is measuring the surfacing of.
    """

    name: str = field(init=False, default="private_information")
    title: str = field(init=False, default="Private information")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: tuple[str, ...] | Unbound = UNBOUND
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="dynamic")
    sensitive: bool = field(init=False, default=True)

    def value_issues(self, value: tuple[str, ...]) -> tuple[ValidationIssue, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
            return (
                ValidationIssue(
                    "prompt.blocks.private_information.value", "must be a non-empty sequence"
                ),
            )
        return ()

    def render(self) -> str:
        facts = "\n".join(f"- {fact}" for fact in self.value)  # type: ignore[union-attr]
        return f"YOUR PRIVATE INFORMATION\n\n{facts}"


@dataclass(frozen=True, slots=True)
class CurrentPositionBlock(PromptBlock[Mapping[str, Any]]):
    """`(X_i^t, R_i^t)` - the agent's own standing public ballot.

    `required=False`: at the first local vote there is no standing ballot yet,
    and an unbound optional block is omitted from the compiled prompt rather
    than rendered empty, which is what lets one family serve both calls.
    """

    name: str = field(init=False, default="current_position")
    title: str = field(init=False, default="Current public position")
    role: MessageRole = field(init=False, default=MessageRole.USER)
    value: Mapping[str, Any] | Unbound = UNBOUND
    required: bool = field(init=False, default=False)
    binding: str = field(init=False, default="dynamic")

    def value_issues(self, value: Mapping[str, Any]) -> tuple[ValidationIssue, ...]:
        if not isinstance(value, Mapping) or "vote" not in value:
            return (
                ValidationIssue("prompt.blocks.current_position.value", "must contain a vote"),
            )
        return ()

    def render(self) -> str:
        reason = self.value.get("reason") or NO_PREVIOUS_REASON  # type: ignore[union-attr]
        return (
            "YOUR CURRENT PUBLIC POSITION\n"
            "\n"
            f"Vote: {self.value['vote']}\n"  # type: ignore[index]
            f"Reason: {reason}"
        )


@dataclass(frozen=True, slots=True)
class SocialInformationBlock(PromptBlock[tuple[str, ...]]):
    """Exactly `q` already-rendered social sources, in scheduler slot order.

    The block takes rendered text rather than source records so that what a
    controlled and an uncontrolled slot look like is decided in exactly one
    place - `render_social_source` - and cannot drift apart here.
    """

    name: str = field(init=False, default="social_information")
    title: str = field(init=False, default="Current social information")
    role: MessageRole = field(init=False, default=MessageRole.USER)
    value: tuple[str, ...] | Unbound = UNBOUND
    required: bool = field(init=False, default=False)
    binding: str = field(init=False, default="dynamic")

    def value_issues(self, value: tuple[str, ...]) -> tuple[ValidationIssue, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
            return (
                ValidationIssue(
                    "prompt.blocks.social_information.value", "must be a non-empty sequence"
                ),
            )
        return ()

    def render(self) -> str:
        sources = "\n\n".join(str(item) for item in self.value)  # type: ignore[union-attr]
        return f"CURRENT SOCIAL INFORMATION\n\n{sources}"


# --------------------------------------------------------------------------
# Response contract
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PublicBallotContract(ResponseContract):
    """`{"vote": ..., "reason": ...}` and nothing else.

    Both fields are required, because both are state: the vote is applied
    immediately as the agent's committed action, and the reason is what the
    next focal agent that draws this one as a social source will read.
    """

    type: str = "hidden_bench_public_ballot"

    def instruction(self) -> str:
        options = " | ".join(self.allowed_values)
        return (
            f"{DECISION_INSTRUCTION}\n"
            "\n"
            "Return only valid JSON:\n"
            "\n"
            "{\n"
            f'  "vote": "<{options}>",\n'
            '  "reason": "<brief public reason>"\n'
            "}"
        )

    def instruction_messages(self) -> tuple[tuple[MessageRole, str], ...]:
        return ((MessageRole.USER, self.instruction()),)

    def validate(self, response: str) -> ValidationResult:
        parsed = extract_json_object(response)
        if parsed is None:
            return ValidationResult.failure(
                ValidationIssue(
                    "response", "must contain a JSON object with vote and reason", response
                )
            )
        if normalize_vote(parsed.get("vote"), self.allowed_values) is None:
            return ValidationResult.failure(
                ValidationIssue(
                    "response.vote",
                    f"must resolve to exactly one of: {', '.join(self.allowed_values)}",
                    parsed.get("vote"),
                )
            )
        reason = parsed.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            return ValidationResult.failure(
                ValidationIssue("response.reason", "must be non-empty text")
            )
        if len(reason.strip()) > MAX_REASON_CHARACTERS:
            return ValidationResult.failure(
                ValidationIssue(
                    "response.reason",
                    f"must be at most {MAX_REASON_CHARACTERS} characters",
                )
            )
        return ValidationResult.success()


@dataclass(frozen=True, slots=True)
class ParsedBallot:
    """One parsed response.  `vote is None` means it did not resolve."""

    vote: str | None
    reason: str | None
    raw_vote: Any


def parse_public_ballot_update(
    response: str, possible_answers: Sequence[str]
) -> ParsedBallot:
    parsed = extract_json_object(response)
    raw_vote = parsed.get("vote") if parsed else None
    raw_reason = parsed.get("reason") if parsed else None
    reason = raw_reason.strip() if isinstance(raw_reason, str) and raw_reason.strip() else None
    return ParsedBallot(
        vote=normalize_vote(raw_vote, tuple(possible_answers)),
        reason=reason,
        raw_vote=raw_vote,
    )


# --------------------------------------------------------------------------
# Full prompt
# --------------------------------------------------------------------------


class PublicBallotPrompt(FullPrompt):
    def concrete_prompt_type(self) -> str:
        return PROMPT_FAMILY


def hidden_bench_public_ballot_prompt(
    possible_answers: Sequence[str] = ("A", "B", "C"),
) -> PublicBallotPrompt:
    return PublicBallotPrompt(
        PROMPT_FAMILY,
        PROMPT_VERSION,
        (
            IdentityBlock(),
            SocialEnvironmentBlock(),
            DecisionBasisBlock(),
            TaskBlock(),
            PrivateInformationBlock(),
            CurrentPositionBlock(),
            SocialInformationBlock(),
        ),
        PublicBallotContract(allowed_values=tuple(possible_answers)),
    )


def build_public_ballot_update_prompt(
    *,
    identity: str,
    scenario: str,
    possible_answers: Sequence[str],
    private_information: Sequence[str],
    current_vote: str | None,
    current_reason: str | None,
    social_sources: Sequence[Mapping[str, Any]] = (),
    vote_visibility: str = "public",
) -> PublicBallotPrompt:
    """Bind one focal update, or - with no sources and no vote - one local vote."""

    prompt = hidden_bench_public_ballot_prompt(possible_answers).bind(
        identity=identity,
        decision_basis=(
            DECISION_BASIS_SOCIAL if social_sources else DECISION_BASIS_INITIAL
        ),
        task={"scenario": sanitize_scenario(scenario), "options": tuple(possible_answers)},
        private_information=tuple(private_information),
    )
    if current_vote is not None:
        prompt = prompt.bind(
            current_position={"vote": current_vote, "reason": current_reason}
        )
    if social_sources:
        prompt = prompt.bind(
            social_information=render_social_sources(
                social_sources, vote_visibility=vote_visibility
            )
        )
    return prompt  # type: ignore[return-value]


__all__ = [
    "DECISION_BASIS_INITIAL",
    "DECISION_BASIS_SOCIAL",
    "DECISION_INSTRUCTION",
    "IMPLEMENTED_VOTE_VISIBILITIES",
    "MAX_REASON_CHARACTERS",
    "NO_PREVIOUS_REASON",
    "PROMPT_FAMILY",
    "PROMPT_VERSION",
    "SOCIAL_ENVIRONMENT",
    "VOTE_VISIBILITIES",
    "ParsedBallot",
    "PublicBallotContract",
    "PublicBallotPrompt",
    "agent_label",
    "build_public_ballot_update_prompt",
    "control_label",
    "hidden_bench_public_ballot_prompt",
    "parse_public_ballot_update",
    "render_control_reason",
    "render_social_source",
    "render_social_sources",
    "sanitize_scenario",
]
