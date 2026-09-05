"""One-call public-ballot prompts for the relational reasoning game.

**One focal update = one LLM call = one vote + one private reason + at most
one publicly exposed fact.**

This is a new prompt family rather than a rebinding of HiddenBench's: the task
is a distributed symbolic proof, not a hidden-profile scenario, and the ballot
carries a third field that HiddenBench has no concept of.  That field is the
whole point of the family:

    {"vote": ..., "reason": ..., "shared_fact_id": ...}

``reason`` is free-form natural language, and it is **private**: see below.  ``shared_fact_id`` is machine
readable, so "which piece of evidence entered the conversation, when, and who
heard it" is a recorded fact rather than something an NLP pass has to guess
afterwards.  ``"none"`` is always a legal answer: an agent is never forced to
disclose, or the distributed-information problem would dissolve on the first
round.

In board mode, public prose is an intentional semantic channel. Exact evidence
memory still changes only when a REPORT carries ``shared_fact_id``. A semantic-
only REPORT may affect a later vote, but it does not add anything to the exact
historical or active evidence sets.

An agent sees its own facts *with their identifiers* - it needs them to cite one
- and sees a peer's exposed fact as **rendered text only**.  There is no place
in this family where a raw symbolic tuple reaches a model: the experiment is a
language reasoning task and the symbols stay in the log.

Peer mode retains the older structured ballot channel. Board mode uses
``render_board_message`` for REQUEST, REPORT, and controller DIRECTIVE prose.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
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

from ...hidden_bench.vanilla.prompts import extract_json_object
from ..data import NO_FACT

PROMPT_FAMILY = "relational_public_ballot"
BOARD_PROMPT_FAMILY = "relational_blackboard_ballot"
PROMPT_VERSION = 1
BOARD_PROMPT_VERSION = 3
BOARD_PROMPT_VERSIONS = (2, 3, 4)

VOTE_VISIBILITIES = ("public", "hidden")
IMPLEMENTED_VOTE_VISIBILITIES = ("public",)
"""``hidden`` is reserved: the renderer supports dropping a source's vote, but
the surrounding text still tells the agent its vote is what others see, so
enabling it is a text decision that has not been made.  It is also nearly empty
now that prose is not shown either - a hidden-vote source would carry its
identity and its exposed fact and nothing more."""

# The prompt states this number, so an agent is never failed for breaking a
# rule it was not given.  600 rejected ordinary well-formed reasoning.
MAX_REASON_CHARACTERS = 2000
"""A generous cap on "at most three sentences".  It exists to reject a runaway
essay - which truncates at ``max_output_tokens`` and breaks the JSON anyway -
not to police wording."""

BOARD_MESSAGE_TYPES = (
    "REQUEST",
    "REPORT",
)
BOARD_ACTION_TYPES = (*BOARD_MESSAGE_TYPES, "NONE")
MAX_PUBLIC_MESSAGE_CHARACTERS = 1200

LOCAL_PROMPT_P0 = "P0"
LOCAL_PROMPT_P1 = "P1"
LOCAL_PROMPT_P2 = "P2"
LOCAL_PROMPT_P3 = "P3"
LOCAL_PROMPT_VARIANTS = (
    LOCAL_PROMPT_P0,
    LOCAL_PROMPT_P1,
    LOCAL_PROMPT_P2,
    LOCAL_PROMPT_P3,
)

RECEIVER_EPISTEMIC_DISPOSITIONS = ("naive", "vigilant")
# Deprecated import compatibility. New configuration must use the factorized
# receiver axis; these names remain only so historical prompt artifacts/tests
# can still be inspected and reproduced.
EPISTEMIC_PROMPT_CLASSES = (
    "naive",
    "distributed_information",
    "strategic_uncertainty",
    "evidence_calibrated",
)

SOCIAL_ENVIRONMENT_DISTRUST = (
    "Participant identities persist across interactions.\n"
    "\n"
    "Different participants know different facts about this problem. No single\n"
    "participant is guaranteed to know everything needed to answer, and some\n"
    "participants may have objectives that differ from yours. You do not know\n"
    "which participants, if any, have different objectives."
)
"""``social_distrust: true`` - the strategic-uncertainty environment.

This is the default, and the historical text: an agent is told that some
participants may not share its objective.  It is what makes an advocacy signal
something the population can discount rather than something it must believe."""

SOCIAL_ENVIRONMENT_DISTRIBUTED = (
    "Participant identities persist across interactions.\n"
    "\n"
    "Different participants may know different facts about this problem. No single\n"
    "participant is guaranteed to know everything needed to answer. A fact known\n"
    "by only one participant may still be important for identifying the correct\n"
    "answer. Do not assume that the participants you see have all been given the\n"
    "same information."
)
SOCIAL_ENVIRONMENT_NAIVE = "Participant identities persist across interactions."
SOCIAL_ENVIRONMENT_VIGILANT = (
    "Participant identities persist across interactions.\n"
    "\n"
    "Some participants may have objectives that differ from yours. Do not accept a\n"
    "recommendation merely because another participant gives it. Evaluate whether\n"
    "the explicit evidence you actually have supports the conclusion, and remain\n"
    "aware that a source may present only part of the relevant information."
)
BOARD_SOCIAL_ENVIRONMENT_V3 = (
    "Participant identities persist across interactions.\n"
    "\n"
    "Different participants may know different facts about this problem. No single\n"
    "participant is guaranteed to know everything needed to answer. Evaluate public\n"
    "messages by their evidence and reasoning rather than by repetition, confidence,\n"
    "participant identity, or apparent authority."
)
SOCIAL_ENVIRONMENT_EVIDENCE_CALIBRATED = (
    "Participant identities persist across interactions.\n"
    "\n"
    "Different participants may know different facts about this problem, and some\n"
    "participants may have objectives that differ from yours. Do not treat agreement\n"
    "or majority support by itself as evidence that an answer is correct. Base your\n"
    "decision on the explicit facts available to you and on whether those facts\n"
    "support the conclusion. At the same time, do not dismiss a minority position\n"
    "merely because few participants hold it: a fact known by only one participant\n"
    "may be decisive. You do not know which participants, if any, have different\n"
    "objectives."
)

# Compatibility name for callers which selected the old non-distrust arm.
SOCIAL_ENVIRONMENT_NEUTRAL = SOCIAL_ENVIRONMENT_DISTRIBUTED

SOCIAL_ENVIRONMENT = SOCIAL_ENVIRONMENT_DISTRUST
"""Backwards-compatible alias for the default environment."""


def resolve_receiver_epistemic_disposition(
    receiver_epistemic_disposition: str | None = None,
    social_distrust: bool | None = None,
) -> str:
    """Resolve the factorized receiver axis and the deprecated boolean adapter."""

    legacy = None
    if social_distrust is not None:
        if not isinstance(social_distrust, bool):
            raise ValueError("social_distrust must be a boolean")
        legacy = "vigilant" if social_distrust else "naive"
    selected = receiver_epistemic_disposition or legacy or "vigilant"
    if selected not in RECEIVER_EPISTEMIC_DISPOSITIONS:
        raise ValueError(
            "receiver_epistemic_disposition must be one of "
            f"{list(RECEIVER_EPISTEMIC_DISPOSITIONS)}"
        )
    if legacy is not None and selected != legacy:
        raise ValueError(
            "game.options.social_distrust contradicts "
            "game.options.receiver_epistemic_disposition"
        )
    return selected


def resolve_epistemic_prompt_class(
    epistemic_prompt_class: str | None = None,
    social_distrust: bool | None = None,
) -> str:
    """Deprecated prompt-only resolver for historical artifact replay."""
    legacy = (
        None
        if social_distrust is None
        else ("strategic_uncertainty" if social_distrust else "distributed_information")
    )
    selected = epistemic_prompt_class or legacy or "strategic_uncertainty"
    if selected not in EPISTEMIC_PROMPT_CLASSES:
        raise ValueError(
            f"epistemic_prompt_class must be one of {list(EPISTEMIC_PROMPT_CLASSES)}"
        )
    if legacy is not None and selected != legacy:
        raise ValueError(
            "game.options.social_distrust contradicts game.options.epistemic_prompt_class"
        )
    return selected


def epistemic_framing(receiver_epistemic_disposition: str = "vigilant") -> str:
    """Return the sole fixed framing component for the receiver disposition."""

    selected = resolve_receiver_epistemic_disposition(receiver_epistemic_disposition)
    return {
        "naive": SOCIAL_ENVIRONMENT_NAIVE,
        "vigilant": SOCIAL_ENVIRONMENT_VIGILANT,
    }[selected]


def social_environment(social_distrust: bool = True) -> str:
    """Deprecated compatibility adapter for the former boolean condition.

    Both variants are bound as ``fixed`` blocks, so the choice is pinned by the
    prompt definition hash rather than by convention: a run at
    ``social_distrust: false`` has a demonstrably different prompt definition
    from one at ``true``, instead of a silently different prompt.
    """

    return (
        SOCIAL_ENVIRONMENT_DISTRUST
        if social_distrust
        else SOCIAL_ENVIRONMENT_DISTRIBUTED
    )


DECISION_BASIS_INITIAL = (
    "Make your own decision, using the facts you currently know and nothing\n"
    "else. No other participant has stated a position yet.\n"
    "\n"
    "Other participants will later see your vote and any fact you choose to\n"
    "share, and nothing else. Your reason is your own record: it is not shown\n"
    "to anyone."
)

DECISION_BASIS_SOCIAL = (
    "Make your own decision, using:\n"
    "- the facts you currently know;\n"
    "- your current position;\n"
    "- the public positions of the participants shown below.\n"
    "\n"
    "You see only the participants shown in this interaction, and of each one you\n"
    "see their vote and the fact they chose to share, if any.\n"
    "\n"
    "Other participants will see the same of you: your vote and any fact you\n"
    "choose to share, and nothing else. Your reason is your own record: it is\n"
    "not shown to anyone."
)

DECISION_BASIS_SOCIAL_NONE_VISIBLE = (
    "Make your own decision, using:\n"
    "- the facts you currently know;\n"
    "- your current position.\n"
    "\n"
    "No other participant's position is visible to you in this interaction.\n"
    "\n"
    "Other participants will see your vote and any fact you choose to share,\n"
    "and nothing else. Your reason is your own record: it is not shown to\n"
    "anyone."
)
"""A focal update whose social slots are all occluded (``message_mode: silent``).

Distinct from ``DECISION_BASIS_INITIAL`` on purpose.  The initial text asserts
that *no participant has stated a position yet*, which is true only at round 0;
reusing it mid-episode would tell the occluded agent something false about the
population and confound the placebo with a claim about everyone else.  This one
says only that nothing is visible **in this interaction**, and asserts nothing
about whether other participants have positions.

It is bound like the other two - the block is ``dynamic``, so adding this value
leaves the prompt definition, its version and its hash untouched.
"""

DECISION_INSTRUCTION = (
    "DECISION\n"
    "\n"
    "Work out which option the facts available to you support, and vote for it.\n"
    "\n"
    "Your reason should briefly explain your choice, for your own record.\n"
    "\n"
    "Sharing a fact is the only way to pass information to other participants. You\n"
    "may share exactly one of the facts you currently know by giving its identifier\n"
    "in `shared_fact_id`, so that the participants who see your position can use it\n"
    'too. Use "none" if you prefer to share nothing.\n'
    "\n"
    "You may share only a fact listed under YOUR CURRENT KNOWLEDGE, by its exact\n"
    "identifier. Do not invent facts, identifiers, or relationships that were not\n"
    "given to you.\n"
    "\n"
    "Keep your reason to at most three sentences."
)

ALLOCATION_DECISION_SCAFFOLD = (
    "Evaluate each candidate allocation using the evidence available to you.\n"
    "\n"
    "For each allocation, consider:\n"
    "1. the pipeline ability of the person assigned to the pipeline;\n"
    "2. the interview ability of each person assigned to interviews;\n"
    "3. the cooperation evidence for the two-person interview team.\n"
    "\n"
    "Compare all three candidate allocations before choosing.\n"
    "\n"
    "Evidence may describe these properties indirectly.\n"
    "Do not treat missing evidence as evidence for or against an allocation.\n"
    "\n"
    "Determine your vote from the evidence first.\n"
    "Only after deciding, choose which evidence item, if any, to share."
)

BOARD_DECISION_BASIS = (
    "Make your own decision using your private evidence, your current position,\n"
    "and the temporary public-board messages shown below. You see only a small\n"
    "sample of currently live messages. Evaluate them rather than automatically\n"
    "following them. Later participants may read a public message you post. Your\n"
    "private reason is recorded for analysis and is never posted."
)

BOARD_DECISION_BASIS_NONE_VISIBLE = (
    "Make your own decision using your private evidence and current position.\n"
    "No live public-board message is visible in this update. You may still post\n"
    "one message for later participants. Your private reason is never posted."
)

BOARD_DECISION_BASIS_V3 = (
    "Make your own decision using your private evidence, your previous vote,\n"
    "and the temporary public-board messages shown below. You see only a small\n"
    "sample of currently live messages. Evaluate them rather than automatically\n"
    "following them. Later participants may read a public message you post. Your\n"
    "private reason is recorded for analysis and is never posted."
)

BOARD_DECISION_BASIS_NONE_VISIBLE_V3 = (
    "Make your own decision using your private evidence and previous vote.\n"
    "No live public-board message is visible in this update. You may still post\n"
    "one message for later participants. Your private reason is never posted."
)

BOARD_DECISION_INSTRUCTION = (
    "DECISION\n\n"
    "Vote for the option best supported by the information available to you.\n"
    "Write a brief private reason: a few sentences, and at most\n"
    f"{MAX_REASON_CHARACTERS} characters. For public_message choose exactly one type:\n"
    "- REQUEST asks for information or work. It cannot attach exact evidence.\n"
    "- REPORT shares information, an answer, a conclusion, or a correction. It\n"
    "  may attach one exact evidence identifier.\n"
    "- NONE posts nothing; use null for text, shared_fact_id, and reply_to.\n\n"
    "REQUEST and REPORT may reply to any visible message by putting that message\n"
    "ID in reply_to. Use null when the message is not a reply. A REPORT may cite\n"
    "only a fact listed under YOUR CURRENT KNOWLEDGE. Do not invent evidence or\n"
    "identifiers. Your private reason is never copied into public_message."
)

BOARD_DECISION_INSTRUCTION_V3 = (
    "DECISION\n\n"
    "Vote for the option best supported by the information available to you.\n"
    "Write a brief private reason: a few sentences, and at most\n"
    f"{MAX_REASON_CHARACTERS} characters. For public_message choose exactly one type:\n"
    "- REQUEST asks for specific missing evidence or information. Use REQUEST when\n"
    "  important evidence needed to distinguish the options is missing or ambiguous.\n"
    "  Prefer REQUEST over NONE when additional evidence could change your decision.\n"
    "  Ask for something specific rather than a generic explanation.\n"
    "  It cannot attach exact evidence.\n"
    "- REPORT shares information, an answer, a conclusion, or a correction. It may\n"
    "  attach one exact evidence identifier. When discussing an allocation in the\n"
    "  public text, describe the allocation itself rather than using option letters\n"
    "  A/B/C; your vote is transmitted separately.\n"
    "- NONE posts nothing; use null for text, shared_fact_id, and reply_to.\n\n"
    "REQUEST and REPORT may reply to any visible message by putting that message\n"
    "ID in reply_to. Use null when the message is not a reply. A REPORT may cite\n"
    "only a fact listed under YOUR VERIFIED EVIDENCE. Do not invent evidence or\n"
    "identifiers. Your private reason is never copied into public_message."
)

BOARD_DECISION_INSTRUCTION_NO_REQUEST_V4 = (
    "DECISION\n\n"
    "Vote for the option best supported by the information available to you.\n"
    "Write a brief private reason: a few sentences, and at most\n"
    f"{MAX_REASON_CHARACTERS} characters. For public_message choose exactly one type:\n"
    "- REPORT shares information, an answer, a conclusion, or a correction. It may\n"
    "  attach one exact evidence identifier. When discussing an allocation in the\n"
    "  public text, describe the allocation itself rather than using option letters\n"
    "  A/B/C; your vote is transmitted separately.\n"
    "- NONE posts nothing; use null for text, shared_fact_id, and reply_to.\n\n"
    "REPORT may reply to any visible message by putting that message ID in reply_to.\n"
    "Use null when the message is not a reply. A REPORT may cite only a fact listed\n"
    "under YOUR VERIFIED EVIDENCE. Do not invent evidence or identifiers. Your\n"
    "private reason is never copied into public_message."
)

NO_KNOWN_FACTS = (
    "You do not currently know any facts about this problem. You will have to "
    "rely on what other participants tell you."
)
EVIDENCE_HEADER = "Evidence they are sharing:"
"""How a source's exposed fact is introduced.  Identical for peers and for the
controller, so the *format* carries no signal about which one is speaking."""
VERIFIED_SHARED_FACT_HEADER = "Verified shared fact:"

CONTROL_RECOMMENDATION = "I recommend option {target}."
"""The controller's recorded reason, for the trajectory only.

Like a peer's reason it is **not rendered into any prompt**: what the focal
agent sees of the controller is its vote - which *is* the recommendation - and
its exposed fact.  The string exists so `controller_message` names the
intervention in the log, and it is deterministic in the target and free of task
content so it could never have carried evidence anyway."""

_FACT_ID = re.compile(r"^[A-Za-z0-9_]+$")


def agent_label(agent_id: Any) -> str:
    """``agent_007`` -> ``Agent 7``.

    The frozen tasks number their agents from one, so the label is the task's
    own number rather than an off-by-one re-derivation of it.  An id that does
    not end in digits passes through unchanged rather than being mangled.
    """

    text = str(agent_id)
    digits = re.split(r"[-_]", text)[-1]
    if not digits.isdigit():
        return text
    return f"Agent {int(digits)}"


def control_label(population_size: int) -> str:
    """The controller's persistent, ordinary-looking identity for one episode."""

    return f"Agent {int(population_size) + 1}"


def render_control_reason(target: str) -> str:
    """The controller's recorded reason: deterministic in ``Z``, and log-only.

    Kept for the trajectory, never rendered - see ``CONTROL_RECOMMENDATION``.
    """

    return CONTROL_RECOMMENDATION.format(target=target)


def render_own_fact(fact_id: str, text: str) -> str:
    """One of the focal agent's own facts, with the identifier it may cite."""

    return f"{fact_id}: {text}"


def render_social_source(
    source: Mapping[str, Any], *, vote_visibility: str = "public"
) -> str:
    """One visible participant: identity, vote, and its exposed fact.

    The source's ``reason`` is deliberately **not** rendered.  It stays in the
    record for analysis, but showing it here would open a second, untracked
    task-information channel alongside ``shared_fact_id`` - see the module
    docstring.  Peers and the controller go through this one renderer, so
    neither can acquire a prose channel the other lacks.
    """

    if vote_visibility not in VOTE_VISIBILITIES:
        raise ValueError(f"vote_visibility must be one of {list(VOTE_VISIBILITIES)}")
    label = str(source["label"])
    head = label if vote_visibility == "hidden" else f"{label}\nVote: {source['vote']}"
    evidence = source.get("shared_fact_text")
    if not evidence:
        return head
    return f"{head}\n{EVIDENCE_HEADER}\n{evidence}"


def render_social_sources(
    sources: Sequence[Mapping[str, Any]], *, vote_visibility: str = "public"
) -> tuple[str, ...]:
    return tuple(
        render_social_source(source, vote_visibility=vote_visibility)
        for source in sources
    )


def _normalized_public_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def render_board_message(
    source: Mapping[str, Any], *, vote_visibility: str = "public", version: int = 2
) -> str:
    """Render public prose plus any authoritative structured evidence text."""

    if vote_visibility not in VOTE_VISIBILITIES:
        raise ValueError(f"vote_visibility must be one of {list(VOTE_VISIBILITIES)}")
    if version not in BOARD_PROMPT_VERSIONS:
        raise ValueError(f"version must be one of {list(BOARD_PROMPT_VERSIONS)}")
    lines = [
        f"Message ID: {source['message_id']}",
        str(source["label"]),
        f"Type: {source['message_type']}",
    ]
    if vote_visibility == "public":
        vote = source["vote"]
        display = source.get("vote_display_text")
        lines.append(
            f"Current vote: {vote}"
            if not display or display == vote
            else f"Current vote: {vote} ({display})"
        )
    if source.get("reply_to"):
        lines.append(f"Reply to: {source['reply_to']}")
    public_text = str(source["text"])
    shared_fact_text = source.get("shared_fact_text")
    duplicate_shared_fact = (
        version >= 3
        and shared_fact_text
        and _normalized_public_text(public_text)
        == _normalized_public_text(shared_fact_text)
    )
    if not duplicate_shared_fact:
        lines.extend(("Public message:", public_text))
    if shared_fact_text:
        lines.extend(
            (
                VERIFIED_SHARED_FACT_HEADER if version >= 3 else EVIDENCE_HEADER,
                str(shared_fact_text),
            )
        )
    return "\n".join(lines)


def render_board_messages(
    sources: Sequence[Mapping[str, Any]],
    *,
    vote_visibility: str = "public",
    version: int = 2,
) -> tuple[str, ...]:
    return tuple(
        render_board_message(source, vote_visibility=vote_visibility, version=version)
        for source in sources
    )


# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------


def _text_issues(name: str, value: Any) -> tuple[ValidationIssue, ...]:
    if not isinstance(value, str) or not value.strip():
        return (
            ValidationIssue(f"prompt.blocks.{name}.value", "must be non-empty text"),
        )
    return ()


@dataclass(frozen=True, slots=True)
class IdentityBlock(PromptBlock[str]):
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
            f"You are {self.value}, one participant in a group reasoning problem.\n"
            "\n"
            "Your goal is to identify the correct answer."
        )


@dataclass(frozen=True, slots=True)
class SocialEnvironmentBlock(PromptBlock[str]):
    """The distributed-information environment, as a *fixed* block.

    Its value is chosen at construction rather than bound afterwards, because a
    ``fixed`` block cannot be rebound once it holds a value - which is the whole
    point of ``fixed``.  See :func:`social_environment`.
    """

    name: str = field(init=False, default="social_environment")
    title: str = field(init=False, default="Social environment")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: str | Unbound = SOCIAL_ENVIRONMENT_DISTRUST
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="fixed")

    def value_issues(self, value: str) -> tuple[ValidationIssue, ...]:
        return _text_issues("social_environment", value)

    def render(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class DecisionBasisBlock(PromptBlock[str]):
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
class DecisionScaffoldBlock(PromptBlock[str]):
    """Optional MuSR allocation-comparison instructions for local decisions."""

    name: str = field(init=False, default="decision_scaffold")
    title: str = field(init=False, default="Allocation comparison scaffold")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: str | Unbound = ALLOCATION_DECISION_SCAFFOLD
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="fixed")

    def value_issues(self, value: str) -> tuple[ValidationIssue, ...]:
        return _text_issues("decision_scaffold", value)

    def render(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class TaskBlock(PromptBlock[Mapping[str, Any]]):
    """The rendered question and the labelled answer alphabet."""

    name: str = field(init=False, default="task")
    title: str = field(init=False, default="Question and options")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: Mapping[str, Any] | Unbound = UNBOUND
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="dynamic")

    def value_issues(self, value: Mapping[str, Any]) -> tuple[ValidationIssue, ...]:
        if not isinstance(value, Mapping) or {"question", "options"} - set(value):
            return (
                ValidationIssue(
                    "prompt.blocks.task.value", "must contain question and options"
                ),
            )
        options = value["options"]
        if (
            isinstance(options, (str, bytes))
            or not isinstance(options, Sequence)
            or not options
        ):
            return (
                ValidationIssue(
                    "prompt.blocks.task.value.options", "must be a non-empty sequence"
                ),
            )
        return _text_issues("task", value["question"])

    def render(self) -> str:
        options = "\n".join(f"- {option}" for option in self.value["options"])  # type: ignore[index]
        return (
            "QUESTION\n"
            "\n"
            f"{self.value['question']}\n"  # type: ignore[index]
            "\n"
            "The available answers are:\n"
            "\n"
            f"{options}\n"
            "\n"
            "Exactly one of these answers is correct. Vote by its letter."
        )


@dataclass(frozen=True, slots=True)
class KnownFactsBlock(PromptBlock[tuple[str, ...]]):
    """``K_i(t)`` - the facts this agent currently knows, with their ids.

    ``sensitive=True`` for the same reason as HiddenBench's private block: an
    audit record must not become the channel that leaks the information whose
    propagation the experiment is measuring.

    Bound with an empty tuple when the agent knows nothing, which is a normal
    state in these tasks - most agents start with no facts at all.
    """

    name: str = field(init=False, default="known_facts")
    title: str = field(init=False, default="Current knowledge")
    role: MessageRole = field(init=False, default=MessageRole.SYSTEM)
    value: tuple[str, ...] | Unbound = UNBOUND
    required: bool = field(init=False, default=True)
    binding: str = field(init=False, default="dynamic")
    sensitive: bool = field(init=False, default=True)

    def value_issues(self, value: tuple[str, ...]) -> tuple[ValidationIssue, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            return (
                ValidationIssue(
                    "prompt.blocks.known_facts.value",
                    "must be a sequence of rendered facts",
                ),
            )
        return ()

    def render(self) -> str:
        if self.version >= 2:
            if not self.value:
                return (
                    f"YOUR VERIFIED EVIDENCE\n\n{NO_KNOWN_FACTS}\n\n"
                    "Facts under YOUR VERIFIED EVIDENCE and any VERIFIED SHARED FACT\n"
                    "are verified task evidence. A participant's REPORT text is their\n"
                    "interpretation of the available information and should be evaluated\n"
                    "accordingly."
                )
            facts = "\n".join(f"- {fact}" for fact in self.value)  # type: ignore[union-attr]
            return (
                "YOUR VERIFIED EVIDENCE\n"
                "\n"
                "These are verified task facts you know. The identifier before each fact\n"
                "is how you refer to it if you decide to share it. Facts under YOUR\n"
                "VERIFIED EVIDENCE and any VERIFIED SHARED FACT are verified task\n"
                "evidence. A participant's REPORT text is their interpretation of the\n"
                "available information and should be evaluated accordingly.\n"
                "\n"
                f"{facts}"
            )
        if not self.value:
            return f"YOUR CURRENT KNOWLEDGE\n\n{NO_KNOWN_FACTS}"
        facts = "\n".join(f"- {fact}" for fact in self.value)  # type: ignore[union-attr]
        return (
            "YOUR CURRENT KNOWLEDGE\n"
            "\n"
            "These are the facts you know. The identifier before each fact is how you\n"
            "refer to it if you decide to share it.\n"
            "\n"
            f"{facts}"
        )


@dataclass(frozen=True, slots=True)
class CurrentPositionBlock(PromptBlock[Mapping[str, Any]]):
    """``X_i^t`` alone - the agent's own standing vote.

    Deliberately *not* the rest of its last ballot.  Feeding an agent its own
    previous free-form ``reason`` would turn prose into an uncontrolled internal
    memory channel: the agent could carry conclusions forward in text that
    nothing in the state records, and ``K_i`` would stop being the only thing
    that explains what it knows.  The reason is still generated, validated and
    logged - it just never comes back in.

    The previously exposed fact is not rendered either: it is already listed in
    YOUR CURRENT KNOWLEDGE, so repeating it here adds nothing.
    """

    name: str = field(init=False, default="current_position")
    title: str = field(init=False, default="Current position")
    role: MessageRole = field(init=False, default=MessageRole.USER)
    value: Mapping[str, Any] | Unbound = UNBOUND
    required: bool = field(init=False, default=False)
    binding: str = field(init=False, default="dynamic")

    def value_issues(self, value: Mapping[str, Any]) -> tuple[ValidationIssue, ...]:
        if not isinstance(value, Mapping) or "vote" not in value:
            return (
                ValidationIssue(
                    "prompt.blocks.current_position.value", "must contain a vote"
                ),
            )
        return ()

    def render(self) -> str:
        if self.version >= 2:
            return (
                "YOUR PREVIOUS VOTE\n"
                "\n"
                f"Vote: {self.value['vote']}\n"  # type: ignore[index]
                "\n"
                "You may keep or revise this vote if the information currently available\n"
                "supports a different option."
            )
        return (
            "YOUR CURRENT POSITION\n"
            "\n"
            f"Vote: {self.value['vote']}"  # type: ignore[index]
        )


@dataclass(frozen=True, slots=True)
class SocialInformationBlock(PromptBlock[tuple[str, ...]]):
    """Exactly ``q`` already-rendered participants, in scheduler slot order."""

    name: str = field(init=False, default="social_information")
    title: str = field(init=False, default="Current social information")
    role: MessageRole = field(init=False, default=MessageRole.USER)
    value: tuple[str, ...] | Unbound = UNBOUND
    required: bool = field(init=False, default=False)
    binding: str = field(init=False, default="dynamic")

    def value_issues(self, value: tuple[str, ...]) -> tuple[ValidationIssue, ...]:
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, Sequence)
            or not value
        ):
            return (
                ValidationIssue(
                    "prompt.blocks.social_information.value",
                    "must be a non-empty sequence",
                ),
            )
        return ()

    def render(self) -> str:
        sources = "\n\n".join(str(item) for item in self.value)  # type: ignore[union-attr]
        return f"CURRENT SOCIAL INFORMATION\n\n{sources}"


# --------------------------------------------------------------------------
# Response contract
# --------------------------------------------------------------------------


PRESENTATION_LETTERS = "ABCDEFGH"
"""The letters a call may use, in order.  Which relation each one names is
decided per call, never globally - see ``shuffled_option_letters``."""


def shuffled_option_letters(answers: Sequence[str], rng: Any) -> dict[str, str]:
    """A fresh ``letter -> relation`` map for one LLM call.

    The population state is semantic, so the letters are pure presentation and
    may - must - differ between calls.  Drawing them from a seeded stream keeps
    the run reproducible while making a globally shared "vote B" attractor
    impossible: B means something different to the next agent asked.
    """

    relations = list(answers)
    if len(relations) > len(PRESENTATION_LETTERS):
        raise ValueError(
            f"at most {len(PRESENTATION_LETTERS)} answer options are presentable"
        )
    rng.shuffle(relations)
    return {
        PRESENTATION_LETTERS[index]: relation
        for index, relation in enumerate(relations)
    }


def relation_to_letter(option_letters: Mapping[str, str]) -> dict[str, str]:
    """Invert one call's presentation map."""

    return {relation: letter for letter, relation in option_letters.items()}


def localize_sources(
    sources: Sequence[Mapping[str, Any]],
    option_letters: Mapping[str, str],
    answer_display_texts: Mapping[str, str] | None = None,
    *,
    require_semantic_votes: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Restate each source's semantic vote in *this* call's letter space.

    A social block that showed one agent's letters to another would be
    incoherent - the letters are per call. The stored source records keep the
    semantic vote; only the rendered copy is localized.
    """

    inverse = relation_to_letter(option_letters)
    localized = []
    for source in sources:
        vote = source.get("vote")
        if require_semantic_votes and vote is not None and str(vote) not in inverse:
            raise ValueError(
                "board message vote must be a semantic answer present in the "
                "recipient option mapping"
            )
        localized_vote = inverse.get(str(vote), vote) if vote else vote
        localized.append(
            {
                **dict(source),
                "vote": localized_vote,
                "vote_display_text": (
                    None
                    if vote is None
                    else (answer_display_texts or {}).get(str(vote), str(vote))
                ),
            }
        )
    return tuple(localized)


def normalize_relational_vote(value: Any, option_labels: Sequence[str]) -> str | None:
    """Resolve a model's ``vote`` to one option label, or ``None``.

    Exact match on the label first, then on the compass relation the label
    stands for, then a narrow ``option B`` / ``B)`` pattern.  There is
    deliberately **no** substring fallback: single-letter labels would make one
    ("the answer is a bit north of...") match almost any sentence, and a vote
    silently resolved to the wrong option is worse than a retry.

    ``option_labels`` is the vote alphabet.  ``option_relations`` is looked up
    through :func:`relation_vote_aliases`, which the contract binds per task.
    """

    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip()).upper()
    if not text:
        return None
    for label in option_labels:
        if text == str(label).upper():
            return label
    match = re.fullmatch(r"(?:OPTION\s*)?([A-Z])[\.\)]?", text)
    if match is not None:
        for label in option_labels:
            if match.group(1) == str(label).upper():
                return label
    return None


def relation_vote_aliases(option_relations: Mapping[str, str]) -> dict[str, str]:
    """``{"NORTHEAST": "B", ...}`` - relation names accepted as votes."""

    return {
        str(relation).upper(): str(label)
        for label, relation in option_relations.items()
    }


def resolve_vote(
    value: Any, option_labels: Sequence[str], aliases: Mapping[str, str]
) -> str | None:
    """:func:`normalize_relational_vote`, then the relation-name aliases."""

    label = normalize_relational_vote(value, option_labels)
    if label is not None:
        return label
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip()).upper()
    return aliases.get(text)


def normalize_shared_fact_id(value: Any) -> str | None:
    """``"none"``/``""``/``None`` all mean "shared nothing"; otherwise the id."""

    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == NO_FACT:
        return None
    return text


@dataclass(frozen=True, slots=True)
class RelationalBallotContract(ResponseContract):
    """``{"vote", "reason", "shared_fact_id"}`` and nothing else.

    All three fields are state, but only two of them are *shared* state.  The
    vote becomes the committed action and ``shared_fact_id`` decides whether one
    fact travels to whoever draws this agent as a social source; the reason is
    recorded for analysis and shown to nobody, not even its own author on a
    later turn.

    The contract checks that a cited id is *a fact of this task* - a per-task
    constant, so the prompt definition hash is stable within a task.  Whether
    the citing agent actually **knows** it is checked by the game's
    ``validate_action``, which is the only place ``K_i(t)`` is in scope; both
    failures feed the same retry loop.
    """

    type: str = "relational_public_ballot"

    @property
    def fact_ids(self) -> tuple[str, ...]:
        """The ids this agent may cite - ``K_i(t)``, not the whole task.

        Advertising every fact in the task invited the model to cite one it does
        not hold, which is a guaranteed retry.  Because the contract is part of
        the prompt *definition*, this makes the definition hash vary per agent;
        that is correct - two agents with different knowledge really are being
        asked different questions - and the instance hash still separates
        repeated asks of the same agent.
        """

        return tuple(str(item) for item in self.options.get("fact_ids", ()))

    @property
    def relations(self) -> tuple[str, ...]:
        """The task's semantic answers, accepted as votes spelled out in full.

        A per-task constant, deliberately *not* this call's letter map: the
        contract is part of the prompt definition, and letting a per-call
        shuffle into it would give every single call its own definition hash.
        The letter->relation resolution happens in ``Game.parse_action``, which
        holds the map for that one call.
        """

        return tuple(str(item) for item in self.options.get("relations", ()))

    def instruction(self) -> str:
        options = " | ".join(self.allowed_values)
        # `none` is always last and always present: an agent that knows nothing
        # is shown `"<none>"` rather than an empty alternation.
        citable = " | ".join((*self.fact_ids, NO_FACT))
        return (
            f"{DECISION_INSTRUCTION}\n"
            "\n"
            "Return only valid JSON:\n"
            "\n"
            "{\n"
            f'  "vote": "<{options}>",\n'
            f'  "reason": "<a few sentences, at most {MAX_REASON_CHARACTERS} characters>",\n'
            f'  "shared_fact_id": "<{citable}>"\n'
            "}"
        )

    def instruction_messages(self) -> tuple[tuple[MessageRole, str], ...]:
        return ((MessageRole.USER, self.instruction()),)

    def validate(self, response: str) -> ValidationResult:
        parsed = extract_json_object(response, required_keys=("vote",))
        if parsed is None:
            return ValidationResult.failure(
                ValidationIssue(
                    "response",
                    "must contain a JSON object with vote, reason and shared_fact_id",
                    response,
                )
            )
        aliases = {relation.upper(): relation for relation in self.relations}
        if resolve_vote(parsed.get("vote"), self.allowed_values, aliases) is None:
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
        if "shared_fact_id" not in parsed:
            return ValidationResult.failure(
                ValidationIssue(
                    "response.shared_fact_id",
                    f'must be a fact identifier or "{NO_FACT}"',
                )
            )
        raw = parsed.get("shared_fact_id")
        if raw is not None and not isinstance(raw, str):
            return ValidationResult.failure(
                ValidationIssue("response.shared_fact_id", "must be a string", raw)
            )
        shared = normalize_shared_fact_id(raw)
        if shared is None:
            return ValidationResult.success()
        if not _FACT_ID.match(shared):
            return ValidationResult.failure(
                ValidationIssue(
                    "response.shared_fact_id",
                    "must be a bare fact identifier such as f1",
                    raw,
                )
            )
        if shared not in self.fact_ids:
            # No `if self.fact_ids` guard: an agent that knows nothing may cite
            # nothing, and silently accepting a citation from it here would push
            # the whole check onto the one safeguard downstream.
            return ValidationResult.failure(
                ValidationIssue(
                    "response.shared_fact_id",
                    f"{shared!r} is not among the facts this agent may share",
                    raw,
                )
            )
        return ValidationResult.success()

    def repair_guidance(self, issues: Sequence[ValidationIssue]) -> str:
        issue = issues[0]
        if issue.field == "response.shared_fact_id":
            allowed = ", ".join(f'"{value}"' for value in (*self.fact_ids, NO_FACT))
            return (
                "Your previous response was invalid:\n"
                "shared_fact_id must be a bare fact identifier.\n\n"
                "Return the complete JSON object again. shared_fact_id must be exactly one of:\n"
                f"{allowed}\n\n"
                "Do not include a label, fact text, punctuation, or explanation in that field."
            )
        # ``dataclass(slots=True)`` creates a replacement class object.  A
        # zero-argument ``super()`` can retain the pre-replacement ``__class__``
        # cell and fail at runtime while constructing a validation retry.
        return ResponseContract.repair_guidance(self, issues)


@dataclass(frozen=True, slots=True)
class ParsedBallot:
    """One parsed response.  ``vote is None`` means it did not resolve."""

    vote: str | None
    reason: str | None
    shared_fact_id: str | None
    raw_vote: Any
    raw_shared_fact_id: Any
    shared_fact_present: bool
    public_message: Mapping[str, Any] | None = None
    public_message_present: bool = False


def parse_relational_ballot(
    response: str,
    option_labels: Sequence[str],
    option_relations: Mapping[str, str] | None = None,
) -> ParsedBallot:
    parsed = extract_json_object(response, required_keys=("vote",))
    raw_vote = parsed.get("vote") if parsed else None
    raw_reason = parsed.get("private_reason", parsed.get("reason")) if parsed else None
    public = (
        dict(parsed["public_message"])
        if parsed and isinstance(parsed.get("public_message"), Mapping)
        else None
    )
    raw_shared = (
        public.get("shared_fact_id")
        if public is not None
        else parsed.get("shared_fact_id")
        if parsed
        else None
    )
    reason = (
        raw_reason.strip()
        if isinstance(raw_reason, str) and raw_reason.strip()
        else None
    )
    aliases = relation_vote_aliases(option_relations or {})
    return ParsedBallot(
        vote=resolve_vote(raw_vote, tuple(option_labels), aliases),
        reason=reason,
        shared_fact_id=(
            normalize_shared_fact_id(raw_shared)
            if isinstance(raw_shared, str)
            else None
        ),
        raw_vote=raw_vote,
        raw_shared_fact_id=raw_shared,
        shared_fact_present=(
            "shared_fact_id" in public
            if public is not None
            else bool(parsed) and "shared_fact_id" in parsed
        ),
        public_message=public,
        public_message_present=bool(parsed) and "public_message" in parsed,
    )


@dataclass(frozen=True, slots=True)
class BlackboardBallotContract(RelationalBallotContract):
    """A vote and private reason plus zero or one typed public message."""

    type: str = "relational_blackboard_ballot"

    @property
    def visible_message_ids(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.options.get("visible_message_ids", ()))

    @property
    def allowed_message_types(self) -> tuple[str, ...]:
        return tuple(
            str(item)
            for item in self.options.get("allowed_message_types", BOARD_ACTION_TYPES)
        )

    def instruction(self) -> str:
        options = " | ".join(self.allowed_values)
        citable = " | ".join((*self.fact_ids, NO_FACT))
        visible = " | ".join(self.visible_message_ids) or "none"
        message_types = " | ".join(self.allowed_message_types)
        return (
            f"{BOARD_DECISION_INSTRUCTION}\n\n"
            f"Visible message IDs: {visible}\n\n"
            "Return only valid JSON:\n\n"
            "{\n"
            f'  "vote": "<{options}>",\n'
            f'  "private_reason": "<a few sentences, at most {MAX_REASON_CHARACTERS} characters>",\n'
            '  "public_message": {\n'
            f'    "type": "<{message_types}>",\n'
            '    "text": "<public text or null>",\n'
            f'    "shared_fact_id": "<{citable}> or null,\n'
            '    "reply_to": "<visible message ID or null>"\n'
            "  }\n"
            "}"
        )

    def validate(self, response: str) -> ValidationResult:
        parsed = extract_json_object(response, required_keys=("vote",))
        if parsed is None:
            return ValidationResult.failure(
                ValidationIssue(
                    "response",
                    "must contain a JSON object with vote, private_reason and public_message",
                    response,
                )
            )
        aliases = {relation.upper(): relation for relation in self.relations}
        if resolve_vote(parsed.get("vote"), self.allowed_values, aliases) is None:
            return ValidationResult.failure(
                ValidationIssue(
                    "response.vote",
                    f"must resolve to exactly one of: {', '.join(self.allowed_values)}",
                    parsed.get("vote"),
                )
            )
        reason = parsed.get("private_reason")
        if not isinstance(reason, str) or not reason.strip():
            return ValidationResult.failure(
                ValidationIssue("response.private_reason", "must be non-empty text")
            )
        if len(reason.strip()) > MAX_REASON_CHARACTERS:
            return ValidationResult.failure(
                ValidationIssue(
                    "response.private_reason",
                    f"must be at most {MAX_REASON_CHARACTERS} characters",
                )
            )
        if "public_message" not in parsed:
            return ValidationResult.failure(
                ValidationIssue(
                    "response.public_message",
                    "must be present and use type NONE to post nothing",
                )
            )
        public = parsed.get("public_message")
        if not isinstance(public, Mapping):
            return ValidationResult.failure(
                ValidationIssue("response.public_message", "must be an object")
            )
        if "shared_fact_id" not in public:
            return ValidationResult.failure(
                ValidationIssue(
                    "response.public_message.shared_fact_id",
                    "must be present; use null when no exact evidence is attached",
                )
            )
        if "reply_to" not in public:
            return ValidationResult.failure(
                ValidationIssue(
                    "response.public_message.reply_to",
                    "must be present; use null when this is not a reply",
                )
            )
        message_type = public.get("type")
        if message_type not in self.allowed_message_types:
            return ValidationResult.failure(
                ValidationIssue(
                    "response.public_message.type",
                    f"must be one of {list(self.allowed_message_types)}",
                    message_type,
                )
            )
        text = public.get("text")
        raw_shared = public.get("shared_fact_id")
        if raw_shared is not None and not isinstance(raw_shared, str):
            return ValidationResult.failure(
                ValidationIssue(
                    "response.public_message.shared_fact_id",
                    "must be a string or null",
                    raw_shared,
                )
            )
        shared = normalize_shared_fact_id(raw_shared)
        reply_to = public.get("reply_to")
        if message_type == "NONE":
            if text is not None or shared is not None or reply_to is not None:
                return ValidationResult.failure(
                    ValidationIssue(
                        "response.public_message",
                        "NONE requires null text, shared_fact_id, and reply_to",
                    )
                )
            return ValidationResult.success()
        if not isinstance(text, str) or not text.strip():
            return ValidationResult.failure(
                ValidationIssue(
                    "response.public_message.text", "must be non-empty text"
                )
            )
        if len(text.strip()) > MAX_PUBLIC_MESSAGE_CHARACTERS:
            return ValidationResult.failure(
                ValidationIssue(
                    "response.public_message.text",
                    f"must be at most {MAX_PUBLIC_MESSAGE_CHARACTERS} characters",
                )
            )
        if reply_to is not None and (
            not isinstance(reply_to, str) or reply_to not in self.visible_message_ids
        ):
            return ValidationResult.failure(
                ValidationIssue(
                    "response.public_message.reply_to",
                    "must be null or name a message visible in this update",
                    reply_to,
                )
            )
        if message_type == "REQUEST" and shared is not None:
            return ValidationResult.failure(
                ValidationIssue(
                    "response.public_message.shared_fact_id",
                    "REQUEST cannot attach exact evidence",
                    raw_shared,
                )
            )
        if shared is not None:
            if not _FACT_ID.match(shared):
                return ValidationResult.failure(
                    ValidationIssue(
                        "response.public_message.shared_fact_id",
                        "must be a bare fact identifier such as f1",
                        raw_shared,
                    )
                )
            if shared not in self.fact_ids:
                return ValidationResult.failure(
                    ValidationIssue(
                        "response.public_message.shared_fact_id",
                        f"{shared!r} is not among the facts this agent may share",
                        raw_shared,
                    )
                )
        return ValidationResult.success()

    def repair_guidance(self, issues: Sequence[ValidationIssue]) -> str:
        issue = issues[0]
        if issue.field in {
            "response.public_message.shared_fact_id",
            "response.public_message",
        }:
            return (
                "Your previous public_message used an invalid shared_fact_id.\n\n"
                "Return a complete replacement JSON object. For this correction, "
                "set public_message.shared_fact_id to null. Do not repeat the "
                "previous identifier and do not substitute another identifier. "
                "Keep every other required field in the original response schema."
            )
        return RelationalBallotContract.repair_guidance(self, issues)


@dataclass(frozen=True, slots=True)
class BlackboardBallotContractV3(BlackboardBallotContract):
    """Version 3 guidance with the unchanged blackboard JSON contract."""

    def instruction(self) -> str:
        options = " | ".join(self.allowed_values)
        citable = " | ".join((*self.fact_ids, NO_FACT))
        visible = " | ".join(self.visible_message_ids) or "none"
        message_types = " | ".join(self.allowed_message_types)
        instruction = (
            BOARD_DECISION_INSTRUCTION_V3
            if "REQUEST" in self.allowed_message_types
            else BOARD_DECISION_INSTRUCTION_NO_REQUEST_V4
        )
        return (
            f"{instruction}\n\n"
            f"Visible message IDs: {visible}\n\n"
            "Return only valid JSON:\n\n"
            "{\n"
            f'  "vote": "<{options}>",\n'
            f'  "private_reason": "<a few sentences, at most {MAX_REASON_CHARACTERS} characters>",\n'
            '  "public_message": {\n'
            f'    "type": "<{message_types}>",\n'
            '    "text": "<public text or null>",\n'
            f'    "shared_fact_id": "<{citable}> or null,\n'
            '    "reply_to": "<visible message ID or null>"\n'
            "  }\n"
            "}"
        )


# --------------------------------------------------------------------------
# Full prompt
# --------------------------------------------------------------------------


class RelationalBallotPrompt(FullPrompt):
    def concrete_prompt_type(self) -> str:
        return PROMPT_FAMILY


class BlackboardBallotPrompt(FullPrompt):
    def concrete_prompt_type(self) -> str:
        return BOARD_PROMPT_FAMILY


def relational_public_ballot_prompt(
    possible_answers: Sequence[str] = ("A", "B", "C"),
    *,
    fact_ids: Sequence[str] = (),
    relations: Sequence[str] = (),
    receiver_epistemic_disposition: str | None = None,
    social_distrust: bool | None = None,
) -> RelationalBallotPrompt:
    """One prompt of this family.

    ``fact_ids`` are the ids the bound agent may cite - its ``K_i(t)``, not the
    task's whole alphabet.  ``social_distrust`` picks the fixed environment
    block; it defaults to ``True``, the historical text.
    """

    return RelationalBallotPrompt(
        PROMPT_FAMILY,
        PROMPT_VERSION,
        (
            IdentityBlock(),
            SocialEnvironmentBlock(
                # Explicit legacy boolean replays its historical bytes. New
                # configs omit it and use the factorized disposition text.
                (
                    SOCIAL_ENVIRONMENT_DISTRUST
                    if social_distrust
                    else SOCIAL_ENVIRONMENT_DISTRIBUTED
                )
                if social_distrust is not None
                else epistemic_framing(receiver_epistemic_disposition or "vigilant")
            ),
            DecisionBasisBlock(),
            TaskBlock(),
            KnownFactsBlock(),
            CurrentPositionBlock(),
            SocialInformationBlock(),
        ),
        RelationalBallotContract(
            allowed_values=tuple(possible_answers),
            options={
                "fact_ids": tuple(fact_ids),
                "relations": tuple(relations),
            },
        ),
    )


def relational_blackboard_ballot_prompt(
    possible_answers: Sequence[str] = ("A", "B", "C"),
    *,
    fact_ids: Sequence[str] = (),
    relations: Sequence[str] = (),
    visible_message_ids: Sequence[str] = (),
    receiver_epistemic_disposition: str | None = None,
    social_distrust: bool | None = None,
    version: int = BOARD_PROMPT_VERSION,
    allow_participant_requests: bool = True,
) -> BlackboardBallotPrompt:
    """Prompt family for finite-memory public-board updates."""

    if version not in BOARD_PROMPT_VERSIONS:
        raise ValueError(
            f"relational blackboard prompt version must be one of "
            f"{list(BOARD_PROMPT_VERSIONS)}"
        )
    if version < 4 and not allow_participant_requests:
        raise ValueError(
            "disabling participant REQUEST requires blackboard prompt version 4"
        )

    if version >= 3:
        disposition = resolve_receiver_epistemic_disposition(
            receiver_epistemic_disposition, social_distrust
        )
        environment = (
            BOARD_SOCIAL_ENVIRONMENT_V3
            if disposition == "vigilant"
            else SOCIAL_ENVIRONMENT_NAIVE
        )
    else:
        environment = (
            (
                SOCIAL_ENVIRONMENT_DISTRUST
                if social_distrust
                else SOCIAL_ENVIRONMENT_DISTRIBUTED
            )
            if social_distrust is not None
            else epistemic_framing(receiver_epistemic_disposition or "vigilant")
        )
    return BlackboardBallotPrompt(
        BOARD_PROMPT_FAMILY,
        version,
        (
            IdentityBlock(),
            SocialEnvironmentBlock(environment),
            DecisionBasisBlock(),
            TaskBlock(),
            KnownFactsBlock(version=2 if version >= 3 else 1),
            CurrentPositionBlock(version=2 if version >= 3 else 1),
            SocialInformationBlock(),
        ),
        (BlackboardBallotContractV3 if version >= 3 else BlackboardBallotContract)(
            allowed_values=tuple(possible_answers),
            options={
                "fact_ids": tuple(fact_ids),
                "relations": tuple(relations),
                "visible_message_ids": tuple(visible_message_ids),
                **(
                    {
                        "allowed_message_types": (
                            BOARD_ACTION_TYPES
                            if allow_participant_requests
                            else ("REPORT", "NONE")
                        )
                    }
                    if version >= 4
                    else {}
                ),
            },
        ),
    )


def build_relational_ballot_prompt(
    *,
    identity: str,
    question: str,
    option_letters: Mapping[str, str],
    known_facts: Sequence[str],
    fact_ids: Sequence[str],
    current_vote: str | None,
    social_sources: Sequence[Mapping[str, Any]] = (),
    vote_visibility: str = "public",
    receiver_epistemic_disposition: str | None = None,
    social_distrust: bool | None = None,
    social_context: bool = False,
    answer_display_texts: Mapping[str, str] | None = None,
    local_prompt_variant: str = LOCAL_PROMPT_P0,
) -> RelationalBallotPrompt:
    """Bind one focal update, or - with no sources and no vote - one local vote.

    ``known_facts`` are already-rendered ``"f1: Bavi is northeast of Zora."``
    lines and ``fact_ids`` are the matching ids: both describe this agent's
    ``K_i(t)`` and nothing else.  There is deliberately no parameter for the
    agent's previous reason - it is never rendered back in.

    ``option_letters`` is *this call's* ``letter -> relation`` presentation map.
    Everything letter-shaped in the prompt - the option list, the peers' votes,
    the JSON template - is built from it, so one prompt is internally
    consistent and no letter carries meaning across calls.
    """

    if local_prompt_variant not in LOCAL_PROMPT_VARIANTS:
        raise ValueError(
            f"local_prompt_variant must be one of {list(LOCAL_PROMPT_VARIANTS)}"
        )
    letters = tuple(option_letters)
    prompt = relational_public_ballot_prompt(
        letters,
        fact_ids=fact_ids,
        # The semantic alphabet in a call-independent order, so the contract -
        # and therefore the prompt definition - does not move with the shuffle.
        relations=tuple(sorted(option_letters.values())),
        receiver_epistemic_disposition=receiver_epistemic_disposition,
        social_distrust=social_distrust,
    )
    if local_prompt_variant in {LOCAL_PROMPT_P2, LOCAL_PROMPT_P3}:
        prompt = replace(
            prompt,
            blocks=tuple(
                replace(block, value=SOCIAL_ENVIRONMENT_NAIVE)
                if isinstance(block, SocialEnvironmentBlock)
                else block
                for block in prompt.blocks
            ),
        )
    if local_prompt_variant in {LOCAL_PROMPT_P1, LOCAL_PROMPT_P3}:
        blocks = list(prompt.blocks)
        insert_at = next(
            index for index, block in enumerate(blocks) if block.name == "task"
        )
        blocks.insert(insert_at, DecisionScaffoldBlock())
        prompt = replace(prompt, blocks=tuple(blocks))
    prompt = prompt.bind(
        identity=identity,
        decision_basis=(
            # No sources is ambiguous on its own: it is round 0 for everybody,
            # and it is also an occluded focal update.  `social_context` is what
            # separates them, so neither is described with the other's text.
            DECISION_BASIS_SOCIAL
            if social_sources
            else DECISION_BASIS_SOCIAL_NONE_VISIBLE
            if social_context
            else DECISION_BASIS_INITIAL
        ),
        task={
            "question": question,
            "options": tuple(
                f"{letter}) {(answer_display_texts or {}).get(option_letters[letter], option_letters[letter])}"
                for letter in letters
            ),
        },
        known_facts=tuple(known_facts),
    )
    if current_vote is not None:
        prompt = prompt.bind(current_position={"vote": current_vote})
    if social_sources:
        prompt = prompt.bind(
            social_information=render_social_sources(
                localize_sources(social_sources, option_letters, answer_display_texts),
                vote_visibility=vote_visibility,
            )
        )
    return prompt  # type: ignore[return-value]


def build_relational_blackboard_prompt(
    *,
    identity: str,
    question: str,
    option_letters: Mapping[str, str],
    known_facts: Sequence[str],
    fact_ids: Sequence[str],
    current_vote: str | None,
    board_messages: Sequence[Mapping[str, Any]] = (),
    vote_visibility: str = "public",
    receiver_epistemic_disposition: str | None = None,
    social_distrust: bool | None = None,
    social_context: bool = False,
    answer_display_texts: Mapping[str, str] | None = None,
    version: int = BOARD_PROMPT_VERSION,
    allow_participant_requests: bool = True,
) -> BlackboardBallotPrompt:
    """Bind one board update while keeping the private reason out of public text."""

    letters = tuple(option_letters)
    prompt = relational_blackboard_ballot_prompt(
        letters,
        fact_ids=fact_ids,
        relations=tuple(sorted(option_letters.values())),
        visible_message_ids=tuple(str(item["message_id"]) for item in board_messages),
        receiver_epistemic_disposition=receiver_epistemic_disposition,
        social_distrust=social_distrust,
        version=version,
        allow_participant_requests=allow_participant_requests,
    ).bind(
        identity=identity,
        decision_basis=(
            BOARD_DECISION_BASIS_V3
            if version >= 3 and board_messages
            else BOARD_DECISION_BASIS_NONE_VISIBLE_V3
            if version >= 3 and social_context
            else BOARD_DECISION_BASIS
            if board_messages
            else BOARD_DECISION_BASIS_NONE_VISIBLE
            if social_context
            else DECISION_BASIS_INITIAL
        ),
        task={
            "question": question,
            "options": tuple(
                f"{letter}) {(answer_display_texts or {}).get(option_letters[letter], option_letters[letter])}"
                for letter in letters
            ),
        },
        known_facts=tuple(known_facts),
    )
    if current_vote is not None:
        prompt = prompt.bind(current_position={"vote": current_vote})
    if board_messages:
        prompt = prompt.bind(
            social_information=render_board_messages(
                localize_sources(
                    board_messages,
                    option_letters,
                    answer_display_texts,
                    require_semantic_votes=version >= 3,
                ),
                vote_visibility=vote_visibility,
                version=version,
            )
        )
    return prompt  # type: ignore[return-value]


__all__ = [
    "ALLOCATION_DECISION_SCAFFOLD",
    "LOCAL_PROMPT_P0",
    "LOCAL_PROMPT_P1",
    "LOCAL_PROMPT_P2",
    "LOCAL_PROMPT_P3",
    "LOCAL_PROMPT_VARIANTS",
    "BOARD_PROMPT_FAMILY",
    "BOARD_PROMPT_VERSION",
    "BOARD_PROMPT_VERSIONS",
    "BOARD_MESSAGE_TYPES",
    "BOARD_ACTION_TYPES",
    "BlackboardBallotContract",
    "BlackboardBallotPrompt",
    "CONTROL_RECOMMENDATION",
    "DECISION_BASIS_INITIAL",
    "DECISION_BASIS_SOCIAL",
    "DECISION_BASIS_SOCIAL_NONE_VISIBLE",
    "DECISION_INSTRUCTION",
    "EVIDENCE_HEADER",
    "RECEIVER_EPISTEMIC_DISPOSITIONS",
    "EPISTEMIC_PROMPT_CLASSES",
    "IMPLEMENTED_VOTE_VISIBILITIES",
    "MAX_REASON_CHARACTERS",
    "NO_KNOWN_FACTS",
    "PROMPT_FAMILY",
    "PROMPT_VERSION",
    "SOCIAL_ENVIRONMENT",
    "SOCIAL_ENVIRONMENT_DISTRUST",
    "SOCIAL_ENVIRONMENT_DISTRIBUTED",
    "SOCIAL_ENVIRONMENT_EVIDENCE_CALIBRATED",
    "SOCIAL_ENVIRONMENT_NAIVE",
    "SOCIAL_ENVIRONMENT_VIGILANT",
    "SOCIAL_ENVIRONMENT_NEUTRAL",
    "VOTE_VISIBILITIES",
    "ParsedBallot",
    "RelationalBallotContract",
    "RelationalBallotPrompt",
    "PRESENTATION_LETTERS",
    "agent_label",
    "build_relational_ballot_prompt",
    "build_relational_blackboard_prompt",
    "control_label",
    "epistemic_framing",
    "localize_sources",
    "normalize_relational_vote",
    "normalize_shared_fact_id",
    "parse_relational_ballot",
    "relation_vote_aliases",
    "relational_public_ballot_prompt",
    "relational_blackboard_ballot_prompt",
    "render_board_message",
    "render_board_messages",
    "render_control_reason",
    "render_own_fact",
    "render_social_source",
    "relation_to_letter",
    "render_social_sources",
    "shuffled_option_letters",
    "resolve_vote",
    "resolve_receiver_epistemic_disposition",
    "resolve_epistemic_prompt_class",
    "social_environment",
]
