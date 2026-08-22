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

**`shared_fact_id` is the only task-information channel between participants.**
A ballot's ``reason`` is written, parsed, stored and analysed - but it is
*never* rendered into another agent's prompt.  If prose were shown to peers, an
agent could pass a fact, or a conclusion derived from one, while reporting
``shared_fact_id: none``, and information would move without appearing in any
``K_i``.  The knowledge state would then be a lower bound on what the population
actually knows rather than an exact record of it, and every epistemic
observable built on it would be unfalsifiable.  So a visible participant shows
exactly three things: who it is, what it votes, and the fact it chose to expose.

An agent sees its own facts *with their identifiers* - it needs them to cite one
- and sees a peer's exposed fact as **rendered text only**.  There is no place
in this family where a raw symbolic tuple reaches a model: the experiment is a
language reasoning task and the symbols stay in the log.

``render_social_source`` is the single renderer for a visible participant, used
for ordinary peers and for the controller alike.  That is deliberate: the
controller occupies one ordinary social slot and must be indistinguishable from
a participant, so its recommendation (its vote) and its injected fact travel
through exactly the same fields a peer's would - and it gets no prose channel a
peer does not have either.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
PROMPT_VERSION = 1

VOTE_VISIBILITIES = ("public", "hidden")
IMPLEMENTED_VOTE_VISIBILITIES = ("public",)
"""``hidden`` is reserved: the renderer supports dropping a source's vote, but
the surrounding text still tells the agent its vote is what others see, so
enabling it is a text decision that has not been made.  It is also nearly empty
now that prose is not shown either - a hidden-vote source would carry its
identity and its exposed fact and nothing more."""

MAX_REASON_CHARACTERS = 600
"""A generous cap on "at most three sentences".  It exists to reject a runaway
essay - which truncates at ``max_output_tokens`` and breaks the JSON anyway -
not to police wording."""

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

SOCIAL_ENVIRONMENT_NEUTRAL = (
    "Participant identities persist across interactions.\n"
    "\n"
    "Different participants know different facts about this problem. No single\n"
    "participant is guaranteed to know everything needed to answer. Every\n"
    "participant is trying to identify the correct answer."
)
"""``social_distrust: false`` - the neutral cooperative baseline.

Same distributed-information statement, no claim about anyone's objectives.
Useful as the clean control arm: any difference against the distrust arm is
then attributable to the strategic warning alone."""

SOCIAL_ENVIRONMENT = SOCIAL_ENVIRONMENT_DISTRUST
"""Backwards-compatible alias for the default environment."""


def social_environment(social_distrust: bool = True) -> str:
    """The environment text for one condition.

    Both variants are bound as ``fixed`` blocks, so the choice is pinned by the
    prompt definition hash rather than by convention: a run at
    ``social_distrust: false`` has a demonstrably different prompt definition
    from one at ``true``, instead of a silently different prompt.
    """

    return SOCIAL_ENVIRONMENT_DISTRUST if social_distrust else SOCIAL_ENVIRONMENT_NEUTRAL

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
    "too. Use \"none\" if you prefer to share nothing.\n"
    "\n"
    "You may share only a fact listed under YOUR CURRENT KNOWLEDGE, by its exact\n"
    "identifier. Do not invent facts, identifiers, or relationships that were not\n"
    "given to you.\n"
    "\n"
    "Keep your reason to at most three sentences."
)

NO_KNOWN_FACTS = (
    "You do not currently know any facts about this problem. You will have to "
    "rely on what other participants tell you."
)
EVIDENCE_HEADER = "Evidence they are sharing:"
"""How a source's exposed fact is introduced.  Identical for peers and for the
controller, so the *format* carries no signal about which one is speaking."""

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


# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------


def _text_issues(name: str, value: Any) -> tuple[ValidationIssue, ...]:
    if not isinstance(value, str) or not value.strip():
        return (ValidationIssue(f"prompt.blocks.{name}.value", "must be non-empty text"),)
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
        if isinstance(options, (str, bytes)) or not isinstance(options, Sequence) or not options:
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
                    "prompt.blocks.known_facts.value", "must be a sequence of rendered facts"
                ),
            )
        return ()

    def render(self) -> str:
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
                ValidationIssue("prompt.blocks.current_position.value", "must contain a vote"),
            )
        return ()

    def render(self) -> str:
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
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
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


def shuffled_option_letters(
    answers: Sequence[str], rng: Any
) -> dict[str, str]:
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
    return {PRESENTATION_LETTERS[index]: relation for index, relation in enumerate(relations)}


def relation_to_letter(option_letters: Mapping[str, str]) -> dict[str, str]:
    """Invert one call's presentation map."""

    return {relation: letter for letter, relation in option_letters.items()}


def localize_sources(
    sources: Sequence[Mapping[str, Any]], option_letters: Mapping[str, str]
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
        localized.append(
            {**dict(source), "vote": inverse.get(str(vote), vote) if vote else vote}
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

    return {str(relation).upper(): str(label) for label, relation in option_relations.items()}


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
            '  "reason": "<brief private reason>",\n'
            f'  "shared_fact_id": "<{citable}>"\n'
            "}"
        )

    def instruction_messages(self) -> tuple[tuple[MessageRole, str], ...]:
        return ((MessageRole.USER, self.instruction()),)

    def validate(self, response: str) -> ValidationResult:
        parsed = extract_json_object(response)
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
                ValidationIssue(
                    "response.shared_fact_id", "must be a string", raw
                )
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


@dataclass(frozen=True, slots=True)
class ParsedBallot:
    """One parsed response.  ``vote is None`` means it did not resolve."""

    vote: str | None
    reason: str | None
    shared_fact_id: str | None
    raw_vote: Any
    raw_shared_fact_id: Any
    shared_fact_present: bool


def parse_relational_ballot(
    response: str,
    option_labels: Sequence[str],
    option_relations: Mapping[str, str] | None = None,
) -> ParsedBallot:
    parsed = extract_json_object(response)
    raw_vote = parsed.get("vote") if parsed else None
    raw_reason = parsed.get("reason") if parsed else None
    raw_shared = parsed.get("shared_fact_id") if parsed else None
    reason = raw_reason.strip() if isinstance(raw_reason, str) and raw_reason.strip() else None
    aliases = relation_vote_aliases(option_relations or {})
    return ParsedBallot(
        vote=resolve_vote(raw_vote, tuple(option_labels), aliases),
        reason=reason,
        shared_fact_id=(
            normalize_shared_fact_id(raw_shared) if isinstance(raw_shared, str) else None
        ),
        raw_vote=raw_vote,
        raw_shared_fact_id=raw_shared,
        shared_fact_present=bool(parsed) and "shared_fact_id" in parsed,
    )


# --------------------------------------------------------------------------
# Full prompt
# --------------------------------------------------------------------------


class RelationalBallotPrompt(FullPrompt):
    def concrete_prompt_type(self) -> str:
        return PROMPT_FAMILY


def relational_public_ballot_prompt(
    possible_answers: Sequence[str] = ("A", "B", "C"),
    *,
    fact_ids: Sequence[str] = (),
    relations: Sequence[str] = (),
    social_distrust: bool = True,
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
            SocialEnvironmentBlock(social_environment(social_distrust)),
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
    social_distrust: bool = True,
    social_context: bool = False,
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

    letters = tuple(option_letters)
    prompt = relational_public_ballot_prompt(
        letters,
        fact_ids=fact_ids,
        # The semantic alphabet in a call-independent order, so the contract -
        # and therefore the prompt definition - does not move with the shuffle.
        relations=tuple(sorted(option_letters.values())),
        social_distrust=social_distrust,
    ).bind(
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
                f"{letter}) {option_letters[letter]}" for letter in letters
            ),
        },
        known_facts=tuple(known_facts),
    )
    if current_vote is not None:
        prompt = prompt.bind(current_position={"vote": current_vote})
    if social_sources:
        prompt = prompt.bind(
            social_information=render_social_sources(
                localize_sources(social_sources, option_letters),
                vote_visibility=vote_visibility,
            )
        )
    return prompt  # type: ignore[return-value]


__all__ = [
    "CONTROL_RECOMMENDATION",
    "DECISION_BASIS_INITIAL",
    "DECISION_BASIS_SOCIAL",
    "DECISION_BASIS_SOCIAL_NONE_VISIBLE",
    "DECISION_INSTRUCTION",
    "EVIDENCE_HEADER",
    "IMPLEMENTED_VOTE_VISIBILITIES",
    "MAX_REASON_CHARACTERS",
    "NO_KNOWN_FACTS",
    "PROMPT_FAMILY",
    "PROMPT_VERSION",
    "SOCIAL_ENVIRONMENT",
    "SOCIAL_ENVIRONMENT_DISTRUST",
    "SOCIAL_ENVIRONMENT_NEUTRAL",
    "VOTE_VISIBILITIES",
    "ParsedBallot",
    "RelationalBallotContract",
    "RelationalBallotPrompt",
    "PRESENTATION_LETTERS",
    "agent_label",
    "build_relational_ballot_prompt",
    "control_label",
    "localize_sources",
    "normalize_relational_vote",
    "normalize_shared_fact_id",
    "parse_relational_ballot",
    "relation_vote_aliases",
    "relational_public_ballot_prompt",
    "render_control_reason",
    "render_own_fact",
    "render_social_source",
    "relation_to_letter",
    "render_social_sources",
    "shuffled_option_letters",
    "resolve_vote",
    "social_environment",
]
