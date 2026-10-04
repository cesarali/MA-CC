"""Post-action communication chooser for the MuSR blackboard controller.

The chooser never decides whether the controller acts.  It is called only after
the existing binary policy has produced ``U_k = 1`` and uses an independent
random-number stream, so it cannot perturb later sensor samples or binary
actions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
from typing import Any

from mas_cc.llm_runtime.messages import Message, MessageRole
from mas_cc.llm_runtime.prompts import CompiledPrompt, ResponseContract
from mas_cc.llm_runtime.providers import CompletionRequest, LLMProvider
from mas_cc.storage import canonical_hash
from .prompts import agent_label


class CommunicationMode(str, Enum):
    REPORT = "REPORT"
    REQUEST = "REQUEST"
    DIRECTIVE = "DIRECTIVE"
    # Spend nothing this round and keep the allowance for a later one. Only
    # meaningful against a whole-game budget, where saving is a real choice;
    # under a per-round quota an unspent message is simply lost.
    HOLD = "HOLD"
    # A round-level composition marker. Individual public messages still use
    # REPORT, REQUEST, or DIRECTIVE.
    MIXED = "MIXED"


COMMUNICATION_POLICY = "contextual_weighted_v1"
COMMUNICATION_POLICY_VERSION = 1
LLM_COMMUNICATION_POLICY = "llm_structured_v1"
LLM_COMMUNICATION_POLICY_VERSION = 1
LLM_AUTHORED_REPORT_ONLY_POLICY = "llm_authored_report_only_v1"
LLM_AUTHORED_REPORT_ONLY_POLICY_VERSION = 1
LLM_AUTHORED_FIXED_REPORT_ONLY_POLICY = "llm_authored_fixed_report_only_v1"
LLM_AUTHORED_FIXED_REPORT_ONLY_POLICY_VERSION = 1
LLM_AUTHORED_FULL_COMMUNICATION_POLICY = "llm_authored_full_communication_v1"
LLM_AUTHORED_FULL_COMMUNICATION_POLICY_VERSION = 1
LLM_AUTHORED_VARIABLE_FULL_COMMUNICATION_POLICY = (
    "llm_authored_variable_full_communication_v1"
)
LLM_AUTHORED_VARIABLE_FULL_COMMUNICATION_POLICY_VERSION = 1
LLM_AUTHORED_MIXED_FULL_COMMUNICATION_POLICY = (
    "llm_authored_mixed_full_communication_v1"
)
LLM_AUTHORED_MIXED_FULL_COMMUNICATION_POLICY_VERSION = 1
COMMUNICATION_POLICIES = (
    COMMUNICATION_POLICY,
    LLM_COMMUNICATION_POLICY,
    LLM_AUTHORED_REPORT_ONLY_POLICY,
)

LLM_CONTROLLER_INSTRUCTION = (
    "Your objective is to increase support for the target allocation through "
    "truthful communication. Review the sampled votes and previous public "
    "discussion. Choose REPORT to supply useful verified evidence, REQUEST to "
    "ask participants for relevant evidence, or DIRECTIVE to focus a comparison. "
    "Choose what best serves the target in the present context. You may select "
    "only supplied eligible fact IDs; never invent or alter verified evidence. "
    "Return the required structured decision."
)


@dataclass(frozen=True, slots=True)
class ControllerVisibleMessage:
    """One public message visible to the communication chooser."""

    message_id: str
    author: str
    message_type: str
    text: str
    vote: str
    shared_fact_id: str | None
    reply_to: str | None
    round_created: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "author": self.author,
            "message_type": self.message_type,
            "text": self.text,
            "vote": self.vote,
            "shared_fact_id": self.shared_fact_id,
            "reply_to": self.reply_to,
            "round_created": self.round_created,
        }


@dataclass(frozen=True, slots=True)
class ControllerVisibleFact:
    """One canonical true fact currently eligible for an LLM REPORT."""

    fact_id: str
    text: str
    prior_post_count: int
    last_post_round: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "text": self.text,
            "prior_post_count": self.prior_post_count,
            "last_post_round": self.last_post_round,
        }


@dataclass(frozen=True, slots=True)
class ControllerCommunicationContext:
    """The explicit, public-information-only input to the chooser."""

    round_index: int
    target: str
    sampled_opinion_counts: Mapping[str, int]
    live_message_type_counts: Mapping[str, int]
    previous_modes: tuple[CommunicationMode, ...] = ()
    sampled_votes: tuple[str, ...] = ()
    previous_board_messages: tuple[ControllerVisibleMessage, ...] = ()
    eligible_facts: tuple[ControllerVisibleFact, ...] = ()
    posting_history: tuple[Mapping[str, Any], ...] = ()
    # Under the episode budget scope this is what REMAINS, not the configured
    # total, so a policy that fills `budget` spends the rest of the allowance.
    budget: int = 0
    budget_scope: str = "per_round"
    round_budget_mode: str = "exact"
    budget_total: int = 0
    budget_spent: int = 0
    horizon: int | None = None
    rounds_remaining: int | None = None
    population: int | None = None
    sensing_mode: str = "votes"
    board_view_complete: bool | None = None
    public_memory: Mapping[str, Any] | None = None
    # The per-fact repost limits actually in force. The prompt used to state
    # them as literals ("within the last round, or three times already"), which
    # was wrong for any config that changed them and would make a controller
    # self-censor against limits that no longer exist.
    report_cooldown_rounds: int | None = None
    report_max_posts_per_fact: int | None = None
    # The controller used to be told "increase support for ALLOCATION_1"
    # without ever learning what ALLOCATION_1 meant, which left it unable to
    # tell whether a given fact helped its target or a rival. These carry the
    # same scenario and question the agents receive, plus what each option is.
    scenario_question: str | None = None
    answer_display_texts: Mapping[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "target": self.target,
            "sampled_votes": list(self.sampled_votes),
            "sampled_opinion_counts": dict(self.sampled_opinion_counts),
            "previous_board_messages": [
                message.to_dict() for message in self.previous_board_messages
            ],
            "eligible_facts": [fact.to_dict() for fact in self.eligible_facts],
            "posting_history": [dict(row) for row in self.posting_history],
            "budget": self.budget,
            "budget_scope": self.budget_scope,
            **({"round_budget_mode": self.round_budget_mode}
               if self.budget_scope == "per_round" else {}),
            **({"budget_total": self.budget_total,
                "budget_spent": self.budget_spent}
               if self.budget_scope == "episode" else {}),
            # `budget` means "this round's quota" under per_round scope and
            # "what is left of the whole-game allowance" under episode scope.
            # Spell the second one out rather than leaving it to be inferred;
            # emit it only where a whole-game remainder actually exists.
            **({"budget_remaining_for_episode":
                max(self.budget_total - self.budget_spent, 0)}
               if self.budget_scope == "episode" else {}),
            **({"horizon": self.horizon} if self.horizon is not None else {}),
            **({"rounds_remaining": self.rounds_remaining}
               if self.rounds_remaining is not None else {}),
            **({"population": self.population} if self.population is not None else {}),
            "sensing_mode": self.sensing_mode,
            **({"board_view_complete": self.board_view_complete}
               if self.board_view_complete is not None else {}),
            **({"public_memory": dict(self.public_memory)}
               if self.public_memory is not None else {}),
            **({"report_cooldown_rounds": self.report_cooldown_rounds}
               if self.report_cooldown_rounds is not None else {}),
            **({"report_max_posts_per_fact": self.report_max_posts_per_fact}
               if self.report_max_posts_per_fact is not None else {}),
        }


@dataclass(frozen=True, slots=True)
class AuthoredControllerMessage:
    """One validated public message in an opt-in mixed controller round."""

    mode: CommunicationMode
    text: str
    fact_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.mode.value,
            "fact_id": self.fact_id,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class CommunicationChoice:
    mode: CommunicationMode
    reason: str
    policy: str = COMMUNICATION_POLICY
    policy_version: int = COMMUNICATION_POLICY_VERSION
    fact_ids: tuple[str, ...] = ()
    text: str | None = None
    report_texts: tuple[str, ...] = ()
    message_texts: tuple[str, ...] = ()
    messages: tuple[AuthoredControllerMessage, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "reason": self.reason,
            "policy": self.policy,
            "policy_version": self.policy_version,
            "fact_ids": list(self.fact_ids),
            "text": self.text,
            "report_texts": list(self.report_texts),
            "message_texts": list(self.message_texts),
            "messages": [message.to_dict() for message in self.messages],
        }


@dataclass(frozen=True, slots=True)
class LLMCommunicationAttempt:
    attempt: int
    request: CompletionRequest
    response: Any | None
    validation_error: str | None
    provider_error: str | None

    @property
    def valid(self) -> bool:
        return self.validation_error is None and self.provider_error is None

    def to_dict(self) -> dict[str, Any]:
        response = self.response
        return {
            "attempt": self.attempt,
            "request": self.request.to_dict(),
            "response": None if response is None else response.to_dict(),
            "raw_output": None if response is None else response.content,
            "validation_error": self.validation_error,
            "provider_error": self.provider_error,
        }


@dataclass(frozen=True, slots=True)
class LLMCommunicationResult:
    choice: CommunicationChoice | None
    attempts: tuple[LLMCommunicationAttempt, ...]


@dataclass(frozen=True, slots=True)
class ControllerCommunicationPrompt:
    """Small compilable prompt used by conservative static preflight."""

    text: str
    family: str = "relational_controller_communication"
    version: int = 1

    def compile(self, token_counter: Any | None = None) -> CompiledPrompt:
        message = Message(MessageRole.USER, self.text)
        token_count = (
            0 if token_counter is None else token_counter.count_tokens(self.text)
        )
        identity = canonical_hash(
            {"family": self.family, "version": self.version, "text": self.text}
        )
        return CompiledPrompt(
            family=self.family,
            version=self.version,
            definition_hash=canonical_hash(
                {"family": self.family, "version": self.version}
            ),
            instance_hash=identity,
            blocks=(),
            omitted_blocks=(),
            messages=(message,),
            response_contract=ResponseContract(self.family),
            tokenizer_name=(
                None if token_counter is None else type(token_counter).__name__
            ),
            message_token_counts=(token_count,),
        )


def _task_preamble(
    context: "ControllerCommunicationContext", *, full_communication: bool = False
) -> str:
    """Scenario, question and what each option actually means.

    Returns "" when the caller supplies neither, so older callers render
    exactly the prompt they rendered before.
    """

    blocks = []
    if context.scenario_question:
        blocks.append(context.scenario_question.strip())
    if context.answer_display_texts:
        lines = "\n".join(
            f"{option}: {text}"
            for option, text in context.answer_display_texts.items()
        )
        blocks.append(f"OPTIONS\n{lines}")
    game = _game_explanation(context, full_communication=full_communication)
    if game:
        blocks.append(game)
    return "\n\n".join(blocks) + "\n\n" if blocks else ""


def _game_explanation(
    context: "ControllerCommunicationContext", *, full_communication: bool = False
) -> str:
    """The situation the controller is acting in, told as a story.

    The controller used to receive a bare objective and a budget, with no
    statement that a population existed, that the episode ran in rounds, or
    that it was posting into their discussion. Kept deliberately qualitative:
    the numbers that change round to round are stated in the status line, and
    repeating the mechanics as figures here only buries the point.
    """

    # "the question above" is incoherent without the scenario block, and the
    # whole framing presumes it, so a caller that supplies no task context gets
    # the plain pre-existing prompt rather than a dangling reference.
    if not context.scenario_question:
        return ""
    if not context.population or context.horizon is None:
        return ""
    if context.sensing_mode == "board":
        board_view = (
            "every message" if context.board_view_complete else "a sample of messages"
        )
        game = (
            f"The episode lasts {context.horizon} days. On each day after the first, "
            f"you see {board_view} posted to the public board on the previous "
            "day, then decide whether to post at dawn. The old board expires "
            f"before the day's {context.population} sequential participant updates. "
            "An agent may update more than once; on each update, that agent "
            "reads some live board messages, votes, and may post a message for "
            "later agents to read that day. Today's messages expire before the "
            "next day. Participants begin with different parts of the evidence "
            "and try to identify the best allocation by the end of the episode."
        )
    else:
        game = (
            "A group of participants is trying to answer the question above "
            "together. Each one holds a different part of the evidence and none of "
            "them can see all of it, so over a series of rounds they share what "
            "they know and vote for the allocation they believe is correct. They "
            "do not know which allocation is correct. Each participant sees only "
            "part of the board each round and can say only so much, so what "
            "circulates depends on what people choose to put there."
        )
    budget_sentence = (
        "Your budget of messages has to last the whole episode, which makes "
        "timing part of the problem."
        if context.budget_scope == "episode"
        else "You receive a new message allowance each day."
    )
    role = (
        "You are not one of them, and they do not know you exist: your messages "
        "appear on the same public board as participant messages. Your objective "
        f"is to steer the group towards {context.target}. You may report eligible "
        "verified facts faithfully, request relevant evidence, or direct attention "
        "to a comparison. Never invent or alter evidence. Choose the message "
        "type, content, and timing that serve your objective. " + budget_sentence
        if full_communication else
        f"You are not one of them, and they do not know you exist: your reports "
        "appear on the same public board as participant reports. Your objective "
        f"is to steer the group towards "
        f"{context.target}. You may only circulate facts from the eligible "
        "list below, and each report must convey its fact faithfully, so your "
        "instruments are which facts you put in front of them, how you "
        "faithfully phrase them, and when. " + budget_sentence
    )
    return "\n".join((
        "THE GAME",
        game,
        "",
        "YOUR ROLE",
        role,
    ))


def _repost_limits(context: "ControllerCommunicationContext") -> str:
    """State the per-fact repost limits that are actually configured.

    Returns "" when neither limit can bind, so a controller given a free hand
    is not told about restrictions that do not exist.
    """

    clauses = []
    cooldown = context.report_cooldown_rounds
    if cooldown:
        clauses.append(
            "within the last round"
            if cooldown == 1
            else f"within the last {cooldown} rounds"
        )
    cap = context.report_max_posts_per_fact
    # Name the cap only if it can actually bind, i.e. if it sits below the most
    # times a single fact could be posted anyway. Measure that against the
    # whole allowance, never the remainder: under an episode budget the
    # remainder shrinks each round, which would announce the cap early and then
    # silently drop it. When the ceiling is unknown, state the cap -- claiming
    # a limit that cannot bite is far safer than hiding one that can.
    if context.budget_scope == "episode":
        ceiling = context.budget_total or None
    elif context.horizon:
        ceiling = context.budget * context.horizon
    else:
        ceiling = None
    if cap is not None and (ceiling is None or cap < ceiling):
        clauses.append(f"{cap} times already")
    if not clauses:
        return ""
    if len(clauses) == 1:
        return f" A fact you posted {clauses[0]} will be refused."
    return " A fact you posted " + ", or ".join(clauses) + ", will be refused."


def _controller_situation(context: "ControllerCommunicationContext") -> str:
    """The facts a controller needs and previously had to infer.

    The budget used to appear only as the literal word "budget" in the
    instruction, with the number buried in the JSON payload. Under the episode
    budget scope the number also changes every round, so it has to be stated.
    """

    lines = []
    if context.horizon is not None and context.rounds_remaining is not None:
        lines.append(
            f"This is day {context.round_index + 1} of {context.horizon}; "
            f"{context.rounds_remaining} days remain after this one."
        )
    if context.budget_scope == "episode":
        # `budget` is what may be posted THIS round, which the runtime caps at
        # the size of the eligible pool. The whole-game remainder is the
        # separate, larger number, and it is the one the sentence is about.
        remaining = max(context.budget_total - context.budget_spent, 0)
        lines.append(
            f"You have {remaining} messages remaining for the whole episode; "
            f"you have already used {context.budget_spent} in previous days."
        )
        if context.rounds_remaining is not None:
            days_left = context.rounds_remaining + 1
            lines.append(
                f"An even pace is about {remaining / days_left:.1f} per remaining "
                "day, not a daily quota."
            )
    else:
        lines.append(f"You may post {context.budget} messages this round.")
    sampled = sum(context.sampled_opinion_counts.values())
    if sampled and context.population and context.sensing_mode != "board":
        lines.append(
            f"The votes below are a sample of {sampled} of the "
            f"{context.population} participants."
        )
    return " ".join(lines)


def render_llm_controller_prompt(
    context: ControllerCommunicationContext,
    allowed_modes: Sequence[CommunicationMode | str],
    policy: str = LLM_COMMUNICATION_POLICY,
) -> str:
    """Render only the explicitly permitted public controller information."""

    allowed = [CommunicationMode(mode).value for mode in allowed_modes]
    payload = context.to_dict()
    payload["allowed_modes"] = allowed
    preamble = _task_preamble(
        context,
        full_communication=policy in {
            LLM_AUTHORED_FULL_COMMUNICATION_POLICY,
            LLM_AUTHORED_VARIABLE_FULL_COMMUNICATION_POLICY,
            LLM_AUTHORED_MIXED_FULL_COMMUNICATION_POLICY,
        },
    )
    if policy in {
        LLM_AUTHORED_REPORT_ONLY_POLICY,
        LLM_AUTHORED_FIXED_REPORT_ONLY_POLICY,
    }:
        # "distinct" in both: duplicate fact IDs are rejected by the parser,
        # so the free-count wording has to say so too.
        count_instruction = (
            f"exactly {context.budget} distinct"
            if policy == LLM_AUTHORED_FIXED_REPORT_ONLY_POLICY
            else f"between 1 and {context.budget} distinct"
        )
        # HOLD can preserve an episode allowance or forgo a fresh daily quota.
        # The two scopes need different wording even though they share a mode.
        hold = CommunicationMode.HOLD.value in allowed
        # The preamble already states the objective. Without one (an older
        # caller that passes no task context) it still has to be said here.
        objective = (
            ""
            if preamble
            else (
                f"Your objective is to increase support for the target "
                f"allocation {context.target} using truthful but strategically "
                f"selected information. "
            )
        )
        if hold and context.budget_scope == "episode":
            this_round = (
                f"{_controller_situation(context)}\n\n"
                "Before spending, weigh the days remaining and read yesterday's "
                "posts as clues to the discussion, not a vote census. Choose "
                "verified facts that could plausibly shift later votes toward "
                f"{context.target}; otherwise HOLD. You do not need to spend the entire "
                f"allowance. You may post up to {context.budget} distinct eligible "
                "verified facts or HOLD; the limit is a ceiling, not a target. "
                "If posting, choose only facts with a specific purpose and write "
                "one concise public REPORT for each. Each report_texts entry "
                "must faithfully communicate only its paired fact_id; do not "
                "invent, alter, exaggerate, or add evidence. Strategic selection "
                "and emphasis are allowed."
            )
        elif hold:
            this_round = (
                f"{_controller_situation(context)}\n\n"
                "Read yesterday's board as clues to the discussion, not a vote "
                "census. Decide whether posting today could move later votes "
                f"toward {context.target}. You may HOLD and post nothing, or post "
                f"exactly {context.budget} distinct eligible verified facts. "
                "Unused slots expire tonight; tomorrow brings a fresh allowance. "
                "If posting, choose facts with a specific purpose and write one "
                "concise public REPORT for each. Each report_texts entry must "
                "faithfully communicate only its paired fact_id; do not invent, "
                "alter, exaggerate, or add evidence. Strategic selection and "
                "emphasis are allowed."
            )
        else:
            this_round = (
                f"{_controller_situation(context)}\n\n"
                f"Choose {count_instruction} eligible verified facts from the table "
                "below and write one concise public REPORT for each. Each "
                "report_texts entry must faithfully communicate only its paired "
                "fact_id; do not invent, alter, exaggerate, or add evidence. "
                "Strategic selection and emphasis are allowed."
            )
        hold_schema = (
            "\n\n"
            + ("To post nothing today and keep the remaining messages for later:"
               if context.budget_scope == "episode" else
               "To post nothing today and forgo today's allowance:")
            + "\n```\n{\n"
            '"mode":"HOLD",\n"fact_ids":[],\n"text":null,\n'
            '"report_texts":[],\n"reason":"'
            + ("<why holding beats posting now>"
               if context.budget_scope == "episode" else "<why not posting now>")
            + '"\n'
            "}\n```"
            if hold
            else ""
        )
        output = (
            "## OUTPUT\n\n"
            "Return one JSON object with exactly these fields:\n```\n{\n"
            '"mode":"REPORT",\n"fact_ids":["<eligible id>"],\n"text":null,\n'
            '"report_texts":["<paired public report>"],\n'
            '"reason":"<why these facts rather than the other eligible ones>"\n'
            "}\n```"
            + hold_schema
            + "\n\nDo not diverge from this output scheme."
        )
        return (
            preamble
            + objective
            + ("THIS ROUND\n" if preamble else "")
            + this_round
            + "\n\n"
            + output
            + "\n\n"
            + render_controller_context_markdown(context, allowed)
        )

    if policy == LLM_AUTHORED_VARIABLE_FULL_COMMUNICATION_POLICY:
        return (
            preamble
            + "THIS ROUND\n"
            + _controller_situation(context)
            + "\n\nRead yesterday's board as clues to the discussion, not a vote census. "
            "Choose only messages with a specific purpose for moving later votes "
            f"toward {context.target}. You may post between 1 and "
            f"{context.budget} messages in one permitted mode, or HOLD and post "
            "nothing today. The limit is a ceiling, not a target. A REPORT must "
            "faithfully communicate only its paired eligible fact; use distinct "
            "fact IDs. A REQUEST asks participants for relevant evidence, and a "
            "DIRECTIVE focuses a comparison without asserting new facts."
            "\n\nOUTPUT\n"
            "Return one JSON object with exactly these fields:\n"
            '{"mode":"REPORT|REQUEST|DIRECTIVE|HOLD","fact_ids":[],"text":null,'
            '"report_texts":[],"message_texts":[],"reason":"..."}\n'
            "For REPORT, provide one fact_id and one faithful report_text per "
            "message, and leave message_texts empty. For REQUEST or DIRECTIVE, "
            "leave fact_ids and report_texts empty and provide one concise "
            "message_text per message. For HOLD, leave every list empty. "
            "Do not diverge from this output scheme.\n\n"
            + render_controller_context_markdown(context, allowed)
        )
    if policy == LLM_AUTHORED_FULL_COMMUNICATION_POLICY:
        hold = CommunicationMode.HOLD.value in allowed
        posting_instruction = (
            "You may HOLD and post nothing, or choose one permitted posting "
            "mode and fill exactly "
            if hold else
            "Choose one permitted mode for this round and fill exactly "
        )
        mode_schema = (
            "REPORT|REQUEST|DIRECTIVE|HOLD"
            if hold else "REPORT|REQUEST|DIRECTIVE"
        )
        return (
            preamble
            + f"{LLM_CONTROLLER_INSTRUCTION}\n\n"
            + f"{_controller_situation(context)} "
            + posting_instruction
            + f"{context.budget} public message slots. "
            + ("Unused slots expire tonight; tomorrow brings a fresh allowance. "
               if hold else "")
            + f"For REPORT, return exactly {context.budget} distinct eligible fact IDs and one "
            "faithful report_text per fact. For REQUEST or DIRECTIVE, fact_ids and "
            "report_texts must be empty and message_texts must contain exactly "
            f"{context.budget} concise authored messages. "
            + ("For HOLD, leave every list empty. " if hold else "")
            + "Return one JSON object with "
            "exactly these fields:\n"
            + '{"mode":"' + mode_schema + '","fact_ids":[],"text":null,'
            '"report_texts":[],"message_texts":[],"reason":"..."}\n\n'
            "CONTROLLER INFORMATION\n"
            + json.dumps(payload, sort_keys=True, ensure_ascii=False)
        )
    if policy == LLM_AUTHORED_MIXED_FULL_COMMUNICATION_POLICY:
        return (
            preamble
            +
            "Your objective is to increase support for the target allocation through "
            "truthful communication. Review the sampled votes and previous public "
            f"discussion. {_controller_situation(context)} Fill exactly "
            f"{context.budget} public message slots. Choose REPORT, REQUEST, or "
            "DIRECTIVE independently for every slot; the types may be mixed or all "
            "the same. A REPORT must contain one distinct eligible fact_id and text "
            "that faithfully communicates only that fact. A REQUEST or DIRECTIVE "
            "must use null fact_id and concise authored text. Return one JSON object "
            "with exactly these fields:\n"
            '{"mode":"MIXED","messages":['
            '{"type":"REPORT|REQUEST|DIRECTIVE","fact_id":null,"text":"..."}'
            '],"reason":"..."}\n\n'
            "CONTROLLER INFORMATION\n"
            + json.dumps(payload, sort_keys=True, ensure_ascii=False)
        )
    return (
        preamble
        + f"{LLM_CONTROLLER_INSTRUCTION}\n\n"
        "Return one JSON object with exactly these fields:\n"
        '{"mode":"REPORT|REQUEST|DIRECTIVE","fact_ids":[],"text":null,"reason":"..."}\n'
        "For REPORT, choose 1 through budget distinct eligible fact IDs. For REQUEST "
        "or DIRECTIVE, fact_ids must be empty. Always set text to null; code renders "
        "the permitted canonical public message.\n\n"
        "CONTROLLER INFORMATION\n"
        + json.dumps(payload, sort_keys=True, ensure_ascii=False)
    )


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A GitHub-style table. Empty rows render as a plain dash, not a husk."""

    if not rows:
        return "None."
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(" --- " for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def _render_public_observation_ledger(memory: Mapping[str, Any]) -> str:
    """Render cumulative public observations without calling them population votes."""

    population = int(memory["population_size"])
    days = tuple(memory["days"])
    participants = tuple(memory["participants"])
    facts = tuple(memory["facts"])
    day_rows = []
    for row in days:
        votes = ", ".join(
            f"{option}: {count}"
            for option, count in row["last_posted_vote_counts"].items()
        )
        day_rows.append((
            str(row["day"]),
            "full" if row["view_complete"] else "sample",
            str(row["messages_seen"]),
            f"{row['distinct_participants_seen']}/{population}",
            votes,
            ", ".join(row["new_participant_fact_ids"]) or "—",
            ", ".join(row["controller_fact_ids"]) or "—",
        ))
    participant_rows = [
        (
            agent_label(row["author_id"]),
            str(row["last_seen_day"]),
            str(row["last_posted_vote"]),
            str(row["observed_post_count"]),
            ", ".join(row["cited_fact_ids"]) or "—",
        )
        for row in participants
    ]
    fact_rows = [
        (
            str(row["fact_id"]),
            str(row["distinct_participant_authors"]),
            str(row["observed_post_count"]),
            str(row["last_seen_day"]),
        )
        for row in facts
    ]
    return "\n\n".join((
        "## Public observation memory\n\n"
        "This is a record of posts you observed, including yesterday's board. "
        "One participant may post more than once; your own posts are separate. "
        "A last posted vote is not a current population vote. You cannot see "
        "silent participants' votes or tell who read a message.",
        _md_table(
            ("day", "view", "messages seen", "distinct participants seen", "their last posted votes", "new participant fact IDs", "your fact IDs"),
            day_rows,
        ),
        "### Last observed post by participant\n\n"
        + _md_table(
            ("participant", "last seen day", "vote in last seen post", "posts seen", "fact IDs cited in observed reports"),
            participant_rows,
        ),
        "### Observed participant fact circulation\n\n"
        + _md_table(
            ("fact ID", "distinct participants citing", "posts seen", "last seen day"),
            fact_rows,
        ),
    ))


def render_controller_context_markdown(
    context: "ControllerCommunicationContext",
    allowed_modes: Sequence[str],
) -> str:
    """The controller's status as readable prose and tables, not a JSON blob.

    The same facts the JSON payload carried, laid out so a reader can scan
    them: which facts are still postable and how often each has been used,
    how the sampled votes split, and what was on the board last round.
    """

    blocks: list[str] = []

    counts = dict(context.sampled_opinion_counts)
    sampled = sum(counts.values())
    vote_rows = [
        (option, str(n), f"{(n / sampled if sampled else 0):.0%}")
        for option, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    vote_note = ""
    if context.sensing_mode == "board":
        vote_note = (
            "These counts are per message, not per participant; your own posts "
            "may be included."
        )
        if context.public_memory is not None:
            vote_note += (
                " The memory table below counts distinct observed participant "
                "authors, but it is still not a population census."
            )
        vote_note += "\n\n"
    blocks.append(
        ("## Votes attached to yesterday's observed board messages\n\n"
         if context.sensing_mode == "board" else "## Votes you sampled\n\n")
        + vote_note
        + (f"You sampled {sampled} of the {context.population} participants.\n\n"
           if sampled and context.population and context.sensing_mode != "board" else "")
        + _md_table(("allocation", "votes", "share"), vote_rows)
    )

    if context.public_memory is not None:
        blocks.append(_render_public_observation_ledger(context.public_memory))

    posted = [row for row in context.eligible_facts if row.prior_post_count]
    blocks.append(
        "## What you have already posted\n\n"
        + ("Nothing yet." if not posted else _md_table(
            ("fact_id", "times posted", "last posted"),
            [(r.fact_id, str(r.prior_post_count),
              "never" if r.last_post_round is None else f"day {r.last_post_round + 1}")
             for r in posted]))
    )

    limits = _repost_limits(context).strip()
    blocks.append(
        "## Facts you may post\n\n"
        + "Every fact below is verified true. You may post only these."
        + (f" {limits}" if limits else "")
        + "\n\n"
        + _md_table(
            ("fact_id", "times posted", "last posted", "fact"),
            [(r.fact_id, str(r.prior_post_count),
              "never" if r.last_post_round is None else f"day {r.last_post_round + 1}",
              r.text) for r in context.eligible_facts])
    )

    if context.previous_board_messages:
        lines = []
        for m in context.previous_board_messages:
            cites = f", cites {m.shared_fact_id}" if m.shared_fact_id else ""
            vote = f"votes {m.vote}" if m.vote else "no vote"
            lines.append(f"**{m.author}** ({vote}{cites})  \n{m.text}")
        board = "\n\n".join(lines)
    else:
        board = "The board was empty."
    blocks.append("## The public board, previous day\n\n" + board)

    return "\n\n".join(blocks)


def parse_llm_communication_choice(
    content: str,
    *,
    context: ControllerCommunicationContext,
    allowed_modes: Sequence[CommunicationMode | str],
    policy: str = LLM_COMMUNICATION_POLICY,
) -> CommunicationChoice:
    """Validate a structured choice against modes, budget, and fact eligibility."""

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response must be valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("response must be one JSON object")
    unknown = set(payload) - {
        "mode",
        "fact_ids",
        "text",
        "report_texts",
        "message_texts",
        "messages",
        "reason",
    }
    if unknown:
        raise ValueError(f"response contains unknown fields: {sorted(unknown)}")
    try:
        mode = CommunicationMode(str(payload.get("mode", "")))
    except ValueError as exc:
        raise ValueError(
            "mode must be REPORT, REQUEST, DIRECTIVE, or MIXED"
        ) from exc
    allowed = {CommunicationMode(value) for value in allowed_modes}
    if (
        policy != LLM_AUTHORED_MIXED_FULL_COMMUNICATION_POLICY
        and mode not in allowed
    ):
        raise ValueError(f"mode {mode.value} is not allowed")
    raw_fact_ids = payload.get("fact_ids", [])
    if not isinstance(raw_fact_ids, list) or any(
        not isinstance(value, str) for value in raw_fact_ids
    ):
        raise ValueError("fact_ids must be a list of strings")
    fact_ids = tuple(raw_fact_ids)
    if len(set(fact_ids)) != len(fact_ids):
        raise ValueError("fact_ids must be distinct")
    text = payload.get("text")
    raw_report_texts = payload.get("report_texts", [])
    raw_message_texts = payload.get("message_texts", [])
    raw_messages = payload.get("messages", [])
    reason = payload.get("reason", "llm_selected")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    eligible = {fact.fact_id for fact in context.eligible_facts}
    authored_messages: tuple[AuthoredControllerMessage, ...] = ()
    if policy == LLM_AUTHORED_MIXED_FULL_COMMUNICATION_POLICY:
        if set(payload) != {"mode", "messages", "reason"}:
            raise ValueError(
                "mixed full-communication response must contain exactly mode, "
                "messages, and reason"
            )
        if mode != CommunicationMode.MIXED:
            raise ValueError("mixed full-communication policy requires mode MIXED")
        if fact_ids or text is not None or raw_report_texts or raw_message_texts:
            raise ValueError(
                "mixed full-communication policy accepts public content only in messages"
            )
        if not isinstance(raw_messages, list) or len(raw_messages) != context.budget:
            raise ValueError(
                "mixed full-communication policy requires exactly budget messages"
            )
        parsed_messages: list[AuthoredControllerMessage] = []
        report_fact_ids: list[str] = []
        for index, raw_message in enumerate(raw_messages):
            if not isinstance(raw_message, Mapping):
                raise ValueError(f"messages[{index}] must be an object")
            if set(raw_message) != {"type", "fact_id", "text"}:
                raise ValueError(
                    f"messages[{index}] must contain exactly type, fact_id, and text"
                )
            try:
                message_mode = CommunicationMode(str(raw_message["type"]))
            except ValueError as exc:
                raise ValueError(
                    f"messages[{index}].type must be REPORT, REQUEST, or DIRECTIVE"
                ) from exc
            if message_mode not in allowed:
                raise ValueError(
                    f"messages[{index}].type {message_mode.value} is not allowed"
                )
            message_text = raw_message["text"]
            if not isinstance(message_text, str) or not message_text.strip():
                raise ValueError(f"messages[{index}].text must be non-empty")
            if len(message_text.strip()) > 1200:
                raise ValueError(
                    f"messages[{index}].text must be at most 1200 characters"
                )
            message_fact_id = raw_message["fact_id"]
            if message_mode == CommunicationMode.REPORT:
                if not isinstance(message_fact_id, str) or message_fact_id not in eligible:
                    raise ValueError(
                        f"messages[{index}] REPORT must select one eligible fact ID"
                    )
                report_fact_ids.append(message_fact_id)
            elif message_fact_id is not None:
                raise ValueError(
                    f"messages[{index}] {message_mode.value} requires null fact_id"
                )
            parsed_messages.append(
                AuthoredControllerMessage(
                    mode=message_mode,
                    fact_id=message_fact_id,
                    text=message_text.strip(),
                )
            )
        if len(set(report_fact_ids)) != len(report_fact_ids):
            raise ValueError("REPORT fact IDs in messages must be distinct")
        authored_messages = tuple(parsed_messages)
    elif "messages" in payload:
        raise ValueError(
            "messages is supported only by the mixed full-communication policy"
        )
    elif mode == CommunicationMode.REPORT:
        if policy in {
            LLM_AUTHORED_FIXED_REPORT_ONLY_POLICY,
            LLM_AUTHORED_FULL_COMMUNICATION_POLICY,
        }:
            if len(fact_ids) != context.budget:
                raise ValueError("LLM-authored REPORT requires exactly budget fact IDs")
            if text is not None:
                raise ValueError("authored report-only policy requires text to be null")
            if not isinstance(raw_report_texts, list) or len(raw_report_texts) != len(fact_ids):
                raise ValueError("report_texts must contain one string per fact ID")
            if any(not isinstance(value, str) or not value.strip() for value in raw_report_texts):
                raise ValueError("every authored report must be non-empty text")
            if any(len(value.strip()) > 1200 for value in raw_report_texts):
                raise ValueError("each authored report must be at most 1200 characters")
            if raw_message_texts:
                raise ValueError("REPORT requires message_texts to be empty")
        elif policy in {
            LLM_AUTHORED_REPORT_ONLY_POLICY,
            LLM_AUTHORED_VARIABLE_FULL_COMMUNICATION_POLICY,
        }:
            if not 1 <= len(fact_ids) <= context.budget:
                raise ValueError("authored variable policy requires 1 through budget fact IDs")
            if text is not None:
                raise ValueError("authored report-only policy requires text to be null")
            if not isinstance(raw_report_texts, list) or len(raw_report_texts) != len(fact_ids):
                raise ValueError("report_texts must contain one string per fact ID")
            if any(not isinstance(value, str) or not value.strip() for value in raw_report_texts):
                raise ValueError("every authored report must be non-empty text")
            if any(len(value.strip()) > 1200 for value in raw_report_texts):
                raise ValueError("each authored report must be at most 1200 characters")
            if raw_message_texts:
                raise ValueError("REPORT requires message_texts to be empty")
        elif not 1 <= len(fact_ids) <= context.budget:
            raise ValueError("REPORT must select between 1 and budget fact IDs")
        if set(fact_ids) - eligible:
            raise ValueError("REPORT selected a fact ID outside the eligible pool")
        if policy != LLM_AUTHORED_REPORT_ONLY_POLICY and text is not None:
            raise ValueError(
                "REPORT text must be null; canonical text is rendered in code"
            )
    elif mode == CommunicationMode.HOLD:
        if fact_ids:
            raise ValueError("HOLD cannot select fact IDs")
        if text is not None:
            raise ValueError("HOLD text must be null")
        if raw_report_texts:
            raise ValueError("HOLD requires report_texts to be empty")
        if raw_message_texts:
            raise ValueError("HOLD requires message_texts to be empty")
    else:
        if fact_ids:
            raise ValueError(f"{mode.value} cannot select fact IDs")
        if text is not None:
            raise ValueError(
                f"{mode.value} text must be null; permitted text is rendered in code"
            )
        if policy in {
            LLM_AUTHORED_REPORT_ONLY_POLICY,
            LLM_AUTHORED_FIXED_REPORT_ONLY_POLICY,
        }:
            raise ValueError("authored report-only policy permits REPORT only")
        if policy in {
            LLM_AUTHORED_FULL_COMMUNICATION_POLICY,
            LLM_AUTHORED_VARIABLE_FULL_COMMUNICATION_POLICY,
        }:
            if raw_report_texts:
                raise ValueError(f"{mode.value} requires report_texts to be empty")
            valid_count = (
                len(raw_message_texts) == context.budget
                if policy == LLM_AUTHORED_FULL_COMMUNICATION_POLICY
                else 1 <= len(raw_message_texts) <= context.budget
            ) if isinstance(raw_message_texts, list) else False
            if (
                not valid_count
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in raw_message_texts
                )
            ):
                raise ValueError(
                    f"LLM-authored {mode.value} requires "
                    + ("exactly budget" if policy == LLM_AUTHORED_FULL_COMMUNICATION_POLICY
                       else "1 through budget")
                    + " message_texts"
                )
            if any(len(value.strip()) > 1200 for value in raw_message_texts):
                raise ValueError("each authored message must be at most 1200 characters")
    return CommunicationChoice(
        mode=mode,
        reason=reason.strip(),
        policy=policy,
        policy_version=(
            LLM_AUTHORED_REPORT_ONLY_POLICY_VERSION
            if policy == LLM_AUTHORED_REPORT_ONLY_POLICY
            else LLM_AUTHORED_FIXED_REPORT_ONLY_POLICY_VERSION
            if policy == LLM_AUTHORED_FIXED_REPORT_ONLY_POLICY
            else LLM_AUTHORED_FULL_COMMUNICATION_POLICY_VERSION
            if policy == LLM_AUTHORED_FULL_COMMUNICATION_POLICY
            else LLM_AUTHORED_VARIABLE_FULL_COMMUNICATION_POLICY_VERSION
            if policy == LLM_AUTHORED_VARIABLE_FULL_COMMUNICATION_POLICY
            else LLM_AUTHORED_MIXED_FULL_COMMUNICATION_POLICY_VERSION
            if policy == LLM_AUTHORED_MIXED_FULL_COMMUNICATION_POLICY
            else LLM_COMMUNICATION_POLICY_VERSION
        ),
        fact_ids=fact_ids,
        text=text,
        report_texts=tuple(str(value).strip() for value in raw_report_texts),
        message_texts=tuple(str(value).strip() for value in raw_message_texts),
        messages=authored_messages,
    )


async def choose_llm_communication(
    *,
    provider: LLMProvider,
    context: ControllerCommunicationContext,
    allowed_modes: Sequence[CommunicationMode | str],
    temperature: float,
    max_output_tokens: int,
    max_retries: int,
    seed_for_attempt: Any,
    policy: str = LLM_COMMUNICATION_POLICY,
) -> LLMCommunicationResult:
    """Ask once, with bounded schema repairs, for post-action communication."""

    base_prompt = render_llm_controller_prompt(context, allowed_modes, policy)
    messages: tuple[Message, ...] = (Message(MessageRole.USER, base_prompt),)
    attempts: list[LLMCommunicationAttempt] = []
    for attempt_index in range(max_retries + 1):
        request = CompletionRequest(
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            seed=int(seed_for_attempt(attempt_index)),
            metadata={
                "decision_stage": "controller_communication",
                "round_index": context.round_index,
                "validation_attempt": attempt_index + 1,
                "validation_retry_bound": max_retries,
            },
        )
        try:
            response = await provider.complete(request)
        except Exception as exc:
            attempts.append(
                LLMCommunicationAttempt(
                    attempt_index + 1, request, None, None, str(exc)
                )
            )
            break
        try:
            choice = parse_llm_communication_choice(
                response.content,
                context=context,
                allowed_modes=allowed_modes,
                policy=policy,
            )
        except (TypeError, ValueError) as exc:
            attempts.append(
                LLMCommunicationAttempt(
                    attempt_index + 1, request, response, str(exc), None
                )
            )
            if attempt_index < max_retries:
                messages = (
                    *messages,
                    Message(
                        MessageRole.USER,
                        "Your previous response was invalid: "
                        f"{exc}. Return the complete JSON object again.",
                    ),
                )
            continue
        attempts.append(
            LLMCommunicationAttempt(attempt_index + 1, request, response, None, None)
        )
        return LLMCommunicationResult(choice=choice, attempts=tuple(attempts))
    return LLMCommunicationResult(choice=None, attempts=tuple(attempts))


def allowed_communication_modes(
    *, allow_requests: bool, allow_directives: bool
) -> tuple[CommunicationMode, ...]:
    """Return the adaptive vocabulary; truthful REPORT is always available."""

    modes = [CommunicationMode.REPORT]
    if allow_requests:
        modes.append(CommunicationMode.REQUEST)
    if allow_directives:
        modes.append(CommunicationMode.DIRECTIVE)
    return tuple(modes)


def choose_communication_mode(
    controller_context: ControllerCommunicationContext,
    allowed_modes: Sequence[CommunicationMode | str],
    rng: Any,
) -> CommunicationChoice:
    """Choose one allowed mode from current public context with a seeded draw.

    The weights favor requests when reports are scarce, reports after requests,
    and coordination after evidence accumulates.  They are contextual rather
    than tied to a particular round number or intervention budget.
    """

    allowed = tuple(CommunicationMode(value) for value in allowed_modes)
    if not allowed:
        raise ValueError("adaptive communication requires at least one allowed mode")
    if len(set(allowed)) != len(allowed):
        raise ValueError("allowed communication modes must be unique")
    if CommunicationMode.MIXED in allowed:
        raise ValueError("MIXED is a round composition, not an allowed message mode")

    counts = controller_context.live_message_type_counts
    reports = int(counts.get(CommunicationMode.REPORT.value, 0))
    requests = int(counts.get(CommunicationMode.REQUEST.value, 0))
    directives = int(counts.get(CommunicationMode.DIRECTIVE.value, 0))
    weights: dict[CommunicationMode, float] = {
        CommunicationMode.REPORT: 2.0 + 1.5 * requests,
        CommunicationMode.REQUEST: 1.0 + (3.0 if reports == 0 else 0.5),
        CommunicationMode.DIRECTIVE: 1.0 + min(3.0, float(reports)) + 0.5 * directives,
    }
    selected = rng.choices(allowed, weights=[weights[mode] for mode in allowed], k=1)[0]
    reason = {
        CommunicationMode.REPORT: "share_verified_evidence",
        CommunicationMode.REQUEST: "seek_missing_public_evidence",
        CommunicationMode.DIRECTIVE: "coordinate_evidence_comparison",
    }[selected]
    return CommunicationChoice(mode=selected, reason=reason)


__all__ = [
    "COMMUNICATION_POLICIES",
    "COMMUNICATION_POLICY",
    "COMMUNICATION_POLICY_VERSION",
    "LLM_COMMUNICATION_POLICY",
    "LLM_COMMUNICATION_POLICY_VERSION",
    "LLM_AUTHORED_REPORT_ONLY_POLICY",
    "LLM_AUTHORED_REPORT_ONLY_POLICY_VERSION",
    "LLM_AUTHORED_FIXED_REPORT_ONLY_POLICY",
    "LLM_AUTHORED_FIXED_REPORT_ONLY_POLICY_VERSION",
    "LLM_AUTHORED_FULL_COMMUNICATION_POLICY",
    "LLM_AUTHORED_FULL_COMMUNICATION_POLICY_VERSION",
    "LLM_AUTHORED_VARIABLE_FULL_COMMUNICATION_POLICY",
    "LLM_AUTHORED_VARIABLE_FULL_COMMUNICATION_POLICY_VERSION",
    "LLM_AUTHORED_MIXED_FULL_COMMUNICATION_POLICY",
    "LLM_AUTHORED_MIXED_FULL_COMMUNICATION_POLICY_VERSION",
    "AuthoredControllerMessage",
    "LLM_CONTROLLER_INSTRUCTION",
    "CommunicationChoice",
    "CommunicationMode",
    "ControllerCommunicationPrompt",
    "ControllerCommunicationContext",
    "ControllerVisibleFact",
    "ControllerVisibleMessage",
    "LLMCommunicationAttempt",
    "LLMCommunicationResult",
    "allowed_communication_modes",
    "choose_communication_mode",
    "choose_llm_communication",
    "parse_llm_communication_choice",
    "render_llm_controller_prompt",
]
