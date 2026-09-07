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


class CommunicationMode(str, Enum):
    REPORT = "REPORT"
    REQUEST = "REQUEST"
    DIRECTIVE = "DIRECTIVE"


COMMUNICATION_POLICY = "contextual_weighted_v1"
COMMUNICATION_POLICY_VERSION = 1
LLM_COMMUNICATION_POLICY = "llm_structured_v1"
LLM_COMMUNICATION_POLICY_VERSION = 1
COMMUNICATION_POLICIES = (COMMUNICATION_POLICY, LLM_COMMUNICATION_POLICY)

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
    budget: int = 0

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
        }


@dataclass(frozen=True, slots=True)
class CommunicationChoice:
    mode: CommunicationMode
    reason: str
    policy: str = COMMUNICATION_POLICY
    policy_version: int = COMMUNICATION_POLICY_VERSION
    fact_ids: tuple[str, ...] = ()
    text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "reason": self.reason,
            "policy": self.policy,
            "policy_version": self.policy_version,
            "fact_ids": list(self.fact_ids),
            "text": self.text,
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


def render_llm_controller_prompt(
    context: ControllerCommunicationContext,
    allowed_modes: Sequence[CommunicationMode | str],
) -> str:
    """Render only the explicitly permitted public controller information."""

    allowed = [CommunicationMode(mode).value for mode in allowed_modes]
    payload = context.to_dict()
    payload["allowed_modes"] = allowed
    return (
        f"{LLM_CONTROLLER_INSTRUCTION}\n\n"
        "Return one JSON object with exactly these fields:\n"
        '{"mode":"REPORT|REQUEST|DIRECTIVE","fact_ids":[],"text":null,"reason":"..."}\n'
        "For REPORT, choose 1 through budget distinct eligible fact IDs. For REQUEST "
        "or DIRECTIVE, fact_ids must be empty. Always set text to null; code renders "
        "the permitted canonical public message.\n\n"
        "CONTROLLER INFORMATION\n"
        + json.dumps(payload, sort_keys=True, ensure_ascii=False)
    )


def parse_llm_communication_choice(
    content: str,
    *,
    context: ControllerCommunicationContext,
    allowed_modes: Sequence[CommunicationMode | str],
) -> CommunicationChoice:
    """Validate a structured choice against modes, budget, and fact eligibility."""

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response must be valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("response must be one JSON object")
    unknown = set(payload) - {"mode", "fact_ids", "text", "reason"}
    if unknown:
        raise ValueError(f"response contains unknown fields: {sorted(unknown)}")
    try:
        mode = CommunicationMode(str(payload.get("mode", "")))
    except ValueError as exc:
        raise ValueError("mode must be REPORT, REQUEST, or DIRECTIVE") from exc
    allowed = {CommunicationMode(value) for value in allowed_modes}
    if mode not in allowed:
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
    reason = payload.get("reason", "llm_selected")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    eligible = {fact.fact_id for fact in context.eligible_facts}
    if mode == CommunicationMode.REPORT:
        if not 1 <= len(fact_ids) <= context.budget:
            raise ValueError("REPORT must select between 1 and budget fact IDs")
        if set(fact_ids) - eligible:
            raise ValueError("REPORT selected a fact ID outside the eligible pool")
        if text is not None:
            raise ValueError(
                "REPORT text must be null; canonical text is rendered in code"
            )
    else:
        if fact_ids:
            raise ValueError(f"{mode.value} cannot select fact IDs")
        if text is not None:
            raise ValueError(
                f"{mode.value} text must be null; permitted text is rendered in code"
            )
    return CommunicationChoice(
        mode=mode,
        reason=reason.strip(),
        policy=LLM_COMMUNICATION_POLICY,
        policy_version=LLM_COMMUNICATION_POLICY_VERSION,
        fact_ids=fact_ids,
        text=text,
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
) -> LLMCommunicationResult:
    """Ask once, with bounded schema repairs, for post-action communication."""

    base_prompt = render_llm_controller_prompt(context, allowed_modes)
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
                response.content, context=context, allowed_modes=allowed_modes
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
