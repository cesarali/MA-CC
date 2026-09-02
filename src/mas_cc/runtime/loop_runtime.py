"""The one provider-agnostic, game-agnostic "ask / validate / retry / log" loop.

Every LLM-backed decision, in every game, goes through the same mechanical
steps: build a request from a compiled prompt, call the provider, check the
response against the game's own rules, retry up to a bound, and hand back a
full record of every attempt. This module owns that loop exactly once.

It knows nothing about what a legal answer looks like for any particular
game — validity is entirely defined by the already-generic `Game` contract
(`prompt.response_contract`, `game.parse_action`, `game.validate_action`).
It also knows nothing about which provider it's talking to; `provider` is
whatever normalized `LLMProvider` the caller constructed. Games are the
reason a call happens, never the mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING, Any, Callable, Mapping

from mas_cc.config import GameConfig
from mas_cc.llm_runtime.providers import CompletionRequest, CompletionResponse, LLMProvider
from mas_cc.llm_runtime.prompts import CompiledPrompt
from mas_cc.llm_runtime.messages import Message, MessageRole
from mas_cc.llm_runtime.exceptions import ValidationError
from mas_cc.llm_runtime.validation import ValidationIssue

if TYPE_CHECKING:
    # Deferred: `mas_cc.games` imports this module (via `runner.py` and
    # `naming_convention/runtime.py`), so a module-level import here would
    # cycle back before `mas_cc.games.protocols` finishes loading. Safe to
    # defer because `from __future__ import annotations` makes every
    # annotation below a lazy string - these names are never evaluated at
    # runtime, only used by type checkers.
    from mas_cc.games.protocols import Action, DecisionRequest, Game, GameState


class DecisionLoopExhausted(RuntimeError):
    """Raised when no attempt validated within the request's retry bound."""


@dataclass(frozen=True, slots=True)
class ValidationAttempt:
    attempt: int
    request: CompletionRequest
    response: CompletionResponse | None
    action: Action | None
    validation_error: str | None
    provider_error: str | None
    validation_issues: tuple[ValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return self.action is not None and self.validation_error is None and self.provider_error is None


@dataclass(frozen=True, slots=True)
class ValidatedDecision:
    action: Action
    attempts: tuple[ValidationAttempt, ...]


async def run_validated_decision(
    *,
    game: Game,
    state: GameState,
    request: DecisionRequest,
    game_config: GameConfig,
    provider: LLMProvider,
    prompt: CompiledPrompt,
    temperature: float,
    max_output_tokens: int,
    seed_for_attempt: Callable[[int], int],
    metadata_for_attempt: Callable[[int], Mapping[str, Any]],
    on_attempt: Callable[[ValidationAttempt], None] | None = None,
    forced_action: "Action | None" = None,
) -> ValidatedDecision:
    """Ask, validate, retry up to `request.retry_bound`, and log every attempt.

    `seed_for_attempt`/`metadata_for_attempt` take the zero-based attempt
    index so each caller keeps its own seed-derivation and metadata shape;
    this function only supplies the loop, not those per-caller details.

    A transport failure (the provider call itself raising) is not retried:
    it is reported through `on_attempt` and re-raised immediately. Only a
    response that fails the game's own validation is retried.

    `forced_action`, when not `None`, short-circuits the loop entirely: no
    `CompletionRequest` is built and `provider` is never called. This is the
    one shared seam every game's runtime uses to let a `Control` (see
    `mas_cc.control`) override a decision before the LLM is ever asked.
    """

    if forced_action is not None:
        return ValidatedDecision(action=forced_action, attempts=())

    attempts: list[ValidationAttempt] = []
    last_error = "unknown validation failure"
    messages = prompt.messages
    for attempt_index in range(request.retry_bound + 1):
        metadata = dict(metadata_for_attempt(attempt_index))
        metadata.update(
            {
                "validation_repair": attempt_index > 0,
                "repair_schema_version": 1,
                "validation_retry_bound": request.retry_bound,
                "correction_message_hash": (
                    hashlib.sha256(messages[-1].content.encode("utf-8")).hexdigest()
                    if attempt_index > 0 else None
                ),
                "effective_messages_hash": hashlib.sha256(
                    json.dumps(
                        [message.to_dict() for message in messages],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
        completion_request = CompletionRequest(
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            seed=seed_for_attempt(attempt_index),
            metadata=metadata,
        )
        try:
            response = await provider.complete(completion_request)
        except Exception as exc:
            failed = ValidationAttempt(attempt_index + 1, completion_request, None, None, None, str(exc))
            if on_attempt is not None:
                on_attempt(failed)
            raise
        action: Action | None = None
        validation_error: str | None = None
        validation_issues: tuple[ValidationIssue, ...] = ()
        try:
            prompt.response_contract.validate(response.content).raise_for_errors(
                context=f"{game.spec.game_type} response contract"
            )
            action = game.parse_action(request, response.content)
            game.validate_action(state, request, action, game_config).raise_for_errors(
                context=f"{game.spec.game_type} action"
            )
        except (TypeError, ValueError) as exc:
            validation_error = str(exc)
            last_error = validation_error
            if isinstance(exc, ValidationError):
                validation_issues = tuple(
                    issue for issue in exc.issues if isinstance(issue, ValidationIssue)
                )
        attempt = ValidationAttempt(
            attempt_index + 1, completion_request, response, action, validation_error, None,
            validation_issues,
        )
        attempts.append(attempt)
        if on_attempt is not None:
            on_attempt(attempt)
        if attempt.valid:
            return ValidatedDecision(action=action, attempts=tuple(attempts))
        if attempt_index < request.retry_bound:
            guidance = prompt.response_contract.repair_guidance(validation_issues)
            messages = (*prompt.messages, Message(MessageRole.USER, guidance))
    raise DecisionLoopExhausted(
        f"{request.agent_id} produced no valid action after {len(attempts)} validation attempts: {last_error}"
    )
