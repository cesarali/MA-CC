"""Concurrent-within-pair runtime for the Phase 6 convention game."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from mas_cc.config import RunConfig
from mas_cc.core import Seed
from mas_cc.llm_providers import CompletionRequest, LLMProvider
from mas_cc.prompts import CompiledPrompt, RegexTokenCounter, TokenCounter
from mas_cc.runtime import DecisionLoopExhausted, ValidationAttempt, run_validated_decision

from .game import NamingConventionGame
from .records import (
    ConventionDecisionOutcome,
    ConventionDecisionRequest,
    ConventionGameResult,
    ConventionInteractionRecord,
    ConventionValidationAttempt,
    InvalidConventionResponse,
    ParsedConventionResponse,
    PrivateMemoryEntry,
)


def _prompt_hash(request: CompletionRequest) -> str:
    canonical = json.dumps(
        request.wire_messages(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _notify(observer: Any | None, method: str, *args: Any, **payload: Any) -> None:
    """Keep the game runtime independent of any observability implementation."""

    callback = getattr(observer, method, None) if observer is not None else None
    if callback is not None:
        callback(*args, **payload)


async def _execute_validated_decision(
    game: NamingConventionGame,
    logical: ConventionDecisionRequest,
    state,
    config: RunConfig,
    provider: LLMProvider,
    token_counter: TokenCounter,
    root_seed: Seed,
    observer: Any | None = None,
) -> ConventionDecisionOutcome:
    """Ask/validate/retry through the shared loop, then wrap the result in
    this game's own typed audit record."""

    prompt = logical.prompt.compile(token_counter)
    _notify(
        observer, "event", "decision_started", round_index=state.turn + 1,
        interaction_id=str(logical.interaction_id), agent_id=str(logical.agent_id),
        decision_stage=logical.stage, prompt_definition_hash=prompt.definition_hash,
        prompt_instance_hash=prompt.instance_hash,
    )
    top_k = config.llm_provider.options.get("top_k")

    def _seed_for_attempt(attempt_index: int) -> int:
        return int(
            root_seed.derive(
                f"naming-convention-request:{logical.interaction_id}:"
                f"{logical.agent_id}:{attempt_index + 1}"
            )
        )

    def _metadata_for_attempt(attempt_index: int) -> dict[str, Any]:
        return {
            "game_type": game.spec.game_type,
            "game_version": game.spec.version,
            "interaction_id": str(logical.interaction_id),
            "decision_stage": logical.stage,
            "agent_id": str(logical.agent_id),
            "validation_attempt": attempt_index + 1,
            "prompt_family": config.prompt.prompt_family,
            "prompt_version": config.prompt.prompt_version,
            "prompt_definition_hash": prompt.definition_hash,
            "prompt_instance_hash": prompt.instance_hash,
            "requested_sampling": {
                "temperature": config.llm_provider.temperature,
                "top_k": top_k,
                "max_tokens": config.llm_provider.max_output_tokens,
            },
            "parameters_sent_by_normalized_request": [
                "temperature",
                "max_output_tokens",
                "seed",
            ],
            "unsupported_or_adapter_omitted_parameters": (
                ["top_k"] if top_k is not None else []
            ),
        }

    def _on_attempt(attempt: ValidationAttempt) -> None:
        _notify(
            observer, "record_attempt", round_index=state.turn + 1,
            game_id=game.spec.game_type, request=attempt.request, prompt=prompt,
            response=attempt.response, attempt=attempt.attempt,
            valid=attempt.valid, validation_error=attempt.validation_error,
            provider_error=(
                RuntimeError(attempt.provider_error) if attempt.provider_error else None
            ),
            observation=logical.observation.to_dict(),
        )

    try:
        decision = await run_validated_decision(
            game=game, state=state, request=logical, game_config=config.game,
            provider=provider, prompt=prompt,
            temperature=config.llm_provider.temperature,
            max_output_tokens=config.llm_provider.max_output_tokens,
            seed_for_attempt=_seed_for_attempt,
            metadata_for_attempt=_metadata_for_attempt,
            on_attempt=_on_attempt,
        )
    except DecisionLoopExhausted as exc:
        raise InvalidConventionResponse(str(exc)) from exc

    attempts = tuple(
        ConventionValidationAttempt(
            attempt=attempt.attempt,
            completion_request=attempt.request,
            response=attempt.response,
            parsed=(
                None
                if attempt.action is None
                else ParsedConventionResponse(
                    raw_text=attempt.response.content,
                    value=attempt.action.value,
                    reason=attempt.action.metadata.get("parsed_reason"),
                    parser_mode=str(attempt.action.metadata["parser_mode"]),
                )
            ),
            valid=attempt.valid,
            validation_error=attempt.validation_error,
        )
        for attempt in decision.attempts
    )
    final_request = decision.attempts[-1].request
    return ConventionDecisionOutcome(
        request=logical,
        action=decision.action,
        parsed_reason=decision.action.metadata.get("parsed_reason"),
        parser_mode=str(decision.action.metadata["parser_mode"]),
        prompt_hash=_prompt_hash(final_request),
        prompt_definition_hash=prompt.definition_hash,
        prompt_instance_hash=prompt.instance_hash,
        compiled_prompt=prompt,
        attempts=attempts,
    )


async def run_naming_convention_game(
    game: NamingConventionGame,
    config: RunConfig,
    provider: LLMProvider,
    *,
    token_counter: TokenCounter | None = None,
    observer: Any | None = None,
) -> ConventionGameResult:
    """Run sequential pairs with a two-request concurrency barrier per pair."""

    rules = game.rules(config.game)
    if config.prompt.prompt_family != rules.prompt_contract:
        raise ValueError(
            "resolved prompt family does not match game.options.prompt_contract"
        )
    if config.prompt.message_mode is not None and config.prompt.message_mode != (
        "merge_consecutive_roles"
    ):
        raise ValueError("naming convention prompt requires merge_consecutive_roles")
    if config.prompt.block_separator is not None and config.prompt.block_separator != "\n\n":
        raise ValueError("naming convention prompt requires the registered block separator")
    configured_actions = tuple(
        str(item)
        for item in config.prompt.response_contract.get("allowed_values", ())
    )
    if set(configured_actions) != set(rules.actions):
        raise ValueError(
            "prompt response-contract actions must match the game action pool"
        )
    selected_counter = token_counter or RegexTokenCounter()
    root_seed = Seed(config.execution.seed)
    pair_rng = root_seed.derive("naming-convention-pair-sampling").create_random()
    state = game.initialize(config.game, config.execution.seed)
    initial_state = state
    _notify(observer, "event", "run_started", game_type=game.spec.game_type)
    interactions: list[ConventionInteractionRecord] = []
    termination = game.detect_termination(state, config.game)

    while termination is None:
        pair = game.select_participants(state, config.game, pair_rng)
        observations = game.construct_observations(state, pair, config.game)
        requests = game.build_decision_requests(state, observations, config.game)
        pre_memories = tuple(
            tuple(
                PrivateMemoryEntry.from_mapping(item)
                for item in request.visible_memory
            )
            for request in requests
        )
        # Both coroutines receive the same immutable pre-interaction state.  No
        # transition occurs until both validated decisions cross this barrier.
        decision_1, decision_2 = await asyncio.gather(
            _execute_validated_decision(
                game, requests[0], state, config, provider, selected_counter, root_seed, observer
            ),
            _execute_validated_decision(
                game, requests[1], state, config, provider, selected_counter, root_seed, observer
            ),
        )
        transition = game.apply_transition(
            state,
            pair,
            (decision_1.action, decision_2.action),
            config.game,
        )
        next_state = transition.next_state
        rules = game.rules(config.game)
        post_memories = tuple(
            next_state.convention_agent(agent_id).visible_history(rules.memory_size)
            for agent_id in pair
        )
        interaction = ConventionInteractionRecord(
            interaction_id=transition.interaction_id,
            interaction_index=state.turn + 1,
            selected_agents=pair,
            pre_visible_memories=(pre_memories[0], pre_memories[1]),
            decisions=(decision_1, decision_2),
            transition=transition,
            post_visible_memories=(post_memories[0], post_memories[1]),
        )
        interactions.append(interaction)
        _notify(
            observer, "record_interaction", round_index=state.turn + 1,
            interaction=interaction, state=next_state.to_dict(),
            prompt_definitions={
                "pair_decision": decision.prompt_definition_hash
                for decision in interaction.decisions
            },
        )
        state = next_state
        termination = game.detect_termination(state, config.game)

    decisions = tuple(
        decision for interaction in interactions for decision in interaction.decisions
    )
    result = ConventionGameResult(
        initial_state=initial_state,
        final_state=state,
        interactions=tuple(interactions),
        termination_reason=termination,
        logical_decisions=len(decisions),
        validation_attempts=sum(decision.validation_attempts for decision in decisions),
        provider_retries=sum(decision.provider_retries for decision in decisions),
    )
    _notify(observer, "event", "game_completed", interactions=len(result.interactions))
    return result


def run_naming_convention_game_sync(*args, **kwargs) -> ConventionGameResult:
    return asyncio.run(run_naming_convention_game(*args, **kwargs))
