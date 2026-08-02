"""Concurrent-within-pair runtime for the Phase 6 convention game."""

from __future__ import annotations

import asyncio
import hashlib
import json

from mas_cc.config import RunConfig
from mas_cc.core import Seed
from mas_cc.llm_providers import CompletionRequest, LLMProvider
from mas_cc.prompts import CompiledPrompt, RegexTokenCounter, TokenCounter

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


async def _execute_validated_decision(
    game: NamingConventionGame,
    logical: ConventionDecisionRequest,
    state,
    config: RunConfig,
    provider: LLMProvider,
    token_counter: TokenCounter,
    root_seed: Seed,
) -> ConventionDecisionOutcome:
    prompt = logical.prompt.compile(token_counter)
    attempts: list[ConventionValidationAttempt] = []
    last_error = "unknown validation failure"
    top_k = config.llm_provider.options.get("top_k")
    for attempt_index in range(logical.retry_bound + 1):
        request_seed = int(
            root_seed.derive(
                f"naming-convention-request:{logical.interaction_id}:"
                f"{logical.agent_id}:{attempt_index + 1}"
            )
        )
        completion_request = CompletionRequest(
            messages=prompt.messages,
            temperature=config.llm_provider.temperature,
            max_output_tokens=config.llm_provider.max_output_tokens,
            seed=request_seed,
            metadata={
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
            },
        )
        response = await provider.complete(completion_request)
        parsed = None
        action = None
        validation_error = None
        try:
            prompt.response_contract.validate(response.content).raise_for_errors(
                context="naming-convention response contract"
            )
            action = game.parse_action(logical, response.content)
            validation = game.validate_action(state, logical, action, config.game)
            validation.raise_for_errors(context="naming-convention action")
            parsed = ParsedConventionResponse(
                raw_text=response.content,
                value=action.value,
                reason=action.metadata.get("parsed_reason"),
                parser_mode=str(action.metadata["parser_mode"]),
            )
        except (TypeError, ValueError) as exc:
            validation_error = str(exc)
            last_error = validation_error
        attempts.append(
            ConventionValidationAttempt(
                attempt=attempt_index + 1,
                completion_request=completion_request,
                response=response,
                parsed=parsed,
                valid=action is not None and validation_error is None,
                validation_error=validation_error,
            )
        )
        if action is not None and validation_error is None:
            return ConventionDecisionOutcome(
                request=logical,
                action=action,
                parsed_reason=parsed.reason,
                parser_mode=parsed.parser_mode,
                prompt_hash=_prompt_hash(completion_request),
                prompt_definition_hash=prompt.definition_hash,
                prompt_instance_hash=prompt.instance_hash,
                compiled_prompt=prompt,
                attempts=tuple(attempts),
            )
    raise InvalidConventionResponse(
        f"{logical.agent_id} returned no valid convention action after "
        f"{len(attempts)} validation attempts: {last_error}"
    )


async def run_naming_convention_game(
    game: NamingConventionGame,
    config: RunConfig,
    provider: LLMProvider,
    *,
    token_counter: TokenCounter | None = None,
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
                game, requests[0], state, config, provider, selected_counter, root_seed
            ),
            _execute_validated_decision(
                game, requests[1], state, config, provider, selected_counter, root_seed
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
        interactions.append(
            ConventionInteractionRecord(
                interaction_id=transition.interaction_id,
                interaction_index=state.turn + 1,
                selected_agents=pair,
                pre_visible_memories=(pre_memories[0], pre_memories[1]),
                decisions=(decision_1, decision_2),
                transition=transition,
                post_visible_memories=(post_memories[0], post_memories[1]),
            )
        )
        state = next_state
        termination = game.detect_termination(state, config.game)

    decisions = tuple(
        decision for interaction in interactions for decision in interaction.decisions
    )
    return ConventionGameResult(
        initial_state=initial_state,
        final_state=state,
        interactions=tuple(interactions),
        termination_reason=termination,
        logical_decisions=len(decisions),
        validation_attempts=sum(decision.validation_attempts for decision in decisions),
        provider_retries=sum(decision.provider_retries for decision in decisions),
    )


def run_naming_convention_game_sync(*args, **kwargs) -> ConventionGameResult:
    return asyncio.run(run_naming_convention_game(*args, **kwargs))
