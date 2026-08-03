"""Small generic game loop used by the Phase 5 reference inspection."""

from __future__ import annotations

import asyncio
from typing import Any

from mas_cc.config import RunConfig
from mas_cc.core import Seed
from mas_cc.llm_providers import LLMProvider
from mas_cc.prompts import RegexTokenCounter, TokenCounter
from mas_cc.runtime import run_validated_decision

from .protocols import DecisionRecord, Game, GameResult, InteractionRecord


async def run_game(
    game: Game,
    config: RunConfig,
    provider: LLMProvider,
    *,
    token_counter: TokenCounter | None = None,
) -> GameResult:
    """Execute a game while keeping every provider call outside transitions."""

    selected_counter = token_counter or RegexTokenCounter()
    root_seed = Seed(config.execution.seed)
    rng = root_seed.derive("participant-selection").create_random()
    state = game.initialize(config.game, config.execution.seed)
    initial_state = state
    interactions: list[InteractionRecord] = []
    termination_reason = game.detect_termination(state, config.game)

    while termination_reason is None:
        participants = game.select_participants(state, config.game, rng)
        observations = game.construct_observations(state, participants, config.game)
        logical_requests = game.build_decision_requests(state, observations, config.game)
        if not logical_requests:
            raise ValueError("game returned no decision requests for an active interaction")
        decisions: list[DecisionRecord] = []

        for logical in logical_requests:
            if not logical.provider_required:
                raise NotImplementedError(
                    "provider-free runtime decisions require a game control resolver"
                )
            if (
                logical.prompt.family != config.prompt.prompt_family
                or logical.prompt.version != config.prompt.prompt_version
            ):
                raise ValueError("bound decision prompt does not match resolved prompt selection")
            if config.prompt.message_mode is not None and getattr(
                logical.prompt, "message_mode", None
            ) != config.prompt.message_mode:
                raise ValueError("bound decision prompt does not match configured message_mode")
            if config.prompt.block_separator is not None and getattr(
                logical.prompt, "block_separator", None
            ) != config.prompt.block_separator:
                raise ValueError("bound decision prompt does not match configured block_separator")
            prompt = logical.prompt.compile(selected_counter)
            expected_contract = config.prompt.response_contract
            if expected_contract:
                if expected_contract.get("type") != prompt.response_contract.type:
                    raise ValueError("bound prompt response contract type does not match config")
                allowed = tuple(expected_contract.get("allowed_values", ()))
                if allowed and set(allowed) != set(prompt.response_contract.allowed_values):
                    raise ValueError("bound prompt response values do not match config")

            def _seed_for_attempt(attempt: int) -> int:
                return int(
                    root_seed.derive(
                        f"{logical.interaction_id}:{logical.stage}:{logical.agent_id}:{attempt}"
                    )
                )

            def _metadata_for_attempt(attempt: int) -> dict[str, Any]:
                return {
                    "game_type": game.spec.game_type,
                    "game_version": game.spec.version,
                    "interaction_id": str(logical.interaction_id),
                    "decision_stage": logical.stage,
                    "agent_id": str(logical.agent_id),
                    "attempt": attempt + 1,
                    "prompt_family": config.prompt.prompt_family,
                    "prompt_version": config.prompt.prompt_version,
                    "prompt_definition_hash": prompt.definition_hash,
                    "prompt_instance_hash": prompt.instance_hash,
                    "response_contract": prompt.response_contract.to_dict(),
                }

            decision = await run_validated_decision(
                game=game, state=state, request=logical, game_config=config.game,
                provider=provider, prompt=prompt,
                temperature=config.llm_provider.temperature,
                max_output_tokens=config.llm_provider.max_output_tokens,
                seed_for_attempt=_seed_for_attempt,
                metadata_for_attempt=_metadata_for_attempt,
            )
            last_attempt = decision.attempts[-1]
            decisions.append(
                DecisionRecord(
                    request=logical,
                    completion_request=last_attempt.request,
                    response=last_attempt.response,
                    action=decision.action,
                    attempts=len(decision.attempts),
                    prompt_definition_hash=prompt.definition_hash,
                    prompt_instance_hash=prompt.instance_hash,
                )
            )

        transition = game.apply_transition(
            state,
            participants,
            tuple(decision.action for decision in decisions),
            config.game,
        )
        if transition.interaction_id != logical_requests[0].interaction_id:
            raise ValueError("game transition returned a mismatched interaction identifier")
        interactions.append(
            InteractionRecord(
                interaction_id=transition.interaction_id,
                turn=state.turn + 1,
                participants=participants,
                decisions=tuple(decisions),
                transition=transition,
            )
        )
        state = transition.next_state
        termination_reason = game.detect_termination(state, config.game)

    return GameResult(
        spec=game.spec,
        seed=config.execution.seed,
        initial_state=initial_state,
        final_state=state,
        interactions=tuple(interactions),
        termination_reason=termination_reason,
    )


def run_game_sync(*args: Any, **kwargs: Any) -> GameResult:
    return asyncio.run(run_game(*args, **kwargs))
