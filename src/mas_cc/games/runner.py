"""Small generic game loop used by the Phase 5 reference inspection."""

from __future__ import annotations

import asyncio
from typing import Any

from mas_cc.config import RunConfig
from mas_cc.core import Seed, ValidationIssue, ValidationResult
from mas_cc.llm_providers import CompletionRequest, LLMProvider
from mas_cc.prompts import PromptComposer, RegexTokenCounter, create_default_prompt_registry

from .protocols import DecisionRecord, Game, GameResult, InteractionRecord


async def run_game(
    game: Game,
    config: RunConfig,
    provider: LLMProvider,
    *,
    composer: PromptComposer | None = None,
) -> GameResult:
    """Execute a game while keeping every provider call outside transitions."""

    selected_composer = composer or PromptComposer(
        create_default_prompt_registry(), RegexTokenCounter()
    )
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
            prompt = selected_composer.compose(config.prompt, logical.prompt_context)
            response = None
            action = None
            completion_request = None
            last_validation = ValidationResult.success()
            for attempt in range(logical.retry_bound + 1):
                request_seed = int(
                    root_seed.derive(
                        f"{logical.interaction_id}:{logical.stage}:{logical.agent_id}:{attempt}"
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
                        "attempt": attempt + 1,
                        "prompt_family": config.prompt.prompt_family,
                        "prompt_version": config.prompt.prompt_version,
                        "response_contract": prompt.response_contract.to_dict(),
                    },
                )
                response = await provider.complete(completion_request)
                contract_result = prompt.response_contract.validate(response.content)
                action = game.parse_action(logical, response.content)
                game_result = game.validate_action(state, logical, action, config.game)
                issues = (*contract_result.issues, *game_result.issues)
                last_validation = ValidationResult(tuple(issues))
                if last_validation.is_valid:
                    decisions.append(
                        DecisionRecord(
                            request=logical,
                            completion_request=completion_request,
                            response=response,
                            action=action,
                            attempts=attempt + 1,
                        )
                    )
                    break
            else:
                if not last_validation.issues:
                    last_validation = ValidationResult.failure(
                        ValidationIssue("response", "could not resolve a valid action")
                    )
                last_validation.raise_for_errors(
                    context=f"{logical.interaction_id} {logical.agent_id} decision"
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
