"""Fidelity mode: run a synthetic episode through the whole real pipeline.

Prompts are constructed and compiled, actions cross the provider boundary and
come back as text, `parse_action` and `validate_action` run, the recorder logs
every attempt and writes the metrics tree. Nothing is stubbed out on the way
through - the only thing that is not real is the agent at the far end, and the
fact that we already know what the answer has to be.

This is the slow mode by design. It is what exercises the adapter, the round
alignment, the serialization, and the artifact path; `simulate()` on the game
is what makes sweeps affordable. Running both on the same seed and demanding
the same trajectory is what makes the fast one trustworthy.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mas_cc.config import RunConfig
from mas_cc.core import Seed
from mas_cc.llm_runtime.prompts import RegexTokenCounter, TokenCounter
from mas_cc.llm_runtime.providers import LLMProvider
from mas_cc.runtime import DecisionLoopExhausted, ValidationAttempt, run_validated_decision

from .protocols import (
    SyntheticDecision,
    SyntheticGame,
    SyntheticGameResult,
    SyntheticInteractionRecord,
)


def _notify(observer: Any | None, method: str, *args: Any, **payload: Any) -> None:
    """Keep the runtime independent of any particular observability implementation."""

    callback = getattr(observer, method, None) if observer is not None else None
    if callback is not None:
        callback(*args, **payload)


async def _execute_decision(
    game: SyntheticGame,
    logical: Any,
    state: Any,
    config: RunConfig,
    provider: LLMProvider,
    token_counter: TokenCounter,
    root_seed: Seed,
    observer: Any | None,
) -> SyntheticDecision:
    prompt = logical.prompt.compile(token_counter)

    def _seed_for_attempt(attempt_index: int) -> int:
        return int(
            root_seed.derive(
                f"synthetic-request:{logical.interaction_id}:{logical.agent_id}:{attempt_index + 1}"
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
        }

    def _on_attempt(attempt: ValidationAttempt) -> None:
        _notify(
            observer, "record_attempt", round_index=state.turn + 1,
            game_id=game.spec.game_type, request=attempt.request, prompt=prompt,
            response=attempt.response, attempt=attempt.attempt, valid=attempt.valid,
            validation_error=attempt.validation_error,
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
        # A synthetic agent cannot legitimately fail validation - it is a
        # lookup table applied to a prompt we generated - so this is a
        # pipeline defect, not a flaky model, and it stops the run.
        raise RuntimeError(
            f"synthetic agent produced no valid action for {logical.agent_id}: {exc}"
        ) from exc

    return SyntheticDecision(
        request=logical,
        action=decision.action,
        compiled_prompt=prompt,
        attempts=decision.attempts,
    )


async def run_synthetic_game(
    game: SyntheticGame,
    config: RunConfig,
    provider: LLMProvider,
    *,
    token_counter: TokenCounter | None = None,
    observer: Any | None = None,
) -> SyntheticGameResult:
    """Run one episode with every agent deciding concurrently within a round."""

    if config.prompt.prompt_family != "synthetic_agent_decision":
        raise ValueError(
            "synthetic games require prompt.prompt_family 'synthetic_agent_decision'; "
            f"got {config.prompt.prompt_family!r}"
        )
    counter = token_counter or RegexTokenCounter()
    root_seed = Seed(config.execution.seed)
    # Seeded even though Game 1 has no pairing to draw: a game that does select
    # participants must take them from here, so that a failure replays exactly.
    participant_rng = root_seed.derive("synthetic-participant-selection").create_random()
    state = game.initialize(config.game, config.execution.seed)
    initial_state = state
    truth = game.ground_truth(config.game)
    _notify(observer, "event", "run_started", game_type=game.spec.game_type)

    interactions: list[SyntheticInteractionRecord] = []
    termination = game.detect_termination(state, config.game)
    while termination is None:
        participants = game.select_participants(state, config.game, participant_rng)
        observations = game.construct_observations(state, participants, config.game)
        requests = game.build_decision_requests(state, observations, config.game)
        if not requests:
            raise ValueError("synthetic game returned no decision requests for an active round")
        # Every request is built from the same frozen pre-round state; no
        # transition happens until all of them have come back.
        decisions = tuple(
            await asyncio.gather(
                *(
                    _execute_decision(
                        game, logical, state, config, provider, counter, root_seed, observer
                    )
                    for logical in requests
                )
            )
        )
        transition = game.apply_transition(
            state, participants, tuple(decision.action for decision in decisions), config.game
        )
        interaction = SyntheticInteractionRecord(
            interaction_id=transition.interaction_id,
            interaction_index=state.turn + 1,
            participants=participants,
            decisions=decisions,
            transition=transition,
        )
        interactions.append(interaction)
        _notify(
            observer, "record_interaction", round_index=state.turn + 1,
            interaction=interaction, state=transition.next_state.to_dict(),
            prompt_definitions={
                "synchronous_report": decisions[0].prompt_definition_hash,
            },
        )
        state = transition.next_state
        termination = game.detect_termination(state, config.game)

    every_decision = tuple(
        decision for interaction in interactions for decision in interaction.decisions
    )
    _notify(observer, "event", "game_completed", interactions=len(interactions))
    return SyntheticGameResult(
        initial_state=initial_state,
        final_state=state,
        interactions=tuple(interactions),
        termination_reason=termination,
        ground_truth=truth,
        logical_decisions=len(every_decision),
        validation_attempts=sum(decision.validation_attempts for decision in every_decision),
        metadata={"seed": config.execution.seed, "mode": "fidelity"},
    )


def run_synthetic_game_sync(*args: Any, **kwargs: Any) -> SyntheticGameResult:
    return asyncio.run(run_synthetic_game(*args, **kwargs))
