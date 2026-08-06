"""Observer-aware runner for both HiddenBench games.

Modelled on `games/synthetic/runtime.py`: every request in one phase step is
built from the same frozen pre-step state, all of them are awaited together, and
only then does a transition happen. That barrier is what makes a simultaneous
vote actually simultaneous - no agent's vote can be influenced by another's,
because none of them exists yet when the prompts are built.

**Why this game family needs its own runner** rather than reusing
`games/runner.py::run_game`: that runner asserts every decision prompt belongs
to `config.prompt.prompt_family`, which assumes a game has exactly one prompt.
HiddenBench has two per game (discussion/vote, or message/commit) and switches
between them by phase, so a single-family check cannot express what a correct
run looks like here. This runner validates the *set* of families a game may use
instead, which is the same guarantee at the right granularity.

It is also what makes a HiddenBench grid analysable at all: only a runtime that
reports each step to an observer drives `RunRecorder`'s streaming metrics, and
`metrics/streaming.csv` is what `mas-cc analysis empowerment --grid-dir` reads.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from mas_cc.config import RunConfig
from mas_cc.core import Seed
from mas_cc.games.protocols import Action, DecisionRequest, Game
from mas_cc.llm_runtime.prompts import CompiledPrompt, RegexTokenCounter, TokenCounter
from mas_cc.llm_runtime.providers import LLMProvider
from mas_cc.runtime import DecisionLoopExhausted, ValidationAttempt, run_validated_decision

from .records import HiddenBenchGameState, HiddenBenchTransition

PROMPT_FAMILIES = {
    "hidden_bench_vanilla": ("hidden_bench_discussion", "hidden_bench_vote"),
    "hidden_bench_naming": ("hidden_bench_naming_message", "hidden_bench_naming_commit"),
}


class HiddenBenchDecisionFailed(RuntimeError):
    """Every validation attempt for one logical decision failed."""


def _notify(observer: Any | None, method: str, *args: Any, **payload: Any) -> None:
    callback = getattr(observer, method, None) if observer is not None else None
    if callback is not None:
        callback(*args, **payload)


@dataclass(frozen=True, slots=True)
class HiddenBenchDecision:
    """One completed decision, with everything needed to audit it."""

    request: DecisionRequest
    action: Action
    compiled_prompt: CompiledPrompt
    attempts: tuple[ValidationAttempt, ...]

    @property
    def validation_attempts(self) -> int:
        return len(self.attempts)

    @property
    def prompt_definition_hash(self) -> str:
        return self.compiled_prompt.definition_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": str(self.request.agent_id),
            "stage": self.request.stage,
            "prompt_family": self.request.prompt.family,
            "prompt_definition_hash": self.compiled_prompt.definition_hash,
            "prompt_instance_hash": self.compiled_prompt.instance_hash,
            "action": self.action.to_dict(),
            "validation_attempts": self.validation_attempts,
            "raw_response": (
                self.attempts[-1].response.content if self.attempts else None
            ),
        }


@dataclass(frozen=True, slots=True)
class HiddenBenchInteractionRecord:
    interaction_id: Any
    interaction_index: int
    phase: str
    participants: tuple[Any, ...]
    decisions: tuple[HiddenBenchDecision, ...]
    transition: HiddenBenchTransition

    def to_dict(self) -> dict[str, Any]:
        return {
            "interaction_id": str(self.interaction_id),
            "interaction_index": self.interaction_index,
            "phase": self.phase,
            "participants": [str(item) for item in self.participants],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "payoffs": dict(self.transition.payoffs),
            "matched": self.transition.matched,
        }


@dataclass(frozen=True, slots=True)
class HiddenBenchGameResult:
    initial_state: HiddenBenchGameState
    final_state: HiddenBenchGameState
    interactions: tuple[HiddenBenchInteractionRecord, ...]
    termination_reason: str
    logical_decisions: int
    validation_attempts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_state": self.initial_state.to_dict(),
            "final_state": self.final_state.to_dict(),
            "interactions": [item.to_dict() for item in self.interactions],
            "termination_reason": self.termination_reason,
            "counters": {
                "logical_decisions": self.logical_decisions,
                "validation_attempts": self.validation_attempts,
            },
        }


async def _execute_decision(
    game: Game,
    logical: DecisionRequest,
    state: HiddenBenchGameState,
    config: RunConfig,
    provider: LLMProvider,
    token_counter: TokenCounter,
    root_seed: Seed,
    observer: Any | None,
) -> HiddenBenchDecision:
    prompt = logical.prompt.compile(token_counter)

    def _seed_for_attempt(attempt_index: int) -> int:
        return int(
            root_seed.derive(
                f"hidden-bench-request:{logical.interaction_id}:{logical.stage}:"
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
            # The prompt family varies by phase, so it is read off the bound
            # prompt rather than off the config, which names only one of them.
            "prompt_family": logical.prompt.family,
            "prompt_version": logical.prompt.version,
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
            # `visible_state` carries this agent's own facts and nothing else,
            # so an audit record never becomes a channel for the information the
            # experiment is trying to keep private.
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
        # Not swallowed into a default vote. A vote that never parsed and is
        # silently counted as wrong is one of the three failure modes the brief
        # (§9.5) names as the likely cause of a bad reproduction number.
        raise HiddenBenchDecisionFailed(
            f"no valid {logical.stage} action from {logical.agent_id}: {exc}"
        ) from exc

    return HiddenBenchDecision(
        request=logical, action=decision.action, compiled_prompt=prompt,
        attempts=decision.attempts,
    )


async def run_hidden_bench_game(
    game: Game,
    config: RunConfig,
    provider: LLMProvider,
    *,
    token_counter: TokenCounter | None = None,
    observer: Any | None = None,
) -> HiddenBenchGameResult:
    """Run one HiddenBench episode, one phase step at a time."""

    families = PROMPT_FAMILIES.get(game.spec.game_type)
    if families is None:
        raise ValueError(f"{game.spec.game_type!r} is not a HiddenBench game")
    if config.prompt.prompt_family not in families:
        raise ValueError(
            f"prompt.prompt_family must be one of {families} for {game.spec.game_type!r}; "
            f"got {config.prompt.prompt_family!r}. This game switches prompt family by phase; "
            "the configured one names the family used for the run's decision-bearing stage."
        )
    counter = token_counter or RegexTokenCounter()
    root_seed = Seed(config.execution.seed)
    participant_rng = root_seed.derive("hidden-bench-participant-selection").create_random()
    state = game.initialize(config.game, config.execution.seed)
    initial_state = state
    _notify(observer, "event", "run_started", game_type=game.spec.game_type)

    interactions: list[HiddenBenchInteractionRecord] = []
    termination = game.detect_termination(state, config.game)
    while termination is None:
        participants = game.select_participants(state, config.game, participant_rng)
        observations = game.construct_observations(state, participants, config.game)
        requests = game.build_decision_requests(state, observations, config.game)
        if not requests:
            raise ValueError("hidden_bench game returned no decision requests for an active step")
        for logical in requests:
            if logical.prompt.family not in families:
                raise ValueError(
                    f"stage {logical.stage!r} bound prompt family {logical.prompt.family!r} "
                    f"is not one of this game's families {families}"
                )
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
        phase = state.phase
        transition = game.apply_transition(
            state, participants, tuple(decision.action for decision in decisions), config.game
        )
        interaction = HiddenBenchInteractionRecord(
            interaction_id=transition.interaction_id,
            interaction_index=state.turn + 1,
            phase=phase,
            participants=participants,
            decisions=decisions,
            transition=transition,
        )
        interactions.append(interaction)
        _notify(
            observer, "record_interaction", round_index=state.turn + 1,
            interaction=interaction, state=transition.next_state.to_dict(),
            prompt_definitions={phase: decisions[0].prompt_definition_hash},
        )
        state = transition.next_state
        termination = game.detect_termination(state, config.game)

    every = tuple(decision for item in interactions for decision in item.decisions)
    _notify(observer, "event", "game_completed", interactions=len(interactions))
    return HiddenBenchGameResult(
        initial_state=initial_state,
        final_state=state,
        interactions=tuple(interactions),
        termination_reason=termination,
        logical_decisions=len(every),
        validation_attempts=sum(decision.validation_attempts for decision in every),
    )


def run_hidden_bench_game_sync(*args: Any, **kwargs: Any) -> HiddenBenchGameResult:
    return asyncio.run(run_hidden_bench_game(*args, **kwargs))
