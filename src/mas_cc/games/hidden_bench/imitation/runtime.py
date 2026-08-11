"""Observer-aware runtime for reasoning and provider-free imitation modes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping

from mas_cc.config import RunConfig
from mas_cc.control import Control, NoneControl
from mas_cc.core import Seed
from mas_cc.games.protocols import Action
from mas_cc.llm_runtime.prompts import RegexTokenCounter, TokenCounter
from mas_cc.llm_runtime.providers import LLMProvider

from ..runtime import HiddenBenchDecision, _execute_decision, _notify
from .classical import sample_jump
from .controller import ADVOCATE_TARGET, controller_from_game_options
from .game import FOCAL_UPDATE, HiddenBenchImitationGame
from .state import ImitationGameState, ImitationTransition

PROMPT_FAMILIES = (
    "hidden_bench_imitation_initial",
    "hidden_bench_imitation_message",
    "hidden_bench_imitation_update",
)


@dataclass(frozen=True, slots=True)
class ImitationInteractionRecord:
    interaction_id: Any
    interaction_index: int
    phase: str
    participants: tuple[Any, ...]
    decisions: tuple[HiddenBenchDecision, ...]
    transition: ImitationTransition

    def to_dict(self) -> dict[str, Any]:
        return {
            "interaction_id": str(self.interaction_id),
            "interaction_index": self.interaction_index,
            "phase": self.phase,
            "participants": [str(item) for item in self.participants],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "payoffs": dict(self.transition.payoffs),
            "matched": self.transition.matched,
            "event": dict(self.transition.event or {}),
        }


@dataclass(frozen=True, slots=True)
class ImitationGameResult:
    initial_state: ImitationGameState
    final_state: ImitationGameState
    initial_decisions: tuple[HiddenBenchDecision, ...]
    interactions: tuple[ImitationInteractionRecord, ...]
    termination_reason: str
    logical_decisions: int
    validation_attempts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_state": self.initial_state.to_dict(),
            "initial_votes": list(self.initial_state.initial_votes),
            "initial_decisions": [item.to_dict() for item in self.initial_decisions],
            "final_state": self.final_state.to_dict(),
            "interactions": [item.to_dict() for item in self.interactions],
            "termination_reason": self.termination_reason,
            "counters": {
                "logical_decisions": self.logical_decisions,
                "validation_attempts": self.validation_attempts,
            },
        }


def _resolved_control(config: RunConfig, control: Control | None) -> Control | None:
    if control is not None and not isinstance(control, NoneControl):
        return control
    nested = controller_from_game_options(config.game.options)
    return nested if nested is not None else control


async def run_hidden_bench_imitation_game(
    game: HiddenBenchImitationGame,
    config: RunConfig,
    provider: LLMProvider,
    *,
    token_counter: TokenCounter | None = None,
    observer: Any | None = None,
    control: Control | None = None,
) -> ImitationGameResult:
    """Run a matched imitation episode without entering the provider in classical mode."""

    rules = game.rules(config.game)
    if config.prompt.prompt_family not in PROMPT_FAMILIES:
        raise ValueError(
            f"prompt.prompt_family must be one of {PROMPT_FAMILIES}; "
            f"got {config.prompt.prompt_family!r}"
        )
    if config.prompt.prompt_version != rules.prompt_version:
        # Two places name the prompt version and both are load-bearing:
        # `prompt.prompt_version` is what preflight prices and what the audit
        # record carries, `game.options.prompt_version` is what the game builds.
        # A silent disagreement would price v1 and run v2.
        raise ValueError(
            f"prompt.prompt_version is {config.prompt.prompt_version} but "
            f"game.options.prompt_version is {rules.prompt_version}; they must match"
        )
    counter = token_counter or RegexTokenCounter()
    root = Seed(config.execution.seed)
    participant_rng = root.derive("imitation-focal-and-peer-selection").create_random()
    transition_rng = root.derive("imitation-classical-transition").create_random()
    sensor_rng = root.derive("imitation-controller-sensor").create_random()
    resolved_control = _resolved_control(config, control)
    state = game.initialize(config.game, config.execution.seed)
    _notify(
        observer,
        "event",
        "run_started",
        game_type=game.spec.game_type,
        dynamics_mode=rules.dynamics_mode,
        seed=config.execution.seed,
    )

    initial_decisions: tuple[HiddenBenchDecision, ...] = ()
    if not state.initial_votes:
        if rules.dynamics_mode != "reasoning":
            raise ValueError("classical initialization must never require provider decisions")
        requests = game.initial_vote_requests(state, config.game)
        initial_decisions = tuple(
            await asyncio.gather(
                *(
                    _execute_decision(
                        game, request, state, config, provider, counter, root, observer
                    )
                    for request in requests
                )
            )
        )
        state = game.apply_initial_votes(
            state, tuple(decision.action for decision in initial_decisions)
        )
    initial_state = state
    _notify(
        observer,
        "event",
        "imitation_initialized",
        initial_votes=list(state.initial_votes),
        provider_decisions=len(initial_decisions),
    )

    interactions: list[ImitationInteractionRecord] = []
    while state.turn < rules.horizon and not state.terminated:
        focal, sampled_peer = game.select_participants(state, config.game, participant_rng)
        signal = None
        if resolved_control is not None:
            signal = resolved_control.interaction_signal(
                agent_id=focal,
                interaction_index=state.turn + 1,
                state=state,
                rng=sensor_rng,
            )
        # Part 3.1: a control event **replaces** the peer conversation, it never
        # adds one.  Adding would give the controlled group more conversations
        # than the uncontrolled group, and any difference could then be "more
        # talking happened" rather than control.  Replacement costs a real peer
        # exchange - which is where facts spread - but that cost is visible in
        # `disclosure_events` afterwards, whereas unequal event counts would be
        # baked into the design and untanglable.
        #
        # Part 3.3: this line sits *above* the mode branch on purpose, so the
        # substitution is identical in reasoning and classical mode.  Classical
        # control additionally tilts the transition weight toward Z, but it
        # consumes the same peer slot, which is what keeps B - D unconfounded.
        peer = None if signal is not None and signal.action == ADVOCATE_TARGET else sampled_peer
        decisions: list[HiddenBenchDecision] = []
        dialogue: list[Mapping[str, Any]] = []

        if rules.dynamics_mode == "reasoning":
            if peer is not None:
                participants = (focal, peer)
                for _ in range(rules.messages_per_agent):
                    requests = game.message_requests(
                        state, participants, tuple(dialogue), config.game
                    )
                    message_decisions = tuple(
                        await asyncio.gather(
                            *(
                                _execute_decision(
                                    game,
                                    request,
                                    state,
                                    config,
                                    provider,
                                    counter,
                                    root,
                                    observer,
                                )
                                for request in requests
                            )
                        )
                    )
                    decisions.extend(message_decisions)
                    dialogue.extend(
                        {
                            "agent_id": str(decision.action.agent_id),
                            "message": decision.action.value,
                            "interaction_index": state.turn + 1,
                            "private": True,
                        }
                        for decision in message_decisions
                    )
            request = game.focal_update_request(
                state,
                focal,
                peer,
                tuple(dialogue),
                None if signal is None else signal.message,
                config.game,
            )
            update = await _execute_decision(
                game, request, state, config, provider, counter, root, observer
            )
            decisions.append(update)
            action = update.action
            classical_metadata = None
        else:
            votes = [str(agent.committed_action) for agent in state.agents]
            focal_index = next(
                index for index, agent in enumerate(state.agents) if agent.agent_id == focal
            )
            jump = sample_jump(
                votes,
                focal_index,
                state.possible_answers,
                rules,
                transition_rng,
                signal,
            )
            action = Action(
                focal,
                jump.destination,
                FOCAL_UPDATE,
                {"kind": "classical_jump", "reaction_id": jump.reaction_id},
            )
            classical_metadata = {
                "classical_reaction_id": jump.reaction_id,
                "classical_source_opinion": jump.source,
                "classical_destination_opinion": jump.destination,
                "classical_transition_rate_or_weight": jump.transition_weight,
                "classical_total_rate_or_normalizer": jump.total_weight,
                "classical_base_weight": jump.base_weight,
                "classical_control_weight": jump.control_weight,
                "classical_candidate_channels": list(jump.candidate_channels),
                "physical_time_increment": None,
                "time_convention": "focal_conditioned_embedded_jump_chain",
            }

        transition = game.apply_event_transition(
            state,
            focal=focal,
            peer=peer,
            action=action,
            config=config.game,
            signal=signal,
            dialogue=tuple(dialogue),
            classical=classical_metadata,
            sampled_peer=sampled_peer,
        )
        participants = (focal,) if peer is None else (focal, peer)
        record = ImitationInteractionRecord(
            interaction_id=transition.interaction_id,
            interaction_index=state.turn + 1,
            phase=FOCAL_UPDATE,
            participants=participants,
            decisions=tuple(decisions),
            transition=transition,
        )
        interactions.append(record)
        prompt_definitions = {
            decision.request.stage: decision.prompt_definition_hash for decision in decisions
        }
        _notify(
            observer,
            "record_interaction",
            round_index=state.turn + 1,
            interaction=record,
            state=transition.next_state.to_dict(),
            prompt_definitions=prompt_definitions,
        )
        _notify(observer, "record_trajectory", record=record)
        _notify(observer, "event", "imitation_transition", **dict(transition.event or {}))
        state = transition.next_state

    termination = state.termination_reason or "max_interactions_reached"
    every = (*initial_decisions, *(decision for item in interactions for decision in item.decisions))
    _notify(observer, "event", "game_completed", interactions=len(interactions))
    return ImitationGameResult(
        initial_state=initial_state,
        final_state=state,
        initial_decisions=initial_decisions,
        interactions=tuple(interactions),
        termination_reason=termination,
        logical_decisions=len(every),
        validation_attempts=sum(decision.validation_attempts for decision in every),
    )


def run_hidden_bench_imitation_game_sync(*args: Any, **kwargs: Any) -> ImitationGameResult:
    return asyncio.run(run_hidden_bench_imitation_game(*args, **kwargs))
