"""Two-clock runtime for the budgeted relational reasoning game.

One population round:

    population state
          |
          v
    controller senses q_c votes  (votes only - never K_i)
          |
          v
    one controller decision  {NO_OP, ADVOCATE_Z}
          |
          v
    if ADVOCATE_Z: preallocate exactly b controlled positions
          |
          v
    N microscopic focal updates, one focal agent each:
        sample focal + q social slots
            -> a controlled slot REPLACES one ordinary peer
            -> render each slot from its (vote, exposed fact)
            -> ONE focal provider call
            -> {vote, reason, shared_fact_id}
            -> apply the vote, publish the ballot, grow K_focal
          |
          v
    next population round

The decision execution itself - ask, validate, retry, record every attempt - is
the repository's shared loop (``mas_cc.runtime.run_validated_decision``); this
module only supplies the seeds, the metadata and the observer notifications
around it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from mas_cc.config import RunConfig
from mas_cc.control import (
    Control,
    InteractionControlSignal,
    NoneControl,
    RoundControlSignal,
)
from mas_cc.core import Seed
from mas_cc.games.protocols import AgentState, DecisionRequest, Game, GameState
from mas_cc.llm_runtime.prompts import CompiledPrompt, RegexTokenCounter, TokenCounter
from mas_cc.llm_runtime.providers import LLMProvider
from mas_cc.runtime import (
    DecisionLoopExhausted,
    ValidationAttempt,
    run_validated_decision,
)
from mas_cc.storage import canonical_hash

from ...hidden_bench.imitation.controller import ADVOCATE_TARGET, NO_OP
from ...hidden_bench.imitation.metrics import population_observables
from .controller import RECOMMENDATION_ONLY, SILENT
from .game import RelationalImitationRoundFeedbackGame
from .initialization import (
    initialization_artifact_path,
    paired_initialization_required,
    physical_initial_state_projection,
    read_initialization_artifact,
)
from .metrics import knowledge_observables, knowledge_strata
from .prompts import PROMPT_FAMILY, agent_label, control_label, render_control_reason
from .state import (
    ACTIVE_FACT_IDS,
    FOCAL_UPDATE,
    RelationalAgentState,
    RelationalGameState,
    RelationalRoundRecord,
    reasoning_fact_ids,
)

PROMPT_FAMILIES = (PROMPT_FAMILY,)

CONTROL_SOURCE_ID = "control-source"
"""The controller's stable identifier in the trajectory.  It is *not* what the
focal agent sees: in the prompt the controller is one more numbered
participant, see ``control_label``."""


class RelationalDecisionFailed(RuntimeError):
    """Every validation attempt for one logical decision failed."""


def _notify(observer: Any | None, method: str, *args: Any, **payload: Any) -> None:
    callback = getattr(observer, method, None) if observer is not None else None
    if callback is not None:
        callback(*args, **payload)


@dataclass(frozen=True, slots=True)
class RelationalDecision:
    """One completed decision, with everything needed to audit it."""

    request: DecisionRequest
    action: Any
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
class RelationalInteractionRecord:
    interaction_id: Any
    interaction_index: int
    phase: str
    participants: tuple[Any, ...]
    decisions: tuple[RelationalDecision, ...]
    transition: Any

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
class RelationalGameResult:
    initial_state: RelationalGameState
    final_state: RelationalGameState
    initial_decisions: tuple[RelationalDecision, ...]
    interactions: tuple[RelationalInteractionRecord, ...]
    rounds: tuple[RelationalRoundRecord, ...]
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
            "rounds": [item.to_dict() for item in self.rounds],
            "termination_reason": self.termination_reason,
            "counters": {
                "logical_decisions": self.logical_decisions,
                "validation_attempts": self.validation_attempts,
                "population_rounds": len(self.rounds),
                "microscopic_updates": len(self.interactions),
            },
        }


async def _execute_decision(
    game: Game,
    logical: DecisionRequest,
    state: RelationalGameState,
    config: RunConfig,
    provider: LLMProvider,
    token_counter: TokenCounter,
    root_seed: Seed,
    observer: Any | None,
) -> RelationalDecision:
    """Run one logical decision through the shared ask/validate/retry loop."""

    prompt = logical.prompt.compile(token_counter)

    def _seed_for_attempt(attempt_index: int) -> int:
        return int(
            root_seed.derive(
                f"relational-request:{logical.interaction_id}:{logical.stage}:"
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
            "prompt_family": logical.prompt.family,
            "prompt_version": logical.prompt.version,
            "prompt_definition_hash": prompt.definition_hash,
            "prompt_instance_hash": prompt.instance_hash,
        }

    def _on_attempt(attempt: ValidationAttempt) -> None:
        _notify(
            observer,
            "record_attempt",
            round_index=state.turn + 1,
            game_id=game.spec.game_type,
            request=attempt.request,
            prompt=prompt,
            response=attempt.response,
            attempt=attempt.attempt,
            valid=attempt.valid,
            validation_error=attempt.validation_error,
            validation_issues=attempt.validation_issues,
            provider_error=(
                RuntimeError(attempt.provider_error) if attempt.provider_error else None
            ),
            # `visible_state` carries this agent's own knowledge set and nothing
            # else, so an audit record never becomes a channel for the facts the
            # experiment is measuring the movement of.
            observation=logical.observation.to_dict(),
        )

    try:
        decision = await run_validated_decision(
            game=game,
            state=state,
            request=logical,
            game_config=config.game,
            provider=provider,
            prompt=prompt,
            temperature=config.llm_provider.temperature,
            max_output_tokens=config.llm_provider.max_output_tokens,
            seed_for_attempt=_seed_for_attempt,
            metadata_for_attempt=_metadata_for_attempt,
            on_attempt=_on_attempt,
        )
    except DecisionLoopExhausted as exc:
        # Never swallowed into a default vote: a ballot that failed to parse and
        # is silently counted as a wrong answer would corrupt every downstream
        # number without leaving a trace.
        raise RelationalDecisionFailed(
            f"no valid {logical.stage} action from {logical.agent_id}: {exc}"
        ) from exc

    return RelationalDecision(
        request=logical,
        action=decision.action,
        compiled_prompt=prompt,
        attempts=decision.attempts,
    )


def _controller_view(state: RelationalGameState) -> GameState:
    """Strip knowledge, reasons, and evidence before the controller senses.

    §9: the controller senses **votes only**.  Building its input from a
    reduced state rather than from a convention is what makes that checkable -
    ``K_i`` is not reachable from the object it is handed.
    """

    task = state.task
    return GameState(
        game_type=state.game_type,
        turn=state.turn,
        agents=tuple(
            AgentState(
                agent_id=agent.agent_id,
                attributes={"committed_action": agent.committed_action},
            )
            for agent in state.agents
        ),
        terminated=state.terminated,
        data={
            "seed": int(state.data["seed"]),
            "task": {
                "task_id": task["task_id"],
                "possible_answers": list(state.possible_answers),
                "correct_answer": state.correct_answer,
            },
        },
    )


def _round_interaction_signal(
    signal: RoundControlSignal | None,
    *,
    controlled_slot: bool,
    message: str | None = None,
) -> InteractionControlSignal | None:
    """Project the one round decision onto one microscopic position."""

    if signal is None:
        return None
    return InteractionControlSignal(
        action=signal.action,
        target=signal.target,
        message=(message if message is not None else signal.message)
        if controlled_slot
        else None,
        observation=signal.observation,
        metadata=signal.metadata,
    )


def build_social_sources(
    state: RelationalGameState,
    sampled_peers: Sequence[Any],
    *,
    replaced_peer_slot: int | None,
    controller_target: str | None,
    population_size: int,
    controller_fact_id: str | None = None,
    controller_transmits: bool = True,
) -> tuple[dict[str, Any], ...]:
    """The visible social inputs, in scheduler slot order.

    A controlled position substitutes ``(X_j, S_j) -> (Z, f_C)`` in exactly one
    slot.  Both kinds of source carry the same fields, which is what makes the
    controller indistinguishable from an ordinary participant in the rendered
    prompt - and what makes an injected fact travel down exactly the same
    channel a peer's would.

    ``controller_transmits=False`` is the occlusion placebo (``message_mode:
    silent``): the controlled slot is **vacated rather than filled**, so the
    focal sees ``q - 1`` sources and no substitute speaker.  Nothing is invented
    to stand in the empty slot - a placeholder vote, or a named participant who
    said nothing, would each be a new social object rather than the absence of
    an old one.  The surviving sources keep their original slot numbers, so the
    record still says which slot was taken away.

    ``reason`` is recorded on each source but **not rendered**: it is the
    speaker's own record, and showing it would open a second task-information
    channel beside ``shared_fact_id``.  See ``prompts.render_social_source``.
    """

    sources: list[dict[str, Any]] = []
    for slot, peer in enumerate(sampled_peers):
        if slot == replaced_peer_slot:
            if not controller_transmits:
                continue
            if controller_target is None:
                raise ValueError(
                    "a controlled social slot requires a controller target"
                )
            sources.append(
                {
                    "slot": slot,
                    "source_id": CONTROL_SOURCE_ID,
                    "source_type": "control",
                    "label": control_label(population_size),
                    "vote": controller_target,
                    "reason": render_control_reason(controller_target),
                    "shared_fact_id": controller_fact_id,
                    "shared_fact_text": (
                        None
                        if controller_fact_id is None
                        else state.fact_text(controller_fact_id)
                    ),
                }
            )
            continue
        agent = state.relational_agent(peer)
        exposed = agent.public_shared_fact_id
        if exposed not in set(reasoning_fact_ids(agent, state.epistemic_persistence)):
            exposed = None
        sources.append(
            {
                "slot": slot,
                "source_id": str(peer),
                "source_type": "ordinary",
                "label": agent_label(peer),
                "vote": agent.committed_action,
                "reason": agent.public_reason,
                "shared_fact_id": exposed,
                "shared_fact_text": (
                    None if exposed is None else state.fact_text(exposed)
                ),
            }
        )
    return tuple(sources)


def apply_epistemic_persistence(
    state: RelationalGameState,
    *,
    persistence: float,
    rng: random.Random | None,
) -> tuple[RelationalGameState, tuple[tuple[str, str], ...]]:
    """Deactivate active facts independently at one population-round boundary.

    The returned pairs are ``(agent_id, fact_id)`` in stable order.  Historical
    knowledge and every non-epistemic part of the state are left untouched.
    """

    if persistence == 1.0:
        if rng is not None:
            raise ValueError("rho=1 persistence must not receive or consume an RNG")
        for agent in state.agents:
            if isinstance(agent, RelationalAgentState):
                reasoning_fact_ids(agent, persistence)
        return state, ()
    if not 0.0 <= persistence <= 1.0:
        raise ValueError("epistemic persistence must be between 0.0 and 1.0")
    if rng is None:
        raise ValueError("finite epistemic persistence requires its dedicated RNG")

    deactivated: list[tuple[str, str]] = []
    replacements: dict[str, RelationalAgentState] = {}
    fact_order = tuple(sorted(state.fact_ids))
    for agent in sorted(state.agents, key=lambda item: str(item.agent_id)):
        if not isinstance(agent, RelationalAgentState):
            raise TypeError("relational state contains a non-relational agent")
        active_before = set(agent.active_fact_ids)
        survivors: set[str] = set()
        for fact_id in fact_order:
            if fact_id not in active_before:
                continue
            if rng.random() < persistence:
                survivors.add(fact_id)
            else:
                deactivated.append((str(agent.agent_id), fact_id))
        active_after = [fact_id for fact_id in state.fact_ids if fact_id in survivors]
        replacements[str(agent.agent_id)] = replace(
            agent,
            attributes={
                **dict(agent.attributes),
                ACTIVE_FACT_IDS: active_after,
            },
        )

    next_state = replace(
        state,
        agents=tuple(replacements[str(agent.agent_id)] for agent in state.agents),
    )
    return next_state, tuple(deactivated)


def sample_controlled_positions(
    population_size: int, intervention_budget: int, rng: Any
) -> tuple[int, ...]:
    """Uniform size-``b`` schedule sampled without replacement, canonical order."""

    if not 0 <= intervention_budget <= population_size:
        raise ValueError("intervention_budget must be between 0 and population_size")
    return tuple(sorted(rng.sample(range(population_size), intervention_budget)))


def _count_vector(values: Mapping[str, Any], options: tuple[str, ...]) -> list[int]:
    return [int(values.get(option, 0)) for option in options]


async def run_relational_imitation_round_feedback_game(
    game: RelationalImitationRoundFeedbackGame,
    config: RunConfig,
    provider: LLMProvider,
    *,
    token_counter: TokenCounter | None = None,
    observer: Any | None = None,
    control: Control | None = None,
) -> RelationalGameResult:
    """Run one episode with exactly one controller decision per population round."""

    rules = game.rules(config.game)
    if config.prompt.prompt_family not in PROMPT_FAMILIES:
        raise ValueError(
            f"prompt.prompt_family must be one of {PROMPT_FAMILIES}; "
            f"got {config.prompt.prompt_family!r}"
        )
    if config.prompt.prompt_version != rules.prompt_version:
        raise ValueError(
            f"prompt.prompt_version is {config.prompt.prompt_version} but "
            f"game.options.prompt_version is {rules.prompt_version}; they must match"
        )

    counter = token_counter or RegexTokenCounter()
    root = Seed(config.execution.seed)
    participant_rng = root.derive("relational-focal-and-peer-selection").create_random()
    sensor_rng = root.derive("relational-controller-sensor-policy").create_random()
    replacement_rng = root.derive(
        "relational-controller-slot-replacement"
    ).create_random()

    resolved_control = (
        None if control is None or isinstance(control, NoneControl) else control
    )
    sensor_sample_size = getattr(resolved_control, "sensor_sample_size", None)
    intervention_budget = int(getattr(resolved_control, "intervention_budget", 0))
    if sensor_sample_size is not None and int(sensor_sample_size) > rules.n_agents:
        raise ValueError(
            "controller sensor_sample_size cannot exceed the population size"
        )
    if not 0 <= intervention_budget <= rules.n_agents:
        raise ValueError("controller intervention_budget must be between 0 and N")

    # One fact per *episode*, resolved once from the frozen task before anything
    # runs: §11 wants the controller's evidence to be a fixed experimental
    # condition, not a second stochastic channel varying slot by slot.
    # `getattr` rather than an isinstance branch: a Control that predates these
    # options - or a test double - simply advocates without evidence, exactly as
    # `recommendation_only` does, instead of failing here.
    controller_fact_id: str | None = None
    message_mode = str(getattr(resolved_control, "message_mode", RECOMMENDATION_ONLY))
    evidence_strategy = getattr(resolved_control, "controller_evidence_strategy", None)
    state = game.initialize(config.game, config.execution.seed)
    resolver = getattr(resolved_control, "resolve_fact_id", None)
    if resolver is not None:
        controller_fact_id = resolver(
            game.load_task(config.game), config.execution.seed
        )

    _notify(
        observer,
        "event",
        "run_started",
        game_type=game.spec.game_type,
        dynamics_mode=rules.dynamics_mode,
        seed=config.execution.seed,
    )

    initial_decisions: tuple[RelationalDecision, ...] = ()
    initialization_artifact_hash: str | None = None
    initialization_repetition: int | None = None
    initialization_source = "provider_free"
    if not state.initial_votes and paired_initialization_required(config):
        artifact, _, state = read_initialization_artifact(
            initialization_artifact_path(config, config.execution.seed),
            game,
            config,
            config.execution.seed,
        )
        initialization_artifact_hash = str(artifact["artifact_hash"])
        initialization_repetition = int(artifact["repetition_index"])
        initialization_source = "paired_artifact"
    elif not state.initial_votes:
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
        initialization_source = "provider_local_vote"
    initial_state = state
    physical_initial_state_hash = canonical_hash(
        physical_initial_state_projection(initial_state)
    )
    _notify(
        observer,
        "event",
        "relational_round_feedback_initialized",
        initial_votes=list(state.initial_votes),
        provider_decisions=len(initial_decisions),
        initialization_source=initialization_source,
        initialization_artifact_hash=initialization_artifact_hash,
        physical_initial_state_hash=physical_initial_state_hash,
    )

    interactions: list[RelationalInteractionRecord] = []
    round_records: list[RelationalRoundRecord] = []
    for round_index in range(rules.rounds):
        if state.terminated:
            break
        options = tuple(state.possible_answers)
        population_before = [str(agent.committed_action) for agent in state.agents]
        active_knowledge_before = knowledge_observables(
            state.agents,
            state.supporting_fact_ids,
            fact_ids_attribute=ACTIVE_FACT_IDS,
        )
        historical_knowledge_before = knowledge_observables(
            state.agents, state.supporting_fact_ids
        )
        strata_before = knowledge_strata(
            state.agents,
            state.supporting_fact_ids,
            state.correct_answer,
            fact_ids_attribute=ACTIVE_FACT_IDS,
        )

        round_signal: RoundControlSignal | None = None
        if resolved_control is not None:
            round_signal = resolved_control.round_signal(
                round_index=round_index,
                state=_controller_view(state),
                rng=sensor_rng,
            )
            if round_signal is not None and not isinstance(
                round_signal, RoundControlSignal
            ):
                raise TypeError(
                    "Control.round_signal must return RoundControlSignal or None"
                )
            if round_signal is None:
                raise ValueError(
                    "the selected control does not implement round-level signaling"
                )

        action = None if round_signal is None else round_signal.action
        if action is not None and action not in {NO_OP, ADVOCATE_TARGET}:
            raise ValueError("round controller action must be NO_OP or ADVOCATE_Z")
        target = None if round_signal is None else round_signal.target
        analysis_target = state.correct_answer if target is None else target
        if analysis_target not in options:
            raise ValueError("controller target is outside the task option alphabet")
        probability = (
            None
            if round_signal is None
            else round_signal.metadata.get("advocacy_probability")
        )
        schedule_seed = int(root.derive(f"relational-schedule:{round_index}"))
        controlled_positions = (
            sample_controlled_positions(
                rules.n_agents,
                intervention_budget,
                root.derive(f"relational-schedule:{round_index}").create_random(),
            )
            if action == ADVOCATE_TARGET
            else ()
        )
        controlled_set = frozenset(controlled_positions)
        schedule_hash = hashlib.sha256(
            json.dumps(list(controlled_positions), separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        # Evidence only ever reaches a focal agent through an actuated slot, so
        # a NO_OP round transmits nothing even under recommendation_plus_fact.
        round_controller_fact = (
            controller_fact_id if action == ADVOCATE_TARGET else None
        )
        before_obs = population_observables(
            population_before, options, state.correct_answer, analysis_target
        )

        peer_exposures = 0
        controller_exposures = 0
        new_peer_facts = 0
        new_controller_facts = 0
        reactivated_peer_facts = 0
        reactivated_controller_facts = 0
        # Controlled updates where the focal was not already standing on the
        # controller's target, and how many of those moved onto it. Counted
        # online because the microscopic trajectory is not retained under the
        # compact artifact profile.
        controlled_off_target = 0
        controlled_adoptions = 0

        for within_round_index in range(rules.n_agents):
            selected = game.select_participants(state, config.game, participant_rng)
            focal, sampled_peers = selected[0], tuple(selected[1:])
            controlled_slot = within_round_index in controlled_set
            replaced_peer_slot = (
                replacement_rng.randrange(len(sampled_peers))
                if controlled_slot
                else None
            )
            replaced_peer = (
                None
                if replaced_peer_slot is None
                else sampled_peers[replaced_peer_slot]
            )
            effective_peers = tuple(
                peer
                for slot, peer in enumerate(sampled_peers)
                if slot != replaced_peer_slot
            )
            social_sources = build_social_sources(
                state,
                sampled_peers,
                replaced_peer_slot=replaced_peer_slot,
                controller_target=target,
                population_size=rules.n_agents,
                controller_fact_id=round_controller_fact,
                controller_transmits=message_mode != SILENT,
            )
            request = game.ballot_request(state, focal, social_sources, config.game)
            update = await _execute_decision(
                game, request, state, config, provider, counter, root, observer
            )
            focal_action = update.action

            control_source = next(
                (
                    source
                    for source in social_sources
                    if source["source_type"] == "control"
                ),
                None,
            )
            micro_signal = _round_interaction_signal(
                round_signal,
                # An occluded position (`message_mode: silent`) is controlled but
                # transmits nothing, so it records no controller message. Without
                # this, `_round_interaction_signal` would fall back to the round
                # signal's own legacy text and the trajectory would claim words
                # that were never put in front of anybody.
                controlled_slot=controlled_slot and control_source is not None,
                message=None
                if control_source is None
                else str(control_source["reason"]),
            )
            micro_index = state.turn + 1
            round_fields = {
                "round_index": round_index,
                "within_round_index": within_round_index,
                "global_update_index": round_index * rules.n_agents
                + within_round_index,
                "round_controller_action": action,
                "round_controller_target": target,
                "round_controller_advocate_probability": probability,
                "controlled_slot": controlled_slot,
                "intervention_budget": intervention_budget,
                "controlled_positions_hash_or_id": schedule_hash,
                "controller_message_mode": message_mode,
                "receiver_epistemic_disposition": rules.receiver_epistemic_disposition,
                "controller_evidence_strategy": evidence_strategy,
                "controller_episode_fact_id": controller_fact_id,
            }
            transition = game.apply_round_event_transition(
                state,
                focal=focal,
                action=focal_action,
                config=config.game,
                social_sources=social_sources,
                round_fields=round_fields,
                signal=micro_signal,
                sampled_peers=sampled_peers,
                effective_peers=effective_peers,
                replaced_peer=replaced_peer,
                replaced_peer_slot=replaced_peer_slot,
            )
            event = transition.event or {}
            peer_exposures += int(event.get("peer_fact_exposures", 0))
            controller_exposures += int(event.get("controller_fact_exposures", 0))
            new_peer_facts += int(event.get("new_peer_facts", 0))
            new_controller_facts += int(event.get("new_controller_facts", 0))
            reactivated_peer_facts += int(event.get("reactivated_peer_facts", 0))
            reactivated_controller_facts += int(
                event.get("reactivated_controller_facts", 0)
            )
            if controlled_slot and target is not None:
                if event.get("vote_before") != target:
                    controlled_off_target += 1
                    if event.get("vote_after") == target:
                        controlled_adoptions += 1

            record = RelationalInteractionRecord(
                interaction_id=transition.interaction_id,
                interaction_index=micro_index,
                phase=FOCAL_UPDATE,
                participants=(focal, *effective_peers),
                decisions=(update,),
                transition=transition,
            )
            interactions.append(record)
            _notify(
                observer,
                "record_interaction",
                round_index=micro_index,
                interaction=record,
                state=transition.next_state.to_dict(),
                prompt_definitions={
                    update.request.stage: update.prompt_definition_hash
                },
            )
            _notify(observer, "record_trajectory", record=record)
            _notify(
                observer,
                "event",
                "relational_round_feedback_transition",
                **dict(transition.event or {}),
            )
            state = transition.next_state

        population_after = [str(agent.committed_action) for agent in state.agents]
        after_obs = population_observables(
            population_after, options, state.correct_answer, analysis_target
        )
        active_knowledge_after_interactions = knowledge_observables(
            state.agents,
            state.supporting_fact_ids,
            fact_ids_attribute=ACTIVE_FACT_IDS,
        )
        historical_knowledge_after = knowledge_observables(
            state.agents, state.supporting_fact_ids
        )

        persistence_seed: int | None = None
        deactivated: tuple[tuple[str, str], ...] = ()
        if rules.epistemic_persistence != 1.0:
            persistence_stream = root.derive(
                f"relational-epistemic-persistence:{round_index}"
            )
            persistence_seed = int(persistence_stream)
            state, deactivated = apply_epistemic_persistence(
                state,
                persistence=rules.epistemic_persistence,
                rng=persistence_stream.create_random(),
            )

        active_knowledge_after = knowledge_observables(
            state.agents,
            state.supporting_fact_ids,
            fact_ids_attribute=ACTIVE_FACT_IDS,
        )
        strata_after = knowledge_strata(
            state.agents,
            state.supporting_fact_ids,
            state.correct_answer,
            fact_ids_attribute=ACTIVE_FACT_IDS,
        )
        deactivated_supporting = sum(
            1 for _, fact_id in deactivated if fact_id in set(state.supporting_fact_ids)
        )
        sensor = {} if round_signal is None else dict(round_signal.observation)
        raw_sensor_counts = sensor.get("sampled_opinion_counts", {})
        sensor_counts = (
            _count_vector(raw_sensor_counts, options)
            if isinstance(raw_sensor_counts, Mapping)
            else [0 for _ in options]
        )
        target_index = options.index(analysis_target)
        q_c = None if round_signal is None else int(sensor.get("sample_size", 0))
        sensor_target_share = None if not q_c else sensor_counts[target_index] / q_c
        round_event = {
            "episode_id": f"{state.task['task_id']}-{state.data['seed']}",
            "round_index": round_index,
            "seed": int(state.data["seed"]),
            "task_id": state.task["task_id"],
            "task_seed": state.task["task_seed"],
            "reasoning_depth": state.task["reasoning_depth"],
            "K": len(options),
            "N": rules.n_agents,
            "dynamics_mode": rules.dynamics_mode,
            "social_group_size": rules.social_group_size,
            "sensor_sample_size": q_c,
            "intervention_budget": intervention_budget,
            "sensing_fraction": None if q_c is None else q_c / rules.n_agents,
            "actuation_fraction": intervention_budget / rules.n_agents,
            "controller_enabled": round_signal is not None,
            "controller_action": action,
            "controller_target": target,
            "analysis_target": analysis_target,
            "controller_policy": (
                None if round_signal is None else round_signal.metadata.get("policy")
            ),
            "controller_threshold": (
                None if round_signal is None else round_signal.metadata.get("threshold")
            ),
            "controller_beta": (
                None if round_signal is None else round_signal.metadata.get("beta")
            ),
            "controller_advocate_probability": probability,
            "controller_advocacy_probability": probability,
            "controller_message_mode": message_mode,
            "receiver_epistemic_disposition": rules.receiver_epistemic_disposition,
            "controller_evidence_strategy": evidence_strategy,
            "epistemic_persistence": rules.epistemic_persistence,
            "epistemic_persistence_seed": persistence_seed,
            "epistemic_condition": (
                None
                if evidence_strategy is None
                else f"{rules.receiver_epistemic_disposition}_{evidence_strategy}"
            ),
            "derived_epistemic_condition": (
                None
                if evidence_strategy is None
                else f"{rules.receiver_epistemic_disposition}_{evidence_strategy}"
            ),
            "controller_target_semantics": (
                None
                if resolved_control is None
                else str(getattr(resolved_control, "target", None))
            ),
            "controller_fact_id": round_controller_fact,
            "controller_episode_fact_id": controller_fact_id,
            "controller_fact_text": (
                None
                if round_controller_fact is None
                else state.fact_text(round_controller_fact)
            ),
            "sensor_agent_ids": list(sensor.get("sampled_agent_ids", ())),
            "sensor_observed_opinions": list(sensor.get("sampled_opinions", ())),
            "sensor_count_vector": sensor_counts if round_signal is not None else None,
            "sensor_target_share": sensor_target_share,
            "controlled_positions": list(controlled_positions),
            "controlled_position_count": len(controlled_positions),
            "controlled_positions_seed": schedule_seed,
            "controlled_positions_hash_or_id": schedule_hash,
            # --- votes ------------------------------------------------------
            "population_state_before": population_before,
            "population_state_after": population_after,
            "agent_ids": [str(agent.agent_id) for agent in state.agents],
            "initial_vote_vector": list(initial_state.initial_votes),
            "initial_active_fact_ids_by_agent": [
                list(agent.active_fact_ids) for agent in initial_state.agents
            ],
            "initial_known_fact_ids_by_agent": [
                list(agent.known_fact_ids) for agent in initial_state.agents
            ],
            "initial_task_id": str(initial_state.task["task_id"]),
            "initialization_source": initialization_source,
            "initialization_repetition": initialization_repetition,
            "initialization_artifact_hash": initialization_artifact_hash,
            "physical_initial_state_hash": physical_initial_state_hash,
            "initial_knowledge_class_by_agent": [
                len(set(agent.initial_fact_ids) & set(state.supporting_fact_ids))
                for agent in state.agents
            ],
            "occupation_counts_before": _count_vector(
                before_obs["occupation_counts"], options
            ),
            "occupation_counts_after": _count_vector(
                after_obs["occupation_counts"], options
            ),
            "truth_vote_share_before": before_obs["p_truth"],
            "truth_vote_share": after_obs["p_truth"],
            "controller_target_share_before": before_obs["p_ctrl"],
            "controller_target_share": after_obs["p_ctrl"],
            "m_truth_before": before_obs["m_truth"],
            "m_truth_after": after_obs["m_truth"],
            "m_ctrl_before": before_obs["m_ctrl"],
            "m_ctrl_after": after_obs["m_ctrl"],
            "m_order_before": before_obs["m_order"],
            "m_order_after": after_obs["m_order"],
            "H_vote_before": before_obs["H_vote"],
            "H_vote_after": after_obs["H_vote"],
            "delta_m_truth": float(after_obs["m_truth"] - before_obs["m_truth"]),
            "delta_m_ctrl": float(after_obs["m_ctrl"] - before_obs["m_ctrl"]),
            "delta_m_order": float(after_obs["m_order"] - before_obs["m_order"]),
            "delta_H_vote": float(after_obs["H_vote"] - before_obs["H_vote"]),
            # --- knowledge (§17) --------------------------------------------
            "supporting_fact_ids": list(state.supporting_fact_ids),
            "mean_supporting_fact_coverage_before": active_knowledge_before[
                "mean_supporting_fact_coverage"
            ],
            "mean_supporting_fact_coverage": active_knowledge_after[
                "mean_supporting_fact_coverage"
            ],
            "full_proof_agent_share_before": active_knowledge_before[
                "full_proof_agent_share"
            ],
            "full_proof_agent_share": active_knowledge_after["full_proof_agent_share"],
            "supporting_fact_reach_before": historical_knowledge_before[
                "supporting_fact_reach"
            ],
            "supporting_fact_reach": historical_knowledge_after[
                "supporting_fact_reach"
            ],
            "mean_known_fact_count": historical_knowledge_after[
                "mean_known_fact_count"
            ],
            "active_mean_supporting_fact_coverage_before": active_knowledge_before[
                "mean_supporting_fact_coverage"
            ],
            "active_mean_supporting_fact_coverage_after_interactions": (
                active_knowledge_after_interactions["mean_supporting_fact_coverage"]
            ),
            "active_mean_supporting_fact_coverage_after": active_knowledge_after[
                "mean_supporting_fact_coverage"
            ],
            "active_full_proof_agent_share_before": active_knowledge_before[
                "full_proof_agent_share"
            ],
            "active_full_proof_agent_share_after_interactions": (
                active_knowledge_after_interactions["full_proof_agent_share"]
            ),
            "active_full_proof_agent_share_after": active_knowledge_after[
                "full_proof_agent_share"
            ],
            "active_supporting_fact_reach_before": active_knowledge_before[
                "supporting_fact_reach"
            ],
            "active_supporting_fact_reach_after_interactions": (
                active_knowledge_after_interactions["supporting_fact_reach"]
            ),
            "active_supporting_fact_reach_after": active_knowledge_after[
                "supporting_fact_reach"
            ],
            "active_mean_fact_count_before": active_knowledge_before[
                "mean_known_fact_count"
            ],
            "active_mean_fact_count_after_interactions": (
                active_knowledge_after_interactions["mean_known_fact_count"]
            ),
            "active_mean_fact_count_after": active_knowledge_after[
                "mean_known_fact_count"
            ],
            "historical_mean_supporting_fact_coverage_before": (
                historical_knowledge_before["mean_supporting_fact_coverage"]
            ),
            "historical_mean_supporting_fact_coverage_after": (
                historical_knowledge_after["mean_supporting_fact_coverage"]
            ),
            "historical_full_proof_agent_share_before": historical_knowledge_before[
                "full_proof_agent_share"
            ],
            "historical_full_proof_agent_share_after": historical_knowledge_after[
                "full_proof_agent_share"
            ],
            "persistence_deactivated_fact_count": len(deactivated),
            "persistence_deactivated_supporting_fact_count": deactivated_supporting,
            "peer_fact_exposures": peer_exposures,
            "controller_fact_exposures": controller_exposures,
            "new_peer_facts": new_peer_facts,
            "new_controller_facts": new_controller_facts,
            "reactivated_peer_fact_count": reactivated_peer_facts,
            "reactivated_controller_fact_count": reactivated_controller_facts,
            # --- self-contained round summary (§ compact artifact profile) ---
            # These make round_trajectory.jsonl sufficient on its own: under
            # `results_only` the microscopic trajectory is not retained, so
            # anything the r-scan or the control comparison needs has to be
            # here rather than derivable from it.
            "vote_entropy": after_obs["H_vote"],
            "vote_entropy_before": before_obs["H_vote"],
            **{
                key: value
                for key, value in strata_after.items()
                if key.startswith(("knowledge_share_k", "truth_share_k"))
            },
            **{
                f"{key}_before": value
                for key, value in strata_before.items()
                if key.startswith(("knowledge_share_k", "truth_share_k"))
            },
            "knowledge_stratum_counts": strata_after["knowledge_stratum_counts"],
            "truth_counts_by_stratum": strata_after["truth_counts_by_stratum"],
            # E_k = (n_k^(0), ..., n_k^(L)): how many agents hold exactly j of
            # the L supporting facts at the START of the round, i.e. aligned
            # with `occupation_counts_before` rather than with the post-round
            # strata above. Counts, not shares, because this is a conditioning
            # state for discrete estimators - `share * N` would round-trip
            # through a float for no reason.
            "knowledge_stratum_counts_before": strata_before[
                "knowledge_stratum_counts"
            ],
            "truth_counts_by_stratum_before": strata_before["truth_counts_by_stratum"],
            "reasoning_depth_L": len(state.supporting_fact_ids),
            # Controlled-update response. The rate is conditional on the focal
            # not already standing on the target; both counts are kept so the
            # unconditional form can be recomputed.
            "controlled_update_count": len(controlled_positions),
            "controlled_off_target_count": controlled_off_target,
            "controlled_adoption_count": controlled_adoptions,
            "controlled_target_adoption_rate": (
                controlled_adoptions / controlled_off_target
                if controlled_off_target
                else None
            ),
            "possible_answers": list(options),
            "correct_answer": state.correct_answer,
            "correct_relation": state.task["correct_relation"],
        }
        round_record = RelationalRoundRecord(round_index=round_index, event=round_event)
        round_records.append(round_record)
        _notify(observer, "record_round_trajectory", record=round_record)
        _notify(
            observer,
            "record_round_boundary",
            round_index=round_index,
            state=state.to_dict(),
            prompt_definitions={update.request.stage: update.prompt_definition_hash},
        )
        _notify(observer, "event", "relational_round_feedback", **round_event)

    termination = state.termination_reason or "max_rounds_reached"
    every = (
        *initial_decisions,
        *(decision for item in interactions for decision in item.decisions),
    )
    _notify(
        observer,
        "event",
        "game_completed",
        interactions=len(interactions),
        population_rounds=len(round_records),
    )
    return RelationalGameResult(
        initial_state=initial_state,
        final_state=state,
        initial_decisions=initial_decisions,
        interactions=tuple(interactions),
        rounds=tuple(round_records),
        termination_reason=termination,
        logical_decisions=len(every),
        validation_attempts=sum(decision.validation_attempts for decision in every),
    )


def run_relational_imitation_round_feedback_game_sync(
    *args: Any, **kwargs: Any
) -> RelationalGameResult:
    return asyncio.run(run_relational_imitation_round_feedback_game(*args, **kwargs))


__all__ = [
    "CONTROL_SOURCE_ID",
    "PROMPT_FAMILIES",
    "RelationalDecision",
    "RelationalDecisionFailed",
    "RelationalGameResult",
    "RelationalInteractionRecord",
    "apply_epistemic_persistence",
    "build_social_sources",
    "run_relational_imitation_round_feedback_game",
    "run_relational_imitation_round_feedback_game_sync",
    "sample_controlled_positions",
]
