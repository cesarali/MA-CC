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
    expire public memory; persist K_active for the new day
          |
          v
    dawn_only board mode:
        if ADVOCATE_Z, append exactly b factless DIRECTIVEs now
          |
          v
    N autonomous focal updates, one focal agent each:
        sample focal + q live board messages
            -> no slot replacement and no daytime controller injection
            -> render each slot from its (vote, exposed fact)
            -> ONE focal provider call
            -> {vote, reason, shared_fact_id}
            -> apply the vote, publish the ballot, grow K_focal
          |
          v
    record n_(k+1), K_active, K_hist, board, and exposure observables

The historical microscopic peer/direct-board modes remain available for old
configs. ``controller_timing: dawn_only`` selects the frozen night/dawn/day
protocol and deliberately never constructs their controlled-position schedule.

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
from collections import Counter
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
from .adaptive_communication import (
    CommunicationMode,
    ControllerCommunicationContext,
    allowed_communication_modes,
    choose_communication_mode,
)
from .controller import (
    ADAPTIVE_COMMUNICATION,
    COORDINATION_REQUEST,
    DIRECT_RECOMMENDATION,
    RECOMMENDATION_ONLY,
    SILENT,
    TRUTHFUL_STRATEGIC_REPORT,
    TIMING_DAWN_ONLY,
    TIMING_MICROSCOPIC,
)
from .game import RelationalImitationRoundFeedbackGame
from .initialization import (
    initialization_artifact_path,
    paired_initialization_required,
    physical_initial_state_projection,
    read_initialization_artifact,
)
from .metrics import knowledge_observables, knowledge_strata
from .prompts import (
    BOARD_PROMPT_FAMILY,
    PROMPT_FAMILY,
    agent_label,
    control_label,
    render_control_reason,
)
from .state import (
    ACTIVE_FACT_IDS,
    FOCAL_UPDATE,
    MESSAGE_DIRECTIVE,
    MESSAGE_REQUEST,
    MESSAGE_REPORT,
    SOCIAL_MODE_BOARD,
    SOCIAL_MODE_PEER,
    BlackboardMessage,
    RelationalAgentState,
    RelationalGameState,
    RelationalRoundRecord,
    reasoning_fact_ids,
)

PROMPT_FAMILIES = (PROMPT_FAMILY, BOARD_PROMPT_FAMILY)

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


def _message_source(
    state: RelationalGameState, message: BlackboardMessage, round_index: int
) -> dict[str, Any]:
    fact_id = message.shared_fact_id
    return {
        "message_id": message.message_id,
        "source_id": message.author_id,
        "source_type": (
            "control" if message.author_kind == "controller" else "ordinary"
        ),
        "author_kind": message.author_kind,
        "label": (
            control_label(len(state.agents))
            if message.author_kind == "controller"
            else agent_label(message.author_id)
        ),
        "message_type": message.message_type,
        "text": message.text,
        "vote": message.vote,
        "shared_fact_id": fact_id,
        "shared_fact_text": None if fact_id is None else state.fact_text(fact_id),
        "reply_to": message.reply_to,
        "round_created": message.round_created,
        "micro_step_created": message.micro_step_created,
        "expires_after_round": message.expires_after_round,
        "message_age_rounds": round_index - message.round_created,
        "message_age_micro_steps": state.turn - message.micro_step_created,
    }


def _transient_recommendation_source(
    state: RelationalGameState,
    *,
    target: str,
    round_index: int,
    within_round_index: int,
) -> dict[str, Any]:
    return {
        "message_id": f"direct-r{round_index:04d}-u{within_round_index:04d}",
        "source_id": CONTROL_SOURCE_ID,
        "source_type": "control",
        "author_kind": "controller",
        "label": control_label(len(state.agents)),
        "message_type": MESSAGE_DIRECTIVE,
        "text": render_control_reason(target),
        "vote": target,
        "shared_fact_id": None,
        "shared_fact_text": None,
        "reply_to": None,
        "round_created": round_index,
        "micro_step_created": state.turn,
        "expires_after_round": round_index,
        "message_age_rounds": 0,
        "message_age_micro_steps": 0,
        "transient": True,
    }


def _append_controller_request(
    state: RelationalGameState,
    *,
    target: str,
    text: str,
    round_index: int,
    lifetime_rounds: int,
) -> tuple[RelationalGameState, BlackboardMessage]:
    """Historical coordination path, which intentionally posts DIRECTIVE."""

    return _append_controller_public_message(
        state,
        target=target,
        text=text,
        message_type=MESSAGE_DIRECTIVE,
        round_index=round_index,
        lifetime_rounds=lifetime_rounds,
    )


def _append_controller_public_message(
    state: RelationalGameState,
    *,
    target: str,
    text: str,
    message_type: str,
    round_index: int,
    lifetime_rounds: int,
) -> tuple[RelationalGameState, BlackboardMessage]:
    """Publish one factless controller REQUEST or DIRECTIVE."""

    if message_type not in {MESSAGE_REQUEST, MESSAGE_DIRECTIVE}:
        raise ValueError("factless controller message must be REQUEST or DIRECTIVE")
    board = state.blackboard
    message = BlackboardMessage(
        message_id=f"m{len(board.messages) + 1:06d}",
        author_id=CONTROL_SOURCE_ID,
        message_type=message_type,
        text=text,
        vote=target,
        shared_fact_id=None,
        reply_to=None,
        round_created=round_index,
        micro_step_created=state.turn,
        expires_after_round=round_index + lifetime_rounds - 1,
        author_kind="controller",
    )
    board = board.append(message)
    return replace(
        state, data={**dict(state.data), "blackboard": board.to_list()}
    ), message


def _append_controller_report(
    state: RelationalGameState,
    *,
    target: str,
    fact_id: str,
    round_index: int,
    lifetime_rounds: int,
) -> tuple[RelationalGameState, BlackboardMessage]:
    """Publish canonical evidence through the same REPORT schema peers use."""

    canonical_text = state.controller_report_text(fact_id)
    board = state.blackboard
    message = BlackboardMessage(
        message_id=f"m{len(board.messages) + 1:06d}",
        author_id=CONTROL_SOURCE_ID,
        message_type=MESSAGE_REPORT,
        text=canonical_text,
        vote=target,
        shared_fact_id=fact_id,
        reply_to=None,
        round_created=round_index,
        micro_step_created=state.turn,
        expires_after_round=round_index + lifetime_rounds - 1,
        author_kind="controller",
    )
    board = board.append(message)
    return replace(
        state, data={**dict(state.data), "blackboard": board.to_list()}
    ), message


def _descends_from_controller(
    state: RelationalGameState, message: BlackboardMessage
) -> bool:
    parent = message.reply_to
    visited: set[str] = set()
    while parent is not None and parent not in visited:
        visited.add(parent)
        ancestor = state.blackboard.find(parent)
        if ancestor is None:
            return False
        if ancestor.author_kind == "controller":
            return True
        parent = ancestor.reply_to
    return False


def _origin_directive_id(
    state: RelationalGameState, message: BlackboardMessage
) -> str | None:
    """Return the first controller DIRECTIVE in ``message``'s reply ancestry."""

    current: BlackboardMessage | None = message
    visited: set[str] = set()
    while current is not None and current.message_id not in visited:
        visited.add(current.message_id)
        if (
            current.author_kind == "controller"
            and current.message_type == MESSAGE_DIRECTIVE
        ):
            return current.message_id
        current = (
            None
            if current.reply_to is None
            else state.blackboard.find(current.reply_to)
        )
    return None


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
    board_rng = root.derive("relational-blackboard-sampling").create_random()

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
    actuation_mode = str(
        getattr(resolved_control, "controller_actuation_mode", DIRECT_RECOMMENDATION)
    )
    controller_timing = str(
        getattr(resolved_control, "controller_timing", TIMING_MICROSCOPIC)
    )
    if rules.social_mode == SOCIAL_MODE_PEER and actuation_mode == COORDINATION_REQUEST:
        raise ValueError(
            "coordination_request requires game.options.social_mode: board"
        )
    if rules.social_mode == SOCIAL_MODE_PEER and actuation_mode in {
        TRUTHFUL_STRATEGIC_REPORT,
        ADAPTIVE_COMMUNICATION,
    }:
        raise ValueError(f"{actuation_mode} requires game.options.social_mode: board")
    dawn_blackboard = (
        rules.social_mode == SOCIAL_MODE_BOARD
        and actuation_mode
        in {COORDINATION_REQUEST, TRUTHFUL_STRATEGIC_REPORT, ADAPTIVE_COMMUNICATION}
        and controller_timing == TIMING_DAWN_ONLY
    )
    evidence_strategy = getattr(resolved_control, "controller_evidence_strategy", None)
    state = game.initialize(config.game, config.execution.seed)
    task = game.load_task(config.game)
    truthful_validator = getattr(
        resolved_control, "validate_truthful_report_task", None
    )
    if truthful_validator is not None:
        truthful_validator(task, config.execution.seed)
    resolver = getattr(resolved_control, "resolve_fact_id", None)
    if resolver is not None and actuation_mode != TRUTHFUL_STRATEGIC_REPORT:
        controller_fact_id = resolver(task, config.execution.seed)

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
    _notify(observer, "record_semantic_initialization", state=initial_state.to_dict())

    interactions: list[RelationalInteractionRecord] = []
    round_records: list[RelationalRoundRecord] = []
    retain_result_history = bool(getattr(observer, "retain_result_history", True))
    logical_decisions = len(initial_decisions)
    validation_attempts = sum(
        decision.validation_attempts for decision in initial_decisions
    )
    selected_report_rounds: dict[str, list[int]] = {}
    previous_communication_modes: list[CommunicationMode] = []
    for round_index in range(0 if rules.initialization_only else rules.rounds):
        if state.terminated:
            break
        options = tuple(state.possible_answers)
        population_before = [str(agent.committed_action) for agent in state.agents]

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

        # Frozen dawn-board protocol: the vote sensor and U draw happen first;
        # then the previous day's public board expires and private active memory
        # persists into the new day. Historical evidence is never touched.
        _, night_expired_message_ids = state.blackboard.expire(round_index - 1)
        persistence_seed: int | None = None
        deactivated: tuple[tuple[str, str], ...] = ()
        if dawn_blackboard and rules.epistemic_persistence != 1.0:
            persistence_stream = root.derive(
                f"relational-epistemic-persistence:{round_index}"
            )
            persistence_seed = int(persistence_stream)
            state, deactivated = apply_epistemic_persistence(
                state,
                persistence=rules.epistemic_persistence,
                rng=persistence_stream.create_random(),
            )

        active_knowledge_before = knowledge_observables(
            state.agents,
            state.supporting_fact_ids,
            fact_ids_attribute=ACTIVE_FACT_IDS,
            supporting_fact_groups=state.task.get("supporting_fact_groups", {}),
        )
        historical_knowledge_before = knowledge_observables(
            state.agents,
            state.supporting_fact_ids,
            supporting_fact_groups=state.task.get("supporting_fact_groups", {}),
        )
        strata_before = knowledge_strata(
            state.agents,
            state.supporting_fact_ids,
            state.correct_answer,
            fact_ids_attribute=ACTIVE_FACT_IDS,
            supporting_fact_groups=state.task.get("supporting_fact_groups", {}),
        )

        schedule_seed = (
            None
            if dawn_blackboard
            else int(root.derive(f"relational-schedule:{round_index}"))
        )
        controlled_positions = (
            sample_controlled_positions(
                rules.n_agents,
                intervention_budget,
                root.derive(f"relational-schedule:{round_index}").create_random(),
            )
            if action == ADVOCATE_TARGET and not dawn_blackboard
            else ()
        )
        controlled_set = frozenset(controlled_positions)
        schedule_hash = (
            None
            if dawn_blackboard
            else hashlib.sha256(
                json.dumps(list(controlled_positions), separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
        )
        # Evidence only ever reaches a focal agent through an actuated slot, so
        # a NO_OP round transmits nothing even under recommendation_plus_fact.
        round_controller_fact = (
            controller_fact_id
            if action == ADVOCATE_TARGET and rules.social_mode == SOCIAL_MODE_PEER
            else None
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
        round_board_sizes: list[int] = []
        round_created_message_ids: list[str] = []
        round_controller_post_ids: list[str] = []
        round_controller_readers: set[str] = set()
        round_controller_exposure_count = 0
        round_controller_exposed_updates = 0
        round_controller_repeat_exposures = 0
        round_eligible_message_counts: list[int] = []
        round_eligible_directive_counts: list[int] = []
        round_eligible_controller_report_counts: list[int] = []
        round_message_read_count = 0
        round_lineage_events: list[dict[str, Any]] = []
        round_controller_report_adoptions = 0
        round_controller_report_off_target_exposures = 0
        round_controller_report_selection: list[dict[str, Any]] = []
        allowed_controller_modes: tuple[CommunicationMode, ...] = ()
        communication_choice = None
        executed_communication_mode: CommunicationMode | None = None
        request_topic: str | None = None
        directive_topic: str | None = None

        if actuation_mode == ADAPTIVE_COMMUNICATION:
            allowed_controller_modes = allowed_communication_modes(
                allow_requests=bool(
                    getattr(resolved_control, "allow_controller_requests", True)
                ),
                allow_directives=bool(
                    getattr(resolved_control, "allow_controller_directives", True)
                ),
            )
        elif actuation_mode == TRUTHFUL_STRATEGIC_REPORT:
            allowed_controller_modes = (CommunicationMode.REPORT,)
        elif actuation_mode == COORDINATION_REQUEST:
            allowed_controller_modes = (CommunicationMode.DIRECTIVE,)

        if (
            dawn_blackboard
            and action == ADVOCATE_TARGET
            and target is not None
            and resolved_control is not None
            and actuation_mode == ADAPTIVE_COMMUNICATION
        ):
            sampled_counts = dict(round_signal.observation).get(
                "sampled_opinion_counts", {}
            )
            live_counts = Counter(
                message.message_type
                for message in state.blackboard.live_messages(round_index)
            )
            communication_choice = choose_communication_mode(
                ControllerCommunicationContext(
                    round_index=round_index,
                    target=target,
                    sampled_opinion_counts={
                        str(key): int(value)
                        for key, value in dict(sampled_counts).items()
                    },
                    live_message_type_counts=dict(live_counts),
                    previous_modes=tuple(previous_communication_modes),
                ),
                allowed_controller_modes,
                root.derive(
                    f"relational-controller-communication:{round_index}"
                ).create_random(),
            )
            previous_communication_modes.append(communication_choice.mode)

        # Dawn is one atomic board perturbation. Every directive exists before
        # the first focal decision and the controller performs no daytime work.
        if (
            dawn_blackboard
            and action == ADVOCATE_TARGET
            and target is not None
            and resolved_control is not None
        ):
            chosen_mode = (
                communication_choice.mode
                if communication_choice is not None
                else CommunicationMode.REPORT
                if actuation_mode == TRUTHFUL_STRATEGIC_REPORT
                else CommunicationMode.DIRECTIVE
            )
            executed_communication_mode = chosen_mode
            if chosen_mode == CommunicationMode.REPORT:
                live_fact_counts = Counter(
                    message.shared_fact_id
                    for message in state.blackboard.live_messages(round_index)
                    if message.message_type == MESSAGE_REPORT
                    and message.shared_fact_id is not None
                )
                selector_name = (
                    "select_adaptive_truthful_reports"
                    if actuation_mode == ADAPTIVE_COMMUNICATION
                    else "select_truthful_reports"
                )
                selections = getattr(resolved_control, selector_name)(
                    task,
                    episode_seed=config.execution.seed,
                    round_index=round_index,
                    live_fact_counts=live_fact_counts,
                    selected_rounds=selected_report_rounds,
                )
                if (
                    actuation_mode == TRUTHFUL_STRATEGIC_REPORT
                    and len(selections) != intervention_budget
                ):
                    raise ValueError(
                        "truthful strategic selector did not return exactly b reports"
                    )
                for selection in selections:
                    state, controller_post = _append_controller_report(
                        state,
                        target=target,
                        fact_id=selection.fact_id,
                        round_index=round_index,
                        lifetime_rounds=rules.board_message_lifetime_rounds,
                    )
                    selected_report_rounds.setdefault(selection.fact_id, []).append(
                        round_index
                    )
                    round_controller_report_selection.append(selection.to_dict())
                    round_controller_post_ids.append(controller_post.message_id)
                    round_created_message_ids.append(controller_post.message_id)
            else:
                text_method = (
                    "coordination_request_text"
                    if chosen_mode == CommunicationMode.REQUEST
                    else "coordination_directive_text"
                    if actuation_mode == ADAPTIVE_COMMUNICATION
                    else "coordination_request_text"
                )
                coordination_text = getattr(resolved_control, text_method)(
                    target,
                    dict(round_signal.observation).get("sampled_opinion_counts", {}),
                    state.answer_display_texts,
                )
                message_type = (
                    MESSAGE_REQUEST
                    if chosen_mode == CommunicationMode.REQUEST
                    else MESSAGE_DIRECTIVE
                )
                if message_type == MESSAGE_REQUEST:
                    request_topic = coordination_text
                else:
                    directive_topic = coordination_text
                # In adaptive mode b is a maximum. A factless act is posted
                # once rather than duplicated merely to fill all slots.
                post_count = (
                    1
                    if actuation_mode == ADAPTIVE_COMMUNICATION
                    else intervention_budget
                )
                for _ in range(post_count):
                    state, controller_post = _append_controller_public_message(
                        state,
                        target=target,
                        text=coordination_text,
                        message_type=message_type,
                        round_index=round_index,
                        lifetime_rounds=rules.board_message_lifetime_rounds,
                    )
                    round_controller_post_ids.append(controller_post.message_id)
                    round_created_message_ids.append(controller_post.message_id)

        _notify(
            observer,
            "record_semantic_round_start",
            round_index=round_index,
            state=state.to_dict(),
            expired_message_ids=list(night_expired_message_ids),
            deactivated_pairs=[
                {"agent_id": agent_id, "fact_id": fact_id}
                for agent_id, fact_id in deactivated
            ],
            controller={
                "enabled": round_signal is not None,
                "action": action,
                "target": target,
                "probability": probability,
                "sensor": None
                if round_signal is None
                else dict(round_signal.observation),
                "directive_ids": list(round_controller_post_ids),
                "post_ids": list(round_controller_post_ids),
                "request_ids": [
                    message.message_id
                    for message in state.blackboard.messages
                    if message.message_id in set(round_controller_post_ids)
                    and message.message_type == MESSAGE_REQUEST
                ],
                "report_ids": [
                    message.message_id
                    for message in state.blackboard.messages
                    if message.message_id in set(round_controller_post_ids)
                    and message.message_type == MESSAGE_REPORT
                ],
                "selected_fact_ids": [
                    row["fact_id"] for row in round_controller_report_selection
                ],
                "allowed_message_modes": [
                    mode.value for mode in allowed_controller_modes
                ],
                "chosen_message_mode": (
                    None
                    if executed_communication_mode is None
                    else executed_communication_mode.value
                ),
                "requested_b": intervention_budget,
                "actual_posts": len(round_controller_post_ids),
            },
        )

        for within_round_index in range(rules.n_agents):
            controlled_slot = within_round_index in controlled_set
            controller_post: BlackboardMessage | None = None
            board_before = len(state.blackboard.live_messages(round_index))
            if rules.social_mode == SOCIAL_MODE_PEER:
                selected = game.select_participants(state, config.game, participant_rng)
                focal, sampled_peers = selected[0], tuple(selected[1:])
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
            else:
                focal = participant_rng.choice(
                    [agent.agent_id for agent in state.agents]
                )
                sampled_peers = ()
                effective_peers = ()
                replaced_peer = None
                replaced_peer_slot = None
                if (
                    controlled_slot
                    and actuation_mode == COORDINATION_REQUEST
                    and controller_timing == TIMING_MICROSCOPIC
                    and target is not None
                    and resolved_control is not None
                ):
                    request_text = getattr(
                        resolved_control, "coordination_request_text"
                    )(
                        target,
                        dict(round_signal.observation).get(
                            "sampled_opinion_counts", {}
                        ),
                        state.answer_display_texts,
                    )
                    state, controller_post = _append_controller_request(
                        state,
                        target=target,
                        text=request_text,
                        round_index=round_index,
                        lifetime_rounds=rules.board_message_lifetime_rounds,
                    )
                    round_controller_post_ids.append(controller_post.message_id)
                    round_created_message_ids.append(controller_post.message_id)
                ordinary_limit = rules.social_group_size
                if controlled_slot and actuation_mode == DIRECT_RECOMMENDATION:
                    ordinary_limit -= 1
                eligible_messages = tuple(
                    message
                    for message in state.blackboard.live_messages(round_index)
                    if not rules.board_exclude_self_authored
                    or message.author_id != str(focal)
                )
                eligible_directives = sum(
                    message.message_type == MESSAGE_DIRECTIVE
                    and message.author_kind == "controller"
                    for message in eligible_messages
                )
                eligible_controller_reports = sum(
                    message.message_type == MESSAGE_REPORT
                    and message.author_kind == "controller"
                    for message in eligible_messages
                )
                round_eligible_message_counts.append(len(eligible_messages))
                round_eligible_directive_counts.append(eligible_directives)
                round_eligible_controller_report_counts.append(
                    eligible_controller_reports
                )
                sampled_messages = state.blackboard.sample_live(
                    round_index,
                    max(0, ordinary_limit),
                    board_rng,
                    exclude_author_id=(
                        str(focal) if rules.board_exclude_self_authored else None
                    ),
                )
                round_message_read_count += len(sampled_messages)
                sources = [
                    _message_source(state, message, round_index)
                    for message in sampled_messages
                ]
                if (
                    controlled_slot
                    and actuation_mode == DIRECT_RECOMMENDATION
                    and target is not None
                    and message_mode != SILENT
                ):
                    sources.insert(
                        0,
                        _transient_recommendation_source(
                            state,
                            target=target,
                            round_index=round_index,
                            within_round_index=within_round_index,
                        ),
                    )
                social_sources = tuple(sources)
            request = game.ballot_request(state, focal, social_sources, config.game)
            update = await _execute_decision(
                game, request, state, config, provider, counter, root, observer
            )
            logical_decisions += 1
            validation_attempts += update.validation_attempts
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
                else str(control_source.get("reason", control_source.get("text"))),
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
                "controller_actuation_mode": actuation_mode,
                "chosen_controller_message_mode": (
                    None
                    if executed_communication_mode is None
                    else executed_communication_mode.value
                ),
                "controller_timing": controller_timing,
                "protocol_phase": "day",
            }
            sampled_controller_ids = [
                str(source["message_id"])
                for source in social_sources
                if source.get("author_kind") == "controller"
            ]
            sampled_controller_report_ids = [
                str(source["message_id"])
                for source in social_sources
                if source.get("author_kind") == "controller"
                and source.get("message_type") == MESSAGE_REPORT
            ]
            if sampled_controller_ids:
                round_controller_exposure_count += len(sampled_controller_ids)
                round_controller_exposed_updates += 1
                round_controller_readers.add(str(focal))
                round_controller_repeat_exposures += max(
                    0, len(sampled_controller_ids) - 1
                )
            board_fields = {
                "sampled_message_ids": [
                    source.get("message_id") for source in social_sources
                ]
                if rules.social_mode == SOCIAL_MODE_BOARD
                else [],
                "sampled_message_authors": [
                    source.get("source_id") for source in social_sources
                ]
                if rules.social_mode == SOCIAL_MODE_BOARD
                else [],
                "sampled_message_types": [
                    source.get("message_type") for source in social_sources
                ]
                if rules.social_mode == SOCIAL_MODE_BOARD
                else [],
                "sampled_message_ages": [
                    {
                        "micro_steps": source.get("message_age_micro_steps", 0),
                        "rounds": source.get("message_age_rounds", 0),
                    }
                    for source in social_sources
                ]
                if rules.social_mode == SOCIAL_MODE_BOARD
                else [],
                "sampled_controller_message_ids": sampled_controller_ids,
                "sampled_controller_report_ids": sampled_controller_report_ids,
                "board_size_before": board_before,
                "eligible_board_message_count": (
                    len(eligible_messages)
                    if rules.social_mode == SOCIAL_MODE_BOARD
                    else 0
                ),
                "eligible_directive_count": (
                    eligible_directives if rules.social_mode == SOCIAL_MODE_BOARD else 0
                ),
                "eligible_directive_share": (
                    eligible_directives / len(eligible_messages)
                    if rules.social_mode == SOCIAL_MODE_BOARD and eligible_messages
                    else 0.0
                ),
                "eligible_controller_report_count": (
                    eligible_controller_reports
                    if rules.social_mode == SOCIAL_MODE_BOARD
                    else 0
                ),
                "controller_message_posted": controller_post is not None,
                "controller_message_id": (
                    None if controller_post is None else controller_post.message_id
                ),
                "controller_message_directly_exposed": any(
                    source.get("source_type") == "control" and source.get("transient")
                    for source in social_sources
                ),
                "theory_status": (
                    "reference_only"
                    if rules.social_mode == SOCIAL_MODE_BOARD
                    else "matched_reference"
                ),
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
                board_fields=board_fields,
            )
            event = transition.event or {}
            if sampled_controller_report_ids and target is not None:
                if event.get("vote_before") != target:
                    round_controller_report_off_target_exposures += 1
                    if event.get("vote_after") == target:
                        round_controller_report_adoptions += 1
            acquired_ids = set(event.get("new_peer_fact_ids", ()))
            refreshed_ids = set(event.get("reactivated_peer_fact_ids", ()))
            for source in social_sources:
                message_id = source.get("message_id")
                fact_id = source.get("shared_fact_id")
                if message_id is None or fact_id is None:
                    continue
                message = state.blackboard.find(str(message_id))
                if message is None or message.message_type != "REPORT":
                    continue
                origin_directive_id = _origin_directive_id(state, message)
                if origin_directive_id is None:
                    continue
                event_type = (
                    "acquisition"
                    if fact_id in acquired_ids
                    else "refresh"
                    if fact_id in refreshed_ids
                    else None
                )
                if event_type is not None:
                    round_lineage_events.append(
                        {
                            "origin_directive_id": origin_directive_id,
                            "reply_message_id": message.message_id,
                            "reply_author": message.author_id,
                            "reply_shared_fact_id": str(fact_id),
                            "downstream_reader": str(focal),
                            "event_type": event_type,
                            "round": round_index,
                            "microscopic_update": within_round_index,
                        }
                    )
            if event.get("new_message_id") is not None:
                round_created_message_ids.append(str(event["new_message_id"]))
            if rules.social_mode == SOCIAL_MODE_BOARD:
                board_after = len(
                    transition.next_state.blackboard.live_messages(round_index)
                )
                round_board_sizes.extend((board_before, board_after))
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
            if retain_result_history:
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
            supporting_fact_groups=state.task.get("supporting_fact_groups", {}),
        )
        historical_knowledge_after = knowledge_observables(
            state.agents,
            state.supporting_fact_ids,
            supporting_fact_groups=state.task.get("supporting_fact_groups", {}),
        )

        if not dawn_blackboard and rules.epistemic_persistence != 1.0:
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
            supporting_fact_groups=state.task.get("supporting_fact_groups", {}),
        )
        strata_after = knowledge_strata(
            state.agents,
            state.supporting_fact_ids,
            state.correct_answer,
            fact_ids_attribute=ACTIVE_FACT_IDS,
            supporting_fact_groups=state.task.get("supporting_fact_groups", {}),
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
        controller_sensor_y = (
            None
            if round_signal is None
            else {
                "sample_size": q_c,
                "sampled_agent_ids": list(sensor.get("sampled_agent_ids", ())),
                "sampled_votes": list(sensor.get("sampled_opinions", ())),
                "vote_counts": {
                    option: sensor_counts[index] for index, option in enumerate(options)
                },
                "target": analysis_target,
                "target_count": sensor_counts[target_index],
                "target_share": sensor_target_share,
            }
        )
        controller_sampled_u = (
            None if round_signal is None else int(action == ADVOCATE_TARGET)
        )
        controller_injection_global_indices = [
            round_index * rules.n_agents + position for position in controlled_positions
        ]
        messages_created = tuple(
            message
            for message in state.blackboard.messages
            if message.message_id in set(round_created_message_ids)
        )
        message_type_counts = dict(
            Counter(message.message_type for message in messages_created)
        )
        direct_replies = sum(
            1
            for message in messages_created
            if message.reply_to in set(round_controller_post_ids)
        )
        controller_descendants = sum(
            1
            for message in messages_created
            if message.author_kind == "agent"
            and _descends_from_controller(state, message)
        )
        direct_report_replies = sum(
            message.message_type == "REPORT"
            for message in messages_created
            if message.reply_to in set(round_controller_post_ids)
        )
        direct_evidence_report_replies = sum(
            message.message_type == "REPORT" and message.shared_fact_id is not None
            for message in messages_created
            if message.reply_to in set(round_controller_post_ids)
        )
        expired_message_ids = night_expired_message_ids
        if not dawn_blackboard:
            _, expired_message_ids = state.blackboard.expire(round_index)
        surviving_message_count = len(state.blackboard.live_messages(round_index + 1))
        eligible_message_opportunities = sum(round_eligible_message_counts)
        eligible_directive_opportunities = sum(round_eligible_directive_counts)
        eligible_controller_report_opportunities = sum(
            round_eligible_controller_report_counts
        )
        round_event = {
            "episode_id": f"{state.task['task_id']}-{state.data['seed']}",
            "round_index": round_index,
            "seed": int(state.data["seed"]),
            "task_id": state.task["task_id"],
            "task_family": state.task.get("task_family", rules.task_family),
            "task_seed": state.task["task_seed"],
            "reasoning_depth": state.task["reasoning_depth"],
            "K": len(options),
            "N": rules.n_agents,
            "dynamics_mode": rules.dynamics_mode,
            "social_group_size": rules.social_group_size,
            "social_mode": rules.social_mode,
            "board_sampling": rules.board_sampling,
            "message_lifetime_rounds": rules.board_message_lifetime_rounds,
            "board_exclude_self_authored": rules.board_exclude_self_authored,
            "sensor_sample_size": q_c,
            "intervention_budget": intervention_budget,
            "b": intervention_budget,
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
            "controller_action_probability": probability,
            # Stable transition contract for the existing MI/CMI adapters.
            # Blackboard channel observables below are additive to these fields.
            "n_k": _count_vector(before_obs["occupation_counts"], options)[
                options.index(analysis_target)
            ],
            "n_k_plus_1": _count_vector(after_obs["occupation_counts"], options)[
                options.index(analysis_target)
            ],
            "Y_k": sensor_counts[target_index] if round_signal is not None else None,
            "P_U1_given_Y": probability,
            "U_k": controller_sampled_u,
            "controller_sensor_Y": controller_sensor_y,
            "controller_probability_U1_given_Y": probability,
            "controller_sampled_U": controller_sampled_u,
            "controller_injection_within_round_indices": list(controlled_positions),
            "controller_injection_global_update_indices": (
                controller_injection_global_indices
            ),
            "controller_message_mode": message_mode,
            "controller_actuation_mode": actuation_mode,
            "controller_timing": controller_timing,
            "allow_participant_requests": rules.allow_participant_requests,
            "allow_controller_requests": bool(
                getattr(resolved_control, "allow_controller_requests", True)
            ),
            "allow_controller_directives": bool(
                getattr(resolved_control, "allow_controller_directives", True)
            ),
            "allowed_message_modes": [mode.value for mode in allowed_controller_modes],
            "chosen_message_mode": (
                None
                if executed_communication_mode is None
                else executed_communication_mode.value
            ),
            "communication_choice_reason": (
                None if communication_choice is None else communication_choice.reason
            ),
            "communication_policy": (
                getattr(resolved_control, "controller_communication_policy", None)
                if actuation_mode == ADAPTIVE_COMMUNICATION
                else None
            ),
            "communication_policy_version": (
                getattr(
                    resolved_control,
                    "controller_communication_policy_version",
                    None,
                )
                if actuation_mode == ADAPTIVE_COMMUNICATION
                else None
            ),
            "requested_b": intervention_budget,
            "actual_controller_posts": len(round_controller_post_ids),
            "selected_fact_ids": [
                row["fact_id"] for row in round_controller_report_selection
            ],
            "request_topic": request_topic,
            "directive_topic": directive_topic,
            "controller_report_cooldown_rounds": getattr(
                resolved_control, "controller_report_cooldown_rounds", None
            ),
            "controller_report_selection_strategy": getattr(
                resolved_control, "controller_report_selection_strategy", None
            ),
            "protocol": (
                "night_dawn_autonomous_day_v1" if dawn_blackboard else "legacy"
            ),
            "coordinator_public_vote": target,
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
            "directive_message_ids": [
                message.message_id
                for message in messages_created
                if message.author_kind == "controller"
                and message.message_type == MESSAGE_DIRECTIVE
            ],
            "dawn_directive_count": sum(
                message.author_kind == "controller"
                and message.message_type == MESSAGE_DIRECTIVE
                for message in messages_created
            ),
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
            "persistence_deactivated_pairs": [
                {"agent_id": agent_id, "fact_id": fact_id}
                for agent_id, fact_id in deactivated
            ],
            "active_fact_ids_by_agent_after": {
                str(agent.agent_id): list(agent.active_fact_ids)
                for agent in state.agents
            },
            "known_fact_ids_by_agent_after": {
                str(agent.agent_id): list(agent.known_fact_ids)
                for agent in state.agents
            },
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
            # --- finite-memory public board -------------------------------
            "board_messages_created": len(messages_created),
            "board_messages_expired": len(expired_message_ids),
            "night_expired_message_ids": list(night_expired_message_ids),
            "expired_message_count": len(expired_message_ids),
            "surviving_message_count": surviving_message_count,
            "board_peak_size": max(round_board_sizes, default=0),
            "board_mean_size": (
                sum(round_board_sizes) / len(round_board_sizes)
                if round_board_sizes
                else 0.0
            ),
            "message_type_counts": message_type_counts,
            "request_count": message_type_counts.get("REQUEST", 0),
            "report_count": message_type_counts.get("REPORT", 0),
            "directive_count": message_type_counts.get("DIRECTIVE", 0),
            "controller_posts": len(round_controller_post_ids),
            "controller_post_ids": list(round_controller_post_ids),
            "controller_reports_requested": (
                intervention_budget
                if action == ADVOCATE_TARGET
                and (
                    actuation_mode == TRUTHFUL_STRATEGIC_REPORT
                    or communication_choice is not None
                    and communication_choice.mode == CommunicationMode.REPORT
                )
                else 0
            ),
            "controller_reports_admitted": (
                len(round_controller_post_ids)
                if executed_communication_mode == CommunicationMode.REPORT
                else 0
            ),
            "controller_report_ids": (
                list(round_controller_post_ids)
                if executed_communication_mode == CommunicationMode.REPORT
                else []
            ),
            "controller_report_fact_ids": [
                row["fact_id"] for row in round_controller_report_selection
            ],
            "controller_report_selection": round_controller_report_selection,
            "controller_message_exposures": round_controller_exposure_count,
            "controller_report_exposures": (
                controller_exposures
                if executed_communication_mode == CommunicationMode.REPORT
                else 0
            ),
            "controller_report_repeat_exposures": round_controller_repeat_exposures,
            "directive_exposed_focal_updates": round_controller_exposed_updates,
            "realized_directive_exposure_fraction": (
                round_controller_exposed_updates / rules.n_agents
            ),
            "controller_unique_readers": len(round_controller_readers),
            "controller_report_unique_readers": (
                len(round_controller_readers)
                if executed_communication_mode == CommunicationMode.REPORT
                else 0
            ),
            "controller_direct_replies": direct_replies,
            "directive_report_reply_count": direct_report_replies,
            "directive_evidence_report_count": direct_evidence_report_replies,
            "controller_reply_descendants": controller_descendants,
            "total_eligible_board_message_reads": round_message_read_count,
            "eligible_message_opportunities": eligible_message_opportunities,
            "eligible_directive_opportunities": eligible_directive_opportunities,
            "eligible_controller_report_opportunities": (
                eligible_controller_report_opportunities
            ),
            "controller_report_share_among_eligible_messages": (
                eligible_controller_report_opportunities
                / eligible_message_opportunities
                if eligible_message_opportunities
                else 0.0
            ),
            "controller_report_read_share": (
                controller_exposures / round_message_read_count
                if round_message_read_count
                else 0.0
            ),
            "controller_report_fact_acquisitions": (
                new_controller_facts
                if executed_communication_mode == CommunicationMode.REPORT
                else 0
            ),
            "controller_report_fact_reactivations": (
                reactivated_controller_facts
                if executed_communication_mode == CommunicationMode.REPORT
                else 0
            ),
            "controller_report_off_target_exposures": (
                round_controller_report_off_target_exposures
            ),
            "controller_report_target_adoptions": round_controller_report_adoptions,
            "controller_report_target_adoption_rate": (
                round_controller_report_adoptions
                / round_controller_report_off_target_exposures
                if round_controller_report_off_target_exposures
                else None
            ),
            "peer_report_exposures": peer_exposures,
            "peer_report_exposures_with_controller_actuation": (
                peer_exposures if action == ADVOCATE_TARGET else 0
            ),
            "peer_report_exposures_without_controller_actuation": (
                peer_exposures if action != ADVOCATE_TARGET else 0
            ),
            "directive_share_among_eligible_messages": (
                eligible_directive_opportunities / eligible_message_opportunities
                if eligible_message_opportunities
                else 0.0
            ),
            "directive_lineage_events": list(round_lineage_events),
            "directive_attributed_acquisitions": sum(
                row["event_type"] == "acquisition" for row in round_lineage_events
            ),
            "directive_attributed_refreshes": sum(
                row["event_type"] == "refresh" for row in round_lineage_events
            ),
            "peer_evidence_exposures": peer_exposures,
            "new_evidence_acquisitions": new_peer_facts + new_controller_facts,
            "theory_status": (
                "reference_only"
                if rules.social_mode == SOCIAL_MODE_BOARD
                else "matched_reference"
            ),
            "theory_skip_reason": (
                "finite-memory q-message board is not contemporaneous q-peer sampling"
                if rules.social_mode == SOCIAL_MODE_BOARD
                else None
            ),
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
            "reasoning_depth_L": int(state.task["reasoning_depth"]),
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
        logical_decisions=logical_decisions,
        validation_attempts=validation_attempts,
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
