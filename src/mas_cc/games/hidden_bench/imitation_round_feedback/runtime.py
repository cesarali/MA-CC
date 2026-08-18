"""Two-clock runtime for budgeted round-level HiddenBench feedback.

The slow clock - one controller decision per population round, `q_c` sensing, a
soft policy, an exact budget `b` of randomly placed controlled positions - is
unchanged.  The fast clock's reasoning kernel is not: one microscopic update is

    sample focal + q social slots
        -> render each slot from its current (vote, public reason)
        -> ONE focal provider call
        -> {vote, reason}
        -> apply both immediately

with no peer-dialogue generation anywhere.  Control still consumes one of the
`q` social slots rather than adding one, so a controlled and an uncontrolled
update cost exactly one call each and show exactly `q` sources.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mas_cc.config import RunConfig
from mas_cc.control import (
    Control,
    InteractionControlSignal,
    NoneControl,
    RoundControlSignal,
)
from mas_cc.core import Seed
from mas_cc.games.protocols import Action, AgentState, GameState
from mas_cc.llm_runtime.prompts import RegexTokenCounter, TokenCounter
from mas_cc.llm_runtime.providers import LLMProvider

from ..imitation.controller import ADVOCATE_TARGET, NO_OP
from ..imitation.game import FOCAL_UPDATE
from ..imitation.metrics import population_observables
from ..imitation.runtime import (
    PROMPT_FAMILIES as LEGACY_PROMPT_FAMILIES,
    ImitationInteractionRecord,
)
from ..imitation.state import ImitationGameState
from ..runtime import HiddenBenchDecision, _execute_decision, _notify
from .classical import classical_transition
from .controller import EVIDENCE_SHARED_FACT, select_shared_evidence_fact
from .game import HiddenBenchImitationRoundFeedbackGame
from .prompts import (
    PROMPT_FAMILY,
    agent_label,
    control_label,
    render_control_reason,
)
from .state import CLASSICAL_KERNEL_RULE, RoundFeedbackRecord, get_public_reason

REASONING_PROMPT_FAMILIES = (PROMPT_FAMILY,)
CLASSICAL_PROMPT_FAMILIES = (*LEGACY_PROMPT_FAMILIES, PROMPT_FAMILY)
"""Classical mode never reaches a provider, so the historical family names stay
acceptable there and the shipped classical configs keep loading unchanged."""

CONTROL_SOURCE_ID = "control-source"
"""The controller's stable identifier in the trajectory.  It is *not* what the
focal agent sees: in the prompt the controller is one more numbered
participant, see `control_label`."""


@dataclass(frozen=True, slots=True)
class RoundFeedbackGameResult:
    initial_state: ImitationGameState
    final_state: ImitationGameState
    initial_decisions: tuple[HiddenBenchDecision, ...]
    interactions: tuple[ImitationInteractionRecord, ...]
    rounds: tuple[RoundFeedbackRecord, ...]
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


def _controller_view(state: ImitationGameState) -> GameState:
    """Strip evidence, rationales, transcript, and private memory before sensing."""

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
    """Project the round decision onto one microscopic position.

    `message` overrides the controller's own text with what the focal agent was
    actually shown, so `controller_message` in the trajectory is the rendered
    public reason rather than a template the reasoning path no longer uses.
    """

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
    state: ImitationGameState,
    sampled_peers: tuple[Any, ...],
    *,
    replaced_peer_slot: int | None,
    controller_target: str | None,
    population_size: int,
    controller_evidence_fact: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """The `q` visible social inputs, in scheduler slot order.

    A controlled position substitutes `(X_j, R_j) -> (Z, R_C(Z))` in exactly one
    slot.  Both kinds of source carry the same four visible fields, which is
    what makes the controller indistinguishable from an ordinary participant in
    the rendered prompt.

    `controller_evidence_fact` is the round's already-selected shared fact under
    `evidence_mode: shared_fact`, or `None`.  It only ever changes `R_C(Z)`; the
    slot count, the field set, and the vote are untouched.
    """

    sources: list[dict[str, Any]] = []
    for slot, peer in enumerate(sampled_peers):
        if slot == replaced_peer_slot:
            if controller_target is None:
                raise ValueError("a controlled social slot requires a controller target")
            sources.append(
                {
                    "slot": slot,
                    "source_id": CONTROL_SOURCE_ID,
                    "source_type": "control",
                    "label": control_label(population_size),
                    "vote": controller_target,
                    "reason": render_control_reason(
                        controller_target, evidence_fact=controller_evidence_fact
                    ),
                }
            )
            continue
        agent = state.hidden_bench_agent(peer)
        sources.append(
            {
                "slot": slot,
                "source_id": str(peer),
                "source_type": "ordinary",
                "label": agent_label(peer),
                "vote": agent.committed_action,
                "reason": get_public_reason(agent),
            }
        )
    return tuple(sources)


def _influence_slots(sources: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """The historical per-slot row, derived from the same source records."""

    return tuple(
        {
            "slot": int(source["slot"]),
            "kind": "controller" if source["source_type"] == "control" else "peer",
            "peer_agent_id": (
                None if source["source_type"] == "control" else str(source["source_id"])
            ),
            "peer_vote_before": (
                None if source["source_type"] == "control" else source["vote"]
            ),
            "message": source["reason"],
        }
        for source in sources
    )


def _count_vector(values: Mapping[str, Any], options: tuple[str, ...]) -> list[int]:
    return [int(values.get(option, 0)) for option in options]


def sample_controlled_positions(
    population_size: int, intervention_budget: int, rng: Any
) -> tuple[int, ...]:
    """Uniform size-b schedule sampled without replacement, in canonical order."""

    if not 0 <= intervention_budget <= population_size:
        raise ValueError("intervention_budget must be between 0 and population_size")
    return tuple(
        sorted(rng.sample(range(population_size), intervention_budget))
    )


async def run_hidden_bench_imitation_round_feedback_game(
    game: HiddenBenchImitationRoundFeedbackGame,
    config: RunConfig,
    provider: LLMProvider,
    *,
    token_counter: TokenCounter | None = None,
    observer: Any | None = None,
    control: Control | None = None,
) -> RoundFeedbackGameResult:
    """Run one episode with exactly one controller decision per population round."""

    rules = game.rules(config.game)
    families = (
        REASONING_PROMPT_FAMILIES
        if rules.dynamics_mode == "reasoning"
        else CLASSICAL_PROMPT_FAMILIES
    )
    if config.prompt.prompt_family not in families:
        raise ValueError(
            f"prompt.prompt_family must be one of {families} in "
            f"{rules.dynamics_mode} mode; got {config.prompt.prompt_family!r}"
        )
    if config.prompt.prompt_version != rules.prompt_version:
        raise ValueError(
            f"prompt.prompt_version is {config.prompt.prompt_version} but "
            f"game.options.prompt_version is {rules.prompt_version}; they must match"
        )

    counter = token_counter or RegexTokenCounter()
    root = Seed(config.execution.seed)
    participant_rng = root.derive("round-feedback-focal-and-peer-selection").create_random()
    transition_rng = root.derive("round-feedback-classical-transition").create_random()
    sensor_rng = root.derive("round-feedback-controller-sensor-policy").create_random()
    replacement_rng = root.derive("round-feedback-controller-slot-replacement").create_random()
    resolved_control = None if control is None or isinstance(control, NoneControl) else control
    sensor_sample_size = getattr(resolved_control, "sensor_sample_size", None)
    intervention_budget = int(getattr(resolved_control, "intervention_budget", 0))
    # `getattr` rather than an attribute: a Control that predates this option -
    # or a test double - simply renders the historical fact-free advocacy.
    evidence_mode = str(getattr(resolved_control, "evidence_mode", "none"))
    if sensor_sample_size is not None and int(sensor_sample_size) > rules.n_agents:
        raise ValueError("controller sensor_sample_size cannot exceed the population size")
    if not 0 <= intervention_budget <= rules.n_agents:
        raise ValueError("controller intervention_budget must be between 0 and N")

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
        "imitation_round_feedback_initialized",
        initial_votes=list(state.initial_votes),
        provider_decisions=len(initial_decisions),
    )

    interactions: list[ImitationInteractionRecord] = []
    round_records: list[RoundFeedbackRecord] = []
    for round_index in range(rules.rounds):
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
            if round_signal is not None and not isinstance(round_signal, RoundControlSignal):
                raise TypeError("Control.round_signal must return RoundControlSignal or None")
            if round_signal is None:
                raise ValueError(
                    "the selected control does not implement round-level signaling"
                )

        action = None if round_signal is None else round_signal.action
        if action is not None and action not in {NO_OP, ADVOCATE_TARGET}:
            raise ValueError(
                "round controller action must be NO_OP or ADVOCATE_Z"
            )
        target = None if round_signal is None else round_signal.target
        analysis_target = state.correct_answer if target is None else target
        if analysis_target not in options:
            raise ValueError("controller target is outside the task option alphabet")
        probability = (
            None
            if round_signal is None
            else round_signal.metadata.get("advocacy_probability")
        )
        schedule_seed = int(root.derive(f"round-feedback-schedule:{round_index}"))
        if action == ADVOCATE_TARGET:
            controlled_positions = sample_controlled_positions(
                rules.n_agents,
                intervention_budget,
                root.derive(f"round-feedback-schedule:{round_index}").create_random(),
            )
        else:
            controlled_positions = ()
        # One fact per controller decision, not per actuated slot: the round is
        # the clock the controller acts on, so a single citation makes the whole
        # round's advocacy one message rather than `b` different ones.
        #
        # Reasoning mode only.  Classical dynamics renders no controller text at
        # all - a controlled slot is a kernel term, not a message - so drawing a
        # fact there would log evidence no agent was ever shown.
        evidence_fact_index: int | None = None
        evidence_fact: str | None = None
        if (
            rules.dynamics_mode == "reasoning"
            and action == ADVOCATE_TARGET
            and evidence_mode == EVIDENCE_SHARED_FACT
            and target is not None
        ):
            evidence_fact_index, evidence_fact = select_shared_evidence_fact(
                state,
                target,
                root.derive(
                    f"round-feedback-controller-evidence:{round_index}"
                ).create_random(),
            )
        controlled_set = frozenset(controlled_positions)
        schedule_hash = hashlib.sha256(
            json.dumps(list(controlled_positions), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        before_obs = population_observables(
            population_before, options, state.correct_answer, analysis_target
        )

        for within_round_index in range(rules.n_agents):
            selected = game.select_participants(state, config.game, participant_rng)
            focal, sampled_peers = selected[0], tuple(selected[1:])
            controlled_slot = within_round_index in controlled_set
            replaced_peer_slot = (
                replacement_rng.randrange(len(sampled_peers)) if controlled_slot else None
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
            decisions: list[HiddenBenchDecision] = []
            dialogue: list[Mapping[str, Any]] = []
            focal_agent = state.hidden_bench_agent(focal)
            focal_reason_before = get_public_reason(focal_agent)

            if rules.dynamics_mode == "reasoning":
                social_sources = build_social_sources(
                    state,
                    sampled_peers,
                    replaced_peer_slot=replaced_peer_slot,
                    controller_target=target,
                    population_size=rules.n_agents,
                    controller_evidence_fact=evidence_fact,
                )
                influence_slots = _influence_slots(social_sources)
                request = game.public_ballot_request(
                    state, focal, social_sources, config.game
                )
                update = await _execute_decision(
                    game, request, state, config, provider, counter, root, observer
                )
                decisions.append(update)
                focal_action = update.action
                classical_metadata = None
                # The agent's own new public reason is the only thing said at
                # this event, so it - and nothing else - enters the transcript
                # and the disclosure counters.
                dialogue.append(
                    {
                        "agent_id": str(focal),
                        "message": str(focal_action.metadata.get("reason") or ""),
                        "interaction_index": state.turn + 1,
                        "private": False,
                        "kind": "public_reason",
                    }
                )
            else:
                social_sources = ()
                peer_opinions = tuple(
                    str(state.hidden_bench_agent(peer).committed_action)
                    for peer in sampled_peers
                )
                influence_slots = tuple(
                    {
                        "slot": slot,
                        "kind": "controller" if slot == replaced_peer_slot else "peer",
                        "peer_agent_id": (
                            None if slot == replaced_peer_slot else str(peer)
                        ),
                        "peer_vote_before": (
                            None if slot == replaced_peer_slot else peer_opinions[slot]
                        ),
                        "message": (
                            round_signal.message
                            if slot == replaced_peer_slot and round_signal is not None
                            else None
                        ),
                    }
                    for slot, peer in enumerate(sampled_peers)
                )
                focal_opinion = str(state.hidden_bench_agent(focal).committed_action)
                jump = classical_transition(
                    population_state=[
                        str(agent.committed_action) for agent in state.agents
                    ],
                    focal_agent_id=str(focal),
                    focal_opinion=focal_opinion,
                    peer_agent_ids=[str(peer) for peer in sampled_peers],
                    peer_opinions=peer_opinions,
                    controlled_slot=controlled_slot,
                    controller_target=target,
                    replaced_peer_slot=replaced_peer_slot,
                    rng=transition_rng,
                )
                focal_action = Action(
                    focal,
                    jump.destination,
                    FOCAL_UPDATE,
                    {
                        "kind": "classical_unanimity_q_voter",
                        "reaction_id": (
                            f"unanimity-q-voter:"
                            f"{'K1' if controlled_slot else 'K0'}:"
                            f"{'switch' if jump.changed else 'self'}"
                        ),
                    },
                )
                classical_metadata = {
                    "classical_kernel": rules.classical_kernel,
                    "classical_kernel_rule": CLASSICAL_KERNEL_RULE,
                    "classical_reaction_id": focal_action.metadata["reaction_id"],
                    "classical_source_opinion": jump.source,
                    "classical_destination_opinion": jump.destination,
                    # Conditional on the already-sampled social context the
                    # strict-unanimity response is deterministic.
                    "classical_transition_rate_or_weight": 1.0,
                    "classical_total_rate_or_normalizer": 1.0,
                    "classical_base_weight": None,
                    "classical_control_weight": None,
                    "classical_candidate_channels": [],
                    "physical_time_increment": None,
                    "time_convention": "round_position_strict_unanimity_q_voter",
                    "classical_social_context_consumed": True,
                    "classical_effective_input_opinions": list(
                        jump.effective_input_opinions
                    ),
                    "classical_unanimous_opinion": jump.unanimous_opinion,
                    "classical_focal_changed": jump.changed,
                }

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
                controlled_slot=controlled_slot,
                message=None if control_source is None else str(control_source["reason"]),
            )
            micro_index = state.turn + 1
            round_fields = {
                "round_index": round_index,
                "within_round_index": within_round_index,
                "global_update_index": round_index * rules.n_agents + within_round_index,
                "microscopic_event_index": micro_index,
                "round_controller_action": action,
                "round_controller_target": target,
                "round_controller_advocate_probability": probability,
                "controlled_slot": controlled_slot,
                "intervention_budget": intervention_budget,
                "controlled_positions_hash_or_id": schedule_hash,
                "sampled_peer_agent_ids": [str(peer) for peer in sampled_peers],
                "effective_peer_agent_ids": [str(peer) for peer in effective_peers],
                "replaced_peer_agent_id": (
                    None if replaced_peer is None else str(replaced_peer)
                ),
                "controller_advocate_probability": probability,
                # Everything needed to reconstruct exactly what this focal saw
                # and produced, without re-rendering the prompt.  The matching
                # `focal_vote_before`/`focal_vote_after` are already written by
                # the inherited transition.
                "vote_visibility": rules.vote_visibility,
                "social_sources": [dict(source) for source in social_sources],
                "focal_reason_before": focal_reason_before,
                "focal_reason_after": focal_action.metadata.get("reason"),
            }
            transition = game.apply_round_event_transition(
                state,
                round_fields=round_fields,
                focal=focal,
                peers=effective_peers,
                action=focal_action,
                config=config.game,
                signal=micro_signal,
                dialogue=tuple(dialogue),
                classical=classical_metadata,
                sampled_peers=sampled_peers,
                replaced_peer=replaced_peer,
                replaced_peer_slot=replaced_peer_slot,
                influence_slots=influence_slots,
            )
            record = ImitationInteractionRecord(
                interaction_id=transition.interaction_id,
                interaction_index=micro_index,
                phase=FOCAL_UPDATE,
                participants=(focal, *effective_peers),
                decisions=tuple(decisions),
                transition=transition,
            )
            interactions.append(record)
            prompt_definitions = {
                decision.request.stage: decision.prompt_definition_hash
                for decision in decisions
            }
            _notify(
                observer,
                "record_interaction",
                round_index=micro_index,
                interaction=record,
                state=transition.next_state.to_dict(),
                prompt_definitions=prompt_definitions,
            )
            _notify(observer, "record_trajectory", record=record)
            _notify(
                observer,
                "event",
                "imitation_round_feedback_transition",
                **dict(transition.event or {}),
            )
            state = transition.next_state

        population_after = [str(agent.committed_action) for agent in state.agents]
        after_obs = population_observables(
            population_after, options, state.correct_answer, analysis_target
        )
        sensor = {} if round_signal is None else dict(round_signal.observation)
        raw_sensor_counts = sensor.get("sampled_opinion_counts", {})
        sensor_counts = (
            _count_vector(raw_sensor_counts, options)
            if isinstance(raw_sensor_counts, Mapping)
            else [0 for _ in options]
        )
        target_index = options.index(analysis_target)
        truth_index = options.index(state.correct_answer)
        q_c = None if round_signal is None else int(sensor.get("sample_size", 0))
        sensor_target_share = (
            None if not q_c else sensor_counts[target_index] / q_c
        )
        round_event = {
            "episode_id": f"{state.task['task_id']}-{state.data['seed']}",
            "round_index": round_index,
            "seed": int(state.data["seed"]),
            "task_id": state.task["task_id"],
            "K": len(options),
            "N": rules.n_agents,
            "dynamics_mode": rules.dynamics_mode,
            "classical_kernel": (
                rules.classical_kernel if rules.dynamics_mode == "classical" else None
            ),
            "classical_kernel_rule": (
                CLASSICAL_KERNEL_RULE
                if rules.dynamics_mode == "classical"
                else None
            ),
            "social_group_size": rules.social_group_size,
            "sensor_sample_size": q_c,
            "intervention_budget": intervention_budget,
            "sensing_fraction": None if q_c is None else q_c / rules.n_agents,
            "actuation_fraction": intervention_budget / rules.n_agents,
            "controller_enabled": round_signal is not None,
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
            "controller_action": action,
            "controller_evidence_mode": evidence_mode,
            "controller_evidence_fact": evidence_fact,
            "controller_evidence_fact_index": evidence_fact_index,
            "controller_advocate_probability": probability,
            "controller_advocacy_probability": probability,
            "sensor_agent_ids": list(sensor.get("sampled_agent_ids", ())),
            "sensor_observed_opinions": list(sensor.get("sampled_opinions", ())),
            "sensor_count_vector": sensor_counts if round_signal is not None else None,
            "sensor_target_share": sensor_target_share,
            "controlled_positions": list(controlled_positions),
            "controlled_position_count": len(controlled_positions),
            "controlled_positions_seed": schedule_seed,
            "controlled_positions_hash_or_id": schedule_hash,
            "population_state_before": population_before,
            "occupation_counts_before": _count_vector(
                before_obs["occupation_counts"], options
            ),
            "population_shares_before": [
                float(before_obs["population_shares"][option]) for option in options
            ],
            "target_count_before": int(
                before_obs["occupation_counts"][analysis_target]
            ),
            "truth_count_before": int(
                before_obs["occupation_counts"][state.correct_answer]
            ),
            "m_ctrl_before": before_obs["m_ctrl"],
            "m_truth_before": before_obs["m_truth"],
            "m_order_before": before_obs["m_order"],
            "H_vote_before": before_obs["H_vote"],
            "population_state_after": population_after,
            "occupation_counts_after": _count_vector(
                after_obs["occupation_counts"], options
            ),
            "population_shares_after": [
                float(after_obs["population_shares"][option]) for option in options
            ],
            "target_count_after": int(after_obs["occupation_counts"][analysis_target]),
            "truth_count_after": int(
                after_obs["occupation_counts"][state.correct_answer]
            ),
            "m_ctrl_after": after_obs["m_ctrl"],
            "m_truth_after": after_obs["m_truth"],
            "m_order_after": after_obs["m_order"],
            "H_vote_after": after_obs["H_vote"],
            "delta_m_ctrl": float(after_obs["m_ctrl"] - before_obs["m_ctrl"]),
            "delta_m_truth": float(after_obs["m_truth"] - before_obs["m_truth"]),
            "delta_m_order": float(after_obs["m_order"] - before_obs["m_order"]),
            "delta_H_vote": float(after_obs["H_vote"] - before_obs["H_vote"]),
            "possible_answers": list(options),
            "correct_answer": state.correct_answer,
        }
        round_record = RoundFeedbackRecord(round_index=round_index, event=round_event)
        round_records.append(round_record)
        _notify(observer, "record_round_trajectory", record=round_record)
        _notify(observer, "event", "imitation_round_feedback", **round_event)

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
    return RoundFeedbackGameResult(
        initial_state=initial_state,
        final_state=state,
        initial_decisions=initial_decisions,
        interactions=tuple(interactions),
        rounds=tuple(round_records),
        termination_reason=termination,
        logical_decisions=len(every),
        validation_attempts=sum(decision.validation_attempts for decision in every),
    )


def run_hidden_bench_imitation_round_feedback_game_sync(
    *args: Any, **kwargs: Any
) -> RoundFeedbackGameResult:
    return asyncio.run(run_hidden_bench_imitation_round_feedback_game(*args, **kwargs))


__all__ = [
    "CONTROL_SOURCE_ID",
    "RoundFeedbackGameResult",
    "build_social_sources",
    "run_hidden_bench_imitation_round_feedback_game",
    "run_hidden_bench_imitation_round_feedback_game_sync",
    "sample_controlled_positions",
]
