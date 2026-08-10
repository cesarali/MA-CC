"""HiddenBench semantic reasoning on an explicit one-focal imitation state space."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace
from typing import Any, Mapping, Sequence

from mas_cc.config import GameConfig
from mas_cc.control import InteractionControlSignal
from mas_cc.core import AgentId, InteractionId, Seed
from mas_cc.games.protocols import Action, DecisionRequest, Game, GameSpec, Observation
from mas_cc.llm_runtime.validation import ValidationIssue, ValidationResult
from mas_cc.planning import DecisionStagePlan, GameCallPlan, InteractionCount, PromptScenario

from ..data import assign, assert_union_invariant, disclosed_facts, load_task_set
from ..records import HiddenBenchAgentState
from ..vanilla.prompts import extract_json_object, normalize_vote, shuffled_information
from .metrics import behavioral_transition_metrics, population_observables
from .prompts import bind_initial_prompt, bind_message_prompt, bind_update_prompt
from .state import ImitationGameState, ImitationRules, ImitationTransition

INITIAL_VOTE = "initial_vote"
PAIR_MESSAGE = "pair_message"
FOCAL_UPDATE = "focal_update"


class HiddenBenchImitationGame(Game):
    spec = GameSpec(
        game_type="hidden_bench_imitation",
        version=1,
        description=(
            "HiddenBench evidence under one-focal LLM reasoning or provider-free "
            "multi-opinion imitation dynamics."
        ),
        game_family="choice",
        minimum_population=2,
        supported_topologies=("complete",),
    )

    def rules(self, config: GameConfig) -> ImitationRules:
        return ImitationRules.from_config(config)

    def initialize(self, config: GameConfig, seed: int) -> ImitationGameState:
        rules = self.rules(config)
        root = Seed(seed)
        task_set = load_task_set(
            rules.task_set,
            corpus_root=rules.corpus_root,
            scheme=rules.assignment_scheme,
            n_agents=rules.n_agents,
        )
        task = task_set.by_name(rules.task_id) if rules.task_id else task_set.tasks[0]
        assignment = assign(
            task,
            rules.n_agents,
            rules.assignment_scheme,
            root.derive("hidden-bench-imitation-assignment").create_random(),
            profile=rules.profile,
            prebuilt=task_set.prebuilt_allocations.get(task.task_id),
        )
        assert_union_invariant(task, assignment)
        initial_votes = self._provider_free_initial_votes(rules, task.possible_answers, root)
        agents = []
        for index, (agent_id, info) in enumerate(assignment.items()):
            presented = (
                shuffled_information(
                    info.shared,
                    info.private,
                    seed=int(root.derive(f"imitation-fact-shuffle:{agent_id}")),
                )
                if rules.shuffle_facts
                else (*info.shared, *info.private)
            )
            vote = None if initial_votes is None else initial_votes[index]
            agents.append(
                HiddenBenchAgentState(
                    agent_id=agent_id,
                    score=0.0,
                    memory=(),
                    attributes={
                        "shared_information": list(info.shared),
                        "private_information": list(info.private),
                        "presented_information": list(presented),
                        "evidence_types": list(info.evidence_types),
                        "transformation": info.transformation,
                        "known_facts": sorted(info.evidence_types),
                        "committed_action": vote,
                        "pre_vote": vote,
                        "post_vote": vote,
                        "rationales": [],
                    },
                )
            )
        return ImitationGameState(
            game_type=self.spec.game_type,
            turn=0,
            agents=tuple(agents),
            terminated=False,
            data={
                "seed": seed,
                "phase": INITIAL_VOTE if initial_votes is None else FOCAL_UPDATE,
                "dynamics_mode": rules.dynamics_mode,
                "task": {
                    "task_id": task.task_id,
                    "name": task.name,
                    "description": task.description_for(rules.n_agents),
                    "possible_answers": list(task.possible_answers),
                    "correct_answer": task.correct_answer,
                    "hidden_information": list(task.unshared_information),
                    "n_agents_native": task.n_agents_native,
                    "source": task.source,
                },
                "rules": rules.to_dict(),
                "assignment": {str(key): value.to_dict() for key, value in assignment.items()},
                "initial_votes": [] if initial_votes is None else list(initial_votes),
                "transcript": [],
                "evaluator_history": [],
                "event_history": [],
                "physical_time": 0.0,
                "termination_reason": None,
            },
        )

    @staticmethod
    def _provider_free_initial_votes(
        rules: ImitationRules, options: Sequence[str], root: Seed
    ) -> tuple[str, ...] | None:
        if rules.initial_votes is not None:
            votes = rules.initial_votes
        elif rules.dynamics_mode == "reasoning" and rules.initialization_mode == "local_vote":
            return None
        else:
            distribution = rules.initial_distribution or {
                option: 1.0 for option in options
            }
            unknown = set(distribution) - set(options)
            if unknown:
                raise ValueError(f"initial_distribution contains unknown options: {sorted(unknown)}")
            labels = list(options)
            weights = [float(distribution.get(option, 0.0)) for option in labels]
            rng = root.derive("imitation-provider-free-initialization").create_random()
            votes = tuple(rng.choices(labels, weights=weights, k=rules.n_agents))
        unknown_votes = set(votes) - set(options)
        if unknown_votes:
            raise ValueError(f"initial_votes contains unknown options: {sorted(unknown_votes)}")
        return tuple(votes)

    def apply_initial_votes(
        self, state: ImitationGameState, actions: tuple[Action, ...]
    ) -> ImitationGameState:
        if len(actions) != len(state.agents):
            raise ValueError("initialization requires one local vote per population agent")
        by_agent = {action.agent_id: action for action in actions}
        agents = []
        votes = []
        for agent in state.agents:
            action = by_agent[agent.agent_id]
            votes.append(action.value)
            attributes = dict(agent.attributes)
            attributes.update(
                {
                    "committed_action": action.value,
                    "pre_vote": action.value,
                    "post_vote": action.value,
                    "rationales": [
                        {
                            "phase": INITIAL_VOTE,
                            "vote": action.value,
                            "rationale": action.metadata.get("rationale"),
                        }
                    ],
                }
            )
            agents.append(replace(agent, attributes=attributes))
        return ImitationGameState(
            game_type=state.game_type,
            turn=0,
            agents=tuple(agents),
            terminated=False,
            data={**dict(state.data), "phase": FOCAL_UPDATE, "initial_votes": votes},
        )

    # ---- generic game protocol / reasoning request construction --------

    def select_participants(
        self, state: ImitationGameState, config: GameConfig, rng: random.Random
    ) -> tuple[AgentId, ...]:
        focal, peer = rng.sample([agent.agent_id for agent in state.agents], 2)
        return focal, peer

    def construct_observations(
        self,
        state: ImitationGameState,
        participants: tuple[AgentId, ...],
        config: GameConfig,
    ) -> tuple[Observation, ...]:
        interaction_id = InteractionId(f"interaction-{state.turn + 1:04d}")
        rules = self.rules(config)
        return tuple(
            Observation(
                agent_id=agent_id,
                interaction_id=interaction_id,
                participants=participants,
                visible_state=self._visible_state(state, agent_id, rules),
            )
            for agent_id in participants
        )

    def _visible_state(
        self,
        state: ImitationGameState,
        agent_id: AgentId,
        rules: ImitationRules | None = None,
    ) -> dict[str, Any]:
        agent = state.hidden_bench_agent(agent_id)
        history = tuple(agent.memory)
        if rules is not None and rules.memory_size > 0:
            history = history[-rules.memory_size :]
        return {
            "scenario": state.task["description"],
            "presented_information": list(agent.presented_information),
            "possible_answers": list(state.possible_answers),
            "current_vote": agent.committed_action,
            "private_history": [dict(item) for item in history],
        }

    def initial_vote_requests(self, state: ImitationGameState, config: GameConfig) -> tuple[DecisionRequest, ...]:
        rules = self.rules(config)
        interaction_id = InteractionId("initial-local-votes")
        requests = []
        participants = tuple(agent.agent_id for agent in state.agents)
        for agent in state.agents:
            observation = Observation(
                agent_id=agent.agent_id,
                interaction_id=interaction_id,
                participants=participants,
                visible_state=self._visible_state(state, agent.agent_id, rules),
            )
            visible = observation.visible_state
            requests.append(
                DecisionRequest(
                    agent_id=agent.agent_id,
                    interaction_id=interaction_id,
                    stage=INITIAL_VOTE,
                    observation=observation,
                    prompt=bind_initial_prompt(
                        scenario=str(visible["scenario"]),
                        information=visible["presented_information"],
                        possible_answers=visible["possible_answers"],
                    ),
                    retry_bound=rules.invalid_response_retries,
                )
            )
        return tuple(requests)

    def message_requests(
        self,
        state: ImitationGameState,
        participants: tuple[AgentId, AgentId],
        dialogue: Sequence[Mapping[str, Any]],
        config: GameConfig,
    ) -> tuple[DecisionRequest, ...]:
        rules = self.rules(config)
        observations = self.construct_observations(state, participants, config)
        requests = []
        for observation in observations:
            visible = observation.visible_state
            own_dialogue = [
                {
                    "is_self": str(item["agent_id"]) == str(observation.agent_id),
                    "message": str(item["message"]),
                }
                for item in dialogue
            ]
            requests.append(
                DecisionRequest(
                    agent_id=observation.agent_id,
                    interaction_id=observation.interaction_id,
                    stage=PAIR_MESSAGE,
                    observation=observation,
                    prompt=bind_message_prompt(
                        scenario=str(visible["scenario"]),
                        information=visible["presented_information"],
                        history=visible["private_history"],
                        dialogue=own_dialogue,
                        allow_relay=rules.allow_relay,
                    ),
                    retry_bound=rules.invalid_response_retries,
                )
            )
        return tuple(requests)

    def focal_update_request(
        self,
        state: ImitationGameState,
        focal: AgentId,
        peer: AgentId | None,
        dialogue: Sequence[Mapping[str, Any]],
        controller_message: str | None,
        config: GameConfig,
    ) -> DecisionRequest:
        rules = self.rules(config)
        participants = (focal,) if peer is None else (focal, peer)
        interaction_id = InteractionId(f"interaction-{state.turn + 1:04d}")
        visible = self._visible_state(state, focal, rules)
        observation = Observation(
            agent_id=focal,
            interaction_id=interaction_id,
            participants=participants,
            visible_state=visible,
        )
        own_dialogue = [
            {"is_self": str(item["agent_id"]) == str(focal), "message": str(item["message"])}
            for item in dialogue
        ]
        return DecisionRequest(
            agent_id=focal,
            interaction_id=interaction_id,
            stage=FOCAL_UPDATE,
            observation=observation,
            prompt=bind_update_prompt(
                scenario=str(visible["scenario"]),
                information=visible["presented_information"],
                history=self._bounded_history(state.hidden_bench_agent(focal), rules),
                possible_answers=visible["possible_answers"],
                current_vote=str(visible["current_vote"]),
                dialogue=own_dialogue,
                controller_message=controller_message,
            ),
            retry_bound=rules.invalid_response_retries,
        )

    @staticmethod
    def _bounded_history(
        agent: HiddenBenchAgentState, rules: ImitationRules
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(agent.memory) if rules.memory_size == 0 else tuple(agent.memory[-rules.memory_size :])

    def build_decision_requests(
        self,
        state: ImitationGameState,
        observations: tuple[Observation, ...],
        config: GameConfig,
    ) -> tuple[DecisionRequest, ...]:
        if state.phase == INITIAL_VOTE:
            return self.initial_vote_requests(state, config)
        return self.message_requests(
            state,
            (observations[0].agent_id, observations[1].agent_id),
            (),
            config,
        )

    def parse_action(self, request: DecisionRequest, response: str) -> Action:
        if request.stage == PAIR_MESSAGE:
            value = response.strip() or "<empty>"
            return Action(request.agent_id, value, request.stage, {"kind": "message"})
        answers = tuple(str(item) for item in request.observation.visible_state["possible_answers"])
        parsed = extract_json_object(response)
        raw_vote = parsed.get("vote") if parsed else None
        rationale = parsed.get("rationale") if parsed else None
        vote = normalize_vote(raw_vote, answers)
        return Action(
            request.agent_id,
            vote if vote is not None else (str(raw_vote) if raw_vote is not None else "<unparsed>"),
            request.stage,
            {
                "kind": "commitment",
                "raw_vote": raw_vote,
                "rationale": rationale if isinstance(rationale, str) else None,
                "resolved": vote is not None,
            },
        )

    def validate_action(
        self,
        state: ImitationGameState,
        request: DecisionRequest,
        action: Action,
        config: GameConfig,
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []
        if action.agent_id != request.agent_id:
            issues.append(ValidationIssue("action.agent_id", "must match request agent"))
        if request.stage == PAIR_MESSAGE:
            if action.value == "<empty>":
                issues.append(ValidationIssue("action.value", "message must not be empty"))
        elif not action.metadata.get("resolved") or action.value not in state.possible_answers:
            issues.append(
                ValidationIssue("action.value", f"must resolve to one of {list(state.possible_answers)}")
            )
        return ValidationResult(tuple(issues))

    # ---- event transition ----------------------------------------------

    def apply_event_transition(
        self,
        state: ImitationGameState,
        *,
        focal: AgentId,
        peer: AgentId | None,
        action: Action,
        config: GameConfig,
        signal: InteractionControlSignal | None,
        dialogue: Sequence[Mapping[str, Any]] = (),
        classical: Mapping[str, Any] | None = None,
    ) -> ImitationTransition:
        rules = self.rules(config)
        before = [str(agent.committed_action) for agent in state.agents]
        focal_index = next(index for index, agent in enumerate(state.agents) if agent.agent_id == focal)
        before_focal = before[focal_index]
        if action.value not in state.possible_answers:
            raise ValueError("focal transition destination is outside the task option alphabet")
        agents = list(state.agents)
        focal_agent = state.hidden_bench_agent(focal)
        received = self._received_message(focal, peer, dialogue, signal)
        focal_attributes = dict(focal_agent.attributes)
        focal_attributes["committed_action"] = action.value
        focal_attributes["post_vote"] = action.value
        focal_attributes["rationales"] = [
            *focal_agent.rationales,
            {
                "phase": FOCAL_UPDATE,
                "event": state.turn + 1,
                "vote": action.value,
                "rationale": action.metadata.get("rationale"),
            },
        ]
        if peer is not None:
            peer_messages = [
                str(item["message"])
                for item in dialogue
                if str(item["agent_id"]) == str(peer)
            ]
            heard_facts = disclosed_facts(state.hidden_information, peer_messages)
            focal_attributes["known_facts"] = sorted(
                set(focal_attributes.get("known_facts", ()))
                | {index for index, flag in enumerate(heard_facts) if flag}
            )
        focal_memory = (*focal_agent.memory, {
            "event": state.turn + 1,
            "partner_id": None if peer is None else str(peer),
            "received_message": received,
            "own_vote_before": before_focal,
            "own_vote_after": action.value,
            "controller_action": None if signal is None else signal.action,
        })
        agents[focal_index] = replace(
            focal_agent, attributes=focal_attributes, memory=focal_memory
        )
        if peer is not None:
            peer_index = next(index for index, agent in enumerate(state.agents) if agent.agent_id == peer)
            peer_agent = state.hidden_bench_agent(peer)
            heard = self._received_message(peer, focal, dialogue, None)
            focal_messages = [
                str(item["message"])
                for item in dialogue
                if str(item["agent_id"]) == str(focal)
            ]
            heard_facts = disclosed_facts(state.hidden_information, focal_messages)
            peer_attributes = dict(peer_agent.attributes)
            peer_attributes["known_facts"] = sorted(
                set(peer_attributes.get("known_facts", ()))
                | {index for index, flag in enumerate(heard_facts) if flag}
            )
            agents[peer_index] = replace(
                peer_agent,
                attributes=peer_attributes,
                memory=(*peer_agent.memory, {
                    "event": state.turn + 1,
                    "partner_id": str(focal),
                    "received_message": heard,
                    "own_vote_before": peer_agent.committed_action,
                    "own_vote_after": peer_agent.committed_action,
                    "controller_action": None,
                }),
            )
        after = list(before)
        after[focal_index] = action.value
        next_index = state.turn + 1
        target = None if signal is None else signal.target
        # The no-control cells in a matched experiment still need the same
        # target-directed population projection.  In the v1 pilot the target is
        # the correct answer, while controller/sensor diagnostics remain NA.
        analysis_target = state.correct_answer if target is None else target
        before_obs = population_observables(
            before, state.possible_answers, state.correct_answer, analysis_target
        )
        after_obs = population_observables(
            after, state.possible_answers, state.correct_answer, analysis_target
        )
        messages = [str(item["message"]) for item in dialogue]
        disclosed = disclosed_facts(state.hidden_information, messages)
        all_messages = [str(item["message"]) for item in state.transcript] + messages
        disclosed_all = disclosed_facts(state.hidden_information, all_messages)
        disclosure_reach = [
            sum(
                index in set(agent.attributes.get("known_facts", ()))
                for agent in agents
            )
            for index in range(len(state.hidden_information))
        ]
        sensor = {} if signal is None else dict(signal.observation)
        sensor_counts = Counter(sensor.get("sampled_opinions", ()))
        sensor_count_vector = {
            option: sensor_counts.get(option, 0) for option in state.possible_answers
        }
        behavioral = behavioral_transition_metrics(
            before_obs,
            after_obs,
            focal_opinion_before=before_focal,
            focal_opinion_after=action.value,
            target=analysis_target,
            controller_action=None if signal is None else signal.action,
            sensor_count_vector=sensor_count_vector,
            sensor_sample_size=sensor.get("sample_size"),
        )
        event: dict[str, Any] = {
            "episode_id": f"{state.task['task_id']}-{state.data['seed']}",
            "interaction_index": next_index,
            "tau": next_index / len(state.agents),
            "seed": int(state.data["seed"]),
            "task_id": state.task["task_id"],
            "K": len(state.possible_answers),
            "N": len(state.agents),
            "dynamics_mode": rules.dynamics_mode,
            "population_state_before": before,
            "occupation_counts_before": before_obs["occupation_counts"],
            "population_shares_before": before_obs["population_shares"],
            "p_truth_before": before_obs["p_truth"],
            "p_ctrl_before": before_obs["p_ctrl"],
            "m_truth_before": before_obs["m_truth"],
            "m_ctrl_before": before_obs["m_ctrl"],
            "m_order_before": before_obs["m_order"],
            "H_vote_before": before_obs["H_vote"],
            "focal_agent_id": str(focal),
            "focal_opinion_before": before_focal,
            "peer_agent_id": None if peer is None else str(peer),
            "peer_opinion_before": None if peer is None else state.hidden_bench_agent(peer).committed_action,
            "controller_enabled": signal is not None,
            "controller_target": target,
            "analysis_target": analysis_target,
            "sensor_sample_size": sensor.get("sample_size"),
            "sensor_agent_ids": sensor.get("sampled_agent_ids", []),
            "sensor_observed_opinions": sensor.get("sampled_opinions", []),
            "sensor_count_vector": sensor_count_vector,
            "controller_policy": None if signal is None else signal.metadata.get("policy"),
            "controller_action": None if signal is None else signal.action,
            "controller_applied": bool(signal is not None and signal.message is not None),
            "focal_opinion_after": action.value,
            "population_state_after": after,
            "occupation_counts_after": after_obs["occupation_counts"],
            "population_shares_after": after_obs["population_shares"],
            "p_truth": after_obs["p_truth"],
            "p_ctrl": after_obs["p_ctrl"],
            "m_truth": after_obs["m_truth"],
            "m_ctrl": after_obs["m_ctrl"],
            "m_order": after_obs["m_order"],
            "H_vote": after_obs["H_vote"],
            "possible_answers": list(state.possible_answers),
            "correct_answer": state.correct_answer,
            "disclosed_hidden_facts_this_event": list(disclosed),
            "disclosed_hidden_facts": list(disclosed_all),
            "disclosure_reach": disclosure_reach,
            "unshared_disclosure_rate": (
                sum(disclosed_all) / len(disclosed_all) if disclosed_all else 0.0
            ),
            **behavioral,
        }
        if classical is not None:
            event.update(dict(classical))
        else:
            event.update(
                {
                    "classical_reaction_id": None,
                    "classical_source_opinion": None,
                    "classical_destination_opinion": None,
                    "classical_transition_rate_or_weight": None,
                    "classical_total_rate_or_normalizer": None,
                    "physical_time_increment": None,
                }
            )
        reason = self._termination_reason(after, next_index, rules)
        event_summary = {
            **event,
            "round_index": next_index,
            "selected_agents": [str(focal)],
            "success": action.value == state.correct_answer,
            "phase": FOCAL_UPDATE,
        }
        data = {
            **dict(state.data),
            "phase": FOCAL_UPDATE,
            "transcript": [*state.transcript, *dialogue],
            "evaluator_history": [*state.evaluator_history, event_summary],
            "event_history": [*state.data.get("event_history", ()), event],
            "termination_reason": reason,
        }
        next_state = ImitationGameState(
            game_type=state.game_type,
            turn=next_index,
            agents=tuple(agents),
            terminated=reason is not None,
            data=data,
        )
        return ImitationTransition(
            interaction_id=InteractionId(f"interaction-{next_index:04d}"),
            actions=(action,),
            payoffs={str(focal): 0.0},
            next_state=next_state,
            matched=(None if peer is None else action.value == state.hidden_bench_agent(peer).committed_action),
            termination_reason=reason,
            event=event,
        )

    @staticmethod
    def _received_message(
        listener: AgentId,
        speaker: AgentId | None,
        dialogue: Sequence[Mapping[str, Any]],
        signal: InteractionControlSignal | None,
    ) -> str | None:
        if signal is not None and signal.message is not None:
            return signal.message
        if speaker is None:
            return None
        heard = [str(item["message"]) for item in dialogue if str(item["agent_id"]) == str(speaker)]
        return "\n".join(heard) if heard else None

    @staticmethod
    def _termination_reason(votes: Sequence[str], turn: int, rules: ImitationRules) -> str | None:
        if turn >= rules.horizon:
            return "max_interactions_reached"
        if rules.stop_on_consensus and len(set(votes)) == 1:
            return "consensus_reached"
        return None

    def apply_transition(
        self,
        state: ImitationGameState,
        participants: tuple[AgentId, ...],
        actions: tuple[Action, ...],
        config: GameConfig,
    ) -> ImitationTransition:
        return self.apply_event_transition(
            state,
            focal=participants[0],
            peer=(participants[1] if len(participants) > 1 else None),
            action=actions[-1],
            config=config,
            signal=None,
        )

    def detect_termination(self, state: ImitationGameState, config: GameConfig) -> str | None:
        self.rules(config)
        if state.terminated:
            return state.termination_reason or "max_interactions_reached"
        return None

    # ---- provider demand ------------------------------------------------

    def call_plan(self, config: GameConfig) -> GameCallPlan:
        rules = self.rules(config)
        options = ("Option A", "Option B", "Option C")
        information = ("An illustrative shared fact.", "An illustrative private fact.")
        initial = PromptScenario(
            "local_initial_vote",
            bind_initial_prompt(
                scenario="A representative HiddenBench scenario.",
                information=information,
                possible_answers=options,
            ),
        )
        message = PromptScenario(
            "private_pair_message",
            bind_message_prompt(
                scenario="A representative HiddenBench scenario.",
                information=information,
                history=(),
                dialogue=(),
                allow_relay=rules.allow_relay,
            ),
        )
        update = PromptScenario(
            "focal_vote_update",
            bind_update_prompt(
                scenario="A representative HiddenBench scenario.",
                information=information,
                history=(),
                possible_answers=options,
                current_vote=options[0],
                dialogue=({"is_self": False, "message": "A representative peer message."},),
            ),
        )
        if rules.dynamics_mode == "classical":
            stages = (
                DecisionStagePlan(
                    name="classical_provider_free_jump",
                    requests_per_interaction=0,
                    provider_free_decisions_per_interaction=rules.horizon,
                    assumptions=("All initialization and jumps are provider-free.",),
                ),
            )
            interaction_count = InteractionCount(1, 1, 1, fixed=1)
        else:
            initialization_calls = 0 if rules.initial_votes is not None else rules.n_agents
            nested_controller = config.options.get("controller", {})
            controlled = isinstance(nested_controller, Mapping) and bool(
                nested_controller.get("enabled", False)
            )
            expected_attempts = (
                1.0
                if rules.invalid_response_retries == 0
                else min(
                    float(1 + rules.invalid_response_retries),
                    1.0 / max(1e-9, 1.0 - rules.expected_validation_failure_rate),
                )
            )
            stages = (
                DecisionStagePlan(
                    name="local_initialization",
                    requests_per_interaction=initialization_calls,
                    retry_bound=rules.invalid_response_retries,
                    expected_attempts_per_request=expected_attempts,
                    concurrency_within_stage=max(1, rules.n_agents),
                    lower_prompt=initial,
                    representative_prompt=initial,
                    maximum_prompt=initial,
                    prompt_scenarios=(initial,),
                    assumptions=(f"Initialization contributes {initialization_calls} request(s).",),
                ),
                DecisionStagePlan(
                    name="private_exchange",
                    requests_per_interaction=(
                        rules.horizon * 2 * rules.messages_per_agent
                    ),
                    retry_bound=rules.invalid_response_retries,
                    expected_attempts_per_request=expected_attempts,
                    concurrency_within_stage=2,
                    lower_prompt=message,
                    representative_prompt=message,
                    maximum_prompt=message,
                    prompt_scenarios=(message,),
                    assumptions=(
                        "This is an exact count without feedback control and a conservative "
                        "maximum when ADVOCATE_Z substitutes for an ordinary peer exchange.",
                    ),
                ),
                DecisionStagePlan(
                    name="focal_update",
                    requests_per_interaction=rules.horizon,
                    retry_bound=rules.invalid_response_retries,
                    expected_attempts_per_request=expected_attempts,
                    concurrency_within_stage=1,
                    lower_prompt=update,
                    representative_prompt=update,
                    maximum_prompt=update,
                    prompt_scenarios=(update,),
                ),
            )
            interaction_count = InteractionCount(1, 1, 1, fixed=1)
        return GameCallPlan(
            game_type=self.spec.game_type,
            game_version=self.spec.version,
            interactions=interaction_count,
            decision_stages=stages,
            stopping_condition_assumptions=(
                f"At most {rules.horizon} elementary focal-agent steps.",
                "Default stop_on_consensus is false; equal horizons are preserved.",
            ),
            metadata={
                "population_size": rules.n_agents,
                "dynamics_mode": rules.dynamics_mode,
                "interactions_per_episode": rules.horizon,
                "messages_per_agent": rules.messages_per_agent,
                "initial_votes_provider_supplied": rules.initial_votes is not None,
                "embedded_jump_chain": rules.dynamics_mode == "classical",
            },
        )
