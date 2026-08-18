"""Relational reasoning under one-focal imitation with round-level feedback.

The dynamics are the HiddenBench round-feedback game's, unchanged in shape: one
controller decision per population round, ``N`` microscopic focal updates per
round, exactly one focal agent updating at each microscopic position, one
provider call per update returning one atomic public ballot, and control
consuming one of the ``q`` social slots rather than adding one.

What is new is a second state variable.  A ballot carries a machine-readable
``shared_fact_id``, and whichever focal agents actually *see* that ballot
acquire the fact:

    K_i(t+1) = K_i(t) u {facts exposed to i at this interaction}

so the trajectory records exactly which interaction moved which piece of
evidence to whom, separately for peer-carried and controller-injected facts.
Nothing else ever writes to ``K``.

This game does not subclass the HiddenBench one.  It shares its *shape*, not its
state: HiddenBench's rules, corpus loading, evidence allocation, and
``disclosed_facts`` string matching have no meaning on a frozen symbolic task,
and inheriting them would have meant overriding almost every one of them.  The
generic infrastructure - decision loop, seeds, prompt kernel, recorder,
controller sensing/policy - is reused as-is; see ``runtime.py``.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from mas_cc.config import GameConfig
from mas_cc.control import InteractionControlSignal
from mas_cc.core import AgentId, InteractionId, Seed
from mas_cc.games.protocols import Action, DecisionRequest, Game, GameSpec, Observation
from mas_cc.llm_runtime.validation import ValidationIssue, ValidationResult
from mas_cc.planning import DecisionStagePlan, GameCallPlan, InteractionCount, PromptScenario

from ...hidden_bench.imitation.metrics import population_observables
from ..data import RelationalTask, load_relational_task
from .metrics import knowledge_observables
from .prompts import (
    MAX_REASON_CHARACTERS,
    PROMPT_FAMILY,
    agent_label,
    build_relational_ballot_prompt,
    parse_relational_ballot,
    render_own_fact,
    shuffled_option_letters,
)
from .state import (
    CONTROLLER_SOURCE,
    FOCAL_UPDATE,
    GAME_TYPE,
    INITIAL_SOURCE,
    INITIAL_VOTE,
    PEER_SOURCE,
    RelationalAgentState,
    RelationalGameState,
    RelationalRoundRecord,
    RelationalRules,
    RelationalTransition,
)

BALLOT_STAGES = (INITIAL_VOTE, FOCAL_UPDATE)
"""Both provider-facing stages return the same three-field ballot; they differ
only in whether a standing position and social sources are rendered."""


class RelationalImitationRoundFeedbackGame(Game):
    """One-focal imitation on a frozen relational task, under a two-clock runtime."""

    spec = GameSpec(
        game_type=GAME_TYPE,
        version=1,
        description=(
            "Distributed relational reasoning under atomic public ballots with "
            "explicit evidence transmission, one sensed controller action and an "
            "exact randomized actuation budget per population round."
        ),
        game_family="choice",
        minimum_population=2,
        supported_topologies=("complete",),
    )

    # ---- configuration and initialization -------------------------------

    def rules(self, config: GameConfig) -> RelationalRules:
        return RelationalRules.from_config(config)

    def load_task(self, config: GameConfig) -> RelationalTask:
        """The frozen task this configuration selects, fully validated.

        Never generated, never repaired: §3's startup contract is enforced by
        ``data.load_relational_task``, which raises rather than patching.
        """

        rules = self.rules(config)
        return load_relational_task(
            rules.task_dataset_dir,
            rules.task_id,
            population_size=rules.n_agents,
        )

    def initialize(self, config: GameConfig, seed: int) -> RelationalGameState:
        rules = self.rules(config)
        task = self.load_task(config)
        root = Seed(seed)
        initial_votes = self._provider_free_initial_votes(rules, task, root)
        agents = []
        for index, agent_id in enumerate(task.agent_ids):
            # K_i(0) is the frozen assignment, verbatim and in task fact order.
            known = task.known_facts(agent_id)
            vote = None if initial_votes is None else initial_votes[index]
            agents.append(
                RelationalAgentState(
                    agent_id=AgentId(agent_id),
                    score=0.0,
                    memory=(),
                    attributes={
                        "known_fact_ids": list(known),
                        "initial_fact_ids": list(known),
                        "fact_provenance": {
                            fact_id: {
                                "source": INITIAL_SOURCE,
                                "round_index": None,
                                "within_round_index": None,
                                "from": None,
                            }
                            for fact_id in known
                        },
                        "committed_action": vote,
                        "public_reason": None,
                        "public_shared_fact_id": None,
                    },
                )
            )
        return RelationalGameState(
            game_type=self.spec.game_type,
            turn=0,
            agents=tuple(agents),
            terminated=False,
            data={
                "seed": seed,
                "phase": INITIAL_VOTE if initial_votes is None else FOCAL_UPDATE,
                "dynamics_mode": rules.dynamics_mode,
                "task": task.to_dict(),
                "rules": rules.to_dict(),
                "initial_votes": [] if initial_votes is None else list(initial_votes),
                "evaluator_history": [],
                "event_history": [],
                "termination_reason": None,
            },
        )

    @staticmethod
    def _provider_free_initial_votes(
        rules: RelationalRules, task: RelationalTask, root: Seed
    ) -> tuple[str, ...] | None:
        """``None`` means "ask the provider", i.e. ``local_vote`` on ``K_i(0)``.

        Provider-free votes are drawn in the **semantic** alphabet, like every
        other vote in this game; `initialization.initial_votes` must therefore
        name relations, not presentation letters.
        """

        options = task.semantic_answers
        if rules.initial_votes is not None:
            votes = rules.initial_votes
        elif rules.initialization_mode == "local_vote":
            return None
        else:
            distribution = rules.initial_distribution or {option: 1.0 for option in options}
            unknown = sorted(set(distribution) - set(options))
            if unknown:
                raise ValueError(f"initial_distribution contains unknown options: {unknown}")
            labels = list(options)
            weights = [float(distribution.get(option, 0.0)) for option in labels]
            rng = root.derive("relational-provider-free-initialization").create_random()
            votes = tuple(rng.choices(labels, weights=weights, k=rules.n_agents))
        unknown_votes = sorted(set(votes) - set(options))
        if unknown_votes:
            raise ValueError(f"initial_votes contains unknown options: {unknown_votes}")
        return tuple(votes)

    # ---- provider-facing requests ---------------------------------------

    def option_letters(
        self,
        state: RelationalGameState,
        focal: AgentId,
        *,
        stage: str = FOCAL_UPDATE,
    ) -> dict[str, str]:
        """This call's ``letter -> relation`` presentation map.

        Derived from the episode seed, the stage, the agent and the update
        index, so it replays exactly and is **stable across retries** of the
        same decision - a retry must not silently change what "B" means.
        """

        stream = Seed(int(state.data["seed"])).derive(
            f"relational-option-order:{stage}:{focal}:{state.turn}"
        ).create_random()
        return shuffled_option_letters(state.possible_answers, stream)

    def _citable_fact_ids(
        self, state: RelationalGameState, agent: RelationalAgentState
    ) -> tuple[str, ...]:
        """``K_i(t)`` in the task's own fact order - what the agent may share."""

        known = set(agent.known_fact_ids)
        return tuple(fact_id for fact_id in state.fact_ids if fact_id in known)

    def _known_fact_lines(
        self, state: RelationalGameState, agent: RelationalAgentState
    ) -> tuple[str, ...]:
        """``K_i`` rendered with identifiers, in the task's own fact order."""

        return tuple(
            render_own_fact(fact_id, state.fact_text(fact_id))
            for fact_id in self._citable_fact_ids(state, agent)
        )

    def ballot_request(
        self,
        state: RelationalGameState,
        focal: AgentId,
        social_sources: Sequence[Mapping[str, Any]],
        config: GameConfig,
        *,
        stage: str = FOCAL_UPDATE,
        interaction_id: InteractionId | None = None,
    ) -> DecisionRequest:
        """The single provider call behind one focal update."""

        rules = self.rules(config)
        agent = state.relational_agent(focal)
        resolved_id = interaction_id or InteractionId(f"interaction-{state.turn + 1:04d}")
        shared_before = agent.public_shared_fact_id
        letters = self.option_letters(state, focal, stage=stage)
        observation = Observation(
            agent_id=focal,
            interaction_id=resolved_id,
            participants=(focal,),
            visible_state={
                "question": state.task["question"],
                # The semantic alphabet, and this call's presentation of it.
                "possible_answers": list(state.possible_answers),
                "option_letters": dict(letters),
                # `K_i` alone: an audit record must never become the channel
                # that leaks the information whose spread is being measured.
                "known_fact_ids": list(agent.known_fact_ids),
                "current_vote": agent.committed_action,
                # Recorded, not rendered: the prompt shows this agent only its
                # vote.  These two are here so an audit row still describes the
                # agent's own standing ballot.
                "current_reason": agent.public_reason,
                "current_shared_fact_id": shared_before,
                "social_sources": [dict(item) for item in social_sources],
            },
        )
        return DecisionRequest(
            agent_id=focal,
            interaction_id=resolved_id,
            stage=stage,
            observation=observation,
            prompt=build_relational_ballot_prompt(
                identity=agent_label(focal),
                question=str(state.task["question"]),
                option_letters=letters,
                known_facts=self._known_fact_lines(state, agent),
                # `K_i(t)` and nothing wider: the JSON instruction advertises
                # exactly the ids this agent may legally cite.
                fact_ids=self._citable_fact_ids(state, agent),
                current_vote=agent.committed_action,
                social_sources=social_sources,
                vote_visibility=rules.vote_visibility,
                social_distrust=rules.social_distrust,
            ),
            retry_bound=rules.invalid_response_retries,
        )

    def initial_vote_requests(
        self, state: RelationalGameState, config: GameConfig
    ) -> tuple[DecisionRequest, ...]:
        """``local_vote``: one decision per agent from ``K_i(0)`` alone."""

        interaction_id = InteractionId("initial-local-votes")
        return tuple(
            self.ballot_request(
                state,
                agent.agent_id,
                (),
                config,
                stage=INITIAL_VOTE,
                interaction_id=interaction_id,
            )
            for agent in state.agents
        )

    def select_participants(
        self, state: RelationalGameState, config: GameConfig, rng: random.Random
    ) -> tuple[AgentId, ...]:
        """One focal agent and ``q`` distinct peers, drawn in one sample."""

        rules = self.rules(config)
        return tuple(
            rng.sample(
                [agent.agent_id for agent in state.agents], rules.social_group_size + 1
            )
        )

    def construct_observations(
        self,
        state: RelationalGameState,
        participants: tuple[AgentId, ...],
        config: GameConfig,
    ) -> tuple[Observation, ...]:
        """Only the focal agent ever observes: peers contribute standing ballots."""

        return (self.ballot_request(state, participants[0], (), config).observation,)

    def build_decision_requests(
        self,
        state: RelationalGameState,
        observations: tuple[Observation, ...],
        config: GameConfig,
    ) -> tuple[DecisionRequest, ...]:
        if state.phase == INITIAL_VOTE:
            return self.initial_vote_requests(state, config)
        return (self.ballot_request(state, observations[0].agent_id, (), config),)

    # ---- parsing and validation -----------------------------------------

    def parse_action(self, request: DecisionRequest, response: str) -> Action:
        """Resolve the returned letter to the semantic relation it named here.

        The model answers in *this call's* letter space; the action, and
        therefore every piece of persistent and socially visible state, carries
        the relation.  A letter never leaves this method.
        """

        visible = request.observation.visible_state
        letters = {str(key): str(value) for key, value in visible["option_letters"].items()}
        # `parse_relational_ballot` resolves a letter against the letter
        # alphabet, and also accepts the relation name spelled out - which is
        # already semantic and needs no translation.
        ballot = parse_relational_ballot(response, tuple(letters), letters)
        vote = None if ballot.vote is None else letters.get(ballot.vote, ballot.vote)
        return Action(
            request.agent_id,
            vote
            if vote is not None
            else (str(ballot.raw_vote) if ballot.raw_vote is not None else "<unparsed>"),
            request.stage,
            {
                "kind": "relational_ballot",
                "raw_vote": ballot.raw_vote,
                # The letter this agent actually typed, kept next to the
                # relation it resolved to so the mapping is auditable.
                "presented_letter": ballot.vote,
                "option_letters": dict(letters),
                "reason": ballot.reason,
                "shared_fact_id": ballot.shared_fact_id,
                "raw_shared_fact_id": ballot.raw_shared_fact_id,
                "shared_fact_present": ballot.shared_fact_present,
                "resolved": vote is not None,
            },
        )

    def validate_action(
        self,
        state: RelationalGameState,
        request: DecisionRequest,
        action: Action,
        config: GameConfig,
    ) -> ValidationResult:
        """Vote in the alphabet, reason non-empty, and **evidence honesty**.

        The citation check is here rather than in the response contract because
        this is the only place ``K_i(t)`` is in scope.  An agent that cites a
        fact it does not know fails validation and the normal retry loop asks
        again - it is never silently downgraded to "shared nothing", which would
        turn a hallucination into an invisible non-event.
        """

        issues: list[ValidationIssue] = []
        if action.agent_id != request.agent_id:
            issues.append(ValidationIssue("action.agent_id", "must match request agent"))
        if not action.metadata.get("resolved") or action.value not in state.possible_answers:
            issues.append(
                ValidationIssue(
                    "action.value", f"must resolve to one of {list(state.possible_answers)}"
                )
            )
        reason = action.metadata.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            issues.append(ValidationIssue("action.reason", "must be non-empty text"))
        elif len(reason.strip()) > MAX_REASON_CHARACTERS:
            issues.append(
                ValidationIssue(
                    "action.reason", f"must be at most {MAX_REASON_CHARACTERS} characters"
                )
            )
        if not action.metadata.get("shared_fact_present"):
            issues.append(
                ValidationIssue(
                    "action.shared_fact_id",
                    'must be present; use "none" to share nothing',
                )
            )
        shared = action.metadata.get("shared_fact_id")
        if shared is not None:
            known = set(state.relational_agent(request.agent_id).known_fact_ids)
            if str(shared) not in known:
                issues.append(
                    ValidationIssue(
                        "action.shared_fact_id",
                        f"{shared!r} is not among the facts this agent knows "
                        f"({sorted(known) or 'none'})",
                        shared,
                    )
                )
        return ValidationResult(tuple(issues))

    # ---- initial votes ---------------------------------------------------

    def apply_initial_votes(
        self, state: RelationalGameState, actions: tuple[Action, ...]
    ) -> RelationalGameState:
        """Commit the first ballots.  No fact moves: nobody has seen anyone yet."""

        if len(actions) != len(state.agents):
            raise ValueError("initialization requires one local vote per population agent")
        by_agent = {action.agent_id: action for action in actions}
        agents = []
        votes: list[str] = []
        for agent in state.agents:
            action = by_agent[agent.agent_id]
            votes.append(action.value)
            shared = action.metadata.get("shared_fact_id")
            attributes = {
                **dict(agent.attributes),
                "committed_action": action.value,
                "public_reason": action.metadata.get("reason"),
                "public_shared_fact_id": None if shared is None else str(shared),
            }
            agents.append(replace(agent, attributes=attributes))
        return RelationalGameState(
            game_type=state.game_type,
            turn=0,
            agents=tuple(agents),
            terminated=False,
            data={**dict(state.data), "phase": FOCAL_UPDATE, "initial_votes": votes},
        )

    # ---- the focal transition -------------------------------------------

    def apply_transition(
        self,
        state: RelationalGameState,
        participants: tuple[AgentId, ...],
        actions: tuple[Action, ...],
        config: GameConfig,
    ) -> RelationalTransition:
        return self.apply_round_event_transition(
            state,
            focal=participants[0],
            action=actions[-1],
            config=config,
            social_sources=(),
            round_fields={},
            signal=None,
        )

    def apply_round_event_transition(
        self,
        state: RelationalGameState,
        *,
        focal: AgentId,
        action: Action,
        config: GameConfig,
        social_sources: Sequence[Mapping[str, Any]] = (),
        round_fields: Mapping[str, Any] | None = None,
        signal: InteractionControlSignal | None = None,
        sampled_peers: Sequence[AgentId] = (),
        effective_peers: Sequence[AgentId] = (),
        replaced_peer: AgentId | None = None,
        replaced_peer_slot: int | None = None,
    ) -> RelationalTransition:
        """Apply one microscopic update: the vote, the ballot, and ``K`` growth.

        Exactly one agent's state changes here.  The focal's vote and public
        ballot are replaced, and its knowledge set grows by the facts the
        sources in *this* prompt exposed to it - nothing propagates to anyone
        who was not shown the ballot.
        """

        rules = self.rules(config)
        if action.value not in state.possible_answers:
            raise ValueError("focal transition destination is outside the task option alphabet")
        options = state.possible_answers
        before = [str(agent.committed_action) for agent in state.agents]
        focal_index = next(
            index for index, agent in enumerate(state.agents) if agent.agent_id == focal
        )
        before_focal = before[focal_index]
        focal_agent = state.relational_agent(focal)
        known_before = focal_agent.known_fact_ids

        exposures = self._exposures(social_sources)
        peer_exposed = tuple(fact for fact, kind, _ in exposures if kind == PEER_SOURCE)
        controller_exposed = tuple(
            fact for fact, kind, _ in exposures if kind == CONTROLLER_SOURCE
        )
        known_set = set(known_before)
        new_peer: list[str] = []
        new_controller: list[str] = []
        provenance = dict(focal_agent.fact_provenance)
        round_index = None if round_fields is None else round_fields.get("round_index")
        within_round_index = (
            None if round_fields is None else round_fields.get("within_round_index")
        )
        # Slot order is the tie-break: if two slots expose the same fact, both
        # count as exposures but only the first is an acquisition, and it is the
        # one that owns the provenance entry.
        for fact_id, kind, source_id in exposures:
            if fact_id in known_set:
                continue
            known_set.add(fact_id)
            (new_controller if kind == CONTROLLER_SOURCE else new_peer).append(fact_id)
            provenance[fact_id] = {
                "source": kind,
                "round_index": round_index,
                "within_round_index": within_round_index,
                "from": source_id,
            }
        order = state.fact_ids
        known_after = tuple(fact_id for fact_id in order if fact_id in known_set)

        shared_after = action.metadata.get("shared_fact_id")
        shared_after = None if shared_after is None else str(shared_after)
        if shared_after is not None and shared_after not in set(known_before):
            # Belt and braces: `validate_action` already rejects this, and the
            # runtime never hand-builds an action.  Reaching here would mean an
            # agent published evidence it never held.
            raise ValueError(
                f"agent {focal} cannot share fact {shared_after!r}: it is not in K_i(t)"
            )

        agents = list(state.agents)
        agents[focal_index] = replace(
            focal_agent,
            attributes={
                **dict(focal_agent.attributes),
                "committed_action": action.value,
                "public_reason": action.metadata.get("reason"),
                "public_shared_fact_id": shared_after,
                "known_fact_ids": list(known_after),
                "fact_provenance": provenance,
            },
            memory=(
                *focal_agent.memory,
                {
                    "event": state.turn + 1,
                    "peer_ids": [str(peer) for peer in effective_peers],
                    "own_vote_before": before_focal,
                    "own_vote_after": action.value,
                    "own_shared_fact_id": shared_after,
                    "peer_exposed_fact_ids": list(peer_exposed),
                    "controller_fact_id": (
                        controller_exposed[0] if controller_exposed else None
                    ),
                    "controller_action": None if signal is None else signal.action,
                },
            ),
        )

        after = list(before)
        after[focal_index] = action.value
        next_index = state.turn + 1
        target = None if signal is None else signal.target
        analysis_target = state.correct_answer if target is None else target
        before_obs = population_observables(
            before, options, state.correct_answer, analysis_target
        )
        after_obs = population_observables(after, options, state.correct_answer, analysis_target)
        knowledge_after = knowledge_observables(agents, state.supporting_fact_ids)
        truth_increment = int(action.value == state.correct_answer) - int(
            before_focal == state.correct_answer
        )
        sensor = {} if signal is None else dict(signal.observation)

        event: dict[str, Any] = {
            "episode_id": f"{state.task['task_id']}-{state.data['seed']}",
            "interaction_index": next_index,
            "microscopic_event_index": next_index,
            "seed": int(state.data["seed"]),
            "task_id": state.task["task_id"],
            "K": len(options),
            "N": len(state.agents),
            "population_size": len(state.agents),
            "social_group_size": rules.social_group_size,
            "dynamics_mode": rules.dynamics_mode,
            "prompt_family": PROMPT_FAMILY,
            "prompt_version": rules.prompt_version,
            # --- who acted, and on what social input -----------------------
            "focal_agent_id": str(focal),
            "focal_opinion_before": before_focal,
            "focal_vote_before": before_focal,
            "focal_opinion_after": action.value,
            "focal_vote_after": action.value,
            "vote_before": before_focal,
            "vote_after": action.value,
            "sampled_peer_ids": [str(peer) for peer in sampled_peers],
            "effective_peer_ids": [str(peer) for peer in effective_peers],
            "replaced_peer_id": None if replaced_peer is None else str(replaced_peer),
            "replaced_peer_slot": replaced_peer_slot,
            "social_sources": [dict(source) for source in social_sources],
            "focal_reason_before": focal_agent.public_reason,
            "focal_reason_after": action.metadata.get("reason"),
            "focal_shared_fact_id_before": focal_agent.public_shared_fact_id,
            "focal_shared_fact_id": shared_after,
            "vote_visibility": rules.vote_visibility,
            # --- knowledge state (§16) -------------------------------------
            "focal_known_fact_ids_before": list(known_before),
            "focal_known_fact_ids_after": list(known_after),
            "peer_exposed_fact_ids": list(peer_exposed),
            "controller_fact_id": controller_exposed[0] if controller_exposed else None,
            "new_peer_fact_ids": list(new_peer),
            "new_controller_fact_ids": list(new_controller),
            "peer_fact_exposures": len(peer_exposed),
            "controller_fact_exposures": len(controller_exposed),
            "new_peer_facts": len(new_peer),
            "new_controller_facts": len(new_controller),
            "focal_supporting_fact_coverage_before": _coverage(
                known_before, state.supporting_fact_ids
            ),
            "focal_supporting_fact_coverage_after": _coverage(
                known_after, state.supporting_fact_ids
            ),
            # --- controller ------------------------------------------------
            "controller_enabled": signal is not None,
            "controller_action": None if signal is None else signal.action,
            "controller_target": target,
            "analysis_target": analysis_target,
            "controller_message": None if signal is None else signal.message,
            "controller_policy": None if signal is None else signal.metadata.get("policy"),
            "controller_threshold": (
                None if signal is None else signal.metadata.get("threshold")
            ),
            "controller_beta": None if signal is None else signal.metadata.get("beta"),
            "controller_advocacy_probability": (
                None if signal is None else signal.metadata.get("advocacy_probability")
            ),
            "sensor_sample_size": sensor.get("sample_size"),
            "sensor_agent_ids": list(sensor.get("sampled_agent_ids", ())),
            "sensor_observed_opinions": list(sensor.get("sampled_opinions", ())),
            # Per-option sensed counts in the task's own alphabet. Required by
            # the compact scientific writer whenever the controller acted -
            # `results_only` normalizes every micro event through it, and that
            # path is not exercised by the `full` profile.
            "sensor_count_vector": {
                option: _sensed_counts(sensor).get(option, 0) for option in options
            },
            # --- population observables ------------------------------------
            "population_state_before": before,
            "population_state_after": after,
            "occupation_counts_before": before_obs["occupation_counts"],
            "occupation_counts_after": after_obs["occupation_counts"],
            "population_shares_before": before_obs["population_shares"],
            "population_shares_after": after_obs["population_shares"],
            "m_truth_before": before_obs["m_truth"],
            "m_ctrl_before": before_obs["m_ctrl"],
            "m_order_before": before_obs["m_order"],
            "H_vote_before": before_obs["H_vote"],
            "p_truth": after_obs["p_truth"],
            "p_ctrl": after_obs["p_ctrl"],
            "m_truth": after_obs["m_truth"],
            "m_ctrl": after_obs["m_ctrl"],
            "m_order": after_obs["m_order"],
            "H_vote": after_obs["H_vote"],
            "delta_m_truth": float(after_obs["m_truth"] - before_obs["m_truth"]),
            "delta_m_ctrl": float(after_obs["m_ctrl"] - before_obs["m_ctrl"]),
            "delta_m_order": float(after_obs["m_order"] - before_obs["m_order"]),
            "delta_H_vote": float(after_obs["H_vote"] - before_obs["H_vote"]),
            "focal_changed": int(action.value != before_focal),
            "focal_adopted_target": int(
                before_focal != analysis_target and action.value == analysis_target
            ),
            "focal_left_target": int(
                before_focal == analysis_target and action.value != analysis_target
            ),
            "truth_current_increment": truth_increment,
            "truth_switch_toward": truth_increment == 1,
            "truth_switch_away": truth_increment == -1,
            "truth_vote_share": after_obs["p_truth"],
            "mean_supporting_fact_coverage": knowledge_after["mean_supporting_fact_coverage"],
            "full_proof_agent_share": knowledge_after["full_proof_agent_share"],
            "possible_answers": list(options),
            "correct_answer": state.correct_answer,
            "supporting_fact_ids": list(state.supporting_fact_ids),
            **dict(round_fields or {}),
        }

        reason = self._termination_reason(after, next_index, rules)
        event_summary = {
            **event,
            "round_index": event.get("round_index", next_index),
            "selected_agents": [str(focal)],
            "success": action.value == state.correct_answer,
            "phase": FOCAL_UPDATE,
        }
        next_state = RelationalGameState(
            game_type=state.game_type,
            turn=next_index,
            agents=tuple(agents),
            terminated=reason is not None,
            data={
                **dict(state.data),
                "phase": FOCAL_UPDATE,
                "evaluator_history": [*state.evaluator_history, event_summary],
                "event_history": [*state.event_history, event],
                "termination_reason": reason,
            },
        )
        return RelationalTransition(
            interaction_id=InteractionId(f"interaction-{next_index:04d}"),
            actions=(action,),
            payoffs={str(focal): 0.0},
            next_state=next_state,
            matched=(
                None
                if not effective_peers
                else action.value
                in {
                    state.relational_agent(peer).committed_action
                    for peer in effective_peers
                }
            ),
            termination_reason=reason,
            event=event,
        )

    @staticmethod
    def _exposures(
        social_sources: Sequence[Mapping[str, Any]],
    ) -> tuple[tuple[str, str, str | None], ...]:
        """``(fact_id, source_kind, source_id)`` for every fact this focal saw.

        The rendered prompt and this list are built from the same source
        records, so a fact can only be credited to an agent if that agent's
        prompt actually contained it.
        """

        found: list[tuple[str, str, str | None]] = []
        for source in social_sources:
            fact_id = source.get("shared_fact_id")
            if not fact_id:
                continue
            kind = (
                CONTROLLER_SOURCE
                if str(source.get("source_type")) == "control"
                else PEER_SOURCE
            )
            source_id = None if kind == CONTROLLER_SOURCE else str(source.get("source_id"))
            found.append((str(fact_id), kind, source_id))
        return tuple(found)

    @staticmethod
    def _termination_reason(
        votes: Sequence[str], turn: int, rules: RelationalRules
    ) -> str | None:
        if turn >= rules.horizon:
            return "max_rounds_reached"
        # Consensus is only meaningful at a slow-clock boundary: a round that
        # starts always contains exactly N microscopic opportunities.
        if (
            rules.stop_on_consensus
            and turn % rules.n_agents == 0
            and len(set(votes)) == 1
        ):
            return "consensus_reached"
        return None

    def detect_termination(
        self, state: RelationalGameState, config: GameConfig
    ) -> str | None:
        self.rules(config)
        if state.terminated:
            return state.termination_reason or "max_rounds_reached"
        return None

    # ---- provider demand ------------------------------------------------

    def call_plan(self, config: GameConfig) -> GameCallPlan:
        """One provider call per focal update, controlled or not.

        Control replaces a social slot rather than adding a call, so a
        controlled and an uncontrolled update are priced identically.
        """

        rules = self.rules(config)
        letters = {"A": "NORTH", "B": "NORTHEAST", "C": "SOUTHWEST"}
        relations = tuple(letters.values())
        facts = ("f1: Lumo is north of Kavi.", "f2: Kavi is east of Tero.")
        initial = PromptScenario(
            "local_initial_ballot",
            build_relational_ballot_prompt(
                identity="Agent 1",
                question="Where is Lumo relative to Tero?",
                option_letters=letters,
                known_facts=facts,
                fact_ids=("f1", "f2"),
                current_vote=None,
                social_distrust=rules.social_distrust,
            ),
        )
        update = PromptScenario(
            "relational_ballot_update",
            build_relational_ballot_prompt(
                identity="Agent 1",
                question="Where is Lumo relative to Tero?",
                option_letters=letters,
                known_facts=facts,
                fact_ids=("f1", "f2"),
                # A representative source carries a *semantic* vote; the
                # builder localizes it into this call's letters.
                current_vote=relations[0],
                social_sources=tuple(
                    {
                        # A rendered source is identity + vote + exposed fact;
                        # a speaker's reason is never shown, so it is not part
                        # of what this stage has to pay for.
                        "label": f"Agent {slot + 2}",
                        "vote": relations[slot % len(relations)],
                        "shared_fact_text": "Kavi is east of Tero.",
                    }
                    for slot in range(rules.social_group_size)
                ),
                vote_visibility=rules.vote_visibility,
                social_distrust=rules.social_distrust,
            ),
        )
        initialization_calls = (
            rules.n_agents
            if rules.initial_votes is None and rules.initialization_mode == "local_vote"
            else 0
        )
        expected_attempts = (
            1.0
            if rules.invalid_response_retries == 0
            else min(
                float(1 + rules.invalid_response_retries),
                1.0 / max(1e-9, 1.0 - rules.expected_validation_failure_rate),
            )
        )
        return GameCallPlan(
            game_type=self.spec.game_type,
            game_version=self.spec.version,
            interactions=InteractionCount(1, 1, 1, fixed=1),
            decision_stages=(
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
                    assumptions=(
                        f"Initialization contributes {initialization_calls} request(s).",
                    ),
                ),
                DecisionStagePlan(
                    name="relational_ballot_update",
                    requests_per_interaction=rules.horizon,
                    retry_bound=rules.invalid_response_retries,
                    expected_attempts_per_request=expected_attempts,
                    concurrency_within_stage=1,
                    lower_prompt=update,
                    representative_prompt=update,
                    maximum_prompt=update,
                    prompt_scenarios=(update,),
                    assumptions=(
                        "Exactly one provider call per focal update, controlled or "
                        "not: control replaces one social slot rather than adding "
                        "a call.",
                    ),
                ),
            ),
            stopping_condition_assumptions=(
                f"At most {rules.horizon} elementary focal-agent steps.",
                "Default stop_on_consensus is false; equal horizons are preserved.",
            ),
            metadata={
                "population_size": rules.n_agents,
                "dynamics_mode": rules.dynamics_mode,
                "interactions_per_episode": rules.horizon,
                "population_rounds": rules.rounds,
                "social_group_size": rules.social_group_size,
                "vote_visibility": rules.vote_visibility,
                "social_distrust": rules.social_distrust,
                "task_id": rules.task_id,
                "initial_votes_provider_supplied": rules.initial_votes is not None,
            },
        )


def _sensed_counts(sensor: Mapping[str, Any]) -> Counter:
    """Votes the controller's sensor actually observed, counted per option."""

    return Counter(
        str(value) for value in sensor.get("sampled_opinions", ()) if value is not None
    )


def _coverage(known: Sequence[str], supporting: Sequence[str]) -> float:
    if not supporting:
        return 1.0
    return len(set(known) & set(supporting)) / len(set(supporting))


__all__ = [
    "BALLOT_STAGES",
    "FOCAL_UPDATE",
    "INITIAL_VOTE",
    "RelationalImitationRoundFeedbackGame",
    "RelationalRoundRecord",
]
