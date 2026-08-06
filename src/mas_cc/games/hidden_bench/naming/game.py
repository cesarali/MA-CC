"""`hidden_bench_naming` - Hidden Profile information, naming-game interaction (§6).

Keeps HiddenBench's information structure (`Is` shared, `Iu` partitioned, agents
never told the split is asymmetric) and replaces the plenary discussion with the
Ashery/Baronchelli dyadic structure: each round one pair meets privately,
exchanges `m` messages each, then both commit to an option.

**Why this game belongs in this repository.** In the vanilla game every message
is broadcast, so "did the group pool its information" and "did one agent say the
thing" are nearly the same question. Here the transcript is a *local* object: a
fact reaches an agent only along an edge that was actually sampled, so pooling
becomes a diffusion process over an interaction graph, and `disclosure_reach`
measures how far each fact actually travelled. That is the quantity the
empowerment/MI machinery in `analysis/` is built to condition on.

**Phase model.** Per round: `EXCHANGE x m` (both partners speak simultaneously),
then `COMMIT` (both choose simultaneously). `rounds x 2 x (m + 1)` provider
calls per episode, before retries.

**Memory is private and per-agent.** An agent sees its own facts, its own past
partners and what they said to it, and the current conversation. It never sees a
global transcript, another agent's facts, or a conversation it was not in - and
`tests/games/hidden_bench/test_privacy.py` asserts exactly that.

**Open design question, defaulted rather than decided:** `allow_relay` (default
`true`) controls whether an agent may pass on a fact it learned from a partner.
`true` makes this a genuine information-diffusion process; `false` is a much
harder ceiling in which a fact can only travel one hop from its holder. It is a
*prompt-level* instruction, not an enforced constraint - the game cannot stop a
model from relaying, and pretending otherwise would make the `false` condition
silently untrue. Worth a decision before any headline number is quoted from it.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any, Mapping

from mas_cc.config import GameConfig
from mas_cc.core import AgentId, InteractionId, Seed
from mas_cc.games.protocols import Action, DecisionRequest, Game, GameSpec, Observation
from mas_cc.llm_runtime.validation import ValidationIssue, ValidationResult
from mas_cc.planning import DecisionStagePlan, GameCallPlan, InteractionCount, PromptScenario

from ..data import assign, assert_union_invariant, disclosed_facts, load_task_set
from ..records import (
    COMMIT,
    EXCHANGE,
    HiddenBenchAgentState,
    HiddenBenchGameState,
    HiddenBenchTransition,
)
from ..rules import NamingRules
from ..vanilla.prompts import extract_json_object, normalize_vote, shuffled_information
from .prompts import bind_commit_prompt, bind_message_prompt


class HiddenBenchNamingGame(Game):
    """Dyadic Hidden Profile: private pairs, private memory, per-agent commitment."""

    spec = GameSpec(
        game_type="hidden_bench_naming",
        version=1,
        description=(
            "HiddenBench Hidden Profile information under Ashery-style dyadic interaction: "
            "private pairwise exchange, private memory, independent commitment."
        ),
        game_family="choice",
        minimum_population=2,
        supported_topologies=("complete",),
    )

    def rules(self, config: GameConfig) -> NamingRules:
        if config.type != self.spec.game_type:
            raise ValueError(f"expected game type {self.spec.game_type!r}, got {config.type!r}")
        return NamingRules.from_config(config)

    # ---- initialize -----------------------------------------------------

    def initialize(self, config: GameConfig, seed: int) -> HiddenBenchGameState:
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
            root.derive("hidden-bench-assignment").create_random(),
            profile=rules.profile,
            prebuilt=task_set.prebuilt_allocations.get(task.task_id),
        )
        assert_union_invariant(task, assignment)

        agents = []
        for agent_id, info in assignment.items():
            presented = (
                shuffled_information(
                    info.shared, info.private, seed=int(root.derive(f"fact-shuffle:{agent_id}"))
                )
                if rules.shuffle_facts
                else (*info.shared, *info.private)
            )
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
                        "committed_action": None,
                        "pre_vote": None,
                        "post_vote": None,
                        "rationales": [],
                        # Which hidden facts this agent has *been exposed to*,
                        # natively or by a partner. The diffusion front.
                        "known_facts": sorted(info.evidence_types),
                    },
                )
            )

        return HiddenBenchGameState(
            game_type=self.spec.game_type,
            turn=0,
            agents=tuple(agents),
            terminated=False,
            data={
                "seed": seed,
                "phase": EXCHANGE,
                "task": {
                    "task_id": task.task_id,
                    "name": task.name,
                    "description": task.description_for(rules.n_agents),
                    "possible_answers": list(task.possible_answers),
                    "correct_answer": task.correct_answer,
                    "hidden_information": list(task.unshared_information),
                    "decoy": rules.decoy,
                    "n_agents_native": task.n_agents_native,
                    "source": task.source,
                },
                "rules": rules.to_dict(),
                "assignment": {str(k): v.to_dict() for k, v in assignment.items()},
                "round_index": 0,
                "exchange_index": 0,
                "current_pair": None,
                "dialogue": [],
                "transcript": [],
                "evaluator_history": [],
                "termination_reason": None,
            },
        )

    # ---- participants and observations -----------------------------------

    def select_participants(
        self, state: HiddenBenchGameState, config: GameConfig, rng: random.Random
    ) -> tuple[AgentId, ...]:
        """The pair for this round; resampled only when a new round starts.

        `pairing` is the extension point for a graph/network topology - a
        sampler that respects an adjacency structure drops in here and nowhere
        else, which is why the pair is state rather than being redrawn per step.
        """

        if state.terminated:
            raise ValueError("cannot select participants after termination")
        current = state.data.get("current_pair")
        if current is not None:
            return tuple(AgentId(str(item)) for item in current)
        pair = rng.sample([agent.agent_id for agent in state.agents], k=2)
        return (pair[0], pair[1])

    def _past_encounters(
        self, agent: HiddenBenchAgentState, rules: NamingRules
    ) -> tuple[Mapping[str, Any], ...]:
        history = tuple(agent.memory)
        # memory_size 0 means unbounded - remember every partner ever met.
        return history if rules.memory_size == 0 else history[-rules.memory_size :]

    def construct_observations(
        self,
        state: HiddenBenchGameState,
        participants: tuple[AgentId, ...],
        config: GameConfig,
    ) -> tuple[Observation, ...]:
        """Each partner's private view. The information boundary of this game.

        `dialogue` is relabelled per agent (`is_self`) rather than being handed
        over as a shared object with author ids, so an observation contains no
        identity information about the partner at all - only what it said.
        """

        rules = self.rules(config)
        interaction_id = InteractionId(f"interaction-{state.turn + 1:04d}")
        dialogue = list(state.data.get("dialogue", ()))
        observations = []
        for agent_id in participants:
            agent = state.hidden_bench_agent(agent_id)
            observations.append(
                Observation(
                    agent_id=agent_id,
                    interaction_id=interaction_id,
                    participants=participants,
                    visible_state={
                        "phase": state.phase,
                        "scenario": state.task["description"],
                        "presented_information": list(agent.presented_information),
                        "possible_answers": list(state.possible_answers),
                        "extra_prompt": rules.extra_for(agent_id, state),
                        "payoff_mode": rules.payoff.mode,
                        "allow_relay": rules.allow_relay,
                        "past_encounters": [
                            dict(entry) for entry in self._past_encounters(agent, rules)
                        ],
                        "dialogue": [
                            {
                                "is_self": str(item["agent_id"]) == str(agent_id),
                                "message": item["message"],
                            }
                            for item in dialogue
                        ],
                    },
                )
            )
        return tuple(observations)

    @staticmethod
    def _bound_prompt(observation: Observation) -> Any:
        visible = observation.visible_state
        shared = {
            "scenario": str(visible["scenario"]),
            "information": tuple(str(item) for item in visible["presented_information"]),
            "past_encounters": tuple(visible["past_encounters"]),
            "dialogue": tuple(visible["dialogue"]),
            "payoff_mode": str(visible["payoff_mode"]),
            "allow_relay": bool(visible["allow_relay"]),
            "extra": visible.get("extra_prompt"),
        }
        if str(visible["phase"]) == EXCHANGE:
            return bind_message_prompt(**shared)
        return bind_commit_prompt(
            possible_answers=tuple(str(item) for item in visible["possible_answers"]), **shared
        )

    def build_decision_requests(
        self,
        state: HiddenBenchGameState,
        observations: tuple[Observation, ...],
        config: GameConfig,
    ) -> tuple[DecisionRequest, ...]:
        rules = self.rules(config)
        return tuple(
            DecisionRequest(
                agent_id=observation.agent_id,
                interaction_id=observation.interaction_id,
                stage=str(observation.visible_state["phase"]),
                observation=observation,
                prompt=self._bound_prompt(observation),
                provider_required=True,
                retry_bound=rules.invalid_response_retries,
            )
            for observation in observations
        )

    # ---- parse / validate ------------------------------------------------

    def parse_action(self, request: DecisionRequest, response: str) -> Action:
        if request.stage == EXCHANGE:
            text = response.strip() or "<empty>"
            return Action(request.agent_id, text, request.stage, {"kind": "message"})
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
                "raw_response": response,
                "rationale": rationale if isinstance(rationale, str) else None,
                "resolved": vote is not None,
            },
        )

    def validate_action(
        self,
        state: HiddenBenchGameState,
        request: DecisionRequest,
        action: Action,
        config: GameConfig,
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []
        if action.agent_id != request.agent_id:
            issues.append(ValidationIssue("action.agent_id", "must match the focal decision"))
        if request.stage == EXCHANGE:
            if action.value == "<empty>":
                issues.append(ValidationIssue("action.value", "a message must not be empty"))
            return ValidationResult(tuple(issues))
        if not action.metadata.get("resolved"):
            issues.append(
                ValidationIssue(
                    "action.value",
                    f"commitment must resolve to one of: {', '.join(state.possible_answers)}",
                    action.metadata.get("raw_vote"),
                )
            )
        if action.metadata.get("rationale") is None:
            issues.append(ValidationIssue("action.rationale", "a commitment must carry a rationale"))
        return ValidationResult(tuple(issues))

    # ---- transition ------------------------------------------------------

    def apply_transition(
        self,
        state: HiddenBenchGameState,
        participants: tuple[AgentId, ...],
        actions: tuple[Action, ...],
        config: GameConfig,
    ) -> HiddenBenchTransition:
        rules = self.rules(config)
        if tuple(action.agent_id for action in actions) != participants:
            raise ValueError("actions must follow participant order")
        if len(participants) != 2:
            raise ValueError("hidden_bench_naming interacts in pairs")
        next_index = state.turn + 1
        data = dict(state.data)
        data["current_pair"] = [str(item) for item in participants]
        hidden = state.hidden_information

        if state.phase == EXCHANGE:
            dialogue = [
                *state.data.get("dialogue", ()),
                *(
                    {"agent_id": str(action.agent_id), "message": action.value}
                    for action in actions
                ),
            ]
            data["dialogue"] = dialogue
            data["transcript"] = [
                *state.transcript,
                *(
                    {
                        "speaker_id": str(action.agent_id),
                        "agent_id": str(action.agent_id),
                        "message": action.value,
                        "round": int(data["round_index"]) + 1,
                    }
                    for action in actions
                ),
            ]
            exchange_index = int(data["exchange_index"]) + 1
            data["exchange_index"] = exchange_index
            if exchange_index >= rules.messages_per_turn:
                data["phase"] = COMMIT
            # A fact this agent said is now a fact its partner has been exposed
            # to: that is the single step of diffusion this game is built around.
            agents = self._propagate(state, participants, actions, hidden)
            payoffs = {str(action.agent_id): 0.0 for action in actions}
            matched = None
        else:
            agents, payoffs, matched = self._commit(state, participants, actions, rules)
            data["phase"] = EXCHANGE
            data["round_index"] = int(data["round_index"]) + 1
            data["exchange_index"] = 0
            data["dialogue"] = []
            data["current_pair"] = None

        rounds_done = int(data["round_index"])
        interim = HiddenBenchGameState(
            game_type=state.game_type,
            turn=next_index,
            agents=agents,
            terminated=False,
            data={**data, "evaluator_history": state.evaluator_history},
        )
        reason = self._termination_reason(interim, rules, rounds_done)
        data["termination_reason"] = reason
        # One history entry per completed *round*, not per step. That is the
        # contract `metrics/rolling.py` is written against ("one mapping per
        # past round, oldest first, carrying at least `actions` and `success`"),
        # and it is the only framing under which a rolling coordination rate
        # means anything here - a window over steps would count every message
        # turn as a failed coordination, since a message has no outcome.
        # Streaming metrics evaluated during an exchange step therefore read the
        # previous round's summary, which is correct: nothing has resolved yet.
        if state.phase == COMMIT:
            data["evaluator_history"] = [
                *state.evaluator_history,
                self._round_summary(interim, actions, bool(matched), rounds_done),
            ]

        next_state = HiddenBenchGameState(
            game_type=state.game_type,
            turn=next_index,
            agents=agents,
            terminated=reason is not None,
            data=data,
        )
        return HiddenBenchTransition(
            interaction_id=InteractionId(f"interaction-{next_index:04d}"),
            actions=actions,
            payoffs=payoffs,
            next_state=next_state,
            matched=matched,
            termination_reason=reason,
        )

    @staticmethod
    def _propagate(
        state: HiddenBenchGameState,
        participants: tuple[AgentId, ...],
        actions: tuple[Action, ...],
        hidden: tuple[str, ...],
    ) -> tuple[HiddenBenchAgentState, ...]:
        """Mark each listener as exposed to whatever its partner just disclosed."""

        said_by = {
            str(action.agent_id): disclosed_facts(hidden, [action.value]) for action in actions
        }
        updated = []
        for agent in state.agents:
            if agent.agent_id not in participants:
                updated.append(agent)
                continue
            partner = next(item for item in participants if item != agent.agent_id)
            heard = said_by.get(str(partner), ())
            gained = {index for index, flag in enumerate(heard) if flag}
            known = set(agent.attributes.get("known_facts", ())) | gained
            updated.append(
                replace(agent, attributes={**dict(agent.attributes), "known_facts": sorted(known)})
            )
        return tuple(updated)

    def _commit(
        self,
        state: HiddenBenchGameState,
        participants: tuple[AgentId, ...],
        actions: tuple[Action, ...],
        rules: NamingRules,
    ) -> tuple[tuple[HiddenBenchAgentState, ...], dict[str, float], bool]:
        correct = state.correct_answer
        by_agent = {action.agent_id: action for action in actions}
        matched = actions[0].value == actions[1].value
        payoffs = {
            str(action.agent_id): rules.payoff.payoff(
                matched=matched, correct=action.value == correct
            )
            for action in actions
        }
        dialogue = list(state.data.get("dialogue", ()))
        updated = []
        for agent in state.agents:
            action = by_agent.get(agent.agent_id)
            if action is None:
                updated.append(agent)
                continue
            partner = next(item for item in participants if item != agent.agent_id)
            partner_said = [
                str(item["message"]) for item in dialogue if str(item["agent_id"]) == str(partner)
            ]
            entry = {
                "round": int(state.data["round_index"]) + 1,
                "partner_said": partner_said,
                "own_choice": action.value,
                "partner_choice": by_agent[partner].value,
                "payoff": payoffs[str(agent.agent_id)],
                "matched": matched,
            }
            attributes = dict(agent.attributes)
            attributes["committed_action"] = action.value
            attributes["post_vote"] = action.value
            if attributes.get("pre_vote") is None:
                # An agent's very first commitment is its pre-pooling answer,
                # which is what makes y_pre/improvement comparable with vanilla.
                attributes["pre_vote"] = action.value
            attributes["rationales"] = [
                *agent.rationales,
                {
                    "phase": COMMIT,
                    "round": entry["round"],
                    "vote": action.value,
                    "rationale": action.metadata.get("rationale"),
                },
            ]
            updated.append(
                replace(
                    agent,
                    score=agent.score + payoffs[str(agent.agent_id)],
                    memory=(*agent.memory, entry),
                    attributes=attributes,
                )
            )
        return tuple(updated), payoffs, matched

    def _termination_reason(
        self, state: HiddenBenchGameState, rules: NamingRules, rounds_done: int
    ) -> str | None:
        if rounds_done >= rules.rounds:
            return "max_rounds_reached"
        if not rules.stop_on_consensus:
            return None
        # Same threshold semantics as FirstConsensusTime: the *standing* share
        # of the most common committed option, over agents that have committed.
        standing = [agent.committed_action for agent in state.agents]
        committed = [value for value in standing if value is not None]
        if len(committed) < len(standing):
            return None
        top = max(committed.count(value) for value in set(committed))
        return "consensus_reached" if top / len(committed) >= rules.consensus_threshold else None

    def _round_summary(
        self,
        state: HiddenBenchGameState,
        actions: tuple[Action, ...],
        matched: bool,
        round_index: int,
    ) -> dict[str, Any]:
        hidden = state.hidden_information
        messages = [str(item["message"]) for item in state.transcript]
        disclosed = disclosed_facts(hidden, messages)
        # Reach is counted over agents *exposed* to each fact, which includes
        # its native holders - a fact held by four agents has already reached
        # four before anybody speaks.
        reach = []
        for index in range(len(hidden)):
            reach.append(
                sum(index in set(agent.attributes.get("known_facts", ())) for agent in state.agents)
            )
        return {
            "round_index": round_index,
            # `phase` is always COMMIT here; kept so the metric shelf can read
            # phase uniformly across both games.
            "phase": COMMIT,
            "success": matched,
            # `interaction_index` + `actions` mark this entry as a genuine
            # pairwise interaction outcome, which is what opts it in to the
            # binned trajectory tables (observability/recorder.py).
            "interaction_index": round_index,
            "actions": [action.value for action in actions],
            "selected_agents": [str(action.agent_id) for action in actions],
            "correct_answer": state.correct_answer,
            "decoy": state.task.get("decoy"),
            "possible_answers": list(state.possible_answers),
            "transcript_length": len(messages),
            "disclosed_hidden_facts": list(disclosed),
            "unshared_disclosure_rate": (sum(disclosed) / len(disclosed)) if disclosed else 0.0,
            "disclosure_reach": reach,
            # The dyadic analogue of Y_pre. There is no plenary pre-vote here,
            # so "before pooling" is per agent, not per population: an agent's
            # *first* commitment is the one it made on the least evidence it
            # will ever have. Averaged over the agents that have committed at
            # all - agents nobody has met yet are excluded rather than counted
            # as wrong.
            "accuracy_first_commitment": self._first_commitment_accuracy(state),
        }

    @staticmethod
    def _first_commitment_accuracy(state: HiddenBenchGameState) -> float | None:
        firsts = [agent.pre_vote for agent in state.agents if agent.pre_vote is not None]
        if not firsts:
            return None
        return sum(vote == state.correct_answer for vote in firsts) / len(firsts)

    def detect_termination(self, state: HiddenBenchGameState, config: GameConfig) -> str | None:
        self.rules(config)
        if state.terminated:
            return state.termination_reason or "max_rounds_reached"
        return None

    # ---- planning --------------------------------------------------------

    def call_plan(self, config: GameConfig) -> GameCallPlan:
        """`rounds x 2 x (m + 1)` calls per episode, before retries (§6).

        Priced per *episode* for the same reason as the vanilla game: the
        exchange stage runs `m` times a round and the commit stage once, so no
        single per-interaction multiplier is exact. `stop_on_consensus` makes
        the true count a *maximum*, which is why the lower bound is one round.
        """

        rules = self.rules(config)
        information = ("An illustrative shared fact.", "An illustrative private fact.")
        scenario_args = {
            "scenario": "A representative HiddenBench scenario description.",
            "information": information,
            "past_encounters": (),
            "dialogue": (),
            "payoff_mode": rules.payoff.mode,
            "allow_relay": rules.allow_relay,
            "extra": rules.extra_prompt,
        }
        message = PromptScenario(
            "pair_message",
            bind_message_prompt(**scenario_args),
            ("First message of a round: no dialogue and no memory yet - the lower bound.",),
        )
        commit = PromptScenario(
            "pair_commitment",
            bind_commit_prompt(possible_answers=("Option A", "Option B", "Option C"), **scenario_args),
            ("Commitment with an empty conversation - the lower bound.",),
        )
        exchanges = rules.rounds * rules.messages_per_turn * 2
        commits = rules.rounds * 2
        stages = (
            DecisionStagePlan(
                name="pair_exchange",
                requests_per_interaction=exchanges,
                retry_bound=rules.invalid_response_retries,
                expected_attempts_per_request=rules.expected_attempts_per_request,
                concurrency_within_stage=2,
                state_barrier_after_stage=True,
                lower_prompt=message,
                representative_prompt=message,
                maximum_prompt=message,
                prompt_scenarios=(message,),
                assumptions=(
                    f"{rules.rounds} rounds x {rules.messages_per_turn} message(s) x 2 partners.",
                    "Both partners speak from the same frozen pre-step state.",
                ),
            ),
            DecisionStagePlan(
                name="pair_commitment",
                requests_per_interaction=commits,
                retry_bound=rules.invalid_response_retries,
                expected_attempts_per_request=rules.expected_attempts_per_request,
                concurrency_within_stage=2,
                state_barrier_after_stage=True,
                lower_prompt=commit,
                representative_prompt=commit,
                maximum_prompt=commit,
                prompt_scenarios=(commit,),
                assumptions=(f"{rules.rounds} rounds x 2 independent commitments.",),
            ),
        )
        return GameCallPlan(
            game_type=self.spec.game_type,
            game_version=self.spec.version,
            interactions=InteractionCount(lower=1, expected=1, maximum=1, fixed=1),
            decision_stages=stages,
            stopping_condition_assumptions=(
                f"At most {rules.rounds} rounds.",
                (
                    f"Early stop at a standing consensus share of {rules.consensus_threshold}; "
                    "the estimate above is therefore an upper bound."
                    if rules.stop_on_consensus
                    else "No early stopping; every round runs."
                ),
                "The planning unit is one episode; stage counts are per-episode totals.",
            ),
            metadata={
                "population_size": rules.n_agents,
                "profile": rules.profile,
                "assignment_scheme": rules.assignment_scheme,
                "rounds": rules.rounds,
                "messages_per_turn": rules.messages_per_turn,
                "payoff_mode": rules.payoff.mode,
                "allow_relay": rules.allow_relay,
                "interactions_per_episode": rules.rounds * rules.steps_per_round,
                "provider_calls_per_episode": exchanges + commits,
                "validation_retry_bound_per_request": rules.invalid_response_retries,
                "provider_prices_included": False,
            },
        )
