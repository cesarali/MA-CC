"""HiddenBench imitation with one controller decision per population round.

The state machinery and the two-clock controller semantics are inherited from
`hidden_bench_imitation` unchanged.  What this game overrides is the *reasoning
kernel*: instead of generating peer dialogue and then voting in a second call,
one focal update is one provider call returning one atomic public ballot,

    P(X_i^{t+1}, R_i^{t+1} | E_i, X_i^t, R_i^t, {X_j^t, R_j^t}_{j in N_i}),

whose vote is applied immediately and whose reason becomes the public text the
next focal agent that draws this agent as a social source will read.  See
`prompts.py` for the one-call contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from mas_cc.config import GameConfig
from mas_cc.core import AgentId, InteractionId
from mas_cc.games.protocols import Action, DecisionRequest, GameSpec, Observation
from mas_cc.llm_runtime.validation import ValidationIssue, ValidationResult
from mas_cc.planning import DecisionStagePlan, GameCallPlan, InteractionCount, PromptScenario

from ..data import disclosed_facts
from ..imitation.game import FOCAL_UPDATE, INITIAL_VOTE, HiddenBenchImitationGame
from ..imitation.state import ImitationGameState, ImitationTransition
from .prompts import (
    MAX_REASON_CHARACTERS,
    agent_label,
    build_public_ballot_update_prompt,
    parse_public_ballot_update,
)
from .state import GAME_TYPE, RoundFeedbackRules, get_public_reason, set_public_reason

PUBLIC_BALLOT_STAGES = (INITIAL_VOTE, FOCAL_UPDATE)
"""Both provider-facing stages return the same `{vote, reason}` object; they
differ only in whether a standing position and social sources are rendered."""


class HiddenBenchImitationRoundFeedbackGame(HiddenBenchImitationGame):
    """The inherited HiddenBench state machinery under a two-clock runtime."""

    spec = GameSpec(
        game_type=GAME_TYPE,
        version=2,
        description=(
            "HiddenBench one-focal imitation under atomic public ballots, with "
            "one sensed controller action and an exact randomized actuation "
            "budget per population round."
        ),
        game_family="choice",
        minimum_population=2,
        supported_topologies=("complete",),
    )

    def rules(self, config: GameConfig) -> RoundFeedbackRules:
        return RoundFeedbackRules.from_config(config)

    @staticmethod
    def _termination_reason(
        votes: Sequence[str], turn: int, rules: RoundFeedbackRules
    ) -> str | None:
        if turn >= rules.horizon:
            return "max_rounds_reached"
        # Consensus is checked only at a slow-clock boundary.  A round that
        # starts always contains exactly N microscopic opportunities.
        if (
            rules.stop_on_consensus
            and turn % rules.n_agents == 0
            and len(set(votes)) == 1
        ):
            return "consensus_reached"
        return None

    # ---- one-call public ballot -----------------------------------------

    def public_ballot_request(
        self,
        state: ImitationGameState,
        focal: AgentId,
        social_sources: Sequence[Mapping[str, Any]],
        config: GameConfig,
        *,
        stage: str = FOCAL_UPDATE,
        interaction_id: InteractionId | None = None,
    ) -> DecisionRequest:
        """The single provider call behind one focal update."""

        rules = self.rules(config)
        agent = state.hidden_bench_agent(focal)
        resolved_id = interaction_id or InteractionId(f"interaction-{state.turn + 1:04d}")
        observation = Observation(
            agent_id=focal,
            interaction_id=resolved_id,
            participants=(focal,),
            visible_state={
                "scenario": state.task["description"],
                "presented_information": list(agent.presented_information),
                "possible_answers": list(state.possible_answers),
                "current_vote": agent.committed_action,
                "current_reason": get_public_reason(agent),
                "social_sources": [dict(item) for item in social_sources],
            },
        )
        return DecisionRequest(
            agent_id=focal,
            interaction_id=resolved_id,
            stage=stage,
            observation=observation,
            prompt=build_public_ballot_update_prompt(
                identity=agent_label(focal),
                scenario=str(state.task["description"]),
                possible_answers=state.possible_answers,
                private_information=agent.presented_information,
                current_vote=agent.committed_action,
                current_reason=get_public_reason(agent),
                social_sources=social_sources,
                vote_visibility=rules.vote_visibility,
            ),
            retry_bound=rules.invalid_response_retries,
        )

    def initial_vote_requests(
        self, state: ImitationGameState, config: GameConfig
    ) -> tuple[DecisionRequest, ...]:
        """`local_vote` initialization on the same `{vote, reason}` schema."""

        interaction_id = InteractionId("initial-local-votes")
        return tuple(
            self.public_ballot_request(
                state,
                agent.agent_id,
                (),
                config,
                stage=INITIAL_VOTE,
                interaction_id=interaction_id,
            )
            for agent in state.agents
        )

    def parse_action(self, request: DecisionRequest, response: str) -> Action:
        if request.stage not in PUBLIC_BALLOT_STAGES:
            return super().parse_action(request, response)
        answers = tuple(
            str(item) for item in request.observation.visible_state["possible_answers"]
        )
        ballot = parse_public_ballot_update(response, answers)
        return Action(
            request.agent_id,
            ballot.vote
            if ballot.vote is not None
            else (str(ballot.raw_vote) if ballot.raw_vote is not None else "<unparsed>"),
            request.stage,
            {
                "kind": "public_ballot",
                "raw_vote": ballot.raw_vote,
                "reason": ballot.reason,
                # The inherited transition persists `rationale`, and there is
                # only one reason in this game, so the two names are the same
                # string rather than two independently drifting fields.
                "rationale": ballot.reason,
                "resolved": ballot.vote is not None,
            },
        )

    def validate_action(
        self,
        state: ImitationGameState,
        request: DecisionRequest,
        action: Action,
        config: GameConfig,
    ) -> ValidationResult:
        if request.stage not in PUBLIC_BALLOT_STAGES:
            return super().validate_action(state, request, action, config)
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
        return ValidationResult(tuple(issues))

    def apply_initial_votes(
        self, state: ImitationGameState, actions: tuple[Action, ...]
    ) -> ImitationGameState:
        """The inherited vote application, plus the first public reason."""

        applied = super().apply_initial_votes(state, actions)
        reasons = {action.agent_id: action.metadata.get("reason") for action in actions}
        return ImitationGameState(
            game_type=applied.game_type,
            turn=applied.turn,
            agents=tuple(
                replace(
                    agent,
                    attributes=set_public_reason(
                        agent.attributes, reasons.get(agent.agent_id)
                    ),
                )
                for agent in applied.agents
            ),
            terminated=applied.terminated,
            data=dict(applied.data),
        )

    def apply_round_event_transition(
        self,
        state: ImitationGameState,
        *,
        round_fields: Mapping[str, Any],
        **transition_fields: Any,
    ) -> ImitationTransition:
        """Apply the inherited focal transition and enrich its persisted row."""

        transition = super().apply_event_transition(state, **transition_fields)
        event = {**dict(transition.event or {}), **dict(round_fields)}
        next_state = self._publish_focal_ballot(
            transition.next_state,
            focal=transition_fields["focal"],
            action=transition_fields["action"],
            social_sources=round_fields.get("social_sources", ()),
            hidden_information=state.hidden_information,
        )
        event_history = list(next_state.data.get("event_history", ()))
        if event_history:
            event_history[-1] = event
        evaluator_history = list(next_state.data.get("evaluator_history", ()))
        if evaluator_history:
            evaluator_history[-1] = {**dict(evaluator_history[-1]), **dict(round_fields)}
        enriched_state = ImitationGameState(
            game_type=next_state.game_type,
            turn=next_state.turn,
            agents=next_state.agents,
            terminated=next_state.terminated,
            data={
                **dict(next_state.data),
                "event_history": event_history,
                "evaluator_history": evaluator_history,
            },
        )
        return replace(transition, next_state=enriched_state, event=event)

    @staticmethod
    def _publish_focal_ballot(
        state: ImitationGameState,
        *,
        focal: AgentId,
        action: Action,
        social_sources: Sequence[Mapping[str, Any]],
        hidden_information: Sequence[str],
    ) -> ImitationGameState:
        """Commit `R_i` next to the already-committed `X_i`.

        Also credits the focal with the hidden facts it just read: the public
        reasons of its social sources are the only channel information travels
        on now, so `known_facts` - and therefore `disclosure_reach` - has to be
        driven by what the focal actually saw in this prompt.
        """

        reason = action.metadata.get("reason")
        if reason is None:
            return state
        read = [
            str(item["reason"]) for item in social_sources if item.get("reason")
        ]
        heard = disclosed_facts(hidden_information, read) if read else ()
        agents = []
        for agent in state.agents:
            if agent.agent_id != focal:
                agents.append(agent)
                continue
            attributes = set_public_reason(agent.attributes, str(reason))
            attributes["known_facts"] = sorted(
                set(attributes.get("known_facts", ()))
                | {index for index, flag in enumerate(heard) if flag}
            )
            agents.append(replace(agent, attributes=attributes))
        return ImitationGameState(
            game_type=state.game_type,
            turn=state.turn,
            agents=tuple(agents),
            terminated=state.terminated,
            data=dict(state.data),
        )

    # ---- provider demand ------------------------------------------------

    def call_plan(self, config: GameConfig) -> GameCallPlan:
        """One provider call per focal update, and no peer-dialogue stage.

        The parent plan prices `2 q m` message calls per update, which this
        game does not make.  Pricing the old shape here would inflate every
        preflight by an order of magnitude and reserve budget for calls that
        never happen.
        """

        rules = self.rules(config)
        options = ("Option A", "Option B", "Option C")
        information = ("An illustrative shared fact.", "An illustrative private fact.")
        initial = PromptScenario(
            "local_initial_ballot",
            build_public_ballot_update_prompt(
                identity="Agent 1",
                scenario="A representative HiddenBench scenario.",
                possible_answers=options,
                private_information=information,
                current_vote=None,
                current_reason=None,
            ),
        )
        update = PromptScenario(
            "public_ballot_update",
            build_public_ballot_update_prompt(
                identity="Agent 1",
                scenario="A representative HiddenBench scenario.",
                possible_answers=options,
                private_information=information,
                current_vote=options[0],
                current_reason="A representative previous public reason.",
                social_sources=tuple(
                    {
                        "label": f"Agent {slot + 2}",
                        "vote": options[slot % len(options)],
                        "reason": "A representative public reason.",
                    }
                    for slot in range(rules.social_group_size)
                ),
                vote_visibility=rules.vote_visibility,
            ),
        )
        if rules.dynamics_mode == "classical":
            return GameCallPlan(
                game_type=self.spec.game_type,
                game_version=self.spec.version,
                interactions=InteractionCount(1, 1, 1, fixed=1),
                decision_stages=(
                    DecisionStagePlan(
                        name="classical_provider_free_jump",
                        requests_per_interaction=0,
                        provider_free_decisions_per_interaction=rules.horizon,
                        assumptions=("All initialization and jumps are provider-free.",),
                    ),
                ),
                stopping_condition_assumptions=(
                    f"At most {rules.horizon} elementary focal-agent steps.",
                ),
                metadata=self._plan_metadata(rules),
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
                    name="public_ballot_update",
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
            metadata=self._plan_metadata(rules),
        )

    @staticmethod
    def _plan_metadata(rules: RoundFeedbackRules) -> dict[str, Any]:
        return {
            "population_size": rules.n_agents,
            "dynamics_mode": rules.dynamics_mode,
            "interactions_per_episode": rules.horizon,
            "population_rounds": rules.rounds,
            "social_group_size": rules.social_group_size,
            "vote_visibility": rules.vote_visibility,
            "initial_votes_provider_supplied": rules.initial_votes is not None,
            "embedded_jump_chain": rules.dynamics_mode == "classical",
        }


__all__ = ["PUBLIC_BALLOT_STAGES", "HiddenBenchImitationRoundFeedbackGame"]
