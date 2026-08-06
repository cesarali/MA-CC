"""`hidden_bench_vanilla` - the paper's protocol, reproduced (brief §5).

Li, Naito & Shirado, *Systematic Failures in Collective Reasoning under
Distributed Information in Multi-Agent LLMs* (ICML 2026). Implements §1.1's
formal model, §1.2's two profile conditions, and §1.5's prompts.

**Phase model.** `PRE_VOTE -> DISCUSS x (T x N) -> POST_VOTE`.

**What `T` counts.** `rounds: T` is `T` *full round-robin passes*, so the
discussion contains `T x N` speaking turns. The paper's own runner instead
treats one round as one speaking event; the brief (§5) asks for passes, and
picks the unit deliberately because it is the one that keeps "how much airtime
did each agent get" constant as `N` varies - which is exactly what a group-size
ablation needs. The two units coincide only at `N = 1`, so a reproduction that
wants the paper's `T in {5,10,15,20}` literally should set
`rounds_are_speaking_turns: true`, which switches the unit back and is what
`configs/runs/hidden_bench_vanilla.yaml` does.

**Full Profile is a config flag, not a game.** `profile: full` with `rounds: 0`
runs `PRE_VOTE -> POST_VOTE` with every agent holding all of `Iu`; its pre-vote
accuracy is `Y_full`. Keeping it here rather than in a second game is what makes
`Y_full` and `Y_post` comparable - same prompts, same parser, same seeds.

**`horizon` is not used by this game.** The interaction count is fully
determined by `rounds`, `population_size` and `profile`; `call_plan` reports the
derived number and that is what pricing and budget preflight consume. The
configs set `horizon` to the derived total for readability only.
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
    DISCUSS,
    POST_VOTE,
    PRE_VOTE,
    HiddenBenchAgentState,
    HiddenBenchGameState,
    HiddenBenchTransition,
)
from ..rules import VanillaRules
from .prompts import (
    bind_discussion_prompt,
    bind_vote_prompt,
    extract_json_object,
    normalize_vote,
    shuffled_information,
)


class HiddenBenchVanillaGame(Game):
    """N agents, round-robin plenary discussion, a vote before and after."""

    spec = GameSpec(
        game_type="hidden_bench_vanilla",
        version=1,
        description=(
            "HiddenBench Hidden Profile task with round-robin group discussion and "
            "pre/post-discussion votes (Li, Naito & Shirado, ICML 2026)."
        ),
        # The action alphabet *is* the option set, so the whole choice-family
        # metric shelf in metrics/generic.py applies without adaptation.
        game_family="choice",
        minimum_population=2,
        supported_topologies=("complete",),
    )

    def rules(self, config: GameConfig) -> VanillaRules:
        if config.type != self.spec.game_type:
            raise ValueError(f"expected game type {self.spec.game_type!r}, got {config.type!r}")
        return VanillaRules.from_config(config)

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
            # One shuffle per agent, fixed for the whole episode: an agent's
            # facts must not silently reorder between its pre-vote and its
            # speaking turn, or the two measurements are of different prompts.
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
                    },
                )
            )

        # Fixed per episode and recorded: who speaks in which slot is a
        # confound if it is redrawn, and unanalysable if it is not written down.
        speaking_order = [str(agent.agent_id) for agent in agents]
        root.derive("speaking-order").create_random().shuffle(speaking_order)

        return HiddenBenchGameState(
            game_type=self.spec.game_type,
            turn=0,
            agents=tuple(agents),
            terminated=False,
            data={
                "seed": seed,
                "phase": PRE_VOTE,
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
                "speaking_order": speaking_order,
                "speaker_index": 0,
                "discussion_round": 0,
                "transcript": [],
                "evaluator_history": [],
                "termination_reason": None,
            },
        )

    # ---- participants and observations -----------------------------------

    def select_participants(
        self, state: HiddenBenchGameState, config: GameConfig, rng: random.Random
    ) -> tuple[AgentId, ...]:
        if state.terminated:
            raise ValueError("cannot select participants after termination")
        if state.phase in {PRE_VOTE, POST_VOTE}:
            return tuple(agent.agent_id for agent in state.agents)
        order = state.data["speaking_order"]
        return (AgentId(str(order[int(state.data["speaker_index"])])),)

    def construct_observations(
        self,
        state: HiddenBenchGameState,
        participants: tuple[AgentId, ...],
        config: GameConfig,
    ) -> tuple[Observation, ...]:
        """The information boundary of the whole benchmark lives here.

        An observation carries the focal agent's *own* facts plus the public
        transcript, and nothing else. No other agent's `private_information`
        reaches an observation by any route except that agent having said it
        out loud, which is the entire experimental manipulation.
        """

        rules = self.rules(config)
        interaction_id = InteractionId(f"interaction-{state.turn + 1:04d}")
        transcript = [
            {"speaker_id": item["speaker_id"], "message": item["message"]}
            for item in state.transcript
        ]
        observations = []
        for agent_id in participants:
            agent = state.hidden_bench_agent(agent_id)
            visible: dict[str, Any] = {
                "phase": state.phase,
                "scenario": state.task["description"],
                "presented_information": list(agent.presented_information),
                "possible_answers": list(state.possible_answers),
                "extra_prompt": rules.extra_for(agent_id, state),
            }
            if state.phase == DISCUSS:
                visible["transcript"] = transcript
                visible["is_first_speaker"] = not transcript
            elif state.phase == POST_VOTE:
                visible["transcript"] = transcript
            observations.append(
                Observation(
                    agent_id=agent_id,
                    interaction_id=interaction_id,
                    participants=participants,
                    visible_state=visible,
                )
            )
        return tuple(observations)

    # ---- prompts ---------------------------------------------------------

    @staticmethod
    def _bound_prompt(observation: Observation) -> Any:
        visible = observation.visible_state
        phase = str(visible["phase"])
        information = tuple(str(item) for item in visible["presented_information"])
        if phase == DISCUSS:
            return bind_discussion_prompt(
                scenario=str(visible["scenario"]),
                information=information,
                extra=visible.get("extra_prompt"),
                transcript=tuple(visible.get("transcript", ())),
                is_first_speaker=bool(visible["is_first_speaker"]),
            )
        return bind_vote_prompt(
            scenario=str(visible["scenario"]),
            information=information,
            possible_answers=tuple(str(item) for item in visible["possible_answers"]),
            transcript=tuple(visible.get("transcript", ())) or None,
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
        if request.stage == DISCUSS:
            text = response.strip()
            if not text:
                # Action.value must be non-empty; an empty turn is a validation
                # failure, not a crash, so it reaches validate_action as a
                # sentinel and is retried through the normal path.
                text = "<empty>"
            return Action(request.agent_id, text, request.stage, {"kind": "message"})
        answers = tuple(str(item) for item in request.observation.visible_state["possible_answers"])
        parsed = extract_json_object(response)
        raw_vote = parsed.get("vote") if parsed else None
        rationale = parsed.get("rationale") if parsed else None
        vote = normalize_vote(raw_vote, answers)
        return Action(
            request.agent_id,
            # Keep the raw string when it did not resolve: validate_action
            # rejects it, and a rejected vote must never be silently recorded
            # as a wrong answer (brief §9.5).
            vote if vote is not None else (str(raw_vote) if raw_vote is not None else "<unparsed>"),
            request.stage,
            {
                "kind": "vote",
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
        if request.stage == DISCUSS:
            if action.value == "<empty>":
                issues.append(ValidationIssue("action.value", "a discussion turn must not be empty"))
            return ValidationResult(tuple(issues))
        if not action.metadata.get("resolved"):
            issues.append(
                ValidationIssue(
                    "action.value",
                    f"vote must resolve to one of: {', '.join(state.possible_answers)}",
                    action.metadata.get("raw_vote"),
                )
            )
        elif action.value not in state.possible_answers:
            issues.append(ValidationIssue("action.value", "vote is not an option", action.value))
        if action.metadata.get("rationale") is None:
            issues.append(ValidationIssue("action.rationale", "a vote must carry a rationale"))
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
        phase = state.phase
        next_index = state.turn + 1
        data = dict(state.data)
        by_agent = {action.agent_id: action for action in actions}

        if phase == DISCUSS:
            action = actions[0]
            order = list(data["speaking_order"])
            speaker_index = int(data["speaker_index"])
            data["transcript"] = [
                *state.transcript,
                {
                    # `speaker_id` is the *slot*, not the agent id: it is what
                    # the paper's transcript lines show, and it keeps the
                    # transcript free of any agent-identity leakage.
                    "speaker_id": speaker_index,
                    "agent_id": str(action.agent_id),
                    "message": action.value,
                    "round": int(data["discussion_round"]) + 1,
                },
            ]
            speaker_index += 1
            if speaker_index >= len(order):
                speaker_index = 0
                data["discussion_round"] = int(data["discussion_round"]) + 1
            data["speaker_index"] = speaker_index
            if rules.discussion_turns_complete(
                int(data["discussion_round"]), speaker_index, len(order)
            ):
                data["phase"] = POST_VOTE
            payoffs = {str(action.agent_id): 0.0}
            matched = None
            agents = state.agents
        else:
            key = "pre_vote" if phase == PRE_VOTE else "post_vote"
            correct = state.correct_answer
            agents_list = []
            for agent in state.agents:
                action = by_agent.get(agent.agent_id)
                if action is None:
                    agents_list.append(agent)
                    continue
                attributes = dict(agent.attributes)
                attributes[key] = action.value
                attributes["committed_action"] = action.value
                attributes["rationales"] = [
                    *agent.rationales,
                    {
                        "phase": phase,
                        "vote": action.value,
                        "rationale": action.metadata.get("rationale"),
                    },
                ]
                agents_list.append(
                    replace(
                        agent,
                        score=agent.score + (1.0 if action.value == correct else 0.0),
                        attributes=attributes,
                    )
                )
            agents = tuple(agents_list)
            payoffs = {
                str(action.agent_id): 1.0 if action.value == correct else 0.0 for action in actions
            }
            # `matched` is the majority-rule outcome (§1.3); the average rule is
            # recoverable from the per-agent payoffs, so nothing is lost by
            # reporting the stricter one here.
            matched = sum(payoffs.values()) / len(payoffs) > 0.5
            if phase == PRE_VOTE:
                # Under `profile: full, rounds: 0` this jumps straight to the
                # post-vote, which is how Y_full is measured with no discussion.
                data["phase"] = DISCUSS if rules.has_discussion else POST_VOTE

        terminated = phase == POST_VOTE
        reason = "post_vote_recorded" if terminated else None
        data["termination_reason"] = reason
        data["evaluator_history"] = [
            *state.evaluator_history,
            self._round_summary(state, phase, data, next_index),
        ]

        next_state = HiddenBenchGameState(
            game_type=state.game_type,
            turn=next_index,
            agents=agents,
            terminated=terminated,
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

    def _round_summary(
        self, state: HiddenBenchGameState, phase: str, data: Mapping[str, Any], round_index: int
    ) -> dict[str, Any]:
        """One dict per completed interaction, carried in `RoundView.recent_history`.

        This is how episode-invariant facts (`correct_answer`, `decoy`) and
        phase reach metrics that only ever see a `RoundView`. Computing the
        disclosure rate here rather than in the metric keeps it out of the
        per-metric hot path and means the number is recorded even when metrics
        are disabled.
        """

        messages = [str(item["message"]) for item in data.get("transcript", ())]
        disclosed = disclosed_facts(state.hidden_information, messages)
        return {
            "round_index": round_index,
            "phase": phase,
            "correct_answer": state.correct_answer,
            "decoy": state.task.get("decoy"),
            "possible_answers": list(state.possible_answers),
            "transcript_length": len(messages),
            "disclosed_hidden_facts": list(disclosed),
            "unshared_disclosure_rate": (sum(disclosed) / len(disclosed)) if disclosed else 0.0,
        }

    def detect_termination(self, state: HiddenBenchGameState, config: GameConfig) -> str | None:
        self.rules(config)
        if state.terminated:
            return state.termination_reason or "post_vote_recorded"
        return None

    # ---- planning --------------------------------------------------------

    def call_plan(self, config: GameConfig) -> GameCallPlan:
        """Exactly `N + N*T + N` provider calls per episode, before retries (§5)."""

        rules = self.rules(config)
        n = rules.n_agents
        turns = rules.total_discussion_turns
        total_interactions = 1 + turns + 1

        scenario_information = ("An illustrative shared fact.", "An illustrative private fact.")
        vote = PromptScenario(
            "vote",
            bind_vote_prompt(
                scenario="A representative HiddenBench scenario description.",
                information=scenario_information,
                possible_answers=("Option A", "Option B", "Option C"),
            ),
            ("Pre-discussion vote: no transcript is attached.",),
        )
        discussion = PromptScenario(
            "discussion_turn",
            bind_discussion_prompt(
                scenario="A representative HiddenBench scenario description.",
                information=scenario_information,
                extra=rules.extra_prompt,
                transcript=(),
                is_first_speaker=True,
            ),
            ("First speaker: the transcript is empty, so this is the lower bound.",),
        )
        # `InteractionCount` is 1 here, and each stage's
        # `requests_per_interaction` is its whole-episode total. That is
        # deliberate: `GameCallPlan.provider_requests` computes
        # `interactions x sum(requests_per_interaction)`, which assumes every
        # stage runs in every interaction - true for a pair game, false for a
        # phase-structured one where the pre-vote happens once and the
        # discussion happens `turns` times. Pricing the *episode* as the unit
        # is the only framing that makes the product exact, and
        # `plan.interactions` is not read anywhere except that multiplication.
        # The true round count is in `metadata.interactions_per_episode`.
        stages = (
            DecisionStagePlan(
                name="pre_vote",
                requests_per_interaction=n,
                retry_bound=rules.invalid_response_retries,
                expected_attempts_per_request=rules.expected_attempts_per_request,
                concurrency_within_stage=n,
                state_barrier_after_stage=True,
                lower_prompt=vote,
                representative_prompt=vote,
                maximum_prompt=vote,
                prompt_scenarios=(vote,),
                assumptions=("All N agents vote once, concurrently, from one frozen state.",),
            ),
            DecisionStagePlan(
                name="discussion",
                requests_per_interaction=turns,
                retry_bound=rules.invalid_response_retries,
                expected_attempts_per_request=rules.expected_attempts_per_request,
                concurrency_within_stage=1,
                state_barrier_after_stage=True,
                lower_prompt=discussion,
                representative_prompt=discussion,
                maximum_prompt=discussion,
                prompt_scenarios=(discussion,),
                assumptions=(
                    f"{turns} sequential speaking turns "
                    f"({'T counted as speaking turns' if rules.rounds_are_speaking_turns else f'T={rules.rounds} full round-robin passes x N={n}'}).",
                    "Each turn's prompt grows with the transcript; this scenario is the empty-transcript lower bound.",
                ),
            ),
            DecisionStagePlan(
                name="post_vote",
                requests_per_interaction=n,
                retry_bound=rules.invalid_response_retries,
                expected_attempts_per_request=rules.expected_attempts_per_request,
                concurrency_within_stage=n,
                state_barrier_after_stage=True,
                lower_prompt=vote,
                representative_prompt=vote,
                maximum_prompt=vote,
                prompt_scenarios=(vote,),
                assumptions=("All N agents vote once more with the full transcript attached.",),
            ),
        )
        return GameCallPlan(
            game_type=self.spec.game_type,
            game_version=self.spec.version,
            interactions=InteractionCount(lower=1, expected=1, maximum=1, fixed=1),
            decision_stages=stages,
            stopping_condition_assumptions=(
                "Fixed structure: one pre-vote step, "
                f"{turns} discussion steps, one post-vote step.",
                "There is no early stopping; the post-vote always runs.",
                "The planning unit is one episode; stage counts are per-episode totals.",
            ),
            metadata={
                "population_size": n,
                "interactions_per_episode": total_interactions,
                "profile": rules.profile,
                "assignment_scheme": rules.assignment_scheme,
                "rounds": rules.rounds,
                "discussion_turns": turns,
                "provider_calls_per_episode": n + turns + n,
                "validation_retry_bound_per_request": rules.invalid_response_retries,
                "provider_prices_included": False,
            },
        )
