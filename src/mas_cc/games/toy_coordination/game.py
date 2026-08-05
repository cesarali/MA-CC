"""Minimal deterministic two-action coordination game."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

from mas_cc.config import GameConfig
from mas_cc.core import AgentId, InteractionId
from mas_cc.llm_runtime.validation import ValidationIssue, ValidationResult
from mas_cc.planning import (
    DecisionStagePlan,
    GameCallPlan,
    InteractionCount,
    PromptScenario,
)

from .prompts import ToyCoordinationFullPrompt, bind_toy_prompt

from ..protocols import (
    Action,
    AgentState,
    DecisionRequest,
    Game,
    GameSpec,
    GameState,
    Observation,
    Transition,
)


class ToyCoordinationGame(Game):
    """A finite-horizon pairwise matching game used as an architecture fixture."""

    spec = GameSpec(
        game_type="toy_coordination",
        version=1,
        description="Pairwise agents choose A or B and receive one point when they match.",
        game_family="choice",
        minimum_population=2,
        supported_topologies=("complete",),
    )

    @staticmethod
    def _actions(config: GameConfig) -> tuple[str, ...]:
        raw = config.options.get("actions", ("A", "B"))
        if isinstance(raw, (str, bytes)):
            raise ValueError("game.options.actions must be a list containing A and B")
        actions = tuple(str(item) for item in raw)
        if actions != ("A", "B"):
            raise ValueError("toy_coordination supports exactly the ordered actions A and B")
        return actions

    def _validate_config(self, config: GameConfig) -> None:
        if config.type != self.spec.game_type:
            raise ValueError(f"expected game type {self.spec.game_type!r}")
        if config.population_size < self.spec.minimum_population:
            raise ValueError("toy_coordination requires at least two agents")
        if config.topology not in self.spec.supported_topologies:
            raise ValueError("toy_coordination currently supports only complete topology")
        self._actions(config)
        retry_bound = config.options.get("decision_retry_bound", 0)
        if isinstance(retry_bound, bool) or not isinstance(retry_bound, int) or retry_bound < 0:
            raise ValueError("game.options.decision_retry_bound must be a non-negative integer")

    def initialize(self, config: GameConfig, seed: int) -> GameState:
        self._validate_config(config)
        agents = tuple(
            AgentState(AgentId(f"agent-{index:03d}"), attributes={"available_actions": ["A", "B"]})
            for index in range(config.population_size)
        )
        return GameState(
            game_type=self.spec.game_type,
            turn=0,
            agents=agents,
            data={
                "seed": seed,
                "horizon": config.horizon,
                "matches": 0,
                "topology": config.topology,
            },
        )

    def select_participants(
        self, state: GameState, config: GameConfig, rng: random.Random
    ) -> tuple[AgentId, ...]:
        if state.terminated:
            raise ValueError("cannot select participants after termination")
        selected = rng.sample([agent.agent_id for agent in state.agents], k=2)
        return tuple(selected)

    def construct_observations(
        self, state: GameState, participants: tuple[AgentId, ...], config: GameConfig
    ) -> tuple[Observation, ...]:
        interaction_id = InteractionId(f"interaction-{state.turn + 1:04d}")
        return tuple(
            Observation(
                agent_id=agent_id,
                interaction_id=interaction_id,
                participants=participants,
                visible_state={
                    "interaction_number": state.turn + 1,
                    "horizon": config.horizon,
                    "counterpart_ids": [
                        str(participant) for participant in participants if participant != agent_id
                    ],
                    "counterpart_actions_visible": False,
                },
            )
            for agent_id in participants
        )

    def _bound_prompt(
        self, state: GameState, observation: Observation, config: GameConfig
    ) -> ToyCoordinationFullPrompt:
        agent = state.agent(observation.agent_id)
        return bind_toy_prompt(
            horizon=config.horizon,
            agent_id=str(agent.agent_id),
            score=agent.score,
            memory=agent.memory,
            interaction=observation.visible_state,
        )

    def build_decision_requests(
        self, state: GameState, observations: tuple[Observation, ...], config: GameConfig
    ) -> tuple[DecisionRequest, ...]:
        retry_bound = int(config.options.get("decision_retry_bound", 0))
        return tuple(
            DecisionRequest(
                agent_id=observation.agent_id,
                interaction_id=observation.interaction_id,
                stage="simultaneous_choice",
                observation=observation,
                prompt=self._bound_prompt(state, observation, config),
                provider_required=True,
                retry_bound=retry_bound,
            )
            for observation in observations
        )

    def parse_action(self, request: DecisionRequest, response: str) -> Action:
        return Action(
            agent_id=request.agent_id,
            value=response.strip(),
            stage=request.stage,
            metadata={"interaction_id": str(request.interaction_id)},
        )

    def validate_action(
        self, state: GameState, request: DecisionRequest, action: Action, config: GameConfig
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []
        if action.agent_id != request.agent_id:
            issues.append(ValidationIssue("action.agent_id", "must match the decision agent"))
        if action.stage != request.stage:
            issues.append(ValidationIssue("action.stage", "must match the decision stage"))
        if action.value not in self._actions(config):
            issues.append(
                ValidationIssue("action.value", "must be exactly A or B", action.value)
            )
        if request.agent_id not in tuple(agent.agent_id for agent in state.agents):
            issues.append(ValidationIssue("action.agent_id", "is not present in game state"))
        return ValidationResult(tuple(issues))

    def apply_transition(
        self,
        state: GameState,
        participants: tuple[AgentId, ...],
        actions: tuple[Action, ...],
        config: GameConfig,
    ) -> Transition:
        if len(participants) != 2 or len(actions) != 2:
            raise ValueError("toy_coordination transitions require exactly two participants/actions")
        if tuple(action.agent_id for action in actions) != participants:
            raise ValueError("actions must follow participant order")
        allowed = set(self._actions(config))
        if any(action.value not in allowed for action in actions):
            raise ValueError("cannot apply an invalid toy_coordination action")

        matched = actions[0].value == actions[1].value
        payoff = 1.0 if matched else 0.0
        action_by_agent = {action.agent_id: action.value for action in actions}
        updated_agents: list[AgentState] = []
        for agent in state.agents:
            if agent.agent_id not in participants:
                updated_agents.append(agent)
                continue
            other = next(item for item in participants if item != agent.agent_id)
            memory = (
                *agent.memory,
                {
                    "interaction_number": state.turn + 1,
                    "own_action": action_by_agent[agent.agent_id],
                    "counterpart_action": action_by_agent[other],
                    "payoff": payoff,
                },
            )
            updated_agents.append(replace(agent, score=agent.score + payoff, memory=memory))

        next_turn = state.turn + 1
        terminated = next_turn >= config.horizon
        reason = "finite_horizon_reached" if terminated else None
        next_state = GameState(
            game_type=state.game_type,
            turn=next_turn,
            agents=tuple(updated_agents),
            terminated=terminated,
            data={
                **dict(state.data),
                "matches": int(state.data.get("matches", 0)) + int(matched),
            },
        )
        return Transition(
            interaction_id=InteractionId(f"interaction-{next_turn:04d}"),
            actions=actions,
            payoffs={str(agent_id): payoff for agent_id in participants},
            next_state=next_state,
            matched=matched,
            termination_reason=reason,
        )

    def detect_termination(self, state: GameState, config: GameConfig) -> str | None:
        if state.terminated or state.turn >= config.horizon:
            return "finite_horizon_reached"
        return None

    def call_plan(self, config: GameConfig) -> GameCallPlan:
        self._validate_config(config)
        representative_state = self.initialize(config, seed=0)
        representative_observation = Observation(
            AgentId("agent-000"),
            InteractionId("interaction-0001"),
            (AgentId("agent-000"), AgentId("agent-001")),
            {
                "interaction_number": 1,
                "horizon": config.horizon,
                "counterpart_ids": ["agent-001"],
                "counterpart_actions_visible": False,
            },
        )
        representative = self._bound_prompt(
            representative_state, representative_observation, config
        )

        maximum_memory = tuple(
            {
                "interaction_number": index,
                "own_action": "A",
                "counterpart_action": "B",
                "payoff": 0.0,
            }
            for index in range(1, config.horizon)
        )
        maximum_agent = replace(
            representative_state.agent(AgentId("agent-000")),
            score=float(config.horizon - 1),
            memory=maximum_memory,
        )
        maximum_state = replace(
            representative_state,
            turn=config.horizon - 1,
            agents=(maximum_agent, *representative_state.agents[1:]),
        )
        maximum_observation = replace(
            representative_observation,
            interaction_id=InteractionId(f"interaction-{config.horizon:04d}"),
            visible_state={
                "interaction_number": config.horizon,
                "horizon": config.horizon,
                "counterpart_ids": ["agent-001"],
                "counterpart_actions_visible": False,
            },
        )
        maximum = self._bound_prompt(maximum_state, maximum_observation, config)
        retry_bound = int(config.options.get("decision_retry_bound", 0))
        return GameCallPlan(
            game_type=self.spec.game_type,
            game_version=self.spec.version,
            interactions=InteractionCount(
                fixed=config.horizon,
                lower=config.horizon,
                expected=config.horizon,
                maximum=config.horizon,
            ),
            decision_stages=(
                DecisionStagePlan(
                    name="simultaneous_choice",
                    requests_per_interaction=2,
                    retry_bound=retry_bound,
                    lower_prompt=PromptScenario(
                        "first_interaction",
                        representative,
                        ("No interaction memory has accumulated.",),
                    ),
                    representative_prompt=PromptScenario(
                        "first_interaction",
                        representative,
                        ("No interaction memory has accumulated.",),
                    ),
                    maximum_prompt=PromptScenario(
                        "final_interaction_full_memory",
                        maximum,
                        (
                            "Every prior interaction is retained in bounded game memory.",
                            "All memory entries use the longest action/payoff fixture used by this game.",
                        ),
                    ),
                    prompt_scenarios=(
                        PromptScenario("first_interaction", representative),
                        PromptScenario("final_interaction_full_memory", maximum),
                    ),
                    assumptions=("Both selected agents require one independent decision.",),
                ),
            ),
            stopping_condition_assumptions=(
                f"The run stops after exactly {config.horizon} interactions.",
                "Each interaction selects exactly two distinct agents.",
            ),
            metadata={
                "population_size": config.population_size,
                "topology": config.topology,
                "provider_prices_included": False,
            },
        )
