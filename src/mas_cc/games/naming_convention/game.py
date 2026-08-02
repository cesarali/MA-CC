"""Ashery–Aiello–Baronchelli repeated naming-convention dynamics."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any, Mapping

from mas_cc.config import GameConfig
from mas_cc.core import AgentId, InteractionId, Seed, ValidationIssue, ValidationResult
from mas_cc.games.protocols import Action, GameSpec, Observation
from mas_cc.planning import (
    DecisionStagePlan,
    GameCallPlan,
    InteractionCount,
    PromptScenario,
)

from .parsing import parse_convention_response
from .prompts import NamingConventionFullPrompt, bind_naming_convention_prompt
from .records import (
    ConventionAgentState,
    ConventionDecisionRequest,
    ConventionGameState,
    ConventionTransition,
    PrivateMemoryEntry,
)


@dataclass(frozen=True, slots=True)
class NamingConventionGameSpec:
    population_size: int
    actions: tuple[str, ...]
    memory_size: int
    success_payoff: int
    failure_payoff: int
    max_interactions: int
    topology: str
    pair_sampling: str
    simultaneous_pair_decisions: bool
    randomize_presented_action_order: bool
    prompt_contract: str
    response_contract: str
    parser_contract: str
    invalid_response_retries: int
    expected_validation_failure_rate: float
    stop_on_convergence: bool

    @classmethod
    def from_config(cls, config: GameConfig) -> "NamingConventionGameSpec":
        options = config.options
        raw_actions = options.get("actions", ("Q", "M"))
        if isinstance(raw_actions, (str, bytes)):
            raise ValueError("game.options.actions must be a list of unique labels")
        actions = tuple(str(item) for item in raw_actions)
        result = cls(
            population_size=config.population_size,
            actions=actions,
            memory_size=_integer(options, "memory_size", 5, minimum=0),
            success_payoff=_integer(options, "success_payoff", 100),
            failure_payoff=_integer(options, "failure_payoff", -50),
            max_interactions=config.horizon,
            topology=config.topology,
            pair_sampling=str(options.get("pair_sampling", "uniform_two_distinct")),
            simultaneous_pair_decisions=_boolean(
                options, "simultaneous_pair_decisions", True
            ),
            randomize_presented_action_order=_boolean(
                options, "randomize_presented_action_order", True
            ),
            prompt_contract=str(
                options.get("prompt_contract", "naming_convention_decision")
            ),
            response_contract=str(options.get("response_contract", "json_reason")),
            parser_contract=str(
                options.get("parser_contract", "tolerant_paper_object_v1")
            ),
            invalid_response_retries=_integer(
                options, "invalid_response_retries", 2, minimum=0
            ),
            expected_validation_failure_rate=float(
                options.get("expected_validation_failure_rate", 0.0)
            ),
            stop_on_convergence=_boolean(options, "stop_on_convergence", False),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("naming_convention requires at least two agents")
        if len(self.actions) < 2 or len(set(self.actions)) != len(self.actions):
            raise ValueError("naming_convention requires at least two unique actions")
        if any(not action.strip() or "\n" in action or "\r" in action for action in self.actions):
            raise ValueError("convention actions must be non-empty single-line strings")
        if self.topology != "complete":
            raise ValueError("the Phase 6 paper profile supports complete topology only")
        if self.pair_sampling != "uniform_two_distinct":
            raise ValueError("pair_sampling must be uniform_two_distinct")
        if not self.simultaneous_pair_decisions:
            raise ValueError("naming_convention requires simultaneous pair decisions")
        if not self.randomize_presented_action_order:
            raise ValueError("the paper-faithful profile requires randomized action order")
        if self.prompt_contract != "naming_convention_decision":
            raise ValueError("prompt_contract must be naming_convention_decision")
        if self.response_contract != "json_reason":
            raise ValueError("the Phase 6 paper-faithful response_contract is json_reason")
        if self.success_payoff != 100 or self.failure_payoff != -50:
            raise ValueError("the Phase 6 paper-faithful payoff profile is +100/-50")
        if self.parser_contract not in {
            "strict_json_reason_v1",
            "tolerant_paper_object_v1",
        }:
            raise ValueError("unsupported naming-convention parser contract")
        if not math.isfinite(self.expected_validation_failure_rate) or not (
            0 <= self.expected_validation_failure_rate <= 1
        ):
            raise ValueError("expected_validation_failure_rate must be between zero and one")
        if self.stop_on_convergence:
            raise ValueError("Phase 6 base profile uses fixed-horizon stopping")

    @property
    def expected_attempts_per_request(self) -> float:
        probability = self.expected_validation_failure_rate
        return sum(probability**attempt for attempt in range(self.invalid_response_retries + 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "population_size": self.population_size,
            "actions": list(self.actions),
            "memory_size": self.memory_size,
            "success_payoff": self.success_payoff,
            "failure_payoff": self.failure_payoff,
            "max_interactions": self.max_interactions,
            "topology": self.topology,
            "pair_sampling": self.pair_sampling,
            "simultaneous_pair_decisions": self.simultaneous_pair_decisions,
            "randomize_presented_action_order": self.randomize_presented_action_order,
            "prompt_contract": self.prompt_contract,
            "response_contract": self.response_contract,
            "parser_contract": self.parser_contract,
            "invalid_response_retries": self.invalid_response_retries,
            "expected_validation_failure_rate": self.expected_validation_failure_rate,
            "stop_on_convergence": self.stop_on_convergence,
        }


def _integer(
    options: Mapping[str, Any], name: str, default: int, minimum: int | None = None
) -> int:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"game.options.{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"game.options.{name} must be at least {minimum}")
    return value


def _boolean(options: Mapping[str, Any], name: str, default: bool) -> bool:
    value = options.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"game.options.{name} must be a boolean")
    return value


class NamingConventionGame:
    """Pure game mechanics and private-view construction; no provider execution."""

    spec = GameSpec(
        game_type="naming_convention",
        version=1,
        description=(
            "Repeated symmetric coordination with private finite memory from Ashery et al. 2025."
        ),
        minimum_population=2,
        supported_topologies=("complete",),
    )

    def rules(self, config: GameConfig) -> NamingConventionGameSpec:
        if config.type != self.spec.game_type:
            raise ValueError(f"expected game type {self.spec.game_type!r}")
        return NamingConventionGameSpec.from_config(config)

    def initialize(self, config: GameConfig, seed: int) -> ConventionGameState:
        rules = self.rules(config)
        agents = tuple(
            ConventionAgentState(
                agent_id=AgentId(f"agent-{index:03d}"),
                score=0,
                memory=(),
                attributes={
                    "available_actions": list(rules.actions),
                    "committed_action": None,
                },
            )
            for index in range(rules.population_size)
        )
        return ConventionGameState(
            game_type=self.spec.game_type,
            turn=0,
            agents=agents,
            terminated=False,
            data={
                "seed": seed,
                "action_pool": list(rules.actions),
                "topology": {"type": "complete"},
                "evaluator_history": [],
                "termination_reason": None,
            },
        )

    def select_participants(
        self, state: ConventionGameState, config: GameConfig, rng: random.Random
    ) -> tuple[AgentId, AgentId]:
        self.rules(config)
        if state.terminated:
            raise ValueError("cannot select participants after termination")
        pair = rng.sample([agent.agent_id for agent in state.agents], k=2)
        return pair[0], pair[1]

    def _presented_actions(
        self, state: ConventionGameState, agent_id: AgentId, rules: NamingConventionGameSpec
    ) -> tuple[str, ...]:
        actions = list(rules.actions)
        seed = Seed(int(state.data["seed"])).derive(
            f"naming-convention-action-order:{state.turn + 1}:{agent_id}"
        )
        seed.create_random().shuffle(actions)
        return tuple(actions)

    def construct_observations(
        self,
        state: ConventionGameState,
        participants: tuple[AgentId, AgentId],
        config: GameConfig,
    ) -> tuple[Observation, Observation]:
        rules = self.rules(config)
        interaction_id = InteractionId(f"interaction-{state.turn + 1:04d}")
        observations: list[Observation] = []
        for agent_id in participants:
            agent = state.convention_agent(agent_id)
            visible = agent.visible_history(rules.memory_size)
            observations.append(
                Observation(
                    agent_id=agent_id,
                    interaction_id=interaction_id,
                    participants=participants,
                    visible_state={
                        "visible_memory": [entry.to_dict() for entry in visible],
                        "visible_score": sum(entry.payoff for entry in visible),
                        "local_round": len(visible) + 1,
                        "presented_actions": list(
                            self._presented_actions(state, agent_id, rules)
                        ),
                        "counterpart_current_action_visible": False,
                    },
                )
            )
        return observations[0], observations[1]

    @staticmethod
    def _bound_prompt(
        observation: Observation, rules: NamingConventionGameSpec
    ) -> NamingConventionFullPrompt:
        visible_memory = tuple(observation.visible_state["visible_memory"])
        return bind_naming_convention_prompt(
            presented_actions=tuple(observation.visible_state["presented_actions"]),
            visible_memory=visible_memory,
            visible_score=int(observation.visible_state["visible_score"]),
            local_round=int(observation.visible_state["local_round"]),
            allowed_actions=rules.actions,
        )

    def build_decision_requests(
        self,
        state: ConventionGameState,
        observations: tuple[Observation, Observation],
        config: GameConfig,
    ) -> tuple[ConventionDecisionRequest, ConventionDecisionRequest]:
        rules = self.rules(config)
        requests = tuple(
            ConventionDecisionRequest(
                agent_id=observation.agent_id,
                interaction_id=observation.interaction_id,
                stage="pair_decision",
                observation=observation,
                prompt=self._bound_prompt(observation, rules),
                provider_required=True,
                retry_bound=rules.invalid_response_retries,
                parser_contract=rules.parser_contract,
            )
            for observation in observations
        )
        return requests[0], requests[1]

    def parse_action(self, request: ConventionDecisionRequest, response: str) -> Action:
        parser_contract = request.parser_contract
        parsed = parse_convention_response(
            response, request.presented_actions, parser_contract
        )
        return Action(
            request.agent_id,
            parsed.value,
            request.stage,
            {
                "parsed_reason": parsed.reason,
                "parser_mode": parsed.parser_mode,
                "presented_actions": list(request.presented_actions),
            },
        )

    def validate_action(
        self,
        state: ConventionGameState,
        request: ConventionDecisionRequest,
        action: Action,
        config: GameConfig,
    ) -> ValidationResult:
        rules = self.rules(config)
        issues: list[ValidationIssue] = []
        if action.agent_id != request.agent_id:
            issues.append(ValidationIssue("action.agent_id", "must match the focal decision"))
        if action.value not in rules.actions:
            issues.append(
                ValidationIssue("action.value", "must be one configured action", action.value)
            )
        if action.value not in request.presented_actions:
            issues.append(
                ValidationIssue("action.value", "must be present in the audited action order")
            )
        return ValidationResult(tuple(issues))

    def apply_transition(
        self,
        state: ConventionGameState,
        participants: tuple[AgentId, AgentId],
        actions: tuple[Action, Action],
        config: GameConfig,
    ) -> ConventionTransition:
        rules = self.rules(config)
        if len(set(participants)) != 2:
            raise ValueError("convention transition requires two distinct agents")
        if tuple(action.agent_id for action in actions) != participants:
            raise ValueError("actions must follow selected-pair order")
        if any(action.value not in rules.actions for action in actions):
            raise ValueError("invalid convention action cannot enter a transition")
        for agent_id in participants:
            state.convention_agent(agent_id)

        success = actions[0].value == actions[1].value
        payoff = rules.success_payoff if success else rules.failure_payoff
        by_agent = {action.agent_id: action.value for action in actions}
        updated: list[ConventionAgentState] = []
        for agent in state.agents:
            if not isinstance(agent, ConventionAgentState):
                raise TypeError("convention state contains incompatible agent state")
            if agent.agent_id not in participants:
                updated.append(agent)
                continue
            other = next(item for item in participants if item != agent.agent_id)
            entry = PrivateMemoryEntry(
                agent_local_interaction_index=len(agent.memory) + 1,
                own_action=by_agent[agent.agent_id],
                partner_action=by_agent[other],
                payoff=payoff,
                success=success,
            )
            updated.append(
                replace(
                    agent,
                    score=agent.lifetime_score + payoff,
                    memory=(*agent.memory, entry.to_dict()),
                )
            )

        next_index = state.turn + 1
        terminated = next_index >= rules.max_interactions
        reason = "fixed_horizon_reached" if terminated else None
        evaluator_summary = {
            "interaction_index": next_index,
            "selected_agents": [str(item) for item in participants],
            "actions": [action.value for action in actions],
            "success": success,
            "payoff": payoff,
        }
        next_state = ConventionGameState(
            game_type=state.game_type,
            turn=next_index,
            agents=tuple(updated),
            terminated=terminated,
            data={
                **dict(state.data),
                "evaluator_history": (*state.evaluator_history, evaluator_summary),
                "termination_reason": reason,
            },
        )
        return ConventionTransition(
            interaction_id=InteractionId(f"interaction-{next_index:04d}"),
            actions=actions,
            payoffs={str(agent_id): payoff for agent_id in participants},
            next_state=next_state,
            matched=success,
            termination_reason=reason,
        )

    def detect_termination(
        self, state: ConventionGameState, config: GameConfig
    ) -> str | None:
        rules = self.rules(config)
        if state.terminated or state.turn >= rules.max_interactions:
            return "fixed_horizon_reached"
        return None

    def _prompt_scenario(
        self, rules: NamingConventionGameSpec, memory_entries: int, name: str
    ) -> PromptScenario:
        history = tuple(
            PrivateMemoryEntry(
                agent_local_interaction_index=index,
                own_action=rules.actions[(index - 1) % len(rules.actions)],
                partner_action=rules.actions[index % len(rules.actions)],
                payoff=rules.failure_payoff,
                success=False,
            ).to_dict()
            for index in range(1, memory_entries + 1)
        )
        observation = Observation(
            agent_id=AgentId("scenario-agent"),
            interaction_id=InteractionId(f"scenario-{name}"),
            participants=(AgentId("scenario-agent"), AgentId("anonymous-partner")),
            visible_state={
                "visible_memory": history,
                "visible_score": sum(int(item["payoff"]) for item in history),
                "local_round": len(history) + 1,
                "presented_actions": list(rules.actions),
                "counterpart_current_action_visible": False,
            },
        )
        return PromptScenario(
            name,
            self._bound_prompt(observation, rules),
            (f"Visible private memory contains exactly {memory_entries} entries.",),
        )

    def call_plan(self, config: GameConfig) -> GameCallPlan:
        rules = self.rules(config)
        representative_size = min(
            rules.memory_size,
            _integer(config.options, "representative_memory_size", 1, minimum=0),
        )
        empty = self._prompt_scenario(rules, 0, "empty_memory")
        representative = self._prompt_scenario(
            rules, representative_size, "representative_memory"
        )
        maximum = self._prompt_scenario(rules, rules.memory_size, "maximum_memory")
        return GameCallPlan(
            game_type=self.spec.game_type,
            game_version=self.spec.version,
            interactions=InteractionCount(
                lower=rules.max_interactions,
                expected=rules.max_interactions,
                maximum=rules.max_interactions,
                fixed=rules.max_interactions,
            ),
            decision_stages=(
                DecisionStagePlan(
                    name="pair_decision",
                    requests_per_interaction=2,
                    forced_decisions_per_interaction=0,
                    provider_free_decisions_per_interaction=0,
                    retry_bound=rules.invalid_response_retries,
                    expected_attempts_per_request=rules.expected_attempts_per_request,
                    concurrency_within_stage=2,
                    state_barrier_after_stage=True,
                    lower_prompt=empty,
                    representative_prompt=representative,
                    maximum_prompt=maximum,
                    prompt_scenarios=(empty, representative, maximum),
                    assumptions=(
                        "Both focal views are frozen from one pre-interaction state.",
                        "Validation retries repeat only the invalid focal decision.",
                        "Provider transport retries are accounted by the provider layer.",
                    ),
                ),
            ),
            stopping_condition_assumptions=(
                f"Fixed horizon of exactly {rules.max_interactions} pair interactions.",
                "Early convergence stopping is disabled in the Phase 6 base profile.",
                "One project population round equals population_size pair interactions.",
            ),
            metadata={
                "population_size": rules.population_size,
                "population_round_project_convention": (
                    f"interaction_index / {rules.population_size}"
                ),
                "participant_count_per_interaction": 2,
                "base_logical_decisions": 2 * rules.max_interactions,
                "forced_decisions": 0,
                "validation_retry_bound_per_request": rules.invalid_response_retries,
                "expected_validation_failure_rate": rules.expected_validation_failure_rate,
                "provider_prices_included": False,
            },
        )
