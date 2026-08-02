"""Generic records and protocol shared by all MAS-CC games."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from mas_cc.config import GameConfig
from mas_cc.core import AgentId, InteractionId, ValidationResult
from mas_cc.llm_providers import CompletionRequest, CompletionResponse
from mas_cc.planning import GameCallPlan
from mas_cc.prompts import CompilablePrompt


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_thaw(item) for item in value)
    return value


def _non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class GameSpec:
    """Stable identity and structural capabilities of a game implementation."""

    game_type: str
    version: int
    description: str
    minimum_population: int = 2
    supported_topologies: tuple[str, ...] = ("complete",)

    def __post_init__(self) -> None:
        _non_empty(self.game_type, "GameSpec.game_type")
        _non_empty(self.description, "GameSpec.description")
        if self.version < 1:
            raise ValueError("GameSpec.version must be positive")
        if self.minimum_population < 2:
            raise ValueError("GameSpec.minimum_population must be at least two")
        topologies = tuple(self.supported_topologies)
        if not topologies or any(
            not isinstance(item, str) or not item.strip() for item in topologies
        ):
            raise ValueError("GameSpec.supported_topologies must be non-empty")
        object.__setattr__(self, "supported_topologies", topologies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_type": self.game_type,
            "version": self.version,
            "description": self.description,
            "minimum_population": self.minimum_population,
            "supported_topologies": list(self.supported_topologies),
        }


@dataclass(frozen=True, slots=True)
class AgentState:
    """Provider-neutral state held by one agent."""

    agent_id: AgentId
    score: float = 0.0
    memory: tuple[Mapping[str, Any], ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, AgentId):
            raise TypeError("AgentState.agent_id must be an AgentId")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("AgentState.score must be numeric")
        if isinstance(self.memory, (str, bytes)) or not isinstance(self.memory, Sequence):
            raise TypeError("AgentState.memory must be a sequence")
        memory = tuple(self.memory)
        if any(not isinstance(item, Mapping) for item in memory):
            raise TypeError("AgentState.memory items must be mappings")
        if not isinstance(self.attributes, Mapping):
            raise TypeError("AgentState.attributes must be a mapping")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "memory", _freeze(memory))
        object.__setattr__(self, "attributes", _freeze(self.attributes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": str(self.agent_id),
            "score": self.score,
            "memory": _thaw(self.memory),
            "attributes": _thaw(self.attributes),
        }


@dataclass(frozen=True, slots=True)
class GameState:
    """Immutable state at one point in a game trajectory."""

    game_type: str
    turn: int
    agents: tuple[AgentState, ...]
    terminated: bool = False
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_empty(self.game_type, "GameState.game_type")
        if isinstance(self.turn, bool) or not isinstance(self.turn, int) or self.turn < 0:
            raise ValueError("GameState.turn cannot be negative")
        agents = tuple(self.agents)
        if len(agents) < 2:
            raise ValueError("GameState.agents must contain at least two agents")
        if len({agent.agent_id for agent in agents}) != len(agents):
            raise ValueError("GameState.agent identifiers must be unique")
        if not isinstance(self.data, Mapping):
            raise TypeError("GameState.data must be a mapping")
        object.__setattr__(self, "agents", agents)
        object.__setattr__(self, "data", _freeze(self.data))

    def agent(self, agent_id: AgentId) -> AgentState:
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        raise KeyError(str(agent_id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_type": self.game_type,
            "turn": self.turn,
            "terminated": self.terminated,
            "agents": [agent.to_dict() for agent in self.agents],
            "data": _thaw(self.data),
        }


@dataclass(frozen=True, slots=True)
class Observation:
    """Exactly what one agent may observe for a decision."""

    agent_id: AgentId
    interaction_id: InteractionId
    participants: tuple[AgentId, ...]
    visible_state: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, AgentId):
            raise TypeError("Observation.agent_id must be an AgentId")
        if not isinstance(self.interaction_id, InteractionId):
            raise TypeError("Observation.interaction_id must be an InteractionId")
        participants = tuple(self.participants)
        if any(not isinstance(item, AgentId) for item in participants):
            raise TypeError("Observation.participants must contain AgentId values")
        if self.agent_id not in participants:
            raise ValueError("Observation.agent_id must be a participant")
        if len(set(participants)) != len(participants):
            raise ValueError("Observation.participants must be unique")
        if not isinstance(self.visible_state, Mapping):
            raise TypeError("Observation.visible_state must be a mapping")
        object.__setattr__(self, "participants", participants)
        object.__setattr__(self, "visible_state", _freeze(self.visible_state))

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": str(self.agent_id),
            "interaction_id": str(self.interaction_id),
            "participants": [str(item) for item in self.participants],
            "visible_state": _thaw(self.visible_state),
        }


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """A logical game decision before it becomes a provider request."""

    agent_id: AgentId
    interaction_id: InteractionId
    stage: str
    observation: Observation
    prompt: CompilablePrompt
    provider_required: bool = True
    retry_bound: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, AgentId):
            raise TypeError("DecisionRequest.agent_id must be an AgentId")
        if not isinstance(self.interaction_id, InteractionId):
            raise TypeError("DecisionRequest.interaction_id must be an InteractionId")
        _non_empty(self.stage, "DecisionRequest.stage")
        if self.agent_id != self.observation.agent_id:
            raise ValueError("DecisionRequest agent and observation agent must match")
        if self.interaction_id != self.observation.interaction_id:
            raise ValueError("DecisionRequest interaction and observation interaction must match")
        if not isinstance(self.prompt, CompilablePrompt):
            raise TypeError("DecisionRequest.prompt must satisfy CompilablePrompt")
        if (
            isinstance(self.retry_bound, bool)
            or not isinstance(self.retry_bound, int)
            or self.retry_bound < 0
        ):
            raise ValueError("DecisionRequest.retry_bound cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": str(self.agent_id),
            "interaction_id": str(self.interaction_id),
            "stage": self.stage,
            "observation": self.observation.to_dict(),
            "prompt": {
                "family": self.prompt.family,
                "version": self.prompt.version,
                "definition_hash": self.prompt.compile().definition_hash,
                "instance_hash": self.prompt.compile().instance_hash,
            },
            "provider_required": self.provider_required,
            "retry_bound": self.retry_bound,
        }


@dataclass(frozen=True, slots=True)
class Action:
    """One locally validated action proposed by an agent."""

    agent_id: AgentId
    value: str
    stage: str = "choice"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, AgentId):
            raise TypeError("Action.agent_id must be an AgentId")
        _non_empty(self.value, "Action.value")
        _non_empty(self.stage, "Action.stage")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("Action.metadata must be a mapping")
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": str(self.agent_id),
            "value": self.value,
            "stage": self.stage,
            "metadata": _thaw(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Transition:
    """Pure state-transition output, independent of provider execution."""

    interaction_id: InteractionId
    actions: tuple[Action, ...]
    payoffs: Mapping[str, float]
    next_state: GameState
    matched: bool | None = None
    termination_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.interaction_id, InteractionId):
            raise TypeError("Transition.interaction_id must be an InteractionId")
        actions = tuple(self.actions)
        if not actions or any(not isinstance(action, Action) for action in actions):
            raise ValueError("Transition.actions must not be empty")
        if not isinstance(self.payoffs, Mapping):
            raise TypeError("Transition.payoffs must be a mapping")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in self.payoffs.values()
        ):
            raise TypeError("Transition.payoffs values must be numeric")
        if not isinstance(self.next_state, GameState):
            raise TypeError("Transition.next_state must be a GameState")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "payoffs", _freeze(self.payoffs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "interaction_id": str(self.interaction_id),
            "actions": [action.to_dict() for action in self.actions],
            "payoffs": _thaw(self.payoffs),
            "matched": self.matched,
            "termination_reason": self.termination_reason,
            "next_state": self.next_state.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """Inspectable observation-to-prompt-to-response-to-action chain."""

    request: DecisionRequest
    completion_request: CompletionRequest | None
    response: CompletionResponse | None
    action: Action
    attempts: int
    prompt_definition_hash: str | None = None
    prompt_instance_hash: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int) or self.attempts < 1:
            raise ValueError("DecisionRecord.attempts must be a positive integer")
        if self.action.agent_id != self.request.agent_id:
            raise ValueError("DecisionRecord action and request agents must match")

    def to_dict(self) -> dict[str, Any]:
        # Provider timing is intentionally deferred to Phase 7 audit events.  Its
        # omission keeps deterministic mock trajectories byte reproducible.
        response = None
        if self.response is not None:
            response = {
                "content": self.response.content,
                "provider": self.response.provider,
                "model": self.response.model,
                "usage": self.response.usage.to_dict(),
                "finish_reason": self.response.finish_reason,
                "request_id": self.response.request_id,
            }
        return {
            "decision_request": self.request.to_dict(),
            "completion_request": (
                None if self.completion_request is None else self.completion_request.to_dict()
            ),
            "response": response,
            "action": self.action.to_dict(),
            "attempts": self.attempts,
            "prompt_definition_hash": self.prompt_definition_hash,
            "prompt_instance_hash": self.prompt_instance_hash,
        }


@dataclass(frozen=True, slots=True)
class InteractionRecord:
    interaction_id: InteractionId
    turn: int
    participants: tuple[AgentId, ...]
    decisions: tuple[DecisionRecord, ...]
    transition: Transition

    def __post_init__(self) -> None:
        participants = tuple(self.participants)
        decisions = tuple(self.decisions)
        if self.turn < 1:
            raise ValueError("InteractionRecord.turn must be positive")
        if self.interaction_id != self.transition.interaction_id:
            raise ValueError("InteractionRecord and transition identifiers must match")
        if tuple(decision.action.agent_id for decision in decisions) != participants:
            raise ValueError("InteractionRecord decisions must follow participant order")
        object.__setattr__(self, "participants", participants)
        object.__setattr__(self, "decisions", decisions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interaction_id": str(self.interaction_id),
            "turn": self.turn,
            "participants": [str(item) for item in self.participants],
            "decisions": [item.to_dict() for item in self.decisions],
            "transition": self.transition.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class GameResult:
    spec: GameSpec
    seed: int
    initial_state: GameState
    final_state: GameState
    interactions: tuple[InteractionRecord, ...]
    termination_reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "interactions", tuple(self.interactions))
        _non_empty(self.termination_reason, "GameResult.termination_reason")
        if self.initial_state.game_type != self.spec.game_type:
            raise ValueError("GameResult initial state does not match its game spec")
        if self.final_state.game_type != self.spec.game_type:
            raise ValueError("GameResult final state does not match its game spec")

    def to_dict(self) -> dict[str, Any]:
        return {
            "game": self.spec.to_dict(),
            "seed": self.seed,
            "initial_state": self.initial_state.to_dict(),
            "final_state": self.final_state.to_dict(),
            "interactions": [item.to_dict() for item in self.interactions],
            "termination_reason": self.termination_reason,
        }


@runtime_checkable
class Game(Protocol):
    """Operational contract implemented by every game."""

    spec: GameSpec

    def initialize(self, config: GameConfig, seed: int) -> GameState: ...

    def select_participants(
        self, state: GameState, config: GameConfig, rng: random.Random
    ) -> tuple[AgentId, ...]: ...

    def construct_observations(
        self, state: GameState, participants: tuple[AgentId, ...], config: GameConfig
    ) -> tuple[Observation, ...]: ...

    def build_decision_requests(
        self, state: GameState, observations: tuple[Observation, ...], config: GameConfig
    ) -> tuple[DecisionRequest, ...]: ...

    def parse_action(self, request: DecisionRequest, response: str) -> Action: ...

    def validate_action(
        self, state: GameState, request: DecisionRequest, action: Action, config: GameConfig
    ) -> ValidationResult: ...

    def apply_transition(
        self, state: GameState, participants: tuple[AgentId, ...], actions: tuple[Action, ...],
        config: GameConfig,
    ) -> Transition: ...

    def detect_termination(self, state: GameState, config: GameConfig) -> str | None: ...

    def call_plan(self, config: GameConfig) -> GameCallPlan: ...
