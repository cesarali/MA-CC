"""Immutable scientific state and audit records for the convention game."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mas_cc.core import AgentId, InteractionId
from mas_cc.core.exceptions import MasCCError
from mas_cc.games.protocols import (
    Action,
    AgentState,
    DecisionRequest,
    GameState,
    Transition,
    _thaw,
)
from mas_cc.llm_providers import CompletionRequest, CompletionResponse
from mas_cc.prompts import CompiledPrompt


class InvalidConventionResponse(MasCCError, RuntimeError):
    """Raised after all validation attempts for one logical decision fail."""


@dataclass(frozen=True, slots=True)
class PrivateMemoryEntry:
    agent_local_interaction_index: int
    own_action: str
    partner_action: str
    payoff: int
    success: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.agent_local_interaction_index, bool)
            or not isinstance(self.agent_local_interaction_index, int)
            or self.agent_local_interaction_index < 1
        ):
            raise ValueError("agent_local_interaction_index must be positive")
        if not self.own_action or not self.partner_action:
            raise ValueError("private memory actions must be non-empty")
        if self.payoff not in {100, -50}:
            raise ValueError("private memory payoff must be +100 or -50")
        if not isinstance(self.success, bool):
            raise TypeError("private memory success must be a boolean")
        if self.success != (self.own_action == self.partner_action):
            raise ValueError("private memory success must agree with its actions")
        if self.payoff != (100 if self.success else -50):
            raise ValueError("private memory payoff must agree with its success flag")

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_local_interaction_index": self.agent_local_interaction_index,
            "own_action": self.own_action,
            "partner_action": self.partner_action,
            "payoff": self.payoff,
            "success": self.success,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PrivateMemoryEntry":
        return cls(
            int(value["agent_local_interaction_index"]),
            str(value["own_action"]),
            str(value["partner_action"]),
            int(value["payoff"]),
            bool(value["success"]),
        )


@dataclass(frozen=True, slots=True)
class ConventionAgentState(AgentState):
    """Complete private agent history plus evaluator-only lifetime state."""

    @property
    def private_history(self) -> tuple[PrivateMemoryEntry, ...]:
        return tuple(PrivateMemoryEntry.from_mapping(item) for item in self.memory)

    @property
    def lifetime_score(self) -> int:
        return int(self.score)

    @property
    def committed_action(self) -> str | None:
        value = self.attributes.get("committed_action")
        return None if value is None else str(value)

    def visible_history(self, memory_size: int) -> tuple[PrivateMemoryEntry, ...]:
        if memory_size == 0:
            return ()
        return self.private_history[-memory_size:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": str(self.agent_id),
            "private_history": [entry.to_dict() for entry in self.private_history],
            "lifetime_score": self.lifetime_score,
            "committed_action": self.committed_action,
            "available_actions": list(self.attributes.get("available_actions", ())),
        }


@dataclass(frozen=True, slots=True)
class ConventionGameState(GameState):
    @property
    def global_interaction_index(self) -> int:
        return self.turn

    @property
    def action_pool(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.data["action_pool"])

    @property
    def evaluator_history(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.data.get("evaluator_history", ()))

    @property
    def termination_reason(self) -> str | None:
        value = self.data.get("termination_reason")
        return None if value is None else str(value)

    def convention_agent(self, agent_id: AgentId) -> ConventionAgentState:
        agent = self.agent(agent_id)
        if not isinstance(agent, ConventionAgentState):
            raise TypeError("convention state contains a non-convention agent")
        return agent

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_type": self.game_type,
            "global_interaction_index": self.global_interaction_index,
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
            "action_pool": list(self.action_pool),
            "topology": _thaw(self.data.get("topology", {})),
            "agents": [agent.to_dict() for agent in self.agents],
            "evaluator_history": _thaw(self.evaluator_history),
        }


@dataclass(frozen=True, slots=True)
class ConventionDecisionRequest(DecisionRequest):
    parser_contract: str = "tolerant_paper_object_v1"

    @property
    def presented_actions(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.observation.visible_state["presented_actions"])

    @property
    def visible_score(self) -> int:
        return int(self.observation.visible_state["visible_score"])

    @property
    def local_round(self) -> int:
        return int(self.observation.visible_state["local_round"])

    @property
    def visible_memory(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.observation.visible_state["visible_memory"])


@dataclass(frozen=True, slots=True)
class ConventionTransition(Transition):
    @property
    def success(self) -> bool:
        return bool(self.matched)

    @property
    def payoff(self) -> int:
        return int(next(iter(self.payoffs.values())))


@dataclass(frozen=True, slots=True)
class ParsedConventionResponse:
    raw_text: str
    value: str
    reason: str | None
    parser_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "parsed_action": self.value,
            "parsed_reason": self.reason,
            "parser_mode": self.parser_mode,
        }


@dataclass(frozen=True, slots=True)
class ConventionValidationAttempt:
    attempt: int
    completion_request: CompletionRequest
    response: CompletionResponse
    parsed: ParsedConventionResponse | None
    valid: bool
    validation_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "completion_request": self.completion_request.to_dict(),
            "provider_response": {
                "raw_text": self.response.content,
                "provider": self.response.provider,
                "model": self.response.model,
                "usage": self.response.usage.to_dict(),
                "finish_reason": self.response.finish_reason,
                "request_id": self.response.request_id,
                "provider_retries": self.response.retries,
            },
            "parser": None if self.parsed is None else self.parsed.to_dict(),
            "valid": self.valid,
            "validation_error": self.validation_error,
        }


@dataclass(frozen=True, slots=True)
class ConventionDecisionOutcome:
    request: ConventionDecisionRequest
    action: Action
    parsed_reason: str | None
    parser_mode: str
    prompt_hash: str
    attempts: tuple[ConventionValidationAttempt, ...]
    prompt_definition_hash: str
    prompt_instance_hash: str
    compiled_prompt: CompiledPrompt
    forced: bool = False

    @property
    def validation_attempts(self) -> int:
        return len(self.attempts)

    @property
    def provider_retries(self) -> int:
        return sum(attempt.response.retries for attempt in self.attempts)

    def to_dict(self) -> dict[str, Any]:
        first = self.attempts[0]
        last = self.attempts[-1]
        return {
            "agent_id": str(self.request.agent_id),
            "local_role": "Player 1",
            "anonymous_partner_role": "Player 2",
            "visible_memory": _thaw(self.request.visible_memory),
            "visible_score": self.request.visible_score,
            "local_round": self.request.local_round,
            "presented_actions": list(self.request.presented_actions),
            "compiled_messages": [message.to_dict() for message in first.completion_request.messages],
            "prompt_contract": self.request.prompt.family,
            "prompt_hash": self.prompt_hash,
            "prompt_definition_hash": self.prompt_definition_hash,
            "prompt_instance_hash": self.prompt_instance_hash,
            "rendered_blocks": self.compiled_prompt.blocks_as_dicts(),
            "prompt_token_counts": {
                "blocks": self.compiled_prompt.block_token_total,
                "messages": self.compiled_prompt.message_token_total,
                "tokenizer": self.compiled_prompt.tokenizer_name,
            },
            "raw_response": last.response.content,
            "parsed_action": self.action.value,
            "parsed_reason": self.parsed_reason,
            "parser_mode": self.parser_mode,
            "validation": {
                "valid": True,
                "logical_decisions": 1,
                "validation_attempts": self.validation_attempts,
                "validation_retries": self.validation_attempts - 1,
                "provider_retries": self.provider_retries,
                "forced_decision": self.forced,
                "permanent_failures": 0,
                "attempts": [attempt.to_dict() for attempt in self.attempts],
            },
            "provider_result": {
                "provider": last.response.provider,
                "model": last.response.model,
                "usage": last.response.usage.to_dict(),
            },
        }


@dataclass(frozen=True, slots=True)
class ConventionInteractionRecord:
    interaction_id: InteractionId
    interaction_index: int
    selected_agents: tuple[AgentId, AgentId]
    pre_visible_memories: tuple[tuple[PrivateMemoryEntry, ...], tuple[PrivateMemoryEntry, ...]]
    decisions: tuple[ConventionDecisionOutcome, ConventionDecisionOutcome]
    transition: ConventionTransition
    post_visible_memories: tuple[tuple[PrivateMemoryEntry, ...], tuple[PrivateMemoryEntry, ...]]

    def to_dict(self) -> dict[str, Any]:
        actions = tuple(decision.action.value for decision in self.decisions)
        return {
            "interaction_id": str(self.interaction_id),
            "interaction_index": self.interaction_index,
            "selected_agents": [str(item) for item in self.selected_agents],
            "pre_interaction_private_memory": {
                "player_1": [entry.to_dict() for entry in self.pre_visible_memories[0]],
                "player_2": [entry.to_dict() for entry in self.pre_visible_memories[1]],
            },
            "presented_action_orders": {
                "player_1": list(self.decisions[0].request.presented_actions),
                "player_2": list(self.decisions[1].request.presented_actions),
            },
            "decisions": {
                "player_1": self.decisions[0].to_dict(),
                "player_2": self.decisions[1].to_dict(),
            },
            "parsed_actions": list(actions),
            "parsed_reasons": [decision.parsed_reason for decision in self.decisions],
            "success": self.transition.success,
            "payoff": self.transition.payoff,
            "post_interaction_private_memory": {
                "player_1": [entry.to_dict() for entry in self.post_visible_memories[0]],
                "player_2": [entry.to_dict() for entry in self.post_visible_memories[1]],
            },
            "visible_scores_before": [
                self.decisions[0].request.visible_score,
                self.decisions[1].request.visible_score,
            ],
            "lifetime_scores_after": [
                self.transition.next_state.convention_agent(agent_id).lifetime_score
                for agent_id in self.selected_agents
            ],
            "forced_decisions": [decision.forced for decision in self.decisions],
        }


@dataclass(frozen=True, slots=True)
class ConventionGameResult:
    initial_state: ConventionGameState
    final_state: ConventionGameState
    interactions: tuple[ConventionInteractionRecord, ...]
    termination_reason: str
    logical_decisions: int
    validation_attempts: int
    provider_retries: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_state": self.initial_state.to_dict(),
            "final_state": self.final_state.to_dict(),
            "interactions": [item.to_dict() for item in self.interactions],
            "termination_reason": self.termination_reason,
            "counters": {
                "logical_decisions": self.logical_decisions,
                "validation_attempts": self.validation_attempts,
                "provider_retries": self.provider_retries,
                "forced_decisions": 0,
                "permanent_failures": 0,
            },
        }
