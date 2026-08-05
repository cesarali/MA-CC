"""Discrete forced-action control: the one mechanism implemented so far."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mas_cc.config import ControlConfig
from mas_cc.core import AgentId
from mas_cc.llm_runtime.exceptions import ConfigurationError
from mas_cc.llm_runtime.validation import ValidationIssue

from .protocols import Control

if TYPE_CHECKING:
    from mas_cc.games.protocols import GameState


@dataclass(frozen=True, slots=True)
class NoneControl(Control):
    """The default, no-op control: never overrides anything."""

    def override(self, *, agent_id: AgentId, interaction_index: int, state: "GameState") -> str | None:
        return None


@dataclass(frozen=True, slots=True)
class ForcedActionControl(Control):
    """Force a fixed set of agents to a single action value.

    Mirrors the legacy `naming_game` committee: `agent_ids` are forced to
    `forced_value` every interaction from the start, unless `until_interaction`
    is set - a "pulse" that expires after that interaction index, after which
    those agents return to ordinary LLM-backed decisions.
    """

    agent_ids: frozenset[AgentId]
    forced_value: str
    until_interaction: int | None = None

    def override(self, *, agent_id: AgentId, interaction_index: int, state: "GameState") -> str | None:
        if agent_id not in self.agent_ids:
            return None
        if self.until_interaction is not None and interaction_index > self.until_interaction:
            return None
        return self.forced_value

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> "ForcedActionControl":
        raw_agent_ids = options.get("agent_ids")
        if not isinstance(raw_agent_ids, Sequence) or isinstance(raw_agent_ids, (str, bytes)) or not raw_agent_ids:
            raise ConfigurationError(
                [ValidationIssue("control.options.agent_ids", "must be a non-empty list of agent IDs")],
                context="control creation",
            )
        try:
            agent_ids = frozenset(AgentId(str(item)) for item in raw_agent_ids)
        except ValueError as exc:
            raise ConfigurationError(
                [ValidationIssue("control.options.agent_ids", f"invalid agent ID: {exc}")],
                context="control creation",
            ) from exc
        forced_value = options.get("forced_value")
        if not isinstance(forced_value, str) or not forced_value.strip():
            raise ConfigurationError(
                [ValidationIssue("control.options.forced_value", "must be a non-empty string")],
                context="control creation",
            )
        until_interaction = options.get("until_interaction")
        if until_interaction is not None and (
            isinstance(until_interaction, bool) or not isinstance(until_interaction, int) or until_interaction < 1
        ):
            raise ConfigurationError(
                [ValidationIssue("control.options.until_interaction", "must be a positive integer or null")],
                context="control creation",
            )
        return cls(agent_ids=agent_ids, forced_value=forced_value, until_interaction=until_interaction)


def create_none_control(config: ControlConfig) -> Control:
    return NoneControl()


def create_forced_action_control(config: ControlConfig) -> Control:
    return ForcedActionControl.from_options(config.options)
