"""Private, immutable inputs made available during prompt rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


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
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_thaw(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class PromptContext:
    """Exactly the information exposed to one agent decision.

    Population state and intervention metadata have no implicit home here.
    Games must deliberately place any permitted information in one of the
    explicit private or current-interaction fields.
    """

    task_description: str
    game_rules: tuple[str, ...]
    private_state: Mapping[str, Any]
    recent_memory: tuple[Mapping[str, Any], ...]
    current_interaction: Mapping[str, Any]
    decision_instruction: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task_description, str) or not self.task_description.strip():
            raise ValueError("PromptContext.task_description must be a non-empty string")
        if isinstance(self.game_rules, (str, bytes)) or not isinstance(
            self.game_rules, Sequence
        ):
            raise TypeError("PromptContext.game_rules must be a sequence of strings")
        rules = tuple(self.game_rules)
        if not rules or any(not isinstance(rule, str) or not rule.strip() for rule in rules):
            raise ValueError("PromptContext.game_rules must contain non-empty strings")
        if not isinstance(self.private_state, Mapping):
            raise TypeError("PromptContext.private_state must be a mapping")
        if isinstance(self.recent_memory, (str, bytes)) or not isinstance(
            self.recent_memory, Sequence
        ):
            raise TypeError("PromptContext.recent_memory must be a sequence of mappings")
        memory = tuple(self.recent_memory)
        if any(not isinstance(item, Mapping) for item in memory):
            raise TypeError("PromptContext.recent_memory items must be mappings")
        if not isinstance(self.current_interaction, Mapping):
            raise TypeError("PromptContext.current_interaction must be a mapping")
        if not isinstance(self.decision_instruction, str) or not self.decision_instruction.strip():
            raise ValueError("PromptContext.decision_instruction must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("PromptContext.metadata must be a mapping")
        object.__setattr__(self, "game_rules", rules)
        object.__setattr__(self, "private_state", _freeze(self.private_state))
        object.__setattr__(self, "recent_memory", _freeze(memory))
        object.__setattr__(self, "current_interaction", _freeze(self.current_interaction))
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_description": self.task_description,
            "game_rules": list(self.game_rules),
            "private_state": _thaw(self.private_state),
            "recent_memory": _thaw(self.recent_memory),
            "current_interaction": _thaw(self.current_interaction),
            "decision_instruction": self.decision_instruction,
            "metadata": _thaw(self.metadata),
        }
