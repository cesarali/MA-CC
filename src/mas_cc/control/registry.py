"""Lazy registry for provider-independent control mechanisms."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field

from mas_cc.config import ControlConfig
from mas_cc.llm_runtime.exceptions import ConfigurationError
from mas_cc.llm_runtime.validation import ValidationIssue

from .protocols import Control

ControlFactory = Callable[[ControlConfig], Control]


@dataclass(slots=True)
class ControlRegistry:
    _entries: dict[str, str | ControlFactory] = field(default_factory=dict)

    def register(self, name: str, factory: str | ControlFactory) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("control registry name must be non-empty")
        if normalized in self._entries:
            raise ValueError(f"control mechanism {normalized!r} is already registered")
        self._entries[normalized] = factory

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def create(self, config: ControlConfig) -> Control:
        name = config.mechanism.strip().lower()
        try:
            factory = self._entries[name]
        except KeyError as exc:
            raise ConfigurationError(
                [
                    ValidationIssue(
                        "control.mechanism",
                        f"unknown control mechanism {config.mechanism!r}; available: {', '.join(self.names())}",
                    )
                ],
                context="control creation",
            ) from exc
        if isinstance(factory, str):
            module_name, separator, attribute = factory.partition(":")
            if not separator:
                raise RuntimeError(f"invalid control registry target {factory!r}")
            factory = getattr(importlib.import_module(module_name), attribute)
        control = factory(config)
        if not isinstance(control, Control):
            raise TypeError(f"control factory for {name!r} returned an incompatible object")
        return control


def create_default_control_registry() -> ControlRegistry:
    registry = ControlRegistry()
    registry.register("none", "mas_cc.control.forced_action:create_none_control")
    registry.register("forced_action", "mas_cc.control.forced_action:create_forced_action_control")
    registry.register(
        "threshold_target",
        "mas_cc.games.hidden_bench.imitation.controller:create_threshold_target_control",
    )
    return registry


def create_control(config: ControlConfig, *, registry: ControlRegistry | None = None) -> Control:
    return (registry or create_default_control_registry()).create(config)
