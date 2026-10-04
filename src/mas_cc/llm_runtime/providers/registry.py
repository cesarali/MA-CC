"""Lazy provider registry and the canonical provider factory."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

from mas_cc.llm_runtime.config import LLMProviderConfig

from ..exceptions import ConfigurationError
from ..validation import ValidationIssue
from .protocols import LLMProvider
from .model_profiles import ModelProfileRegistry, default_model_profile_registry
from .profiles import ProfiledLLMProvider

ProviderFactory = Callable[..., LLMProvider]
OPENAI_COMPATIBLE_PROVIDER_TYPES = frozenset(
    {"deepinfra", "neuralwatt", "openai", "university"}
)


@dataclass(slots=True)
class ProviderRegistry:
    """Maps provider names to lazy module/class references or injected factories."""

    _entries: dict[str, str | ProviderFactory] = field(default_factory=dict)

    def register(self, name: str, factory: str | ProviderFactory) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("provider registry name must be non-empty")
        if normalized in self._entries:
            raise ValueError(f"provider {normalized!r} is already registered")
        self._entries[normalized] = factory

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def create(self, config: LLMProviderConfig, **kwargs: Any) -> LLMProvider:
        name = config.type.strip().lower()
        try:
            factory = self._entries[name]
        except KeyError as exc:
            raise ConfigurationError(
                [
                    ValidationIssue(
                        "llm_provider.type",
                        f"unknown provider {config.type!r}; available: {', '.join(self.names())}",
                    )
                ],
                context="provider creation",
            ) from exc
        if isinstance(factory, str):
            module_name, separator, attribute = factory.partition(":")
            if not separator:
                raise RuntimeError(f"invalid provider registry target {factory!r}")
            factory = getattr(importlib.import_module(module_name), attribute)
        provider = factory(config, **kwargs)
        if not isinstance(provider, LLMProvider):
            raise TypeError(f"provider factory for {name!r} returned an incompatible object")
        return provider


def create_default_provider_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    prefix = "mas_cc.llm_runtime.providers.adapters"
    registry.register("mock", f"{prefix}.mock:MockLLMProvider")
    registry.register("deepinfra", f"{prefix}.deepinfra:DeepInfraProvider")
    registry.register("neuralwatt", f"{prefix}.neuralwatt:NeuralWattProvider")
    registry.register("openai", f"{prefix}.openai:OpenAIProvider")
    registry.register("university", f"{prefix}.university:UniversityProvider")
    registry.register("gemma_local", f"{prefix}.gemma_local:GemmaLocalProvider")
    return registry


def create_llm_provider(
    config: LLMProviderConfig,
    *,
    registry: ProviderRegistry | None = None,
    environment: Mapping[str, str] | None = None,
    profile_registry: ModelProfileRegistry | None = None,
    **adapter_options: Any,
) -> LLMProvider:
    """Construct one adapter; this is the only public provider creation path."""

    selected = registry or create_default_provider_registry()
    provider_type = config.type.strip().lower()
    if provider_type in OPENAI_COMPATIBLE_PROVIDER_TYPES:
        from .load_control import coordinator_from_environment

        if environment is not None:
            adapter_options["environment"] = environment
        adapter_options.setdefault(
            "request_coordinator",
            coordinator_from_environment(config, environment=environment),
        )
    provider = selected.create(config, **adapter_options)
    if provider_type in OPENAI_COMPATIBLE_PROVIDER_TYPES:
        profiles = profile_registry or default_model_profile_registry()
        return ProfiledLLMProvider(provider, provider_type, profiles)
    return provider
