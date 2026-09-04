"""Concrete adapters; importing this package does not construct a provider."""

from .deepinfra import DeepInfraAccountLimits, DeepInfraProvider
from .gemma_local import GemmaLocalProvider, GenerationResult
from .mock import MockLLMProvider
from .openai import OpenAIProvider
from .university import UniversityProvider

__all__ = [
    "DeepInfraAccountLimits",
    "DeepInfraProvider",
    "GemmaLocalProvider",
    "GenerationResult",
    "MockLLMProvider",
    "OpenAIProvider",
    "UniversityProvider",
]
