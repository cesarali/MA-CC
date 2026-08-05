"""Concrete adapters; importing this package does not construct a provider."""

from .gemma_local import GemmaLocalProvider, GenerationResult
from .mock import MockLLMProvider
from .openai import OpenAIProvider
from .university import UniversityProvider

__all__ = [
    "GemmaLocalProvider",
    "GenerationResult",
    "MockLLMProvider",
    "OpenAIProvider",
    "UniversityProvider",
]
