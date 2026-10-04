"""Concrete adapters; importing this package does not construct a provider."""

from .deepinfra import DeepInfraAccountLimits, DeepInfraProvider
from .gemma_local import GemmaLocalProvider, GenerationResult
from .mock import MockLLMProvider
from .neuralwatt import NeuralWattProvider
from .openai import OpenAIProvider
from .university import UniversityProvider

__all__ = [
    "DeepInfraAccountLimits",
    "DeepInfraProvider",
    "GemmaLocalProvider",
    "GenerationResult",
    "MockLLMProvider",
    "NeuralWattProvider",
    "OpenAIProvider",
    "UniversityProvider",
]
