"""The provider boundary consumed by games and experiments."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .capabilities import ProviderCapabilities
from .requests import CompletionRequest
from .responses import CompletionResponse


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str
    capabilities: ProviderCapabilities

    async def complete(self, request: CompletionRequest) -> CompletionResponse: ...

    def close(self) -> None: ...
