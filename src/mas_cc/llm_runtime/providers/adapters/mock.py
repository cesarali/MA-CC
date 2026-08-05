"""Deterministic provider for tests, dry runs, and reproducible smoke checks."""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from collections.abc import Callable

from mas_cc.llm_runtime.config import LLMProviderConfig

from ..capabilities import ProviderCapabilities
from ..requests import CompletionRequest
from ..responses import CompletionResponse, ProviderUsage


class MockLLMProvider:
    name = "mock"
    capabilities = ProviderCapabilities(
        supports_seed=True,
        reports_usage=True,
        max_request_concurrency=None,
    )

    def __init__(
        self,
        config: LLMProviderConfig,
        *,
        response_factory: Callable[[CompletionRequest], str] | None = None,
    ) -> None:
        self.model = config.model
        self._response = str(config.options.get("response", "A"))
        self._latency = float(config.options.get("artificial_latency_seconds", 0.0))
        if self._latency < 0:
            raise ValueError("mock artificial_latency_seconds cannot be negative")
        self._response_factory = response_factory
        self._closed = False

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        if self._closed:
            raise RuntimeError("mock provider is closed")
        started = time.perf_counter()
        await asyncio.sleep(self._latency)
        content = (
            self._response_factory(request)
            if self._response_factory is not None
            else self._response
        )
        if not isinstance(content, str):
            raise TypeError("mock response_factory must return a string")
        input_tokens = sum(_count_tokens(message.content) for message in request.messages)
        output_tokens = _count_tokens(content)
        digest = hashlib.sha256(
            repr((request.to_dict(), self.model, content)).encode("utf-8")
        ).hexdigest()[:16]
        latency = time.perf_counter() - started
        raw = {
            "id": f"mock-{digest}",
            "model": self.model,
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }
        return CompletionResponse(
            content=content,
            provider=self.name,
            model=self.model,
            usage=ProviderUsage(input_tokens, output_tokens, input_tokens + output_tokens),
            finish_reason="stop",
            request_id=raw["id"],
            latency_seconds=latency,
            inference_seconds=latency,
            status_code=200,
            raw_response=raw,
        )

    def close(self) -> None:
        self._closed = True


def _count_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
