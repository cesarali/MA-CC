"""Adapter from MuSR-style generation calls to the MAS-CC provider protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from mas_cc.llm_runtime.messages import Message
from mas_cc.llm_runtime.providers.protocols import LLMProvider
from mas_cc.llm_runtime.providers.requests import CompletionRequest
from mas_cc.llm_runtime.providers.responses import CompletionResponse

from .schemas import PROMPT_VERSION


@dataclass(slots=True)
class MuSRGenerationModel:
    provider: LLMProvider
    temperature: float = 0.7
    max_output_tokens: int = 2_048
    prompt_version: str = PROMPT_VERSION
    audit_sink: Callable[[Mapping[str, Any]], None] | None = None
    calls: list[dict[str, Any]] = field(default_factory=list, init=False)

    async def inference(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        purpose: str = "evidence_generation",
        metadata: Mapping[str, Any] | None = None,
    ) -> CompletionResponse:
        request_metadata = {
            "prompt_family": "musr_team_allocation_generator",
            "prompt_version": self.prompt_version,
            "purpose": purpose,
            **dict(metadata or {}),
        }
        request = CompletionRequest(
            messages=(Message("user", prompt),),
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            seed=seed,
            metadata=request_metadata,
        )
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            response = await self.provider.complete(request)
        except Exception as exc:
            if self.audit_sink is not None:
                self.audit_sink(
                    {
                        "started_at": started_at,
                        "purpose": purpose,
                        "seed": seed,
                        "prompt": prompt,
                        "request": request.to_dict(),
                        "metadata": request_metadata,
                        "raw_response": None,
                        "provider_error": f"{type(exc).__name__}: {exc}",
                    }
                )
            raise
        call = {
            "purpose": purpose,
            "seed": seed,
            "provider": response.provider,
            "model": response.model,
            "usage": response.usage.to_dict(),
            "latency_seconds": response.latency_seconds,
            "retries": response.retries,
            "request_id": response.request_id,
        }
        self.calls.append(call)
        if self.audit_sink is not None:
            self.audit_sink(
                {
                    "started_at": started_at,
                    **call,
                    "prompt": prompt,
                    "request": request.to_dict(),
                    "metadata": request_metadata,
                    "raw_response": response.content,
                    "provider_error": None,
                }
            )
        return response
