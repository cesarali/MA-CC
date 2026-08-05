"""Lazy, serialized local Gemma 4 adapter.

Heavy dependencies and checkpoint loading are isolated in ``_GemmaRuntime``.
Importing this module does not import torch, Transformers, dotenv, or Hugging
Face.  The runtime protocol remains injectable so CPU-only tests exercise the
same normalized provider interface.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mas_cc.llm_runtime.config import LLMProviderConfig
from mas_cc.llm_runtime.exceptions import ConfigurationError
from mas_cc.llm_runtime.validation import ValidationIssue

from ..capabilities import ProviderCapabilities
from ..errors import ProviderError
from ..requests import CompletionRequest
from ..responses import CompletionResponse, ProviderUsage


@dataclass(frozen=True, slots=True)
class GenerationResult:
    content: str
    input_tokens: int
    output_tokens: int
    raw_response: dict[str, Any] | None = None


class GemmaRuntime(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> GenerationResult: ...

    @property
    def diagnostics(self) -> dict[str, Any]: ...


class GemmaLocalProvider:
    name = "gemma_local"
    capabilities = ProviderCapabilities(
        supports_seed=True,
        reports_usage=True,
        supports_parallel_requests=False,
        is_local=True,
        max_request_concurrency=1,
    )

    def __init__(
        self,
        config: LLMProviderConfig,
        *,
        runtime_factory: Callable[[], GemmaRuntime] | None = None,
    ) -> None:
        if config.request_concurrency != 1:
            raise ConfigurationError(
                [ValidationIssue("llm_provider.request_concurrency", "gemma_local requires 1")],
                context="gemma_local provider creation",
            )
        dtype = str(config.options.get("dtype", "bfloat16"))
        device_map = str(config.options.get("device_map", "auto"))
        allow_cpu = config.options.get("allow_cpu", False)
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise ConfigurationError(
                [ValidationIssue("llm_provider.options.dtype", "must be bfloat16, float16, or float32")],
                context="gemma_local provider creation",
            )
        if device_map not in {"auto", "cpu"}:
            raise ConfigurationError(
                [ValidationIssue("llm_provider.options.device_map", "must be 'auto' or 'cpu'")],
                context="gemma_local provider creation",
            )
        if not isinstance(allow_cpu, bool):
            raise ConfigurationError(
                [ValidationIssue("llm_provider.options.allow_cpu", "must be a boolean")],
                context="gemma_local provider creation",
            )
        if device_map == "cpu" and not allow_cpu:
            raise ConfigurationError(
                [ValidationIssue("llm_provider.options.allow_cpu", "must be true for CPU loading")],
                context="gemma_local provider creation",
            )
        self.model = config.model
        self._factory = runtime_factory or (
            lambda: _GemmaRuntime(config.model, dtype, device_map, allow_cpu)
        )
        self._runtime: GemmaRuntime | None = None
        self._load_seconds: float | None = None
        self._init_lock = asyncio.Lock()
        self._inference_lock = asyncio.Semaphore(1)
        self._closed = False

    @property
    def diagnostics(self) -> dict[str, Any]:
        result = {} if self._runtime is None else dict(self._runtime.diagnostics)
        result["loaded"] = self._runtime is not None
        result["load_seconds"] = self._load_seconds
        return result

    async def _get_runtime(self) -> tuple[GemmaRuntime, bool]:
        if self._closed:
            raise ProviderError(
                "gemma_local provider is closed.", provider=self.name, code="closed"
            )
        if self._runtime is None:
            async with self._init_lock:
                if self._runtime is None:
                    started = time.perf_counter()
                    construction = asyncio.create_task(asyncio.to_thread(self._factory))
                    try:
                        self._runtime = await asyncio.shield(construction)
                    except asyncio.CancelledError:
                        self._runtime = await construction
                        self._load_seconds = time.perf_counter() - started
                        raise
                    except Exception as exc:
                        raise ProviderError(
                            "gemma_local model loading failed. Check CUDA, model access, HF_HOME, and optional dependencies.",
                            provider=self.name,
                            code="model_load_failed",
                        ) from exc
                    self._load_seconds = time.perf_counter() - started
                    return self._runtime, True
        return self._runtime, False

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        started = time.perf_counter()
        runtime, loaded_here = await self._get_runtime()
        try:
            async with self._inference_lock:
                inference_started = time.perf_counter()
                operation = asyncio.create_task(
                    asyncio.to_thread(
                        runtime.generate,
                        request.wire_messages(),
                        temperature=request.temperature,
                        max_tokens=request.max_output_tokens,
                        seed=request.seed,
                    )
                )
                try:
                    result = await asyncio.shield(operation)
                except asyncio.CancelledError:
                    await operation
                    raise
                inference_seconds = time.perf_counter() - inference_started
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderError(
                "gemma_local inference failed.",
                provider=self.name,
                code="inference_failed",
            ) from exc
        if not isinstance(result, GenerationResult):
            raise ProviderError(
                "gemma_local runtime returned an invalid result.",
                provider=self.name,
                code="invalid_response",
            )
        latency = time.perf_counter() - started
        usage = ProviderUsage(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.input_tokens + result.output_tokens,
        )
        raw = result.raw_response or {
            "model": self.model,
            "content": result.content,
            "usage": usage.to_dict(),
            "runtime": self.diagnostics,
        }
        return CompletionResponse(
            content=result.content,
            provider=self.name,
            model=self.model,
            usage=usage,
            finish_reason="stop",
            latency_seconds=latency,
            load_seconds=self._load_seconds if loaded_here else None,
            inference_seconds=inference_seconds,
            raw_response=raw,
        )

    def close(self) -> None:
        self._closed = True


class _GemmaRuntime:
    """Transformers runtime; construction is the sole heavy-load boundary."""

    def __init__(
        self, model_id: str, dtype_name: str, device_map: str, allow_cpu: bool
    ) -> None:
        self._load_environment()
        try:
            import torch
            import transformers
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Install the project gemma4 optional dependencies."
            ) from exc
        major = int(transformers.__version__.split(".", 1)[0])
        if major < 5:
            raise RuntimeError("Gemma 4 requires Transformers 5 or newer.")
        if not torch.cuda.is_available() and not allow_cpu:
            raise RuntimeError("CUDA is unavailable; CPU loading was not explicitly requested.")
        dtype = getattr(torch, dtype_name)
        kwargs = {"device_map": device_map, "dtype": dtype}
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForMultimodalLM.from_pretrained(model_id, **kwargs)
        self._model.eval()
        self._device = self._input_device()
        self._generation_calls = 0
        tokenizer = self._tokenizer()
        vocabulary = getattr(self._model.config, "vocab_size", None)
        if vocabulary is not None and len(tokenizer) != vocabulary:
            raise RuntimeError(
                f"Tokenizer/model vocabulary mismatch: {len(tokenizer)} != {vocabulary}."
            )
        self._base_diagnostics = {
            "device": str(self._device),
            "dtype": dtype_name,
            "vocabulary_size": len(tokenizer),
            "cuda_allocated_bytes": (
                torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            ),
        }

    @staticmethod
    def _load_environment() -> None:
        try:
            from dotenv import load_dotenv
        except ImportError:
            load_dotenv = None
        current = Path.cwd().resolve()
        for root in (current, *current.parents):
            env = root / ".env"
            if env.is_file():
                if load_dotenv is not None:
                    load_dotenv(env, override=False)
                break
        if not os.getenv("HF_HOME", "").strip():
            raise RuntimeError("HF_HOME is not configured.")

    @property
    def diagnostics(self) -> dict[str, Any]:
        result = dict(self._base_diagnostics)
        result["generation_calls"] = self._generation_calls
        if self._torch.cuda.is_available():
            result["cuda_peak_allocated_bytes"] = self._torch.cuda.max_memory_allocated()
        return result

    def _tokenizer(self):
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Gemma processor does not expose a tokenizer.")
        return tokenizer

    def _input_device(self):
        for device in (getattr(self._model, "hf_device_map", None) or {}).values():
            if isinstance(device, int):
                return self._torch.device(f"cuda:{device}")
            if isinstance(device, str) and device not in {"cpu", "disk", "meta"}:
                return self._torch.device(device)
        return next(self._model.parameters()).device

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> GenerationResult:
        self._generation_calls += 1
        kwargs = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "add_generation_prompt": True,
        }
        try:
            inputs = self._processor.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except (TypeError, ValueError):
            inputs = self._processor.apply_chat_template(messages, **kwargs)
        inputs = inputs.to(self._device)
        input_tokens = int(inputs["input_ids"].shape[1])
        generation: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation["temperature"] = temperature
            if seed is not None:
                generator = self._torch.Generator(device=self._device)
                generator.manual_seed(seed)
                generation["generator"] = generator
        with self._torch.inference_mode():
            generated = self._model.generate(**inputs, **generation)
        new_tokens = generated[0, input_tokens:]
        content = self._processor.decode(new_tokens, skip_special_tokens=True)
        return GenerationResult(content, input_tokens, int(new_tokens.shape[0]))
