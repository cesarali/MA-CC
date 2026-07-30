"""Lazy, serialized local Gemma 4 client.

Importing this module never imports torch or Transformers.  The small runtime
protocol is intentionally injectable so all application behavior can be tested
without a checkpoint or accelerator.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

from .api_client import _RequestStats
from .local_model_types import (ChoiceScore, ConstrainedDecisionResponse,
                                ConstrainedLLMResponse)
from .models import ConfigurationError, LLMResponse, TokenUsage

MODEL_ID = "google/gemma-4-12B-it"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationResult:
    content: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class ScoringResult:
    prompt_tokens: int
    token_ids: tuple[tuple[int, ...], ...]
    log_likelihoods: tuple[float, ...]


class GemmaRuntime(Protocol):
    def generate(self, messages: list[dict[str, str]], *, temperature: float,
                 max_tokens: int, seed: int | None) -> GenerationResult: ...
    def score(self, messages: list[dict[str, str]], choices: Sequence[str]) -> ScoringResult: ...
    @property
    def diagnostics(self) -> dict[str, Any]: ...


def _validate_messages(messages: list[dict[str, str]]) -> None:
    if not messages:
        raise ValueError("messages must not be empty.")
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("Every message must contain exactly role and content.")
        if any(not isinstance(message[k], str) or not message[k].strip() for k in ("role", "content")):
            raise ValueError("Message role and content must be non-blank strings.")


class GemmaLocalAsyncLLMClient:
    provider_name = "gemma_local"
    concurrency = 1

    def __init__(self, *, model: str = MODEL_ID, runtime_factory: Callable[[], GemmaRuntime] | None = None,
                 dtype: str = "bfloat16", device_map: str = "auto", allow_cpu: bool = False) -> None:
        if model != MODEL_ID:
            raise ConfigurationError(f"Unsupported local model {model!r}; expected {MODEL_ID!r}.")
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise ConfigurationError("dtype must be bfloat16, float16, or float32.")
        if device_map not in {"auto", "cpu"}:
            raise ConfigurationError("device_map must be 'auto' or 'cpu'.")
        if device_map == "cpu" and not allow_cpu:
            raise ConfigurationError("CPU loading requires explicit allow_cpu=True.")
        self.model, self.dtype, self.device_map = model, dtype, device_map
        self._factory = runtime_factory or (lambda: _TransformersGemmaRuntime(model, dtype, device_map, allow_cpu))
        self._runtime: GemmaRuntime | None = None
        self._init_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(1)
        self._stats = _RequestStats()
        self._closed = False

    @property
    def stats(self) -> dict[str, Any]: return self._stats.snapshot()

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {} if self._runtime is None else dict(self._runtime.diagnostics)

    async def _get_runtime(self) -> GemmaRuntime:
        if self._closed:
            raise RuntimeError("client is closed.")
        if self._runtime is None:
            async with self._init_lock:
                if self._runtime is None:
                    self._runtime = await asyncio.to_thread(self._factory)
        return self._runtime

    async def complete(self, messages: list[dict[str, str]], *, temperature: float,
                       max_tokens: int, seed: int | None = None) -> LLMResponse:
        _validate_messages(messages)
        if max_tokens < 1: raise ValueError("max_tokens must be positive.")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative.")
        self._stats.attempt(); started = time.perf_counter()
        try:
            async with self._semaphore:
                runtime = await self._get_runtime()
                result = await asyncio.to_thread(runtime.generate, messages, temperature=temperature,
                                                 max_tokens=max_tokens, seed=seed)
            if result.prompt_tokens < 0 or result.completion_tokens < 0:
                raise RuntimeError("Runtime returned invalid token counts.")
            latency = time.perf_counter() - started
            usage = TokenUsage(result.prompt_tokens, result.completion_tokens,
                               result.prompt_tokens + result.completion_tokens)
            response = LLMResponse(content=result.content, model=self.model,
                                   latency_seconds=latency, usage=usage)
            self._stats.success(latency, usage); return response
        except BaseException:
            self._stats.failure(); raise

    async def complete_constrained(self, messages: list[dict[str, str]], *, choices: Sequence[str],
                                   temperature: float = 1.0, seed: int | None = None) -> ConstrainedLLMResponse:
        del seed  # deterministic argmax is the only phase-one policy
        _validate_messages(messages)
        choice_tuple = tuple(choices)
        if not choice_tuple or any(not isinstance(c, str) or not c.strip() for c in choice_tuple):
            raise ValueError("choices must contain non-blank strings.")
        if len(set(choice_tuple)) != len(choice_tuple): raise ValueError("choices must not contain duplicates.")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("constrained temperature must be finite and positive.")
        self._stats.attempt(); started = time.perf_counter()
        try:
            async with self._semaphore:
                result = await asyncio.to_thread((await self._get_runtime()).score, messages, choice_tuple)
            if len(result.token_ids) != len(choice_tuple) or len(result.log_likelihoods) != len(choice_tuple):
                raise RuntimeError("Runtime returned the wrong number of candidate scores.")
            if any(not ids for ids in result.token_ids): raise ValueError("A choice tokenized to an empty sequence.")
            if any(not math.isfinite(v) for v in result.log_likelihoods):
                raise FloatingPointError("Runtime returned a non-finite sequence score.")
            scaled = [v / temperature for v in result.log_likelihoods]
            peak = max(scaled); weights = [math.exp(v - peak) for v in scaled]; total = math.fsum(weights)
            probabilities = [w / total for w in weights]
            if not all(math.isfinite(p) for p in probabilities) or abs(math.fsum(probabilities) - 1) > 1e-5:
                raise FloatingPointError("Could not normalize finite choice probabilities.")
            scores = tuple(ChoiceScore(c, tuple(ids), ll, p) for c, ids, ll, p in
                           zip(choice_tuple, result.token_ids, result.log_likelihoods, probabilities))
            selected = max(range(len(scores)), key=lambda i: scores[i].probability)
            completion = sum(len(ids) for ids in result.token_ids)
            usage = TokenUsage(result.prompt_tokens * len(choice_tuple), completion,
                               result.prompt_tokens * len(choice_tuple) + completion)
            latency = time.perf_counter() - started
            response = ConstrainedLLMResponse(scores[selected].choice, scores, self.model, latency, usage, temperature)
            self._stats.success(latency, usage); return response
        except BaseException:
            self._stats.failure(); raise

    async def complete_decision(
        self, messages: list[dict[str, str]], *, choices: Sequence[str],
        output_format: str, choice_temperature: float = 1.0,
        selection_policy: str = "argmax", generation_temperature: float = 0.0,
        max_reason_tokens: int = 32, seed: int | None = None,
    ) -> ConstrainedDecisionResponse:
        """Score, select, and optionally explain in one logical request.

        ``usage`` reports scoring prompt work plus candidate and generated reason
        tokens; it is not a claim that every counted token was autoregressively emitted.
        """
        _validate_messages(messages)
        choice_tuple = tuple(choices)
        if output_format not in {"choice_reason", "choice_only"}:
            raise ValueError("output_format must be choice_reason or choice_only.")
        if selection_policy not in {"argmax", "sample"}:
            raise ValueError("selection_policy must be argmax or sample.")
        if not choice_tuple or any(not isinstance(c, str) or not c.strip() for c in choice_tuple):
            raise ValueError("choices must contain non-blank strings.")
        if len(set(choice_tuple)) != len(choice_tuple):
            raise ValueError("choices must not contain duplicates.")
        if not math.isfinite(choice_temperature) or choice_temperature <= 0:
            raise ValueError("choice_temperature must be finite and positive.")
        if not math.isfinite(generation_temperature) or generation_temperature < 0:
            raise ValueError("generation_temperature must be finite and non-negative.")
        if max_reason_tokens < 1:
            raise ValueError("max_reason_tokens must be positive.")
        self._stats.attempt(); started = time.perf_counter()
        try:
            async with self._semaphore:
                runtime = await self._get_runtime()
                scored = await asyncio.to_thread(runtime.score, messages, choice_tuple)
                if len(scored.token_ids) != len(choice_tuple) or len(scored.log_likelihoods) != len(choice_tuple):
                    raise RuntimeError("Runtime returned the wrong number of candidate scores.")
                if any(not ids for ids in scored.token_ids):
                    raise ValueError("A choice tokenized to an empty sequence.")
                if any(not math.isfinite(v) for v in scored.log_likelihoods):
                    raise FloatingPointError("Runtime returned a non-finite sequence score.")
                scaled = [v / choice_temperature for v in scored.log_likelihoods]
                peak = max(scaled); weights = [math.exp(v - peak) for v in scaled]
                total = math.fsum(weights); probabilities = [w / total for w in weights]
                scores = tuple(ChoiceScore(c, tuple(ids), ll, p) for c, ids, ll, p in
                    zip(choice_tuple, scored.token_ids, scored.log_likelihoods, probabilities))
                if selection_policy == "argmax":
                    selected_index = max(range(len(scores)), key=lambda i: scores[i].probability)
                else:
                    selected_index = random.Random(seed).choices(range(len(scores)), weights=probabilities, k=1)[0]
                selected = scores[selected_index].choice
                reason = None; reason_valid = None; reason_tokens = 0
                if output_format == "choice_reason":
                    reason_messages = messages + [{"role": "assistant", "content": f"{selected}\nReason: "}]
                    generated = await asyncio.to_thread(runtime.generate, reason_messages,
                        temperature=generation_temperature, max_tokens=max_reason_tokens, seed=seed)
                    reason = generated.content.strip() or None
                    reason_valid = reason is not None
                    reason_tokens = generated.completion_tokens
                    content = f"{selected}\nReason: {reason or ''}"
                else:
                    content = selected
            candidate_tokens = sum(len(ids) for ids in scored.token_ids)
            prompt_work = scored.prompt_tokens * len(choice_tuple)
            usage = TokenUsage(prompt_work, candidate_tokens + reason_tokens,
                               prompt_work + candidate_tokens + reason_tokens)
            latency = time.perf_counter() - started
            response = ConstrainedDecisionResponse(selected, scores, content, reason,
                reason_valid, output_format, self.model, latency, usage,
                choice_temperature, selection_policy)
            self._stats.success(latency, usage)
            return response
        except BaseException:
            self._stats.failure(); raise

    def close(self) -> None: self._closed = True


class _TransformersGemmaRuntime:
    """Real runtime. Construction is the sole heavy-import/load boundary."""
    def __init__(self, model_id: str, dtype_name: str, device_map: str, allow_cpu: bool) -> None:
        root = Path(__file__).resolve().parents[2]
        env = root / ".env"
        if not env.exists(): raise ConfigurationError("Repository-root .env is required for local Gemma.")
        load_dotenv(env)
        if not os.getenv("HF_HOME", "").strip(): raise ConfigurationError("HF_HOME is not configured.")
        import torch
        import transformers
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        major = int(transformers.__version__.split(".", 1)[0])
        if major < 5: raise ConfigurationError("Gemma 4 requires Transformers 5 or newer.")
        if not torch.cuda.is_available() and not allow_cpu:
            raise RuntimeError("CUDA is unavailable; CPU loading was not explicitly requested.")
        dtype = getattr(torch, dtype_name)
        kwargs = {"device_map": device_map, "dtype" if major >= 5 else "torch_dtype": dtype}
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(model_id, **kwargs)
        self.model.eval(); self.device = self._input_device()
        tokenizer = self._tokenizer()
        model_vocab = getattr(self.model.config, "vocab_size", None)
        if model_vocab is not None and len(tokenizer) != model_vocab:
            raise RuntimeError(f"Tokenizer/model vocabulary mismatch: {len(tokenizer)} != {model_vocab}.")
        self._diagnostics = {"device": str(self.device), "dtype": str(dtype), "vocabulary_size": len(tokenizer),
                             "cuda_allocated_bytes": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0}
        LOGGER.info("Gemma runtime initialized: model=%s device=%s dtype=%s vocabulary=%d",
                    model_id, self.device, dtype_name, len(tokenizer))

    @property
    def diagnostics(self):
        result = dict(self._diagnostics)
        if self.torch.cuda.is_available(): result["cuda_peak_allocated_bytes"] = self.torch.cuda.max_memory_allocated()
        return result

    def _tokenizer(self):
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None: raise RuntimeError("Processor does not expose a tokenizer.")
        return tokenizer

    def _input_device(self):
        for device in (getattr(self.model, "hf_device_map", None) or {}).values():
            if isinstance(device, int): return self.torch.device(f"cuda:{device}")
            if isinstance(device, str) and device not in {"cpu", "disk", "meta"}: return self.torch.device(device)
        return next(self.model.parameters()).device

    def _inputs(self, messages):
        kwargs = dict(tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)
        try: inputs = self.processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except (TypeError, ValueError): inputs = self.processor.apply_chat_template(messages, **kwargs)
        return inputs.to(self.device)

    def generate(self, messages, *, temperature, max_tokens, seed):
        inputs = self._inputs(messages); prompt = inputs["input_ids"].shape[1]
        kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            kwargs["temperature"] = temperature
            if seed is not None:
                generator = self.torch.Generator(device=self.device); generator.manual_seed(seed); kwargs["generator"] = generator
        with self.torch.inference_mode(): generated = self.model.generate(**inputs, **kwargs)
        new = generated[0, prompt:]
        return GenerationResult(self.processor.decode(new, skip_special_tokens=True), prompt, int(new.shape[0]))

    def score(self, messages, choices):
        inputs = self._inputs(messages); prompt_ids = inputs["input_ids"]; prompt = prompt_ids.shape[1]
        tokenizer = self._tokenizer(); all_ids=[]; scores=[]
        with self.torch.inference_mode():
            for choice in choices:
                ids = tokenizer.encode(choice, add_special_tokens=False)
                if not ids: raise ValueError(f"Choice {choice!r} tokenized to an empty sequence.")
                answer = self.torch.tensor([ids], dtype=prompt_ids.dtype, device=self.device)
                combined = self.torch.cat((prompt_ids, answer), dim=1)
                model_inputs = dict(inputs); model_inputs["input_ids"] = combined
                mask = inputs.get("attention_mask")
                model_inputs["attention_mask"] = self.torch.cat((mask, self.torch.ones_like(answer)), dim=1) if mask is not None else self.torch.ones_like(combined)
                logits = self.model(**model_inputs).logits[:, prompt - 1:combined.shape[1] - 1, :].float()
                if not self.torch.isfinite(logits).all(): raise FloatingPointError("Model returned non-finite logits.")
                score = self.torch.log_softmax(logits, -1).gather(-1, answer.unsqueeze(-1)).sum()
                if not self.torch.isfinite(score): raise FloatingPointError("Model returned a non-finite score.")
                all_ids.append(tuple(ids)); scores.append(float(score.item()))
        return ScoringResult(prompt, tuple(all_ids), tuple(scores))
