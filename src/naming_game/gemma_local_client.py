"""Lazy, serialized local Gemma 4 client.

Importing this module never imports torch or Transformers.  The small runtime
protocol is intentionally injectable so all application behavior can be tested
without a checkpoint or accelerator.
"""

from __future__ import annotations

import asyncio
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
from .local_model_types import (
    ChoiceScore,
    ChoiceSelectionPolicy,
    ConstrainedDecisionResponse,
    ConstrainedLLMResponse,
    TextDecisionOutputFormat,
)
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
    # Actual prompt-token work performed by scoring.  A one-token fast path
    # evaluates the prompt once; the teacher-forced fallback evaluates it once
    # per candidate.  ``None`` preserves the historical fake-runtime contract.
    prompt_work_tokens: int | None = None


@dataclass(frozen=True)
class RuntimeDecisionResult:
    scoring: ScoringResult
    selected_index: int
    reason: str | None = None
    reason_valid: bool | None = None
    reason_prompt_tokens: int = 0
    reason_tokens: int = 0


class GemmaRuntime(Protocol):
    def generate(self, messages: list[dict[str, str]], *, temperature: float,
                 max_tokens: int, seed: int | None) -> GenerationResult: ...
    def score(self, messages: list[dict[str, str]], choices: Sequence[str]) -> ScoringResult: ...
    def decide(
        self, messages: list[dict[str, str]], choices: Sequence[str], *,
        output_format: str, choice_temperature: float, selection_policy: str,
        generation_temperature: float, max_reason_tokens: int,
        seed: int | None,
    ) -> RuntimeDecisionResult: ...
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


def _validated_choices(choices: Sequence[str]) -> tuple[str, ...]:
    choice_tuple = tuple(choices)
    if not choice_tuple or any(
        not isinstance(choice, str) or not choice.strip() for choice in choice_tuple
    ):
        raise ValueError("choices must contain non-blank strings.")
    if len(set(choice_tuple)) != len(choice_tuple):
        raise ValueError("choices must not contain duplicates.")
    return choice_tuple


def _validate_scoring_result(result: ScoringResult, choice_count: int) -> None:
    if result.prompt_tokens < 0:
        raise RuntimeError("Runtime returned an invalid prompt token count.")
    if result.prompt_work_tokens is not None and result.prompt_work_tokens < 0:
        raise RuntimeError("Runtime returned an invalid prompt-work token count.")
    if len(result.token_ids) != choice_count or len(result.log_likelihoods) != choice_count:
        raise RuntimeError("Runtime returned the wrong number of candidate scores.")
    if any(not token_ids for token_ids in result.token_ids):
        raise ValueError("A choice tokenized to an empty sequence.")
    if any(
        not isinstance(token_id, int)
        for token_ids in result.token_ids
        for token_id in token_ids
    ):
        raise ValueError("Runtime returned a non-integer candidate token ID.")
    if any(not math.isfinite(value) for value in result.log_likelihoods):
        raise FloatingPointError("Runtime returned a non-finite sequence score.")


def _normalized_probabilities(
    log_likelihoods: Sequence[float], temperature: float
) -> tuple[float, ...]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("choice temperature must be finite and positive.")
    scaled = [value / temperature for value in log_likelihoods]
    peak = max(scaled)
    weights = [math.exp(value - peak) for value in scaled]
    total = math.fsum(weights)
    probabilities = tuple(weight / total for weight in weights)
    if (
        not probabilities
        or not all(math.isfinite(probability) for probability in probabilities)
        or abs(math.fsum(probabilities) - 1.0) > 1e-12
    ):
        raise FloatingPointError("Could not normalize finite choice probabilities.")
    return probabilities


def _selected_index(
    probabilities: Sequence[float], policy: str, seed: int | None
) -> int:
    if policy == "argmax":
        # Python's max keeps the first exact maximum, preserving displayed order.
        return max(range(len(probabilities)), key=probabilities.__getitem__)
    if policy == "sample":
        return random.Random(seed).choices(
            range(len(probabilities)), weights=probabilities, k=1
        )[0]
    raise ValueError("selection_policy must be argmax or sample.")


def _choice_scores(
    choices: Sequence[str], result: ScoringResult, temperature: float
) -> tuple[ChoiceScore, ...]:
    _validate_scoring_result(result, len(choices))
    probabilities = _normalized_probabilities(result.log_likelihoods, temperature)
    return tuple(
        ChoiceScore(choice, tuple(token_ids), log_likelihood, probability)
        for choice, token_ids, log_likelihood, probability in zip(
            choices, result.token_ids, result.log_likelihoods, probabilities
        )
    )


def _prompt_work(result: ScoringResult, choice_count: int) -> int:
    return (
        result.prompt_work_tokens
        if result.prompt_work_tokens is not None
        else result.prompt_tokens * choice_count
    )


def _validated_reason(value: str | None, valid: bool | None) -> tuple[str | None, bool]:
    reason = value.strip() if isinstance(value, str) else ""
    # A runtime may explicitly mark checkpoint/template output malformed.  The
    # choice remains authoritative and the malformed rationale is discarded.
    reason_valid = bool(reason) if valid is None else bool(valid and reason)
    return (reason if reason_valid else None), reason_valid


def _dispatch_candidate_scoring(
    token_ids: Sequence[tuple[int, ...]],
    *,
    score_single_tokens: Callable[[tuple[int, ...]], Sequence[float]],
    score_sequence: Callable[[tuple[int, ...]], float],
) -> tuple[tuple[float, ...], str]:
    """Dispatch to one batched next-token score or full sequence fallback.

    This dependency-free boundary lets CPU fakes prove dispatch and call counts;
    it deliberately makes no claim about real Gemma boundary tokenization.
    """

    candidate_ids = tuple(tuple(ids) for ids in token_ids)
    if candidate_ids and all(len(ids) == 1 for ids in candidate_ids):
        scores = tuple(float(value) for value in score_single_tokens(
            tuple(ids[0] for ids in candidate_ids)
        ))
        if len(scores) != len(candidate_ids):
            raise RuntimeError("Single-token scorer returned the wrong number of scores.")
        return scores, "single_token"
    return tuple(float(score_sequence(ids)) for ids in candidate_ids), "sequence"


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
                    construction = asyncio.create_task(asyncio.to_thread(self._factory))
                    try:
                        self._runtime = await asyncio.shield(construction)
                    except asyncio.CancelledError:
                        # Do not release the initialization lock while a loader
                        # thread is still running; retain its initialized runtime.
                        self._runtime = await construction
                        raise
        return self._runtime

    @staticmethod
    async def _run_blocking(operation: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        task = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # asyncio cannot stop a worker thread.  Await it before allowing the
            # inference semaphore to be released, then preserve cancellation.
            await task
            raise

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
                result = await self._run_blocking(
                    runtime.generate, messages, temperature=temperature,
                    max_tokens=max_tokens, seed=seed
                )
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
        choice_tuple = _validated_choices(choices)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("constrained temperature must be finite and positive.")
        self._stats.attempt(); started = time.perf_counter()
        try:
            async with self._semaphore:
                result = await self._run_blocking(
                    (await self._get_runtime()).score, messages, choice_tuple
                )
            scores = _choice_scores(choice_tuple, result, temperature)
            selected = _selected_index(
                tuple(score.probability for score in scores), "argmax", None
            )
            completion = sum(len(ids) for ids in result.token_ids)
            prompt_work = _prompt_work(result, len(choice_tuple))
            usage = TokenUsage(prompt_work, completion, prompt_work + completion)
            latency = time.perf_counter() - started
            response = ConstrainedLLMResponse(scores[selected].choice, scores, self.model, latency, usage, temperature)
            self._stats.success(latency, usage); return response
        except BaseException:
            self._stats.failure(); raise

    async def complete_decision(
        self, messages: list[dict[str, str]], *, choices: Sequence[str],
        output_format: TextDecisionOutputFormat, choice_temperature: float = 1.0,
        selection_policy: ChoiceSelectionPolicy = "argmax",
        generation_temperature: float = 0.0,
        max_reason_tokens: int = 32, seed: int | None = None,
    ) -> ConstrainedDecisionResponse:
        """Score, select, and optionally explain in one logical request.

        ``usage`` reports scoring prompt work plus candidate and generated reason
        tokens; it is not a claim that every counted token was autoregressively emitted.
        """
        _validate_messages(messages)
        choice_tuple = _validated_choices(choices)
        if output_format not in {"choice_reason", "choice_only"}:
            raise ValueError("output_format must be choice_reason or choice_only.")
        if selection_policy not in {"argmax", "sample"}:
            raise ValueError("selection_policy must be argmax or sample.")
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
                decide = getattr(runtime, "decide", None)
                if callable(decide):
                    runtime_decision = await self._run_blocking(
                        decide,
                        messages,
                        choice_tuple,
                        output_format=output_format,
                        choice_temperature=choice_temperature,
                        selection_policy=selection_policy,
                        generation_temperature=generation_temperature,
                        max_reason_tokens=max_reason_tokens,
                        seed=seed,
                    )
                else:
                    # Compatibility for injected phase-one runtimes.  The real
                    # runtime and combined-contract fakes use ``decide`` so the
                    # exact authoritative prefix is auditable in one operation.
                    scored = await self._run_blocking(runtime.score, messages, choice_tuple)
                    scores = _choice_scores(choice_tuple, scored, choice_temperature)
                    selected_index = _selected_index(
                        tuple(score.probability for score in scores), selection_policy, seed
                    )
                    reason = None
                    reason_valid = None
                    reason_prompt_tokens = reason_tokens = 0
                    if output_format == "choice_reason":
                        prefix = f"{choice_tuple[selected_index]}\nReason: "
                        generate_reason = getattr(runtime, "generate_reason", None)
                        if callable(generate_reason):
                            generated = await self._run_blocking(
                                generate_reason,
                                messages,
                                prefix=prefix,
                                temperature=generation_temperature,
                                max_tokens=max_reason_tokens,
                                seed=seed,
                            )
                        else:
                            generated = await self._run_blocking(
                                runtime.generate,
                                messages + [{"role": "assistant", "content": prefix}],
                                temperature=generation_temperature,
                                max_tokens=max_reason_tokens,
                                seed=seed,
                            )
                        reason = generated.content
                        reason_prompt_tokens = generated.prompt_tokens
                        reason_tokens = generated.completion_tokens
                    runtime_decision = RuntimeDecisionResult(
                        scored,
                        selected_index,
                        reason,
                        reason_valid,
                        reason_prompt_tokens,
                        reason_tokens,
                    )
                scored = runtime_decision.scoring
                scores = _choice_scores(choice_tuple, scored, choice_temperature)
                selected_index = runtime_decision.selected_index
                if (
                    not isinstance(selected_index, int)
                    or isinstance(selected_index, bool)
                    or not 0 <= selected_index < len(scores)
                ):
                    raise RuntimeError("Runtime returned an invalid selected-choice index.")
                expected_index = (
                    _selected_index(
                        tuple(score.probability for score in scores),
                        selection_policy,
                        seed,
                    )
                    if selection_policy == "argmax" or seed is not None
                    else None
                )
                if expected_index is not None and selected_index != expected_index:
                    raise RuntimeError("Runtime selected a choice inconsistent with the requested policy.")
                selected = scores[selected_index].choice
                if output_format == "choice_reason":
                    reason, reason_valid = _validated_reason(
                        runtime_decision.reason, runtime_decision.reason_valid
                    )
                    if runtime_decision.reason_prompt_tokens < 0 or runtime_decision.reason_tokens < 0:
                        raise RuntimeError("Runtime returned invalid reason token counts.")
                    content = f"{selected}\nReason: {reason or ''}"
                else:
                    if (
                        runtime_decision.reason is not None
                        or runtime_decision.reason_tokens
                        or runtime_decision.reason_prompt_tokens
                    ):
                        raise RuntimeError("choice_only runtime generated an unexpected rationale.")
                    reason = None
                    reason_valid = None
                    content = selected
            candidate_tokens = sum(len(ids) for ids in scored.token_ids)
            prompt_work = (
                _prompt_work(scored, len(choice_tuple))
                + runtime_decision.reason_prompt_tokens
            )
            completion_work = candidate_tokens + runtime_decision.reason_tokens
            usage = TokenUsage(
                prompt_work, completion_work, prompt_work + completion_work
            )
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
        self._generation_calls = 0
        self._reason_generation_calls = 0
        self._score_calls = 0
        self._single_token_score_calls = 0
        self._sequence_score_calls = 0
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
        result.update(
            generation_calls=self._generation_calls,
            reason_generation_calls=self._reason_generation_calls,
            score_calls=self._score_calls,
            single_token_score_calls=self._single_token_score_calls,
            sequence_score_calls=self._sequence_score_calls,
        )
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
        self._generation_calls += 1
        inputs = self._inputs(messages)
        return self._generate_from_inputs(
            inputs, temperature=temperature, max_tokens=max_tokens, seed=seed
        )

    def _generate_from_inputs(self, inputs, *, temperature, max_tokens, seed):
        prompt = inputs["input_ids"].shape[1]
        kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            kwargs["temperature"] = temperature
            if seed is not None:
                generator = self.torch.Generator(device=self.device); generator.manual_seed(seed); kwargs["generator"] = generator
        with self.torch.inference_mode(): generated = self.model.generate(**inputs, **kwargs)
        new = generated[0, prompt:]
        return GenerationResult(self.processor.decode(new, skip_special_tokens=True), prompt, int(new.shape[0]))

    def generate_reason(self, messages, *, prefix, temperature, max_tokens, seed):
        """Continue after the exact authoritative ``choice\nReason: `` prefix."""

        self._reason_generation_calls += 1
        inputs = self._inputs(messages)
        prompt_ids = inputs["input_ids"]
        prefix_ids = self._tokenizer().encode(prefix, add_special_tokens=False)
        if not prefix_ids:
            raise ValueError("Authoritative reason prefix tokenized to an empty sequence.")
        prefix_tensor = self.torch.tensor(
            [prefix_ids], dtype=prompt_ids.dtype, device=self.device
        )
        combined = self.torch.cat((prompt_ids, prefix_tensor), dim=1)
        model_inputs = dict(inputs)
        model_inputs["input_ids"] = combined
        mask = inputs.get("attention_mask")
        model_inputs["attention_mask"] = (
            self.torch.cat((mask, self.torch.ones_like(prefix_tensor)), dim=1)
            if mask is not None
            else self.torch.ones_like(combined)
        )
        return self._generate_from_inputs(
            model_inputs,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )

    def score(self, messages, choices):
        # The chat template and assistant boundary are resolved exactly once per
        # decision.  Real boundary renderings remain a GPU validation question.
        self._score_calls += 1
        inputs = self._inputs(messages)
        prompt_ids = inputs["input_ids"]
        prompt = prompt_ids.shape[1]
        tokenizer = self._tokenizer()
        all_ids = tuple(
            tuple(tokenizer.encode(choice, add_special_tokens=False))
            for choice in choices
        )
        for choice, ids in zip(choices, all_ids):
            if not ids:
                raise ValueError(f"Choice {choice!r} tokenized to an empty sequence.")

        def score_single_tokens(ids: tuple[int, ...]) -> tuple[float, ...]:
            with self.torch.inference_mode():
                logits = self.model(**inputs).logits[:, -1, :].float()
                if not self.torch.isfinite(logits).all():
                    raise FloatingPointError("Model returned non-finite logits.")
                log_probs = self.torch.log_softmax(logits, -1)[0]
                values = log_probs[
                    self.torch.tensor(ids, dtype=self.torch.long, device=self.device)
                ]
                return tuple(float(value.item()) for value in values)

        def score_sequence(ids: tuple[int, ...]) -> float:
            with self.torch.inference_mode():
                answer = self.torch.tensor([ids], dtype=prompt_ids.dtype, device=self.device)
                combined = self.torch.cat((prompt_ids, answer), dim=1)
                model_inputs = dict(inputs); model_inputs["input_ids"] = combined
                mask = inputs.get("attention_mask")
                model_inputs["attention_mask"] = self.torch.cat((mask, self.torch.ones_like(answer)), dim=1) if mask is not None else self.torch.ones_like(combined)
                logits = self.model(**model_inputs).logits[:, prompt - 1:combined.shape[1] - 1, :].float()
                if not self.torch.isfinite(logits).all(): raise FloatingPointError("Model returned non-finite logits.")
                score = self.torch.log_softmax(logits, -1).gather(-1, answer.unsqueeze(-1)).sum()
                if not self.torch.isfinite(score): raise FloatingPointError("Model returned a non-finite score.")
                return float(score.item())

        scores, path = _dispatch_candidate_scoring(
            all_ids,
            score_single_tokens=score_single_tokens,
            score_sequence=score_sequence,
        )
        if path == "single_token":
            self._single_token_score_calls += 1
        else:
            self._sequence_score_calls += 1
        prompt_work = prompt if path == "single_token" else prompt * len(all_ids)
        return ScoringResult(prompt, all_ids, scores, prompt_work)

    def decide(
        self, messages, choices, *, output_format, choice_temperature,
        selection_policy, generation_temperature, max_reason_tokens, seed,
    ):
        scored = self.score(messages, choices)
        probabilities = _normalized_probabilities(
            scored.log_likelihoods, choice_temperature
        )
        selected_index = _selected_index(probabilities, selection_policy, seed)
        if output_format == "choice_only":
            return RuntimeDecisionResult(scored, selected_index)
        selected = choices[selected_index]
        generated = self.generate_reason(
            messages,
            prefix=f"{selected}\nReason: ",
            temperature=generation_temperature,
            max_tokens=max_reason_tokens,
            seed=seed,
        )
        reason, valid = _validated_reason(generated.content, None)
        return RuntimeDecisionResult(
            scored,
            selected_index,
            reason,
            valid,
            generated.prompt_tokens,
            generated.completion_tokens,
        )
