"""Stateless asynchronous clients for the University of Potsdam LLM proxy."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import os
import random
import threading
import time
import warnings
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import requests
from dotenv import load_dotenv

from .models import ConfigurationError, LLMResponse, TokenUsage
from .potsdam_network import ensure_windows_vpn_bridge

Message = dict[str, str]
ResponseFactory = Callable[[Sequence[Message]], str | Awaitable[str]]
RequestObserver = Callable[[Sequence[Message]], None | Awaitable[None]]
LatencyFactory = Callable[[Sequence[Message]], float]


class LLMClient(Protocol):
    model: str
    concurrency: int
    provider_name: str

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> LLMResponse: ...

    def close(self) -> None: ...


class LLMAPIError(RuntimeError):
    """A safe provider error that never includes authorization data."""


class _RequestStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.actual_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.retries = 0
        self.latencies: list[float] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.has_prompt_tokens = False
        self.has_completion_tokens = False
        self.has_total_tokens = False

    def attempt(self) -> None:
        with self._lock:
            self.actual_calls += 1

    def retry(self) -> None:
        with self._lock:
            self.retries += 1

    def failure(self) -> None:
        with self._lock:
            self.failed_calls += 1

    def success(self, latency: float, usage: TokenUsage) -> None:
        with self._lock:
            self.successful_calls += 1
            self.latencies.append(latency)
            if usage.prompt_tokens is not None:
                self.has_prompt_tokens = True
                self.prompt_tokens += usage.prompt_tokens
            if usage.completion_tokens is not None:
                self.has_completion_tokens = True
                self.completion_tokens += usage.completion_tokens
            if usage.total_tokens is not None:
                self.has_total_tokens = True
                self.total_tokens += usage.total_tokens

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "actual_calls": self.actual_calls,
                "successful_calls": self.successful_calls,
                "failed_calls": self.failed_calls,
                "retries": self.retries,
                "latencies": list(self.latencies),
                "prompt_tokens": self.prompt_tokens if self.has_prompt_tokens else None,
                "completion_tokens": (
                    self.completion_tokens if self.has_completion_tokens else None
                ),
                "total_tokens": self.total_tokens if self.has_total_tokens else None,
            }


def _find_repository_env(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    while not (current / ".env").exists() and current != current.parent:
        current = current.parent
    env_path = current / ".env"
    if not env_path.exists():
        raise ConfigurationError("Could not find the repository-root .env file.")
    return env_path


class AsyncLLMClient:
    """OpenAI-compatible asynchronous provider adapter.

    The client owns connection-pool and request-accounting state only. It never
    stores messages, conversations, agent IDs, or provider session IDs.
    """

    def __init__(
        self,
        *,
        model: str,
        concurrency: int = 20,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        api_key: str | None = None,
        base_url: str | None = None,
        env_path: Path | None = None,
        backoff_base_seconds: float = 0.5,
        provider_name: str = "university",
        allow_windows_proxy: bool = True,
    ) -> None:
        if concurrency < 1:
            raise ConfigurationError("concurrency must be at least 1.")
        if timeout_seconds <= 0:
            raise ConfigurationError("timeout_seconds must be positive.")
        if max_retries < 0:
            raise ConfigurationError("max_retries cannot be negative.")

        resolved_env_path: Path | None = None
        if api_key is None or base_url is None:
            resolved_env_path = env_path or _find_repository_env()
            load_dotenv(resolved_env_path)
        self._api_key = api_key or os.getenv("POTSDAM_API_KEY")
        configured_url = base_url or os.getenv("BASE_POTSDAM_LLM_URL")
        if not self._api_key:
            raise ConfigurationError("POTSDAM_API_KEY is not configured.")
        if not configured_url:
            raise ConfigurationError("BASE_POTSDAM_LLM_URL is not configured.")

        self.model = model
        self.provider_name = provider_name
        self.concurrency = concurrency
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self._base_url = configured_url.rstrip("/")
        self._chat_url: str | None = None
        self._available_models: tuple[str, ...] | None = None
        self._endpoint_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(concurrency)
        self._session = requests.Session()
        if allow_windows_proxy:
            proxy_url = ensure_windows_vpn_bridge(
                self._base_url,
                repository_root=(
                    resolved_env_path.parent if resolved_env_path is not None else None
                ),
            )
            if proxy_url is not None:
                self._session.proxies["https"] = proxy_url
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
        )
        self._stats = _RequestStats()
        self._jitter = random.Random()

    @property
    def stats(self) -> dict[str, Any]:
        return self._stats.snapshot()

    @property
    def available_models(self) -> tuple[str, ...] | None:
        return self._available_models

    async def validate_model(self) -> tuple[str, ...]:
        await self._ensure_endpoint_and_model()
        return self._available_models or ()

    async def _ensure_endpoint_and_model(self) -> None:
        if self._chat_url is not None:
            return
        async with self._endpoint_lock:
            if self._chat_url is not None:
                return

            models_url = f"{self._base_url}/models"
            response = await asyncio.to_thread(
                self._session.get, models_url, timeout=self.timeout_seconds
            )
            if response.status_code == 404:
                models_url = f"{self._base_url}/v1/models"
                response = await asyncio.to_thread(
                    self._session.get, models_url, timeout=self.timeout_seconds
                )
                chat_url = f"{self._base_url}/v1/chat/completions"
            else:
                chat_url = f"{self._base_url}/chat/completions"

            if response.status_code in (401, 403):
                raise LLMAPIError(
                    f"{self.provider_name} authentication failed with HTTP {response.status_code}."
                )
            try:
                response.raise_for_status()
                payload = response.json()
            except requests.RequestException as exc:
                raise LLMAPIError(
                    f"Could not list {self.provider_name} models (HTTP {response.status_code})."
                ) from exc
            except ValueError as exc:
                raise LLMAPIError(f"The {self.provider_name} model-list response was not valid JSON.") from exc

            entries = payload.get("data", []) if isinstance(payload, dict) else []
            available = sorted(
                entry["id"]
                for entry in entries
                if isinstance(entry, dict) and isinstance(entry.get("id"), str)
            )
            self._available_models = tuple(available)
            if self.model not in self._available_models:
                raise ConfigurationError(
                    f"Model {self.model!r} is not currently listed by the University LLM proxy."
                )
            self._chat_url = chat_url

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> LLMResponse:
        await self._ensure_endpoint_and_model()
        if not messages or any(set(message) != {"role", "content"} for message in messages):
            raise ValueError("Every request must provide complete role/content messages.")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive.")

        payload = {
            "model": self.model,
            "messages": copy.deepcopy(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed

        async with self._semaphore:
            started = time.perf_counter()
            for retry_index in range(self.max_retries + 1):
                self._stats.attempt()
                response: requests.Response | None = None
                try:
                    response = await asyncio.to_thread(
                        self._session.post,
                        self._chat_url,
                        json=payload,
                        timeout=self.timeout_seconds,
                    )
                    if response.status_code == 429 or 500 <= response.status_code < 600:
                        self._stats.failure()
                        if retry_index < self.max_retries:
                            self._stats.retry()
                            await asyncio.sleep(self._retry_delay(response, retry_index))
                            continue
                    response.raise_for_status()
                    body = response.json()
                    content = body["choices"][0]["message"]["content"]
                    if not isinstance(content, str):
                        raise KeyError("choices[0].message.content is not a string")
                    usage = TokenUsage.from_mapping(
                        body.get("usage") if isinstance(body, dict) else None
                    )
                    latency = time.perf_counter() - started
                    self._stats.success(latency, usage)
                    return LLMResponse(
                        content=content,
                        model=(
                            body.get("model", self.model)
                            if isinstance(body, dict)
                            else self.model
                        ),
                        latency_seconds=latency,
                        retries=retry_index,
                        status_code=response.status_code,
                        usage=usage,
                        raw_response=body,
                        finish_reason=(
                            body.get("choices", [{}])[0].get("finish_reason")
                            if isinstance(body, dict) else None
                        ),
                    )
                except (requests.Timeout, requests.ConnectionError) as exc:
                    self._stats.failure()
                    if retry_index < self.max_retries:
                        self._stats.retry()
                        await asyncio.sleep(self._retry_delay(response, retry_index))
                        continue
                    raise LLMAPIError(
                        f"{self.provider_name} request failed after bounded retries."
                    ) from exc
                except requests.HTTPError as exc:
                    # 400/401/403 and other permanent HTTP failures are not retried.
                    if response is not None and not (
                        response.status_code == 429 or response.status_code >= 500
                    ):
                        self._stats.failure()
                    status = response.status_code if response is not None else "unknown"
                    raise LLMAPIError(
                        f"{self.provider_name} chat request failed with HTTP {status}."
                    ) from exc
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    self._stats.failure()
                    raise LLMAPIError(
                        f"The {self.provider_name} response did not match the expected chat schema."
                    ) from exc

        raise AssertionError("unreachable")

    def _retry_delay(
        self, response: requests.Response | None, retry_index: int
    ) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            try:
                if retry_after is not None:
                    return max(0.0, float(retry_after))
            except ValueError:
                pass
        exponential = self.backoff_base_seconds * (2**retry_index)
        return exponential + self._jitter.uniform(0.0, exponential * 0.25)

    def close(self) -> None:
        self._session.close()


class OpenAIAsyncLLMClient(AsyncLLMClient):
    """Official OpenAI chat-completions provider using repository credentials."""

    def __init__(
        self,
        *,
        model: str,
        concurrency: int = 20,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        api_key: str | None = None,
        env_path: Path | None = None,
    ) -> None:
        if api_key is None:
            load_dotenv(env_path or _find_repository_env())
        configured_key = api_key or os.getenv("OPENAI_API_KEY")
        if configured_key is None and os.getenv("OPEN_API_KEY"):
            configured_key = os.getenv("OPEN_API_KEY")
            warnings.warn(
                "OPEN_API_KEY is deprecated; rename it to OPENAI_API_KEY.",
                DeprecationWarning,
                stacklevel=2,
            )
        if not configured_key:
            raise ConfigurationError("OPENAI_API_KEY is not configured.")
        super().__init__(
            model=model,
            concurrency=concurrency,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            api_key=configured_key,
            base_url="https://api.openai.com/v1",
            provider_name="openai",
        )


class MockAsyncLLMClient:
    """Deterministic, latency-configurable client used by tests and benchmarks."""

    def __init__(
        self,
        *,
        model: str = "mock/naming-game",
        concurrency: int = 20,
        artificial_latency: float | LatencyFactory = 0.001,
        seed: int = 1,
        response_factory: ResponseFactory | None = None,
        request_observer: RequestObserver | None = None,
    ) -> None:
        if concurrency < 1:
            raise ConfigurationError("concurrency must be at least 1.")
        self.model = model
        self.provider_name = "mock"
        self.concurrency = concurrency
        self.artificial_latency = artificial_latency
        self.seed = seed
        self.response_factory = response_factory
        self.request_observer = request_observer
        self._semaphore = asyncio.Semaphore(concurrency)
        self._stats = _RequestStats()
        self.active_requests = 0
        self.max_active_requests = 0

    @property
    def stats(self) -> dict[str, Any]:
        return self._stats.snapshot()

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> LLMResponse:
        del temperature, max_tokens, seed
        request_messages = copy.deepcopy(messages)
        if self.request_observer is not None:
            observed = self.request_observer(copy.deepcopy(request_messages))
            if inspect.isawaitable(observed):
                await observed

        async with self._semaphore:
            self.active_requests += 1
            self.max_active_requests = max(self.max_active_requests, self.active_requests)
            self._stats.attempt()
            started = time.perf_counter()
            try:
                latency = (
                    self.artificial_latency(request_messages)
                    if callable(self.artificial_latency)
                    else self.artificial_latency
                )
                if latency < 0:
                    raise ValueError("artificial_latency cannot be negative.")
                await asyncio.sleep(latency)
                if self.response_factory is None:
                    content = self._default_response(request_messages)
                else:
                    content = self.response_factory(request_messages)
                    if inspect.isawaitable(content):
                        content = await content
                if not isinstance(content, str):
                    raise TypeError("Mock response_factory must return a string.")
                elapsed = time.perf_counter() - started
                usage = TokenUsage()
                self._stats.success(elapsed, usage)
                return LLMResponse(
                    content=content,
                    model=self.model,
                    latency_seconds=elapsed,
                    status_code=200,
                    usage=usage,
                )
            except Exception:
                self._stats.failure()
                raise
            finally:
                self.active_requests -= 1

    def _default_response(self, messages: Sequence[Message]) -> str:
        user = messages[-1]["content"]
        fields: dict[str, str] = {}
        for line in user.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                fields[key.strip()] = value.strip()
        action = fields.get("ACTION")
        inventory = json.loads(fields.get("INVENTORY_JSON", "[]"))

        if action in {"speaker_basic", "speaker_reasoning"}:
            if len(inventory) == 1:
                selected = inventory[0]
            else:
                identity = fields.get("AGENT_ID", "")
                digest = hashlib.sha256(
                    f"{self.seed}:{identity}:{','.join(inventory)}".encode()
                ).digest()
                selected = ("A", "B")[digest[0] % 2]
            if action == "speaker_reasoning":
                return json.dumps(
                    {
                        "selected_name": selected,
                        "reason": f"Evidence supports option {selected}.",
                    }
                )
            return json.dumps({"selected_name": selected})

        if action == "listener_basic":
            transmitted = fields.get("TRANSMITTED_NAME")
            return json.dumps({"already_known": transmitted in inventory})

        if action == "listener_reasoning":
            # The mock keeps the listener state; reasoning behavior is tested as
            # an interface only in this benchmark.
            return json.dumps({"new_inventory": inventory})

        # The convention-game prompt intentionally omits agent and population
        # identifiers, so recognize its reference user instruction separately.
        if user.strip() == "Answer saying which action Player 1 should play.":
            system = messages[0]["content"]
            marker = "from the following values: "
            action_line = next(
                (line for line in system.splitlines() if marker in line), None
            )
            if action_line is not None:
                encoded_actions = action_line.split(marker, 1)[1].removesuffix(".")
                actions = json.loads(encoded_actions)
                if isinstance(actions, list) and all(
                    isinstance(item, str) for item in actions
                ):
                    digest = hashlib.sha256(
                        f"{self.seed}:{system}".encode()
                    ).digest()
                    selected = actions[digest[0] % len(actions)]
                    return json.dumps(
                        {"value": selected, "reason": "Seeded mock convention choice."}
                    )

        raise ValueError("Mock request did not contain a recognized ACTION marker.")

    def close(self) -> None:
        return None
