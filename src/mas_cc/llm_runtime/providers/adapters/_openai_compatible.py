"""Shared, bounded-retry transport for OpenAI-compatible chat endpoints."""

from __future__ import annotations

import asyncio
import copy
import os
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mas_cc.llm_runtime.config import LLMProviderConfig
from ..errors import ProviderError

from ..capabilities import ProviderCapabilities
from ..requests import CompletionRequest
from ..responses import CompletionResponse, ProviderUsage
from ..load_control import SharedProviderCoordinator


_OMIT_TEMPERATURE_METADATA_KEY = "_llm_runtime_omit_temperature"


def _load_dotenv_if_available() -> None:
    """Load a repository .env lazily; process environment keeps precedence."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    current = Path.cwd().resolve()
    for root in (current, *current.parents):
        candidate = root / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return


class OpenAICompatibleProvider:
    """Stateless chat-completions adapter with no game or conversation state."""

    name = "openai_compatible"
    capabilities = ProviderCapabilities(
        supports_seed=True,
        reports_usage=True,
        supports_parallel_requests=True,
    )

    def __init__(
        self,
        config: LLMProviderConfig,
        *,
        provider_name: str,
        default_credentials_env: str,
        fixed_base_url: str | None = None,
        default_base_url_env: str | None = None,
        discover_endpoint: bool = False,
        environment: Mapping[str, str] | None = None,
        session: Any | None = None,
        request_coordinator: SharedProviderCoordinator | None = None,
    ) -> None:
        if environment is None:
            _load_dotenv_if_available()
            environment = os.environ
        credentials_env = config.credentials_env or default_credentials_env
        key = environment.get(credentials_env, "").strip()
        if not key:
            raise ProviderError(
                f"{provider_name} credentials are not configured in {credentials_env}.",
                provider=provider_name,
                code="configuration_error",
            )
        if fixed_base_url is None:
            base_env = config.base_url_env or default_base_url_env
            base_url = environment.get(base_env or "", "").strip()
            if not base_url:
                raise ProviderError(
                    f"{provider_name} base URL is not configured in {base_env}.",
                    provider=provider_name,
                    code="configuration_error",
                )
        else:
            base_url = fixed_base_url

        self.name = provider_name
        self.model = config.model
        self._key = key
        self._base_url = base_url.rstrip("/")
        self._timeout = config.timeout_seconds
        self._max_retries = config.max_retries
        self._concurrency = config.request_concurrency
        self._discover_endpoint = discover_endpoint
        self._chat_url: str | None = None if discover_endpoint else f"{self._base_url}/chat/completions"
        self._available_models: tuple[str, ...] | None = None
        self._endpoint_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self._concurrency)
        self._session = session
        self._request_coordinator = request_coordinator
        self._closed = False
        self._jitter = random.Random()

    def _get_session(self):
        if self._session is None:
            try:
                import requests
            except ImportError as exc:
                raise ProviderError(
                    "The requests dependency is required for remote providers.",
                    provider=self.name,
                    code="missing_dependency",
                ) from exc
            self._session = requests.Session()
        return self._session

    async def _coordinated_get(self, url: str, **kwargs: Any) -> Any:
        lease = None
        started = time.perf_counter()
        try:
            if self._request_coordinator is not None:
                lease = await self._request_coordinator.acquire()
            response = await asyncio.to_thread(self._get_session().get, url, **kwargs)
            if lease is not None:
                retryable = self._is_retryable(response.status_code)
                await self._request_coordinator.release(
                    lease, success=not retryable, retryable=retryable,
                    status_code=response.status_code,
                    latency_seconds=time.perf_counter() - started,
                )
                lease = None
            return response
        except Exception:
            if lease is not None:
                await self._request_coordinator.release(
                    lease, success=False, retryable=True, status_code=None,
                    latency_seconds=time.perf_counter() - started,
                )
            raise

    async def discover_models(self) -> tuple[str, ...]:
        """Return advertised model ids while sharing normal endpoint discovery."""

        if self._available_models is not None:
            return self._available_models
        async with self._endpoint_lock:
            if self._available_models is not None:
                return self._available_models
            url = f"{self._base_url}/models"
            try:
                response = await self._coordinated_get(
                    url,
                    headers={"Authorization": f"Bearer {self._key}"},
                    timeout=self._timeout,
                )
                if response.status_code == 404:
                    response = await self._coordinated_get(
                        f"{self._base_url}/v1/models",
                        headers={"Authorization": f"Bearer {self._key}"},
                        timeout=self._timeout,
                    )
                    prefix = f"{self._base_url}/v1"
                else:
                    prefix = self._base_url
                if response.status_code in (401, 403):
                    raise ProviderError(
                        f"{self.name} authentication failed with HTTP {response.status_code}.",
                        provider=self.name,
                        code="authentication_failed",
                        status_code=response.status_code,
                    )
                response.raise_for_status()
                body = response.json()
                entries = body.get("data", []) if isinstance(body, Mapping) else []
                models = {
                    item.get("id") for item in entries if isinstance(item, Mapping)
                }
                self._chat_url = f"{prefix}/chat/completions"
                self._available_models = tuple(
                    sorted(item for item in models if isinstance(item, str) and item)
                )
                return self._available_models
            except ProviderError:
                raise
            except Exception as exc:
                raise self._normalize_transport_error(
                    exc, operation="model discovery"
                ) from exc

    async def _ensure_endpoint(self) -> None:
        if not self._discover_endpoint:
            return
        models = await self.discover_models()
        if self.model not in models:
            raise ProviderError(
                f"Model {self.model!r} is not listed by {self.name}.",
                provider=self.name,
                code="model_unavailable",
            )

    def _raise_if_reasoning_exhausted(
        self, choice: Mapping[str, Any], request: CompletionRequest, status_code: int
    ) -> None:
        """Name the one non-transient cause of an empty `content`.

        Reasoning models (gpt-oss, and anything else that returns a separate
        `reasoning_content`) charge their chain of thought against the same
        `max_tokens` as the answer. When the budget runs out inside the
        reasoning, the body is a well-formed envelope whose `content` is empty -
        indistinguishable, to the schema check, from a truncated proxy reply.

        The difference is that this one is deterministic: every retry burns
        another paid request and fails identically. So it is reported
        immediately, as itself, with the setting that fixes it.
        """

        message = choice.get("message")
        if not isinstance(message, Mapping):
            return
        if choice.get("finish_reason") != "length":
            return
        if not message.get("reasoning_content") and not message.get("reasoning"):
            return
        raise ProviderError(
            f"{self.model!r} spent its whole {request.max_output_tokens}-token "
            "output budget on reasoning and returned no answer. Raise "
            "llm_provider.max_output_tokens; retrying cannot help.",
            provider=self.name,
            code="reasoning_budget_exhausted",
            retryable=False,
            status_code=status_code,
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        if self._closed:
            raise ProviderError(
                f"{self.name} provider is closed.", provider=self.name, code="closed"
            )
        await self._ensure_endpoint()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": copy.deepcopy(request.wire_messages()),
            "max_tokens": request.max_output_tokens,
        }
        if not request.metadata.get(_OMIT_TEMPERATURE_METADATA_KEY, False):
            payload["temperature"] = request.temperature
        if request.seed is not None:
            payload["seed"] = request.seed
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        async with self._semaphore:
            retry = 0
            while True:
                response = None
                lease = None
                attempt_started = time.perf_counter()
                try:
                    if self._request_coordinator is not None:
                        lease = await self._request_coordinator.acquire()
                    response = await asyncio.to_thread(
                        self._get_session().post,
                        self._chat_url,
                        headers=headers,
                        json=payload,
                        timeout=self._timeout,
                    )
                    if self._is_retryable(response.status_code) and self._can_retry(
                        retry, started
                    ):
                        if lease is not None:
                            await self._request_coordinator.release(
                                lease, success=False, retryable=True,
                                status_code=response.status_code,
                                latency_seconds=time.perf_counter() - attempt_started,
                            )
                            lease = None
                        await asyncio.sleep(self._retry_delay(response, retry))
                        retry += 1
                        continue
                    response.raise_for_status()
                    body = response.json()
                    choice = body["choices"][0]
                    content = choice["message"]["content"]
                    if not isinstance(content, str):
                        self._raise_if_reasoning_exhausted(
                            choice, request, response.status_code
                        )
                        raise TypeError("message content is not a string")
                    if lease is not None:
                        await self._request_coordinator.release(
                            lease, success=True, retryable=False,
                            status_code=response.status_code,
                            latency_seconds=time.perf_counter() - attempt_started,
                        )
                        lease = None
                    latency = time.perf_counter() - started
                    return CompletionResponse(
                        content=content,
                        provider=self.name,
                        model=str(body.get("model") or self.model),
                        usage=ProviderUsage.from_mapping(body.get("usage")),
                        finish_reason=choice.get("finish_reason"),
                        request_id=body.get("id"),
                        latency_seconds=latency,
                        inference_seconds=latency,
                        retries=retry,
                        status_code=response.status_code,
                        raw_response=body,
                    )
                except ProviderError:
                    # Already diagnosed and already normalized - in particular
                    # the exhausted reasoning budget above, which must not be
                    # retried into three identical paid failures.
                    if lease is not None:
                        status = getattr(response, "status_code", None)
                        await self._request_coordinator.release(
                            lease, success=False, retryable=self._is_retryable(status),
                            status_code=status,
                            latency_seconds=time.perf_counter() - attempt_started,
                        )
                    raise
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    # Shared OpenAI-compatible proxies occasionally return an
                    # HTTP-200 body that is not a complete Chat Completions
                    # envelope. Never accept that body, but treat it like the
                    # transient upstream fault it is and spend the configured
                    # bounded retries before failing the logical request.
                    if lease is not None:
                        await self._request_coordinator.release(
                            lease, success=False, retryable=True,
                            status_code=getattr(response, "status_code", None),
                            latency_seconds=time.perf_counter() - attempt_started,
                        )
                        lease = None
                    if self._can_retry(retry, started):
                        await asyncio.sleep(self._retry_delay(response, retry))
                        retry += 1
                        continue
                    raise ProviderError(
                        f"The {self.name} response did not match the chat-completions schema.",
                        provider=self.name,
                        code="invalid_response",
                        retryable=True,
                        status_code=getattr(response, "status_code", None),
                    ) from exc
                except Exception as exc:
                    status = getattr(response, "status_code", None)
                    if lease is not None:
                        await self._request_coordinator.release(
                            lease, success=False, retryable=self._is_retryable(status),
                            status_code=status,
                            latency_seconds=time.perf_counter() - attempt_started,
                        )
                        lease = None
                    # `status is None` means the request never produced a
                    # response at all: a connect or read timeout, a dropped
                    # connection, a VPN blip.  That is the *most* common
                    # failure against a shared university proxy and the most
                    # obviously transient, yet it used to be the one case that
                    # skipped the retry loop entirely - so a configured
                    # `max_retries: 2` silently bought nothing, and one slow
                    # generation killed a whole episode.
                    if self._is_retryable(status) and self._can_retry(retry, started):
                        await asyncio.sleep(self._retry_delay(response, retry))
                        retry += 1
                        continue
                    raise self._normalize_transport_error(
                        exc, operation="completion", status_code=status
                    ) from exc
        raise AssertionError("unreachable")

    @staticmethod
    def _is_retryable(status: int | None) -> bool:
        """No response at all, rate limiting, or a server fault."""

        return status is None or status == 429 or status >= 500

    def _can_retry(self, retry: int, started: float) -> bool:
        """Keep coordinated logical requests alive through provider outages."""

        if retry < self._max_retries:
            return True
        if self._request_coordinator is None:
            return False
        return (
            time.perf_counter() - started
            < self._request_coordinator.config.retry_max_elapsed_seconds
        )

    def _normalize_transport_error(
        self, exc: Exception, *, operation: str, status_code: int | None = None
    ) -> ProviderError:
        status = status_code if isinstance(status_code, int) else None
        retryable = self._is_retryable(status)
        if status in (401, 403):
            code = "authentication_failed"
        elif status == 429:
            code = "rate_limited"
        elif status is not None:
            code = "http_error"
        else:
            code = "connection_error"
        suffix = f" with HTTP {status}" if status is not None else ""
        return ProviderError(
            f"{self.name} {operation} failed{suffix}.",
            provider=self.name,
            code=code,
            retryable=retryable,
            status_code=status,
        )

    def _retry_delay(self, response: Any, retry: int) -> float:
        retry_after = getattr(response, "headers", {}).get("Retry-After")
        try:
            if retry_after is not None:
                return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            pass
        # Full jitter prevents synchronized SLURM workers from retrying a
        # degraded shared endpoint in the same wave. Coordinated runs use a
        # long capped schedule; standalone callers retain bounded retries.
        if self._request_coordinator is None:
            initial, maximum = 2.0, 60.0
        else:
            policy = self._request_coordinator.config
            initial = policy.retry_backoff_initial_seconds
            maximum = policy.retry_backoff_max_seconds
        exponent = min(retry, 30)
        return self._jitter.uniform(0.0, min(maximum, initial * (2**exponent)))

    def close(self) -> None:
        self._closed = True
        if self._session is not None:
            self._session.close()
