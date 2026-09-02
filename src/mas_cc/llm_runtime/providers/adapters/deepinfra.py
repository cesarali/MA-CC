"""DeepInfra OpenAI-compatible chat-completions adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from mas_cc.llm_runtime.config import LLMProviderConfig

from ._openai_compatible import OpenAICompatibleProvider
from ..errors import ProviderError
from ..load_control import SharedProviderCoordinator


_JSON_OBJECT_UNSUPPORTED_MODELS = frozenset({"google/gemma-4-E4B-it"})


@dataclass(frozen=True, slots=True)
class DeepInfraAccountLimits:
    """Authenticated per-model capacity reported by DeepInfra."""

    maximum_concurrent_requests: int
    tokens_per_minute: int


class DeepInfraProvider(OpenAICompatibleProvider):
    """Use DeepInfra with isolated routing, credentials, and JSON defaults."""

    _ACCOUNT_LIMITS_URL = "https://api.deepinfra.com/v1/me/rate_limit"
    _DEFAULT_BASE_URL = "https://api.deepinfra.com/v1/openai"
    _MODEL_LIST_URL = "https://api.deepinfra.com/v1/models"

    def __init__(
        self,
        config: LLMProviderConfig,
        *,
        environment: Mapping[str, str] | None = None,
        session: Any | None = None,
        request_coordinator: SharedProviderCoordinator | None = None,
    ) -> None:
        options = dict(config.options)
        if (
            config.model in _JSON_OBJECT_UNSUPPORTED_MODELS
            and options.get("response_format") is not None
        ):
            raise ProviderError(
                f"DeepInfra model {config.model} does not support json_object "
                "response format.",
                provider="deepinfra",
                code="configuration_error",
                retryable=False,
            )
        if "response_format" not in options and "structured_output_tool" not in options:
            # DeepInfra defaults to its fast JSON-object protocol. Keep the
            # exception provider-owned and model-specific: the live E4B
            # contract returned HTTP 405 when response_format was present.
            options["response_format"] = (
                None
                if config.model in _JSON_OBJECT_UNSUPPORTED_MODELS
                else {"type": "json_object"}
            )
            config = replace(config, options=options)
        super().__init__(
            config,
            provider_name="deepinfra",
            default_credentials_env="DEEPINFRA_API_KEY",
            default_base_url_env="DEEPINFRA_BASE_URL",
            fallback_base_url=self._DEFAULT_BASE_URL,
            # DeepInfra's OpenAI-compatible chat route is below /v1/openai,
            # while its model catalogue is exposed separately at /v1/models.
            # The shared transport validates the exact configured model while
            # retaining the separate fixed chat route.
            validate_model=True,
            model_list_url=self._MODEL_LIST_URL,
            environment=environment,
            session=session,
            request_coordinator=request_coordinator,
        )

    async def discover_account_limits(self) -> DeepInfraAccountLimits:
        """Return the authenticated per-model concurrency and TPM ceilings."""

        response = None
        try:
            response = await self._coordinated_get(
                self._ACCOUNT_LIMITS_URL,
                headers={"Authorization": f"Bearer {self._key}"},
                timeout=self._timeout,
            )
            if response.status_code in (401, 403):
                raise ProviderError(
                    f"{self.name} authentication failed with HTTP "
                    f"{response.status_code}.",
                    provider=self.name,
                    code="authentication_failed",
                    retryable=False,
                    status_code=response.status_code,
                )
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, Mapping):
                raise TypeError("account-limit response is not an object")
            concurrency = body.get("rate_limit")
            tokens_per_minute = body.get("tpm_rate_limit")
            if (
                isinstance(concurrency, bool)
                or not isinstance(concurrency, int)
                or concurrency < 1
                or isinstance(tokens_per_minute, bool)
                or not isinstance(tokens_per_minute, int)
                or tokens_per_minute < 1
            ):
                raise TypeError("account-limit response has invalid limits")
            return DeepInfraAccountLimits(concurrency, tokens_per_minute)
        except ProviderError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise ProviderError(
                "The deepinfra account-limit response did not match its schema.",
                provider=self.name,
                code="invalid_response",
                retryable=True,
                status_code=getattr(response, "status_code", None),
            ) from exc
        except Exception as exc:
            raise self._normalize_transport_error(
                exc,
                operation="account limit lookup",
                status_code=getattr(response, "status_code", None),
            ) from exc
