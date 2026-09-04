from __future__ import annotations

import asyncio

import pytest

from mas_cc.llm_runtime.config import LLMProviderConfig
from mas_cc.llm_runtime.messages import Message
from mas_cc.llm_runtime.providers import CompletionRequest
from mas_cc.llm_runtime.providers.errors import ProviderError
from mas_cc.llm_runtime.providers.pricing import OfflinePricingSource
from mas_cc.llm_runtime.providers.registry import (
    create_default_provider_registry,
    create_llm_provider,
)


MODEL = "deepseek-ai/DeepSeek-V4-Flash"


class _Response:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code
        self.headers = {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("upstream error body intentionally hidden")


class _Session:
    def __init__(self, *, gets=(), posts=()):
        self.gets = list(gets)
        self.posts = list(posts)
        self.get_calls = []
        self.post_calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return self.gets.pop(0)

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return self.posts.pop(0)

    def close(self):
        self.closed = True


def _models(*models):
    return _Response({"data": [{"id": model} for model in models]})


def _completion(content='{"status":"ready"}'):
    return _Response(
        {
            "id": "deepinfra-test",
            "model": MODEL,
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        }
    )


def _request():
    return CompletionRequest(
        (Message("system", "Return JSON."), Message("user", "Ready?")),
        temperature=0.0,
        max_output_tokens=16,
        seed=7,
    )


def test_deepinfra_uses_isolated_credentials_routes_and_json_mode():
    session = _Session(gets=[_models(MODEL)], posts=[_completion()])
    provider = create_llm_provider(
        LLMProviderConfig(type="deepinfra", model=MODEL, max_retries=0),
        environment={"DEEPINFRA_API_KEY": "deepinfra-test-secret"},
        session=session,
    )

    response = asyncio.run(provider.complete(_request()))
    provider.close()

    assert response.provider == "deepinfra"
    assert session.get_calls[0][0] == "https://api.deepinfra.com/v1/models"
    assert session.post_calls[0][0] == (
        "https://api.deepinfra.com/v1/openai/chat/completions"
    )
    sent = session.post_calls[0][1]
    assert sent["headers"]["Authorization"] == "Bearer deepinfra-test-secret"
    assert sent["json"]["response_format"] == {"type": "json_object"}
    assert session.closed


def test_deepinfra_json_mode_can_be_disabled():
    session = _Session(gets=[_models(MODEL)], posts=[_completion("ready")])
    provider = create_llm_provider(
        LLMProviderConfig(
            type="deepinfra",
            model=MODEL,
            options={"response_format": None},
        ),
        environment={"DEEPINFRA_API_KEY": "deepinfra-test-secret"},
        session=session,
    )
    assert asyncio.run(provider.complete(_request())).content == "ready"
    provider.close()
    assert "response_format" not in session.post_calls[0][1]["json"]


def test_deepinfra_account_limits_and_non_retryable_payment_error():
    limit_session = _Session(
        gets=[_Response({"rate_limit": 200, "tpm_rate_limit": 1_500_000})]
    )
    provider = create_llm_provider(
        LLMProviderConfig(type="deepinfra", model=MODEL),
        environment={"DEEPINFRA_API_KEY": "deepinfra-test-secret"},
        session=limit_session,
    )
    limits = asyncio.run(provider.discover_account_limits())
    provider.close()
    assert limits.maximum_concurrent_requests == 200
    assert limits.tokens_per_minute == 1_500_000

    payment_session = _Session(
        gets=[_models(MODEL)], posts=[_Response({}, status_code=402)]
    )
    provider = create_llm_provider(
        LLMProviderConfig(type="deepinfra", model=MODEL, max_retries=8),
        environment={"DEEPINFRA_API_KEY": "deepinfra-test-secret"},
        session=payment_session,
    )
    with pytest.raises(ProviderError) as captured:
        asyncio.run(provider.complete(_request()))
    provider.close()
    assert captured.value.code == "payment_required"
    assert captured.value.retryable is False
    assert len(payment_session.post_calls) == 1


def test_deepinfra_is_registered_priced_and_does_not_reuse_openai_key():
    assert "deepinfra" in create_default_provider_registry().names()
    with pytest.raises(ProviderError, match="DEEPINFRA_API_KEY"):
        create_llm_provider(
            LLMProviderConfig(type="deepinfra", model=MODEL),
            environment={"OPENAI_API_KEY": "wrong-provider-secret"},
        )
    quote = OfflinePricingSource().fetch("deepinfra", MODEL)
    assert quote.status == "known"
    assert quote.pricing is not None
    assert quote.pricing.ordinary_input_per_million == pytest.approx(0.09)
    assert quote.pricing.output_per_million == pytest.approx(0.18)
