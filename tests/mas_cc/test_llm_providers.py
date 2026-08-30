import asyncio
import json
import sys
import threading
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

from mas_cc.config import LLMProviderConfig
from mas_cc.llm_runtime.messages import Message
from mas_cc.llm_runtime.providers import (
    BudgetCeiling,
    CompletionRequest,
    CompletionResponse,
    ProviderError,
    ProviderLoadControlConfig,
    ProviderUsage,
    create_llm_provider,
)
from mas_cc.llm_runtime.providers.adapters.gemma_local import (
    GemmaLocalProvider,
    GenerationResult,
)
from mas_cc.planning import LogicalCallSpec, static_preflight


def _request() -> CompletionRequest:
    return CompletionRequest(
        (Message("system", "Choose one option."), Message("user", "A or B?")),
        max_output_tokens=4,
        seed=7,
    )


def test_normalized_records_validate_and_redact_raw_credentials():
    usage = ProviderUsage.from_mapping(
        {
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "total_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 2},
        }
    )
    assert usage.to_dict() == {
        "input_tokens": 3,
        "output_tokens": 1,
        "total_tokens": 4,
        "cached_input_tokens": 2,
    }
    provider = create_llm_provider(
        LLMProviderConfig(
            type="mock",
            model="deterministic-v1",
            options={"response": "B"},
        )
    )
    response = asyncio.run(provider.complete(_request()))
    assert response.provider == "mock"
    assert response.content == "B"
    assert set(response.to_dict()) >= {"provider", "model", "usage", "latency_seconds"}
    redacted = CompletionResponse(
        content="A",
        provider="fake",
        model="fake",
        raw_response={
            "headers": {"Authorization": "Bearer secret"},
            "nested": {"api_key": "secret", "safe": "kept"},
        },
    ).redacted_raw_response()
    assert redacted == {
        "headers": "<redacted>",
        "nested": {"api_key": "<redacted>", "safe": "kept"},
    }


class _Response:
    def __init__(self, status_code, body, *, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"unsafe upstream body is deliberately unavailable")


class _Session:
    def __init__(self, posts, gets=()):
        self.posts = list(posts)
        self.gets = list(gets)
        self.post_calls = []
        self.get_calls = []
        self.closed = False
        self.proxies = {}

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return self.posts.pop(0)

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return self.gets.pop(0)

    def close(self):
        self.closed = True


class _CountingCoordinator:
    def __init__(self):
        self.acquired = 0
        self.outcomes = []
        self.config = ProviderLoadControlConfig()

    async def acquire(self):
        self.acquired += 1
        return SimpleNamespace(token=str(self.acquired))

    async def release(self, lease, **outcome):
        self.outcomes.append((lease.token, outcome))


class _ReleaseFailingCoordinator(_CountingCoordinator):
    async def release(self, lease, **outcome):
        raise RuntimeError("simulated shared-filesystem telemetry failure")


class _HeartbeatCoordinator(_CountingCoordinator):
    def __init__(self):
        super().__init__()
        self.config = ProviderLoadControlConfig(heartbeat_seconds=0.01, lease_seconds=1)
        self.renewals = 0

    async def renew(self, lease, **kwargs):
        self.renewals += 1
        return True


def test_openai_compatible_adapter_retries_and_normalizes_without_wire_metadata():
    body = {
        "id": "req-1",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }
    session = _Session([_Response(500, {}), _Response(200, body)])
    config = LLMProviderConfig(
        type="openai",
        model="gpt-4o-mini",
        credentials_env="TEST_API_KEY",
        max_retries=1,
    )
    provider = create_llm_provider(
        config,
        environment={"TEST_API_KEY": "test-secret"},
        session=session,
    )
    response = asyncio.run(provider.complete(_request()))
    provider.close()
    assert response.content == "A"
    assert response.retries == 1
    assert response.usage.total_tokens == 6
    assert session.closed
    sent = session.post_calls[-1][1]["json"]
    assert sent["messages"] == [
        {"role": "system", "content": "Choose one option."},
        {"role": "user", "content": "A or B?"},
    ]
    assert "metadata" not in sent


def test_cluster_coordinator_accounts_for_each_retry_attempt():
    body = {
        "id": "req-coordinated",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}],
    }
    coordinator = _CountingCoordinator()
    session = _Session(
        [_Response(500, {}, headers={"Retry-After": "0"}), _Response(200, body)]
    )
    provider = create_llm_provider(
        LLMProviderConfig(
            type="openai",
            model="gpt-4o-mini",
            credentials_env="TEST_API_KEY",
            max_retries=1,
        ),
        environment={"TEST_API_KEY": "test-secret"},
        session=session,
        request_coordinator=coordinator,
    )

    assert asyncio.run(provider.complete(_request())).content == "A"
    assert coordinator.acquired == 2
    assert [outcome["success"] for _, outcome in coordinator.outcomes] == [False, True]
    assert coordinator.outcomes[0][1]["status_code"] == 500


def test_coordinated_request_survives_beyond_adapter_retry_count():
    body = {
        "id": "req-recovered",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}],
    }
    coordinator = _CountingCoordinator()
    session = _Session(
        [
            _Response(500, {}, headers={"Retry-After": "0"}),
            _Response(500, {}, headers={"Retry-After": "0"}),
            _Response(200, body),
        ]
    )
    provider = create_llm_provider(
        LLMProviderConfig(
            type="openai",
            model="gpt-4o-mini",
            credentials_env="TEST_API_KEY",
            max_retries=0,
        ),
        environment={"TEST_API_KEY": "test-secret"},
        session=session,
        request_coordinator=coordinator,
    )

    response = asyncio.run(provider.complete(_request()))

    assert response.content == "A"
    assert response.retries == 2
    assert coordinator.acquired == 3
    assert [outcome["success"] for _, outcome in coordinator.outcomes] == [
        False,
        False,
        True,
    ]


def test_coordination_release_failure_does_not_destroy_valid_response():
    body = {
        "id": "req-valid-despite-telemetry",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}],
    }
    provider = create_llm_provider(
        LLMProviderConfig(
            type="openai", model="gpt-4o-mini", credentials_env="TEST_API_KEY"
        ),
        environment={"TEST_API_KEY": "test-secret"},
        session=_Session([_Response(200, body)]),
        request_coordinator=_ReleaseFailingCoordinator(),
    )

    assert asyncio.run(provider.complete(_request())).content == "A"


def test_slow_http_attempt_is_renewed_and_released():
    import time

    body = {
        "id": "req-slow",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}],
    }
    coordinator = _HeartbeatCoordinator()
    session = _Session([_Response(200, body)])
    original_post = session.post

    def slow_post(*args, **kwargs):
        time.sleep(0.04)
        return original_post(*args, **kwargs)

    session.post = slow_post
    provider = create_llm_provider(
        LLMProviderConfig(
            type="openai", model="gpt-4o-mini", credentials_env="TEST_API_KEY"
        ),
        environment={"TEST_API_KEY": "test-secret"},
        session=session,
        request_coordinator=coordinator,
    )
    assert asyncio.run(provider.complete(_request())).content == "A"
    assert coordinator.renewals >= 1
    assert len(coordinator.outcomes) == 1


def test_openai_compatible_adapter_retries_a_transient_malformed_success_body():
    valid = {
        "id": "req-2",
        "model": "gpt-oss",
        "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }
    session = _Session(
        [
            _Response(
                200,
                {"temporarily": "not-a-chat-envelope"},
                headers={"Retry-After": "0"},
            ),
            _Response(200, valid),
        ]
    )
    provider = create_llm_provider(
        LLMProviderConfig(
            type="openai",
            model="gpt-oss",
            credentials_env="TEST_API_KEY",
            max_retries=1,
        ),
        environment={"TEST_API_KEY": "test-secret"},
        session=session,
    )
    response = asyncio.run(provider.complete(_request()))
    assert response.content == "A"
    assert response.retries == 1
    assert len(session.post_calls) == 2


def _reasoning_body(finish_reason, content, reasoning="Let me think about this."):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": "req-3",
        "model": "gpt-oss",
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 884, "completion_tokens": 300, "total_tokens": 1184},
    }


def _reasoning_provider(session, *, max_retries=1):
    return create_llm_provider(
        LLMProviderConfig(
            type="openai",
            model="gpt-oss",
            credentials_env="TEST_API_KEY",
            max_retries=max_retries,
        ),
        environment={"TEST_API_KEY": "test-secret"},
        session=session,
    )


def test_a_reasoning_model_that_ran_out_of_budget_says_so_instead_of_retrying():
    """gpt-oss charges its chain of thought against `max_tokens`.

    When the budget runs out inside the reasoning the envelope is well formed
    but `content` is empty, which is indistinguishable from a flaky proxy to
    the schema check alone - yet it is perfectly deterministic, so retrying it
    only buys identical paid failures.
    """

    session = _Session([_Response(200, _reasoning_body("length", None))])
    provider = _reasoning_provider(session)

    with pytest.raises(ProviderError) as captured:
        asyncio.run(provider.complete(_request()))

    assert captured.value.code == "reasoning_budget_exhausted"
    assert captured.value.retryable is False
    assert "max_output_tokens" in str(captured.value)
    # The whole point: one request, not one per configured retry.
    assert len(session.post_calls) == 1


@pytest.mark.parametrize(
    ("finish_reason", "reasoning"),
    [("stop", "Let me think."), ("length", None)],
    ids=["complete_but_empty", "truncated_without_reasoning"],
)
def test_an_empty_content_that_is_not_a_reasoning_overrun_still_retries(
    finish_reason, reasoning
):
    valid = {
        "id": "req-4",
        "model": "gpt-oss",
        "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }
    session = _Session(
        [
            _Response(
                200,
                _reasoning_body(finish_reason, None, reasoning),
                headers={"Retry-After": "0"},
            ),
            _Response(200, valid),
        ]
    )

    response = asyncio.run(_reasoning_provider(session).complete(_request()))

    assert response.content == "A"
    assert len(session.post_calls) == 2


def test_a_reasoning_model_that_answers_is_read_from_content_not_reasoning():
    body = _reasoning_body("stop", '{"vote": "A", "reason": "Because."}')
    response = asyncio.run(
        _reasoning_provider(_Session([_Response(200, body)])).complete(_request())
    )

    assert response.content == '{"vote": "A", "reason": "Because."}'
    assert response.raw_response["choices"][0]["message"]["reasoning_content"]


def test_university_discovers_v1_endpoint_and_rejects_unlisted_model_safely():
    config = LLMProviderConfig(
        type="university",
        model="wanted",
        credentials_env="TEST_API_KEY",
        base_url_env="TEST_BASE_URL",
        max_retries=0,
    )
    session = _Session(
        [], gets=[_Response(404, {}), _Response(200, {"data": [{"id": "other"}]})]
    )
    provider = create_llm_provider(
        config,
        environment={
            "TEST_API_KEY": "test-secret",
            "TEST_BASE_URL": "https://example.invalid",
        },
        session=session,
    )
    with pytest.raises(ProviderError) as captured:
        asyncio.run(provider.complete(_request()))
    assert captured.value.code == "model_unavailable"
    assert "test-secret" not in str(captured.value)
    assert [call[0] for call in session.get_calls] == [
        "https://example.invalid/models",
        "https://example.invalid/v1/models",
    ]


class _GemmaRuntime:
    diagnostics = {"fake": True}

    def __init__(self):
        self.active = 0
        self.maximum = 0
        self.lock = threading.Lock()

    def generate(self, messages, **kwargs):
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        time.sleep(0.01)
        with self.lock:
            self.active -= 1
        return GenerationResult("B", 5, 1)


def test_gemma_is_lazy_loaded_once_and_serializes_inference():
    count = 0
    runtime = _GemmaRuntime()

    def factory():
        nonlocal count
        count += 1
        return runtime

    config = LLMProviderConfig(
        type="gemma_local", model="google/gemma-4-12B-it", request_concurrency=1
    )
    provider = GemmaLocalProvider(config, runtime_factory=factory)
    assert "transformers" not in sys.modules

    async def run():
        return await asyncio.gather(*(provider.complete(_request()) for _ in range(4)))

    responses = asyncio.run(run())
    assert count == 1
    assert runtime.maximum == 1
    assert {item.content for item in responses} == {"B"}
    assert sum(item.load_seconds is not None for item in responses) == 1
    assert provider.diagnostics["loaded"] is True


def test_static_preflight_counts_calls_cost_runtime_and_budget_without_provider_creation():
    config = LLMProviderConfig(
        type="openai",
        model="gpt-4o-mini",
        credentials_env="MISSING_ON_PURPOSE",
        request_concurrency=2,
        options={"estimated_latency_seconds": 1.5},
    )
    estimate = static_preflight(
        _request(),
        config,
        LogicalCallSpec(5),
        assumed_output_tokens=2,
        budget=BudgetCeiling(1),
    )
    assert (
        estimate.estimated_total_input_tokens
        == estimate.estimated_input_tokens_per_call * 5
    )
    assert estimate.estimated_total_output_tokens == 10
    assert estimate.expected_cost_usd is not None
    assert estimate.conservative_cost_bound_usd == pytest.approx(
        estimate.expected_cost_usd * 1.5
    )
    assert estimate.rough_runtime_seconds == 4.5
    assert estimate.within_budget is True


def test_provider_creation_errors_are_normalized_and_secret_safe():
    config = LLMProviderConfig(
        type="openai", model="gpt-4o-mini", credentials_env="MISSING_KEY"
    )
    with pytest.raises(ProviderError) as captured:
        create_llm_provider(config, environment={})
    assert captured.value.to_dict()["code"] == "configuration_error"
    assert "Bearer" not in str(captured.value)


class _Timeout:
    """A post that never produces a response, the way a read timeout does."""

    def __init__(self, exc):
        self.exc = exc


def _timeout_session(posts):
    session = _Session(posts)
    original = session.post

    def post(url, **kwargs):
        item = session.posts[0]
        if isinstance(item, _Timeout):
            session.posts.pop(0)
            session.post_calls.append((url, kwargs))
            raise item.exc
        return original(url, **kwargs)

    session.post = post
    return session


def test_a_transport_timeout_is_retried_like_any_other_transient_failure():
    """Regression: `max_retries` used to buy nothing against the commonest fault.

    A connect or read timeout produces no response, so `status_code` is None.
    The retry branch required `status is not None`, so the single most likely
    failure against a shared proxy - and the most obviously transient - was the
    one case that skipped retries entirely. One slow generation then killed a
    whole episode, and a failed episode contaminates the aggregate curves.
    """

    body = {
        "id": "req-1",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }
    session = _timeout_session(
        [_Timeout(TimeoutError("read timed out")), _Response(200, body)]
    )
    provider = create_llm_provider(
        LLMProviderConfig(
            type="openai",
            model="gpt-4o-mini",
            credentials_env="TEST_API_KEY",
            max_retries=2,
        ),
        environment={"TEST_API_KEY": "test-secret"},
        session=session,
    )
    response = asyncio.run(provider.complete(_request()))
    provider.close()
    assert response.content == "A"
    assert response.retries == 1
    assert len(session.post_calls) == 2


def test_an_exhausted_timeout_reports_itself_as_retryable_and_never_leaks_the_key():
    session = _timeout_session([_Timeout(TimeoutError("read timed out"))] * 2)
    provider = create_llm_provider(
        LLMProviderConfig(
            type="openai",
            model="gpt-4o-mini",
            credentials_env="TEST_API_KEY",
            max_retries=1,
        ),
        environment={"TEST_API_KEY": "test-secret"},
        session=session,
    )
    with pytest.raises(ProviderError) as captured:
        asyncio.run(provider.complete(_request()))
    assert captured.value.code == "connection_error"
    assert captured.value.retryable is True
    assert captured.value.status_code is None
    assert "test-secret" not in str(captured.value)
    assert len(session.post_calls) == 2


def test_authentication_and_client_errors_are_never_retried():
    """The other half of the rule: retrying a 401 just burns the budget."""

    session = _Session([_Response(401, {}), _Response(400, {})])
    provider = create_llm_provider(
        LLMProviderConfig(
            type="openai",
            model="gpt-4o-mini",
            credentials_env="TEST_API_KEY",
            max_retries=3,
        ),
        environment={"TEST_API_KEY": "test-secret"},
        session=session,
    )
    with pytest.raises(ProviderError) as captured:
        asyncio.run(provider.complete(_request()))
    assert captured.value.code == "authentication_failed"
    assert captured.value.retryable is False
    assert len(session.post_calls) == 1
