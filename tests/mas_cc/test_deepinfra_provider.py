from __future__ import annotations

import asyncio
import json
import os

import pytest

from mas_cc.config import load_run_config
from mas_cc.games import create_game
from mas_cc.llm_runtime.config import LLMProviderConfig
from mas_cc.llm_runtime.messages import Message
from mas_cc.llm_runtime.providers import (
    CompletionRequest,
    OfflinePricingSource,
    ProviderError,
    create_default_provider_registry,
    create_llm_provider,
    default_model_profile_registry,
)


MODEL = "deepseek-ai/DeepSeek-V4-Flash"
RELEASE_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
GEMMA_MODEL = "google/gemma-4-E4B-it"
SMOKE_CONFIG = (
    "configs/runs/relational_reasoning/"
    "relational_imitation_round_feedback_deepinfra_N6_R3_smoke.yaml"
)
GEMMA_SMOKE_CONFIG = (
    "configs/runs/relational_reasoning/"
    "relational_imitation_round_feedback_deepinfra_gemma_N6_R3_smoke.yaml"
)
GEMMA_STUDY08_SMOKE_CONFIG = (
    "configs/runs/relational_reasoning/"
    "population_study_08_deepinfra_gemma_one_episode_smoke.yaml"
)
STUDY09H_RELEASE_SMOKE_CONFIG = (
    "configs/runs/relational_reasoning/"
    "population_study_09h_deepinfra_deepseek_one_episode_smoke.yaml"
)


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


def _request(max_output_tokens: int = 16) -> CompletionRequest:
    return CompletionRequest(
        (
            Message("system", "Reply with one JSON object."),
            Message("user", 'Return {"status":"ready"}.'),
        ),
        temperature=0.0,
        max_output_tokens=max_output_tokens,
        seed=7,
    )


def _completion(content: str = '{"status":"ready"}') -> _Response:
    return _Response(
        {
            "id": "chatcmpl-deepinfra-smoke",
            "model": MODEL,
            "choices": [
                {"message": {"content": content}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 5,
                "total_tokens": 17,
            },
        }
    )


def _models(*model_ids: str) -> _Response:
    return _Response({"data": [{"id": model_id} for model_id in model_ids]})


def _embedded_json_object(content: str) -> dict:
    """Parse one object while tolerating a provider-added Markdown fence."""

    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response does not contain a JSON object")
    value = json.loads(content[start : end + 1])
    if not isinstance(value, dict):
        raise TypeError("response JSON is not an object")
    return value


def test_deepinfra_uses_its_own_key_fixed_chat_route_and_json_default():
    session = _Session(gets=[_models(MODEL)], posts=[_completion()])
    provider = create_llm_provider(
        LLMProviderConfig(type="deepinfra", model=MODEL, max_retries=0),
        environment={"DEEPINFRA_API_KEY": "deepinfra-test-secret"},
        session=session,
    )

    response = asyncio.run(provider.complete(_request()))
    provider.close()

    assert response.provider == "deepinfra"
    assert response.model == MODEL
    assert _embedded_json_object(response.content) == {"status": "ready"}
    assert [call[0] for call in session.get_calls] == [
        "https://api.deepinfra.com/v1/models"
    ]
    assert [call[0] for call in session.post_calls] == [
        "https://api.deepinfra.com/v1/openai/chat/completions"
    ]
    sent = session.post_calls[0][1]
    assert sent["headers"]["Authorization"] == "Bearer deepinfra-test-secret"
    assert sent["json"] == {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Reply with one JSON object."},
            {"role": "user", "content": 'Return {"status":"ready"}.'},
        ],
        "max_tokens": 16,
        "temperature": 0.0,
        "seed": 7,
        "response_format": {"type": "json_object"},
    }
    assert session.closed


def test_deepinfra_json_default_has_an_explicit_provider_opt_out():
    session = _Session(gets=[_models(MODEL)], posts=[_completion("ready")])
    provider = create_llm_provider(
        LLMProviderConfig(
            type="deepinfra",
            model=MODEL,
            max_retries=0,
            options={"response_format": None},
        ),
        environment={"DEEPINFRA_API_KEY": "deepinfra-test-secret"},
        session=session,
    )

    assert asyncio.run(provider.complete(_request())).content == "ready"
    provider.close()

    assert "response_format" not in session.post_calls[0][1]["json"]


def test_deepinfra_e4b_has_a_provider_owned_json_object_exception():
    session = _Session(gets=[_models(GEMMA_MODEL)], posts=[_completion()])
    provider = create_llm_provider(
        LLMProviderConfig(type="deepinfra", model=GEMMA_MODEL, max_retries=0),
        environment={"DEEPINFRA_API_KEY": "deepinfra-test-secret"},
        session=session,
    )

    asyncio.run(provider.complete(_request()))
    provider.close()

    assert provider._response_format is None
    assert "response_format" not in session.post_calls[0][1]["json"]


def test_deepinfra_e4b_rejects_an_explicit_unsupported_json_object_mode():
    with pytest.raises(ProviderError, match="does not support json_object") as captured:
        create_llm_provider(
            LLMProviderConfig(
                type="deepinfra",
                model=GEMMA_MODEL,
                options={"response_format": {"type": "json_object"}},
            ),
            environment={"DEEPINFRA_API_KEY": "deepinfra-test-secret"},
            session=_Session(),
        )

    assert captured.value.code == "configuration_error"
    assert captured.value.retryable is False


def test_deepinfra_account_limits_use_the_provider_metadata_endpoint():
    session = _Session(
        gets=[_Response({"rate_limit": 200, "tpm_rate_limit": 1_500_000})]
    )
    provider = create_llm_provider(
        LLMProviderConfig(type="deepinfra", model=MODEL),
        environment={"DEEPINFRA_API_KEY": "deepinfra-test-secret"},
        session=session,
    )

    limits = asyncio.run(provider.discover_account_limits())
    provider.close()

    assert limits.maximum_concurrent_requests == 200
    assert limits.tokens_per_minute == 1_500_000
    assert [call[0] for call in session.get_calls] == [
        "https://api.deepinfra.com/v1/me/rate_limit"
    ]
    assert session.get_calls[0][1]["headers"]["Authorization"] == (
        "Bearer deepinfra-test-secret"
    )


def test_deepinfra_rejects_a_malformed_account_limit_response():
    provider = create_llm_provider(
        LLMProviderConfig(type="deepinfra", model=MODEL),
        environment={"DEEPINFRA_API_KEY": "deepinfra-test-secret"},
        session=_Session(gets=[_Response({"rate_limit": "many"})]),
    )

    with pytest.raises(ProviderError, match="did not match") as captured:
        asyncio.run(provider.discover_account_limits())
    provider.close()

    assert captured.value.code == "invalid_response"
    assert captured.value.retryable is True


def test_deepinfra_payment_required_is_normalized_and_not_retried():
    session = _Session(
        gets=[_models(GEMMA_MODEL)],
        posts=[_Response({"error": "balance required"}, status_code=402)],
    )
    provider = create_llm_provider(
        LLMProviderConfig(
            type="deepinfra",
            model=GEMMA_MODEL,
            max_retries=8,
        ),
        environment={"DEEPINFRA_API_KEY": "deepinfra-test-secret"},
        session=session,
    )

    with pytest.raises(ProviderError) as captured:
        asyncio.run(provider.complete(_request()))
    provider.close()

    assert captured.value.code == "payment_required"
    assert captured.value.status_code == 402
    assert captured.value.retryable is False
    assert len(session.post_calls) == 1


def test_deepinfra_rejects_a_model_missing_from_the_live_catalogue():
    session = _Session(gets=[_models("deepseek-ai/some-other-model")])
    provider = create_llm_provider(
        LLMProviderConfig(type="deepinfra", model=MODEL, max_retries=0),
        environment={"DEEPINFRA_API_KEY": "deepinfra-test-secret"},
        session=session,
    )

    with pytest.raises(ProviderError, match="is not listed") as captured:
        asyncio.run(provider.complete(_request()))
    provider.close()

    assert captured.value.code == "model_unavailable"
    assert session.post_calls == []


def test_deepinfra_accepts_an_explicit_base_url_environment_setting():
    provider = create_llm_provider(
        LLMProviderConfig(type="deepinfra", model=MODEL),
        environment={
            "DEEPINFRA_API_KEY": "deepinfra-test-secret",
            "DEEPINFRA_BASE_URL": "https://deepinfra.example/v1/openai/",
        },
        session=_Session(),
    )

    assert provider._base_url == "https://deepinfra.example/v1/openai"
    assert provider._chat_url == "https://deepinfra.example/v1/openai/chat/completions"
    provider.close()


def test_deepinfra_does_not_fall_back_to_other_provider_keys_or_urls():
    assert "deepinfra" in create_default_provider_registry().names()
    with pytest.raises(ProviderError, match="DEEPINFRA_API_KEY") as captured:
        create_llm_provider(
            LLMProviderConfig(type="deepinfra", model=MODEL),
            environment={
                "OPENAI_API_KEY": "openai-only-secret",
                "NEURALWATT_API_KEY": "neuralwatt-only-secret",
            },
        )
    assert "openai-only-secret" not in str(captured.value)
    assert "neuralwatt-only-secret" not in str(captured.value)

    openai = create_llm_provider(
        LLMProviderConfig(type="openai", model="gpt-4o-mini"),
        environment={"OPENAI_API_KEY": "openai-test-secret"},
        session=_Session(),
    )
    neuralwatt = create_llm_provider(
        LLMProviderConfig(type="neuralwatt", model="deepseek-v4-flash"),
        environment={"NEURALWATT_API_KEY": "neuralwatt-test-secret"},
        session=_Session(),
    )
    deepinfra = create_llm_provider(
        LLMProviderConfig(type="deepinfra", model=MODEL),
        environment={"DEEPINFRA_API_KEY": "deepinfra-test-secret"},
        session=_Session(),
    )
    assert openai._base_url == "https://api.openai.com/v1"
    assert neuralwatt._base_url == "https://api.neuralwatt.com/v1"
    assert deepinfra._base_url == "https://api.deepinfra.com/v1/openai"
    assert openai._response_format is None
    assert neuralwatt._response_format == {"type": "json_object"}
    assert deepinfra._response_format == {"type": "json_object"}
    openai.close()
    neuralwatt.close()
    deepinfra.close()


def test_deepseek_v4_flash_has_a_dated_deepinfra_profile_and_pricing():
    profile = default_model_profile_registry().get("deepinfra", MODEL)
    assert profile.provider_type == "deepinfra"
    assert profile.model == MODEL
    assert profile.family == "deepseek"
    assert profile.probe_source == "manual"
    assert profile.supports_seed is True
    assert profile.supports_system_messages is True
    assert profile.max_output_tokens_field == "max_tokens"

    quote = OfflinePricingSource().fetch("deepinfra", MODEL)
    assert quote.status == "known"
    assert quote.available is None
    assert quote.pricing is not None
    assert quote.pricing.unit == "USD"
    assert quote.pricing.ordinary_input_per_million == pytest.approx(0.09)
    assert quote.pricing.cached_input_per_million == pytest.approx(0.018)
    assert quote.pricing.output_per_million == pytest.approx(0.18)
    assert quote.pricing.limits.maximum_input_tokens == 1_048_576
    assert quote.pricing.limits.maximum_output_tokens == 65_536


def test_deepseek_v4_flash_0731_has_live_verified_deepinfra_pricing():
    profile = default_model_profile_registry().get("deepinfra", RELEASE_MODEL)
    assert profile.provider_type == "deepinfra"
    assert profile.model == RELEASE_MODEL
    assert profile.family == "deepseek"
    assert profile.supports_seed is True
    assert profile.supports_system_messages is True
    assert profile.max_output_tokens_field == "max_tokens"

    quote = OfflinePricingSource().fetch("deepinfra", RELEASE_MODEL)
    assert quote.status == "known"
    assert quote.pricing is not None
    assert quote.pricing.unit == "USD"
    assert quote.pricing.ordinary_input_per_million == pytest.approx(0.08)
    assert quote.pricing.cached_input_per_million == pytest.approx(0.016)
    assert quote.pricing.output_per_million == pytest.approx(0.18)
    assert quote.pricing.limits.maximum_input_tokens == 1_048_576


def test_gemma_4_e4b_has_a_dated_deepinfra_profile_and_pricing():
    profile = default_model_profile_registry().get("deepinfra", GEMMA_MODEL)
    assert profile.provider_type == "deepinfra"
    assert profile.model == GEMMA_MODEL
    assert profile.family == "gemma"
    assert profile.probe_source == "manual"
    assert profile.supports_seed is True
    assert profile.supports_system_messages is True
    assert profile.max_output_tokens_field == "max_tokens"

    quote = OfflinePricingSource().fetch("deepinfra", GEMMA_MODEL)
    assert quote.status == "known"
    assert quote.available is None
    assert quote.pricing is not None
    assert quote.pricing.unit == "USD"
    assert quote.pricing.ordinary_input_per_million == pytest.approx(0.02)
    assert quote.pricing.cached_input_per_million is None
    assert quote.pricing.output_per_million == pytest.approx(0.10)
    assert quote.pricing.limits.maximum_input_tokens == 131_072
    assert quote.pricing.limits.maximum_output_tokens is None


def test_deepinfra_relational_smoke_config_is_one_n6_r3_episode():
    config = load_run_config(SMOKE_CONFIG, environment={})
    assert config.llm_provider.type == "deepinfra"
    assert config.llm_provider.model == MODEL
    assert config.llm_provider.credentials_env == "DEEPINFRA_API_KEY"
    assert config.llm_provider.request_concurrency == 6
    assert config.llm_provider.max_output_tokens == 4096
    assert config.game.type == "relational_imitation_round_feedback"
    assert config.game.population_size == 6
    assert config.game.options["rounds"] == 3
    assert config.execution.repetitions == 1
    assert config.execution.parallelism == 1
    assert config.execution.seed == 20260830
    assert config.storage.output_dir.startswith("/pscratch/")

    plan = create_game(config.game).call_plan(config.game)
    assert plan.metadata["population_rounds"] == 3
    assert plan.metadata["interactions_per_episode"] == 18
    assert plan.provider_requests.lower == 18
    assert plan.provider_requests.maximum == 36


def test_deepinfra_study09h_release_smoke_is_one_n12_r3_episode():
    config = load_run_config(STUDY09H_RELEASE_SMOKE_CONFIG, environment={})
    assert config.llm_provider.type == "deepinfra"
    assert config.llm_provider.model == RELEASE_MODEL
    assert config.llm_provider.request_concurrency == 10
    assert config.llm_provider.max_output_tokens == 4096
    assert config.game.population_size == 12
    assert config.game.options["task_id"] == "task_0002"
    assert config.game.options["rounds"] == 3
    assert config.execution.repetitions == 1
    assert config.execution.parallelism == 1
    assert config.storage.artifact_profile == "full"

    plan = create_game(config.game).call_plan(config.game)
    assert plan.provider_requests.lower == 48
    assert plan.provider_requests.maximum == 96


def test_deepinfra_gemma_relational_smoke_config_is_one_n6_r3_episode():
    config = load_run_config(GEMMA_SMOKE_CONFIG, environment={})
    assert config.llm_provider.type == "deepinfra"
    assert config.llm_provider.model == GEMMA_MODEL
    assert config.llm_provider.credentials_env == "DEEPINFRA_API_KEY"
    assert config.llm_provider.request_concurrency == 6
    assert config.llm_provider.max_output_tokens == 4096
    assert config.game.type == "relational_imitation_round_feedback"
    assert config.game.population_size == 6
    assert config.game.options["rounds"] == 3
    assert config.execution.repetitions == 1
    assert config.execution.parallelism == 1
    assert config.storage.output_dir.startswith("/pscratch/")


def test_deepinfra_gemma_study08_smoke_is_one_full_shape_episode():
    config = load_run_config(GEMMA_STUDY08_SMOKE_CONFIG, environment={})
    assert config.llm_provider.type == "deepinfra"
    assert config.llm_provider.model == GEMMA_MODEL
    assert config.llm_provider.request_concurrency == 24
    assert config.llm_provider.max_output_tokens == 4096
    assert config.game.type == "relational_imitation_round_feedback"
    assert config.game.population_size == 24
    assert config.game.options["rounds"] == 10
    assert config.game.options["receiver_epistemic_disposition"] == "vigilant"
    assert config.control.options["target"] == "correct"
    assert config.control.options["intervention_budget"] == 12
    assert config.execution.repetitions == 1
    assert config.execution.parallelism == 1
    assert config.experiment.metadata["cells"] == 1
    assert config.experiment.metadata["episodes"] == 1
    assert config.logging.options["prompt_examples"]["count"] == 100
    assert config.logging.options["detailed_prompt_audit"] == {
        "enabled": True,
        "always_log_first_n_rounds": 0,
        "max_logged_prompts_per_game": 20,
        "max_logged_prompts_per_run": 20,
    }
    assert config.storage.artifact_profile == "full"
    assert config.storage.output_dir.startswith("/pscratch/")


@pytest.mark.skipif(
    os.environ.get("MAS_CC_RUN_DEEPINFRA_SMOKE") != "1",
    reason="set MAS_CC_RUN_DEEPINFRA_SMOKE=1 for live DeepInfra checks",
)
def test_deepinfra_live_account_limit_smoke():
    model = os.environ.get("MAS_CC_DEEPINFRA_SMOKE_MODEL", MODEL)
    provider = create_llm_provider(
        LLMProviderConfig(type="deepinfra", model=model, timeout_seconds=30)
    )
    try:
        limits = asyncio.run(provider.discover_account_limits())
    finally:
        provider.close()

    assert limits.maximum_concurrent_requests >= 1
    assert limits.tokens_per_minute >= 1


@pytest.mark.skipif(
    os.environ.get("MAS_CC_RUN_DEEPINFRA_SMOKE") != "1",
    reason="set MAS_CC_RUN_DEEPINFRA_SMOKE=1 for one billable DeepInfra request",
)
def test_deepinfra_live_chat_completion_smoke():
    model = os.environ.get("MAS_CC_DEEPINFRA_SMOKE_MODEL", MODEL)
    provider = create_llm_provider(
        LLMProviderConfig(
            type="deepinfra",
            model=model,
            timeout_seconds=60,
            max_retries=0,
            temperature=0.0,
            max_output_tokens=4096,
        )
    )
    try:
        response = asyncio.run(provider.complete(_request(max_output_tokens=4096)))
    finally:
        provider.close()

    assert response.provider == "deepinfra"
    assert response.model == model
    assert _embedded_json_object(response.content) == {"status": "ready"}
    assert response.status_code == 200
    assert response.usage.total_tokens is not None


@pytest.mark.skipif(
    os.environ.get("MAS_CC_RUN_DEEPINFRA_BURST_SMOKE") != "1",
    reason="set MAS_CC_RUN_DEEPINFRA_BURST_SMOKE=1 for a billable concurrent burst",
)
def test_deepinfra_live_concurrent_burst_smoke():
    burst = int(os.environ.get("MAS_CC_DEEPINFRA_BURST_REQUESTS", "16"))
    assert 1 <= burst <= 200
    model = os.environ.get("MAS_CC_DEEPINFRA_SMOKE_MODEL", MODEL)
    provider = create_llm_provider(
        LLMProviderConfig(
            type="deepinfra",
            model=model,
            timeout_seconds=60,
            max_retries=1,
            request_concurrency=burst,
            temperature=0.0,
            max_output_tokens=64,
        )
    )

    async def run_burst():
        return await asyncio.gather(
            *(provider.complete(_request(max_output_tokens=64)) for _ in range(burst))
        )

    try:
        responses = asyncio.run(run_burst())
    finally:
        provider.close()

    assert len(responses) == burst
    assert all(response.status_code == 200 for response in responses)
    assert all(
        _embedded_json_object(response.content) == {"status": "ready"}
        for response in responses
    )
