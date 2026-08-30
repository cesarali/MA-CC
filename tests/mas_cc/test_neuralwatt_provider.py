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


MODEL = "deepseek-v4-flash"
SMOKE_CONFIG = (
    "configs/runs/relational_reasoning/"
    "relational_imitation_round_feedback_neuralwatt_N6_R3_smoke.yaml"
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


def _request() -> CompletionRequest:
    return CompletionRequest(
        (
            Message("system", "Reply with one short word."),
            Message("user", "Say ready."),
        ),
        temperature=0.0,
        max_output_tokens=16,
        seed=7,
    )


def test_neuralwatt_uses_its_own_key_and_fixed_openai_compatible_endpoint():
    session = _Session(
        gets=[_Response({"data": [{"id": MODEL}]})],
        posts=[
            _Response(
                {
                    "id": "chatcmpl-neuralwatt-smoke",
                    "model": MODEL,
                    "choices": [
                        {"message": {"content": "ready"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 1,
                        "total_tokens": 9,
                    },
                }
            )
        ],
    )
    provider = create_llm_provider(
        LLMProviderConfig(type="neuralwatt", model=MODEL, max_retries=0),
        environment={"NEURALWATT_API_KEY": "neuralwatt-test-secret"},
        session=session,
    )

    response = asyncio.run(provider.complete(_request()))
    provider.close()

    assert response.provider == "neuralwatt"
    assert response.model == MODEL
    assert response.content == "ready"
    assert response.usage.total_tokens == 9
    assert [call[0] for call in session.get_calls] == [
        "https://api.neuralwatt.com/v1/models"
    ]
    assert [call[0] for call in session.post_calls] == [
        "https://api.neuralwatt.com/v1/chat/completions"
    ]
    sent = session.post_calls[0][1]
    assert sent["headers"]["Authorization"] == "Bearer neuralwatt-test-secret"
    assert sent["json"] == {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Reply with one short word."},
            {"role": "user", "content": "Say ready."},
        ],
        "max_tokens": 16,
        "temperature": 0.0,
        "seed": 7,
        "response_format": {"type": "json_object"},
    }
    assert session.closed


def test_neuralwatt_json_object_mode_is_the_provider_default():
    session = _Session(
        gets=[_Response({"data": [{"id": MODEL}]})],
        posts=[
            _Response(
                {
                    "model": MODEL,
                    "choices": [
                        {
                            "message": {"content": '{"vote":"A"}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"total_tokens": 9},
                }
            )
        ],
    )
    provider = create_llm_provider(
        LLMProviderConfig(
            type="neuralwatt",
            model=MODEL,
            max_retries=0,
        ),
        environment={"NEURALWATT_API_KEY": "neuralwatt-test-secret"},
        session=session,
    )

    response = asyncio.run(provider.complete(_request()))
    provider.close()

    assert response.content == '{"vote":"A"}'
    assert session.post_calls[0][1]["json"]["response_format"] == {
        "type": "json_object"
    }


def test_neuralwatt_json_object_default_has_an_explicit_provider_opt_out():
    session = _Session(
        gets=[_Response({"data": [{"id": MODEL}]})],
        posts=[
            _Response(
                {
                    "model": MODEL,
                    "choices": [
                        {"message": {"content": "ready"}, "finish_reason": "stop"}
                    ],
                    "usage": {"total_tokens": 9},
                }
            )
        ],
    )
    provider = create_llm_provider(
        LLMProviderConfig(
            type="neuralwatt",
            model=MODEL,
            max_retries=0,
            options={"response_format": None},
        ),
        environment={"NEURALWATT_API_KEY": "neuralwatt-test-secret"},
        session=session,
    )

    assert asyncio.run(provider.complete(_request())).content == "ready"
    provider.close()

    assert "response_format" not in session.post_calls[0][1]["json"]


def test_openai_compatible_response_format_rejects_unbounded_passthrough():
    with pytest.raises(ProviderError, match="supports only"):
        create_llm_provider(
            LLMProviderConfig(
                type="neuralwatt",
                model=MODEL,
                options={"response_format": {"type": "xml"}},
            ),
            environment={"NEURALWATT_API_KEY": "neuralwatt-test-secret"},
            session=_Session(),
        )


def test_neuralwatt_forced_structured_tool_arguments_become_completion_content():
    tool = {
        "name": "submit_relational_ballot",
        "description": "Submit one relational ballot.",
        "parameters": {
            "type": "object",
            "properties": {
                "vote": {"type": "string", "enum": ["A", "B", "C"]},
                "reason": {"type": "string", "maxLength": 16_384},
                "shared_fact_id": {"type": "string"},
            },
            "required": ["vote", "reason", "shared_fact_id"],
            "additionalProperties": False,
        },
    }
    arguments = '{"vote":"A","reason":"because","shared_fact_id":"none"}'
    session = _Session(
        gets=[_Response({"data": [{"id": MODEL}]})],
        posts=[
            _Response(
                {
                    "model": MODEL,
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "type": "function",
                                        "function": {
                                            "name": "submit_relational_ballot",
                                            "arguments": arguments,
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"total_tokens": 20},
                }
            )
        ],
    )
    provider = create_llm_provider(
        LLMProviderConfig(
            type="neuralwatt",
            model=MODEL,
            max_retries=0,
            options={"structured_output_tool": tool},
        ),
        environment={"NEURALWATT_API_KEY": "neuralwatt-test-secret"},
        session=session,
    )

    response = asyncio.run(provider.complete(_request()))
    provider.close()

    assert response.content == arguments
    payload = session.post_calls[0][1]["json"]
    assert payload["tools"] == [{"type": "function", "function": tool}]
    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_relational_ballot"},
    }
    assert "response_format" not in payload


def test_neuralwatt_does_not_fall_back_to_the_official_openai_key_or_url():
    assert "neuralwatt" in create_default_provider_registry().names()
    with pytest.raises(ProviderError, match="NEURALWATT_API_KEY") as captured:
        create_llm_provider(
            LLMProviderConfig(type="neuralwatt", model=MODEL),
            environment={"OPENAI_API_KEY": "openai-only-secret"},
        )
    assert "openai-only-secret" not in str(captured.value)

    openai = create_llm_provider(
        LLMProviderConfig(type="openai", model="gpt-4o-mini"),
        environment={"OPENAI_API_KEY": "openai-test-secret"},
        session=_Session(),
    )
    neuralwatt = create_llm_provider(
        LLMProviderConfig(type="neuralwatt", model=MODEL),
        environment={"NEURALWATT_API_KEY": "neuralwatt-test-secret"},
        session=_Session(),
    )
    university = create_llm_provider(
        LLMProviderConfig(
            type="university",
            model="gwdg/openai-gpt-oss-120b",
            credentials_env="POTSDAM_TEST_KEY",
            base_url_env="POTSDAM_TEST_URL",
        ),
        environment={
            "POTSDAM_TEST_KEY": "potsdam-test-secret",
            "POTSDAM_TEST_URL": "https://potsdam.example.invalid/v1",
        },
        session=_Session(),
    )
    assert openai._base_url == "https://api.openai.com/v1"
    assert neuralwatt._base_url == "https://api.neuralwatt.com/v1"
    assert openai._response_format is None
    assert university._response_format is None
    assert neuralwatt._response_format == {"type": "json_object"}
    openai.close()
    neuralwatt.close()
    university.close()


def test_deepseek_v4_flash_has_a_dated_profile_and_offline_usd_pricing():
    profile = default_model_profile_registry().get("neuralwatt", MODEL)
    assert profile.provider_type == "neuralwatt"
    assert profile.model == MODEL
    assert profile.family == "deepseek"
    assert profile.probe_source in {"manual", "probe"}
    assert profile.supports_seed is True
    assert profile.supports_system_messages is True
    assert profile.max_output_tokens_field == "max_tokens"

    quote = OfflinePricingSource().fetch("neuralwatt", MODEL)
    assert quote.status == "known"
    # Offline pricing is auditable but does not claim live account availability.
    assert quote.available is None
    assert quote.pricing is not None
    assert quote.pricing.unit == "USD"
    assert quote.pricing.ordinary_input_per_million == pytest.approx(0.14)
    assert quote.pricing.cached_input_per_million == pytest.approx(0.028)
    assert quote.pricing.output_per_million == pytest.approx(0.28)
    assert quote.pricing.limits.maximum_input_tokens == 1_048_560
    assert quote.pricing.limits.maximum_output_tokens == 65_536


def test_neuralwatt_relational_smoke_config_is_one_n6_r3_episode():
    config = load_run_config(SMOKE_CONFIG, environment={})
    assert config.llm_provider.type == "neuralwatt"
    assert config.llm_provider.model == MODEL
    assert config.llm_provider.credentials_env == "NEURALWATT_API_KEY"
    assert config.llm_provider.max_output_tokens == 128
    assert config.game.type == "relational_imitation_round_feedback"
    assert config.game.population_size == 6
    assert config.game.options["rounds"] == 3
    assert config.execution.repetitions == 1
    assert config.execution.parallelism == 1
    assert config.execution.seed == 20260829
    assert config.storage.output_dir.startswith("/pscratch/")

    plan = create_game(config.game).call_plan(config.game)
    assert plan.metadata["population_rounds"] == 3
    assert plan.metadata["interactions_per_episode"] == 18
    assert plan.provider_requests.lower == 18
    assert plan.provider_requests.maximum == 36


@pytest.mark.skipif(
    os.environ.get("MAS_CC_RUN_NEURALWATT_SMOKE") != "1",
    reason="set MAS_CC_RUN_NEURALWATT_SMOKE=1 for one billable NeuralWatt request",
)
def test_neuralwatt_live_chat_completion_smoke():
    provider = create_llm_provider(
        LLMProviderConfig(
            type="neuralwatt",
            model=MODEL,
            timeout_seconds=60,
            max_retries=0,
            temperature=0.0,
            max_output_tokens=16,
        )
    )
    try:
        response = asyncio.run(
            provider.complete(
                CompletionRequest(
                    (
                        Message(
                            "user",
                            'Return only this JSON object: {"status":"ready"}',
                        ),
                    ),
                    temperature=0.0,
                    max_output_tokens=16,
                )
            )
        )
    finally:
        provider.close()

    body = json.loads(response.content)
    assert response.provider == "neuralwatt"
    assert response.model == MODEL
    assert body == {"status": "ready"}
    assert response.status_code == 200
    assert response.usage.total_tokens is not None


@pytest.mark.skipif(
    os.environ.get("MAS_CC_RUN_NEURALWATT_SMOKE") != "1",
    reason="set MAS_CC_RUN_NEURALWATT_SMOKE=1 for one billable NeuralWatt request",
)
def test_neuralwatt_live_forced_relational_ballot_tool_smoke():
    provider = create_llm_provider(
        LLMProviderConfig(
            type="neuralwatt",
            model=MODEL,
            timeout_seconds=60,
            max_retries=0,
            temperature=0.0,
            max_output_tokens=4096,
            options={
                "structured_output_tool": {
                    "name": "submit_relational_ballot",
                    "description": "Submit exactly one relational reasoning ballot.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vote": {"type": "string", "enum": ["A", "B", "C"]},
                            "reason": {"type": "string", "maxLength": 16_384},
                            "shared_fact_id": {"type": "string"},
                        },
                        "required": ["vote", "reason", "shared_fact_id"],
                        "additionalProperties": False,
                    },
                }
            },
        )
    )
    try:
        response = asyncio.run(
            provider.complete(
                CompletionRequest(
                    (
                        Message(
                            "user",
                            "Vote A, explain briefly, and share no fact. Submit the ballot.",
                        ),
                    ),
                    temperature=0.0,
                    max_output_tokens=4096,
                )
            )
        )
    finally:
        provider.close()

    body = json.loads(response.content)
    assert set(body) == {"vote", "reason", "shared_fact_id"}
    assert body["vote"] in {"A", "B", "C"}
    assert isinstance(body["reason"], str) and body["reason"].strip()
    assert isinstance(body["shared_fact_id"], str)
    assert response.status_code == 200
