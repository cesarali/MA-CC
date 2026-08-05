import concurrent.futures
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mas_cc.config import LLMProviderConfig
from mas_cc.llm_runtime.messages import Message
from mas_cc.llm_runtime.providers import (
    BudgetLimits,
    CachedPricingSource,
    CompletionRequest,
    LongContextPricing,
    ProviderError,
    ModelPricing,
    MonetaryAmount,
    OfflinePricingSource,
    ProviderLimits,
    RuntimeBudgetGuard,
    UniversityPricingSource,
    resolve_budget_limits,
    sanitized_snapshot_bytes,
)
from mas_cc.planning import LogicalCallSpec, estimate_cost, static_preflight
from mas_cc.cli.provider_economics import inspect_phase_4_amendment


NOW = "2026-08-02T12:00:00Z"


def pricing(**overrides):
    values = {
        "provider": "university",
        "model": "chat-model",
        "ordinary_input_per_million": 2.0,
        "cached_input_per_million": 0.5,
        "cache_creation_per_million": 3.0,
        "output_per_million": 8.0,
        "unit": "proxy_accounting_unit",
        "unit_source": "fixture metadata",
        "source": "GET /v1/model/info",
        "retrieved_at": NOW,
        "version": "fixture-v1",
    }
    values.update(overrides)
    return ModelPricing(**values)


def money(amount, unit="proxy_accounting_unit"):
    return MonetaryAmount(
        amount, unit, "fixture", "university", "chat-model", "fixture", NOW, "v1"
    )


def request():
    return CompletionRequest((Message("user", "Choose A or B."),), max_output_tokens=10)


def test_cached_input_and_cache_creation_arithmetic_do_not_double_charge():
    cost = pricing().cost(
        1_000_000, 100_000, cached_input_tokens=250_000, cache_creation_tokens=100_000
    )
    # 650k ordinary * 2 + 250k cached * .5 + 100k creation * 3 + 100k output * 8
    assert cost is not None
    assert cost.amount == pytest.approx(2.525)
    assert cost.unit == "proxy_accounting_unit"
    assert cost.to_dict()["unit_source"] == "fixture metadata"


def test_long_context_override_is_selected_per_request():
    quote = pricing(
        long_context=LongContextPricing(
            100, ordinary_input_per_million=4.0, cached_input_per_million=1.0,
            cache_creation_per_million=6.0, output_per_million=16.0,
        )
    )
    short = quote.cost(100, 10)
    long = quote.cost(101, 10)
    assert short is not None and long is not None
    assert short.amount == pytest.approx((100 * 2 + 10 * 8) / 1_000_000)
    assert long.amount == pytest.approx((101 * 4 + 10 * 16) / 1_000_000)


def test_non_chat_modalities_are_rejected_instead_of_token_priced():
    with pytest.raises(ValueError, match="unsupported pricing modality"):
        pricing(modality="image").cost(100, 10)


class Response:
    def __init__(self, body, status_code=200):
        self.body = body
        self.status_code = status_code

    def json(self):
        return self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            error = RuntimeError("upstream body intentionally hidden")
            error.response = self
            raise error


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def live_source(session):
    return UniversityPricingSource(
        LLMProviderConfig(
            type="university", model="chat-model", credentials_env="KEY", base_url_env="BASE"
        ),
        environment={"KEY": "top-secret-value", "BASE": "https://internal.invalid"},
        session=session,
    )


def test_university_live_source_joins_exact_model_and_sanitizes_snapshot():
    session = Session(
        [
            Response({"data": [{"id": "chat-model"}, {"id": "unrelated"}]}),
            Response({"data": [
                {
                    "model_name": "chat-model", "mode": "chat",
                    "input_cost_per_token": 0.000002,
                    "cache_read_input_token_cost": 0.0000005,
                    "cache_creation_input_token_cost": 0.000003,
                    "output_cost_per_token": 0.000008,
                    "input_cost_per_token_above_128k_tokens": 0.000004,
                    "output_cost_per_token_above_128k_tokens": 0.000016,
                    "rpm": 250, "tpm": 250000,
                    "max_input_tokens": 200000, "max_output_tokens": 32000,
                    "api_base": "https://deployment-secret.invalid",
                    "deployment_id": "private-deployment",
                },
                {"model_name": "unrelated", "input_cost_per_token": 99},
            ]}),
            Response({
                "user_id": "private-account", "max_budget": 10, "spend": 3,
                "accounting_unit": "proxy_accounting_unit", "api_key": "top-secret-value",
            }),
        ]
    )
    quote = live_source(session).fetch("university", "chat-model")
    assert quote.status == "known"
    assert quote.available is True
    assert quote.pricing is not None
    assert quote.pricing.cached_input_per_million == pytest.approx(0.5)
    assert quote.pricing.long_context is not None
    assert quote.pricing.limits == ProviderLimits(250, 250000, 200000, 32000)
    assert quote.account_budget is not None
    assert quote.account_budget.remaining.amount == 7
    artifact = sanitized_snapshot_bytes(quote).decode()
    for forbidden in (
        "top-secret-value", "internal.invalid", "deployment-secret", "private-deployment",
        "private-account", "unrelated",
    ):
        assert forbidden not in artifact
    assert [call[0].split("internal.invalid", 1)[1] for call in session.calls] == [
        "/models", "/v1/model/info", "/user/info"
    ]


def test_cached_and_offline_sources_have_explicit_provenance_and_staleness(tmp_path: Path):
    quote = OfflinePricingSource().fetch("openai", "gpt-4o-mini")
    assert quote.mode == "offline" and quote.status == "known"
    path = tmp_path / "snapshot.json"
    data = quote.to_dict()
    data["retrieved_at"] = "2000-01-01T00:00:00Z"
    path.write_text(json.dumps(data), encoding="utf-8")
    cached = CachedPricingSource(path, max_age=timedelta(seconds=1)).fetch(
        "openai", "gpt-4o-mini"
    )
    assert cached.mode == "cached" and cached.status == "stale"
    missing = OfflinePricingSource().fetch("openai", "made-up-model")
    assert missing.status == "unavailable" and missing.pricing is None


def test_preflight_preserves_unit_and_uses_explicit_safe_statuses():
    config = LLMProviderConfig(type="university", model="chat-model")
    from mas_cc.llm_runtime.providers import PricingQuote

    quote = PricingQuote(
        "live", "known", "university", "chat-model", NOW, "fixture", "v1", True,
        pricing(), fresh_until="2099-01-01T00:00:00Z",
    )
    system = BudgetLimits(max_cost=money(1.0))
    run = BudgetLimits(max_cost=money(0.5), max_requests=2, max_input_tokens=1000, max_output_tokens=100)
    result = static_preflight(
        request(), config, LogicalCallSpec(2), pricing_quote=quote,
        system_budget=system, run_budget=run, cached_input_tokens_per_call=1,
    )
    assert result.costs.expected.unit == "proxy_accounting_unit"
    assert result.expected_cost_usd is None
    assert result.launch_status == "permitted"
    assert result.provider_account_budget is None
    unavailable = OfflinePricingSource().fetch("university", "missing")
    blocked = static_preflight(request(), config, LogicalCallSpec(1), pricing_quote=unavailable)
    assert blocked.launch_status == "explicit-override-required"


def test_run_budget_cannot_raise_system_limit():
    with pytest.raises(ValueError, match="cannot raise"):
        resolve_budget_limits(BudgetLimits(max_cost=money(1)), BudgetLimits(max_cost=money(2)))
    with pytest.raises(ValueError, match="same accounting unit"):
        resolve_budget_limits(BudgetLimits(max_cost=money(1)), BudgetLimits(max_cost=money(0.5, "EUR")))


def test_runtime_guard_atomic_reservations_cannot_overspend():
    guard = RuntimeBudgetGuard(
        BudgetLimits(max_cost=money(1), max_requests=10, max_input_tokens=100, max_output_tokens=100)
    )

    def reserve(_):
        try:
            guard.reserve(conservative_cost=money(0.11), input_tokens=1, output_tokens=1)
        except ProviderError as exc:
            return exc.code
        return "reserved"

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(reserve, range(40)))
    assert results.count("reserved") == 9
    assert set(results) == {"reserved", "budget_exhausted"}
    assert guard.status()["used_and_reserved"]["cost"]["amount"] <= 1
    with pytest.raises(ProviderError) as captured:
        RuntimeBudgetGuard(BudgetLimits()).reserve(
            conservative_cost=None, input_tokens=1, output_tokens=1
        )
    assert captured.value.code == "budget_unbounded"


def test_estimate_cost_returns_typed_amount_not_implicit_currency():
    result = estimate_cost(
        pricing(), input_tokens_per_call=100, output_tokens_per_call=10, logical_calls=3
    )
    assert isinstance(result, MonetaryAmount)
    assert result.unit == "proxy_accounting_unit"


def test_phase_4_amendment_bundle_is_complete_and_preflight_only(tmp_path: Path):
    model = "gwdg/qwen3-30b-a3b-instruct-2507"
    session = Session([
        Response({"data": [{"id": model}]}),
        Response({"data": [{
            "model_name": model, "mode": "chat", "input_cost_per_token": 0,
            "output_cost_per_token": 0, "rpm": 2000,
        }]}),
        Response({"max_budget": 20, "spend": 2, "accounting_unit": "proxy_accounting_unit",
                  "user_id": "must-not-escape"}),
    ])
    output = tmp_path / "phase_04_amendment"
    assert inspect_phase_4_amendment(
        "configs/runs/provider_smoke_test.yaml", output,
        pricing_mode="live", run_regressions=False,
        environment={"POTSDAM_API_KEY": "inspection-secret-key",
                     "BASE_POTSDAM_LLM_URL": "https://internal.invalid"},
        session=session,
    )
    expected = {
        "report.md", "manifest.json", "resolved_config.yaml",
        "selected_model_availability.json", "pricing_snapshot.json",
        "pricing_snapshot.sha256", "preflight_estimate.json", "budget_status.json",
        "runtime_guard_scenarios.json", "regression_test_summary.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    combined = "\n".join(path.read_text(encoding="utf-8") for path in output.iterdir())
    assert "inspection-secret-key" not in combined
    assert "internal.invalid" not in combined
    assert "must-not-escape" not in combined
    preflight = json.loads((output / "preflight_estimate.json").read_text())
    assert preflight["launch_status"] == "permitted"
    assert preflight["pricing"]["mode"] == "live"
    assert json.loads((output / "regression_test_summary.json").read_text())["live_completion"] == "not-requested"
    assert len(session.calls) == 3
