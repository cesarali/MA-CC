"""Provider-independent, auditable pricing records and sources.

Importing this module performs no environment reads, file I/O, or network work.
Live access happens only when :meth:`UniversityPricingSource.fetch` is called.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol

from mas_cc.llm_runtime.config import LLMProviderConfig

from .errors import ProviderError


PRICING_STATUSES = {"known", "partial", "stale", "unavailable", "unit-unknown"}
PRICING_MODES = {"live", "cached", "offline"}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return None if number is None else int(number)


def _first(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _find_nested(value: Any, *names: str) -> Any:
    if isinstance(value, Mapping):
        found = _first(value, *names)
        if found is not None:
            return found
        for item in value.values():
            found = _find_nested(item, *names)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_nested(item, *names)
            if found is not None:
                return found
    return None


@dataclass(frozen=True, slots=True)
class MonetaryAmount:
    """An amount whose accounting unit and provenance are never implicit."""

    amount: float
    unit: str
    unit_source: str
    provider: str
    model: str
    source: str
    retrieved_at: str
    version: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.amount) or self.amount < 0:
            raise ValueError("monetary amount must be finite and non-negative")
        for name in ("unit", "unit_source", "provider", "model", "source", "retrieved_at", "version"):
            if not getattr(self, name):
                raise ValueError(f"monetary amount {name} cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "amount": self.amount,
            "unit": self.unit,
            "unit_source": self.unit_source,
            "provider": self.provider,
            "model": self.model,
            "source": self.source,
            "retrieved_at": self.retrieved_at,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class ProviderLimits:
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    maximum_input_tokens: int | None = None
    maximum_output_tokens: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "requests_per_minute": self.requests_per_minute,
            "tokens_per_minute": self.tokens_per_minute,
            "maximum_input_tokens": self.maximum_input_tokens,
            "maximum_output_tokens": self.maximum_output_tokens,
        }


@dataclass(frozen=True, slots=True)
class LongContextPricing:
    threshold_input_tokens: int
    ordinary_input_per_million: float | None = None
    cached_input_per_million: float | None = None
    cache_creation_per_million: float | None = None
    output_per_million: float | None = None

    def __post_init__(self) -> None:
        if self.threshold_input_tokens < 1:
            raise ValueError("long-context threshold must be positive")
        for value in (
            self.ordinary_input_per_million,
            self.cached_input_per_million,
            self.cache_creation_per_million,
            self.output_per_million,
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("long-context rates must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold_input_tokens": self.threshold_input_tokens,
            "ordinary_input_per_million": self.ordinary_input_per_million,
            "cached_input_per_million": self.cached_input_per_million,
            "cache_creation_per_million": self.cache_creation_per_million,
            "output_per_million": self.output_per_million,
        }


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Rates for one exact provider/model and one accounting unit."""

    provider: str
    model: str
    ordinary_input_per_million: float | None
    output_per_million: float | None
    unit: str
    unit_source: str
    source: str
    retrieved_at: str
    version: str
    cached_input_per_million: float | None = None
    cache_creation_per_million: float | None = None
    long_context: LongContextPricing | tuple[LongContextPricing, ...] | None = None
    limits: ProviderLimits = ProviderLimits()
    modality: str = "chat"

    def __post_init__(self) -> None:
        for value in (
            self.ordinary_input_per_million,
            self.cached_input_per_million,
            self.cache_creation_per_million,
            self.output_per_million,
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("pricing rates must be finite and non-negative")
        if isinstance(self.long_context, LongContextPricing):
            object.__setattr__(self, "long_context", (self.long_context,))
        elif self.long_context is not None:
            ordered = tuple(sorted(self.long_context, key=lambda item: item.threshold_input_tokens))
            object.__setattr__(self, "long_context", ordered)

    # Phase 4 v1 compatibility views. They are deliberately unavailable for
    # non-USD quotes, preventing an accounting unit from masquerading as USD.
    @property
    def input_usd_per_million_tokens(self) -> float | None:
        return self.ordinary_input_per_million if self.unit == "USD" else None

    @property
    def output_usd_per_million_tokens(self) -> float | None:
        return self.output_per_million if self.unit == "USD" else None

    def _rates(self, input_tokens: int) -> tuple[float | None, float | None, float | None, float | None]:
        selected = None
        for override in self.long_context or ():
            if input_tokens > override.threshold_input_tokens:
                selected = override
        if selected is not None:
            return (
                selected.ordinary_input_per_million,
                selected.cached_input_per_million,
                selected.cache_creation_per_million,
                selected.output_per_million,
            )
        return (
            self.ordinary_input_per_million,
            self.cached_input_per_million,
            self.cache_creation_per_million,
            self.output_per_million,
        )

    def cost(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        cached_input_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> MonetaryAmount | None:
        if min(input_tokens, output_tokens, cached_input_tokens, cache_creation_tokens) < 0:
            raise ValueError("token counts cannot be negative")
        if cached_input_tokens + cache_creation_tokens > input_tokens:
            raise ValueError("cached and cache-creation tokens cannot exceed total input")
        if self.modality != "chat":
            raise ValueError(f"unsupported pricing modality {self.modality!r} for chat")
        ordinary, cached, creation, output = self._rates(input_tokens)
        required = [ordinary if input_tokens - cached_input_tokens - cache_creation_tokens else 0,
                    cached if cached_input_tokens else 0,
                    creation if cache_creation_tokens else 0,
                    output if output_tokens else 0]
        if any(rate is None for rate in required):
            return None
        uncached = input_tokens - cached_input_tokens - cache_creation_tokens
        amount = (
            uncached * float(ordinary or 0)
            + cached_input_tokens * float(cached or 0)
            + cache_creation_tokens * float(creation or 0)
            + output_tokens * float(output or 0)
        ) / 1_000_000
        return MonetaryAmount(
            amount=amount,
            unit=self.unit,
            unit_source=self.unit_source,
            provider=self.provider,
            model=self.model,
            source=self.source,
            retrieved_at=self.retrieved_at,
            version=self.version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "unit": self.unit,
            "unit_source": self.unit_source,
            "source": self.source,
            "retrieved_at": self.retrieved_at,
            "version": self.version,
            "modality": self.modality,
            "rates_per_million_tokens": {
                "ordinary_input": self.ordinary_input_per_million,
                "cached_input_read": self.cached_input_per_million,
                "cache_creation": self.cache_creation_per_million,
                "output": self.output_per_million,
            },
            "context_threshold_overrides": [item.to_dict() for item in self.long_context or ()],
            "limits": self.limits.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelPricing":
        rates = value.get("rates_per_million_tokens", value)
        long = value.get("context_threshold_overrides", value.get("long_context"))
        limits = value.get("limits", {})
        if not isinstance(rates, Mapping) or not isinstance(limits, Mapping):
            raise ValueError("invalid pricing record")
        return cls(
            provider=str(value["provider"]), model=str(value["model"]),
            ordinary_input_per_million=_number(_first(rates, "ordinary_input", "ordinary_input_per_million", "input_usd_per_million_tokens")),
            cached_input_per_million=_number(_first(rates, "cached_input_read", "cached_input_per_million")),
            cache_creation_per_million=_number(_first(rates, "cache_creation", "cache_creation_per_million")),
            output_per_million=_number(_first(rates, "output", "output_per_million", "output_usd_per_million_tokens")),
            unit=str(value.get("unit", "USD" if "input_usd_per_million_tokens" in rates else "unknown")),
            unit_source=str(value.get("unit_source", "legacy static catalog")),
            source=str(value.get("source", "unknown")),
            retrieved_at=str(value.get("retrieved_at", "1970-01-01T00:00:00Z")),
            version=str(value.get("version", value.get("quote_version", "unknown"))),
            modality=str(value.get("modality", "chat")),
            long_context=_parse_context_overrides(long),
            limits=ProviderLimits(
                requests_per_minute=_integer(limits.get("requests_per_minute")),
                tokens_per_minute=_integer(limits.get("tokens_per_minute")),
                maximum_input_tokens=_integer(limits.get("maximum_input_tokens")),
                maximum_output_tokens=_integer(limits.get("maximum_output_tokens")),
            ),
        )


def _parse_context_overrides(value: Any) -> tuple[LongContextPricing, ...] | None:
    items = [value] if isinstance(value, Mapping) else value
    if not isinstance(items, (list, tuple)):
        return None
    parsed = tuple(
        LongContextPricing(
            threshold_input_tokens=int(item["threshold_input_tokens"]),
            ordinary_input_per_million=_number(item.get("ordinary_input_per_million")),
            cached_input_per_million=_number(item.get("cached_input_per_million")),
            cache_creation_per_million=_number(item.get("cache_creation_per_million")),
            output_per_million=_number(item.get("output_per_million")),
        )
        for item in items if isinstance(item, Mapping)
    )
    return parsed or None


@dataclass(frozen=True, slots=True)
class AccountBudget:
    limit: MonetaryAmount | None = None
    spent: MonetaryAmount | None = None
    remaining: MonetaryAmount | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "limit": None if self.limit is None else self.limit.to_dict(),
            "spent": None if self.spent is None else self.spent.to_dict(),
            "remaining": None if self.remaining is None else self.remaining.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PricingQuote:
    mode: str
    status: str
    provider: str
    model: str
    retrieved_at: str
    source: str
    version: str
    available: bool | None
    pricing: ModelPricing | None
    account_budget: AccountBudget | None = None
    fresh_until: str | None = None
    warning: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in PRICING_MODES:
            raise ValueError(f"unsupported pricing mode {self.mode!r}")
        if self.status not in PRICING_STATUSES:
            raise ValueError(f"unsupported pricing status {self.status!r}")

    @property
    def is_fresh(self) -> bool | None:
        if self.fresh_until is None:
            return None
        return utc_now() <= parse_timestamp(self.fresh_until)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "mode": self.mode,
            "status": self.status,
            "provider": self.provider,
            "model": self.model,
            "retrieved_at": self.retrieved_at,
            "fresh_until": self.fresh_until,
            "source": self.source,
            "version": self.version,
            "available": self.available,
            "pricing": None if self.pricing is None else self.pricing.to_dict(),
            "provider_account_budget": None if self.account_budget is None else self.account_budget.to_dict(),
            "warning": self.warning,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, mode: str | None = None) -> "PricingQuote":
        pricing = value.get("pricing")
        budget = value.get("provider_account_budget")
        parsed_budget = None
        if isinstance(budget, Mapping):
            def money(name: str) -> MonetaryAmount | None:
                item = budget.get(name)
                return None if not isinstance(item, Mapping) else MonetaryAmount(**item)
            parsed_budget = AccountBudget(money("limit"), money("spent"), money("remaining"))
        return cls(
            mode=mode or str(value["mode"]), status=str(value["status"]),
            provider=str(value["provider"]), model=str(value["model"]),
            retrieved_at=str(value["retrieved_at"]), source=str(value["source"]),
            version=str(value["version"]), available=value.get("available"),
            pricing=None if not isinstance(pricing, Mapping) else ModelPricing.from_mapping(pricing),
            account_budget=parsed_budget,
            fresh_until=value.get("fresh_until"), warning=value.get("warning"),
        )


class PricingSource(Protocol):
    mode: str

    def fetch(self, provider: str, model: str) -> PricingQuote: ...


@dataclass(frozen=True, slots=True)
class PricingCatalog:
    version: str
    entries: tuple[ModelPricing, ...]

    def find(self, provider: str, model: str) -> ModelPricing | None:
        return next((entry for entry in self.entries if entry.provider == provider and entry.model == model), None)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PricingCatalog":
        version, entries = value.get("version"), value.get("entries")
        if not isinstance(version, str) or not version or not isinstance(entries, list):
            raise ValueError("pricing catalog requires version and entries")
        return cls(version, tuple(ModelPricing.from_mapping(item) for item in entries))


class OfflinePricingSource:
    mode = "offline"

    def __init__(self, catalog: PricingCatalog | None = None) -> None:
        self.catalog = catalog or default_pricing_catalog()

    def fetch(self, provider: str, model: str) -> PricingQuote:
        pricing = self.catalog.find(provider, model)
        retrieved = pricing.retrieved_at if pricing else "1970-01-01T00:00:00Z"
        return PricingQuote(
            mode=self.mode, status="unavailable" if pricing is None else _pricing_completeness(pricing),
            provider=provider, model=model, retrieved_at=retrieved,
            fresh_until=None, source="versioned MAS-CC static catalog",
            version=self.catalog.version, available=None, pricing=pricing,
            warning="Unknown models are deliberately not assigned guessed rates." if pricing is None else None,
        )


class CachedPricingSource:
    mode = "cached"

    def __init__(self, path: str | Path, *, max_age: timedelta = timedelta(hours=24), allow_stale: bool = False) -> None:
        self.path = Path(path)
        self.max_age = max_age
        self.allow_stale = allow_stale

    def fetch(self, provider: str, model: str) -> PricingQuote:
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("cached pricing snapshot must be a mapping")
        quote = PricingQuote.from_mapping(value, mode="cached")
        if quote.provider != provider or quote.model != model:
            return PricingQuote("cached", "unavailable", provider, model, isoformat(utc_now()),
                                f"sanitized cache {self.path.name}", "cache-mismatch", False, None,
                                warning="Cached snapshot does not match the selected provider/model.")
        fresh_until = parse_timestamp(quote.retrieved_at) + self.max_age
        stale = utc_now() > fresh_until
        status = "stale" if stale else quote.status
        warning = quote.warning
        if stale and not self.allow_stale:
            warning = "Cached quote is stale and policy forbids using it for launch."
        return PricingQuote(
            "cached", status, quote.provider, quote.model, quote.retrieved_at,
            f"sanitized cache {self.path.name}", quote.version, quote.available,
            quote.pricing, quote.account_budget, isoformat(fresh_until), warning,
        )


def _pricing_completeness(pricing: ModelPricing) -> str:
    if pricing.unit == "unknown":
        return "unit-unknown"
    if pricing.modality != "chat":
        return "partial"
    if pricing.ordinary_input_per_million is None or pricing.output_per_million is None:
        return "partial"
    return "known"


class UniversityPricingSource:
    """One-shot read-only source for Potsdam availability, prices, and budget."""

    mode = "live"

    def __init__(self, config: LLMProviderConfig, *, environment: Mapping[str, str] | None = None,
                 session: Any | None = None, include_account_budget: bool = True,
                 freshness: timedelta = timedelta(minutes=15)) -> None:
        self.config = config
        self.environment = environment
        self.session = session
        self.include_account_budget = include_account_budget
        self.freshness = freshness

    def _connection(self) -> tuple[str, str, Any]:
        environment = self.environment
        if environment is None:
            try:
                from dotenv import load_dotenv
                for root in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
                    if (root / ".env").is_file():
                        load_dotenv(root / ".env", override=False)
                        break
            except ImportError:
                pass
            environment = os.environ
        key_name = self.config.credentials_env or "POTSDAM_API_KEY"
        base_name = self.config.base_url_env or "BASE_POTSDAM_LLM_URL"
        key, base = environment.get(key_name, "").strip(), environment.get(base_name, "").strip()
        if not key or not base:
            raise ProviderError(
                f"University pricing credentials/base URL are not configured in {key_name} and {base_name}.",
                provider="university", code="configuration_error",
            )
        session = self.session
        if session is None:
            import requests
            session = requests.Session()
            self.session = session
        return key, base.rstrip("/"), session

    def _get(self, session: Any, url: str, key: str) -> Any:
        response = session.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=self.config.timeout_seconds)
        if response.status_code in (401, 403):
            raise ProviderError(
                f"University pricing query failed with HTTP {response.status_code}.",
                provider="university", code="authentication_failed", status_code=response.status_code,
            )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, Mapping):
            raise ValueError("University metadata response must be a mapping")
        return body

    def fetch_account_budget(self, *, unit: str | None = None) -> AccountBudget | None:
        """Read only the account's own spend, with no pricing round-trips.

        `fetch` costs three requests because it also has to prove the model
        exists and price it. Polling during a run wants none of that — the
        model cannot vanish mid-run and the prices were frozen at launch — so
        this hits `/user/info` alone and is cheap enough to call on a timer.

        ``unit`` pins the accounting unit to the one the launch quote resolved,
        so a polled amount stays comparable with the amount already reserved
        rather than silently re-deriving it from a thinner response body.

        Returns ``None`` when the proxy reports no budget fields, which is not
        an error: it means this account is not spend-tracked.
        """

        key, base, session = self._connection()
        retrieved = utc_now()
        body = self._get(session, f"{base}/user/info", key)
        if unit is None:
            return _account_budget(body, "university", self.config.model, retrieved)
        return _account_budget(
            body, "university", self.config.model, retrieved,
            unit=unit, unit_source="accounting unit pinned to the launch pricing quote",
        )

    def fetch(self, provider: str, model: str) -> PricingQuote:
        if provider != "university":
            raise ValueError("UniversityPricingSource only supports provider='university'")
        key, base, session = self._connection()
        retrieved = utc_now()
        try:
            models = self._get(session, f"{base}/models", key)
        except Exception as exc:
            if getattr(exc, "response", None) is not None and exc.response.status_code == 404:
                models = self._get(session, f"{base}/v1/models", key)
            else:
                raise
        entries = models.get("data", [])
        available = {item.get("id") for item in entries if isinstance(item, Mapping) and isinstance(item.get("id"), str)}
        if model not in available:
            return PricingQuote("live", "unavailable", provider, model, isoformat(retrieved),
                                "University GET /models + GET /v1/model/info", "live-unavailable",
                                False, None, fresh_until=isoformat(retrieved + self.freshness),
                                warning="Selected model is not in the live availability list.")
        info_body = self._get(session, f"{base}/v1/model/info", key)
        info_entries = info_body.get("data", [])
        raw = next((item for item in info_entries if isinstance(item, Mapping)
                    and _first(item, "model_name", "model", "id") == model), None)
        account_body = None
        if self.include_account_budget:
            try:
                account_body = self._get(session, f"{base}/user/info", key)
            except Exception:
                account_body = None
        if raw is None:
            return PricingQuote("live", "partial", provider, model, isoformat(retrieved),
                                "University GET /models + GET /v1/model/info", "live-no-model-info",
                                True, None, _account_budget(account_body, provider, model, retrieved),
                                isoformat(retrieved + self.freshness),
                                "Model is available but has no model-info pricing record.")
        unit, unit_source = _accounting_unit(account_body, info_body)
        pricing = _university_model_pricing(raw, model=model, retrieved=retrieved,
                                            unit=unit, unit_source=unit_source)
        return PricingQuote(
            "live", _pricing_completeness(pricing), provider, model, isoformat(retrieved),
            "University GET /models + GET /v1/model/info", pricing.version, True, pricing,
            _account_budget(account_body, provider, model, retrieved, unit=unit, unit_source=unit_source),
            isoformat(retrieved + self.freshness),
        )


def _accounting_unit(account: Mapping[str, Any] | None, info: Mapping[str, Any]) -> tuple[str, str]:
    explicit = _find_nested(account, "currency", "currency_code", "accounting_unit") if account else None
    if explicit is None:
        explicit = _find_nested(info, "currency", "currency_code", "accounting_unit")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip(), "University metadata field"
    return "proxy_accounting_unit", "University proxy budget semantics; no currency field reported"


def _money(amount: float, *, provider: str, model: str, retrieved: datetime,
           unit: str, unit_source: str) -> MonetaryAmount:
    return MonetaryAmount(amount, unit, unit_source, provider, model, "University GET /user/info",
                          isoformat(retrieved), f"university-account-{retrieved.date().isoformat()}")


def _account_budget(account: Mapping[str, Any] | None, provider: str, model: str,
                    retrieved: datetime, *, unit: str = "proxy_accounting_unit",
                    unit_source: str = "University proxy budget semantics; no currency field reported") -> AccountBudget | None:
    if account is None:
        return None
    maximum = _number(_find_nested(account, "max_budget", "budget"))
    spent = _number(_find_nested(account, "spend"))
    remaining = None if maximum is None or spent is None else max(0.0, maximum - spent)
    if maximum is None and spent is None:
        return None
    build = lambda value: None if value is None else _money(value, provider=provider, model=model,
                                                              retrieved=retrieved, unit=unit, unit_source=unit_source)
    return AccountBudget(build(maximum), build(spent), build(remaining))


def _per_million(raw: Mapping[str, Any], *names: str) -> float | None:
    value = _number(_first(raw, *names))
    return None if value is None else value * 1_000_000


def _university_model_pricing(raw: Mapping[str, Any], *, model: str, retrieved: datetime,
                              unit: str, unit_source: str) -> ModelPricing:
    details = raw.get("model_info")
    details = details if isinstance(details, Mapping) else raw
    params = raw.get("litellm_params")
    params = params if isinstance(params, Mapping) else {}
    mode = str(_first(details, "mode", "model_mode", "modality") or "chat").lower()
    modality = "chat" if mode in {"chat", "completion", "text"} else mode
    thresholds = []
    for suffix, threshold in (("128k", 128_000), ("200k", 200_000), ("272k", 272_000)):
        rates = (
            _per_million(details, f"input_cost_per_token_above_{suffix}_tokens"),
            _per_million(details, f"cache_read_input_token_cost_above_{suffix}_tokens"),
            _per_million(details, f"cache_creation_input_token_cost_above_{suffix}_tokens"),
            _per_million(details, f"output_cost_per_token_above_{suffix}_tokens"),
        )
        if any(value is not None for value in rates):
            thresholds.append(LongContextPricing(threshold, *rates))
    custom_threshold = _integer(_first(details, "long_context_threshold_tokens", "context_threshold_tokens"))
    if custom_threshold is not None:
        rates = (
            _per_million(details, "input_cost_per_token_above_threshold"),
            _per_million(details, "cached_input_cost_per_token_above_threshold"),
            _per_million(details, "cache_creation_cost_per_token_above_threshold"),
            _per_million(details, "output_cost_per_token_above_threshold"),
        )
        if any(value is not None for value in rates):
            thresholds.append(LongContextPricing(custom_threshold, *rates))
    version_value = _first(details, "version", "updated_at", "created_at")
    return ModelPricing(
        provider="university", model=model,
        ordinary_input_per_million=_per_million(details, "input_cost_per_token"),
        cached_input_per_million=_per_million(details, "cache_read_input_token_cost", "cached_input_cost_per_token"),
        cache_creation_per_million=_per_million(details, "cache_creation_input_token_cost", "cache_creation_cost_per_token"),
        output_per_million=_per_million(details, "output_cost_per_token"),
        unit=unit, unit_source=unit_source,
        source="University GET /v1/model/info", retrieved_at=isoformat(retrieved),
        version=str(version_value or f"live-{retrieved.date().isoformat()}"),
        long_context=tuple(thresholds) or None,
        limits=ProviderLimits(
            requests_per_minute=_integer(_first(details, "rpm", "rpm_limit", "requests_per_minute") or _first(params, "rpm")),
            tokens_per_minute=_integer(_first(details, "tpm", "tpm_limit", "tokens_per_minute") or _first(params, "tpm")),
            maximum_input_tokens=_integer(_first(details, "max_input_tokens", "maximum_input_tokens")),
            maximum_output_tokens=_integer(_first(details, "max_output_tokens", "maximum_output_tokens")),
        ),
        modality=modality,
    )


def sanitized_snapshot_bytes(quote: PricingQuote) -> bytes:
    """Canonical bytes containing only the quote's public planning fields."""

    return (json.dumps(quote.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def snapshot_sha256(quote: PricingQuote) -> str:
    return hashlib.sha256(sanitized_snapshot_bytes(quote)).hexdigest()


def default_pricing_catalog() -> PricingCatalog:
    """Small dated catalog; unknown models deliberately remain unknown."""

    retrieved = "2026-08-02T00:00:00Z"
    version = "2026-08-02-provider-economics-v2"
    common = {"retrieved_at": retrieved, "version": version}
    return PricingCatalog(
        version=version,
        entries=(
            ModelPricing("mock", "deterministic-v1", 0.0, 0.0, "USD", "local zero-cost fixture",
                         "MAS-CC static catalog", **common),
            ModelPricing("openai", "gpt-4o-mini", 0.15, 0.60, "USD", "official OpenAI pricing table",
                         "https://developers.openai.com/api/docs/pricing", cached_input_per_million=0.075,
                         **common),
            ModelPricing("university", "gwdg/qwen3-30b-a3b-instruct-2507", 0.0, 0.0,
                         "proxy_accounting_unit", "University proxy model-info snapshot; currency unspecified",
                         "docs/university_llm_api.md dated snapshot", limits=ProviderLimits(requests_per_minute=2000),
                         **common),
            ModelPricing("university", "microsoft/gpt-5.4-nano", 0.20, 1.25,
                         "proxy_accounting_unit", "University proxy model-info snapshot; currency unspecified",
                         "docs/university_llm_api.md dated snapshot", cached_input_per_million=0.02,
                         limits=ProviderLimits(requests_per_minute=250, tokens_per_minute=250000,
                                               maximum_input_tokens=1050000, maximum_output_tokens=128000),
                         **common),
            ModelPricing("gemma_local", "google/gemma-4-12B-it", 0.0, 0.0, "USD",
                         "marginal API cost only; hardware/energy excluded",
                         "MAS-CC static catalog", **common),
        ),
    )
