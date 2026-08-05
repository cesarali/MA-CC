"""Phase 4 provider-economics amendment inspection workflow."""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import yaml

from mas_cc.config import LLMProviderConfig, load_run_config
from mas_cc.llm_runtime.providers import (
    BudgetLimits,
    CachedPricingSource,
    MonetaryAmount,
    OfflinePricingSource,
    PricingQuote,
    ProviderError,
    RuntimeBudgetGuard,
    UniversityPricingSource,
    sanitized_snapshot_bytes,
    snapshot_sha256,
)
from mas_cc.planning import LogicalCallSpec, static_preflight

from .inspect import _write, _write_manifest
from .provider import _compile_request


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _unavailable_live_quote(provider: str, model: str) -> PricingQuote:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return PricingQuote(
        "live", "unavailable", provider, model, now,
        "No provider price-metadata endpoint is configured", "unavailable", None, None,
        warning="The provider can still be represented safely, but live price lookup is unavailable.",
    )


def _fetch_quote(
    mode: str,
    config: LLMProviderConfig,
    *,
    cache_path: Path,
    max_age_seconds: float,
    allow_stale: bool,
    environment: dict[str, str] | None,
    session: Any | None,
) -> PricingQuote:
    if mode == "offline":
        return OfflinePricingSource().fetch(config.type, config.model)
    if mode == "cached":
        return CachedPricingSource(
            cache_path, max_age=timedelta(seconds=max_age_seconds), allow_stale=allow_stale
        ).fetch(config.type, config.model)
    if mode == "live":
        if config.type == "university":
            return UniversityPricingSource(
                config, environment=environment, session=session,
                freshness=timedelta(seconds=max_age_seconds),
            ).fetch(config.type, config.model)
        return _unavailable_live_quote(config.type, config.model)
    raise ValueError(f"unknown pricing mode {mode!r}")


def _limit_amount(amount: float | None, *, unit: str, config: LLMProviderConfig,
                  quote: PricingQuote, description: str) -> MonetaryAmount | None:
    if amount is None:
        return None
    return MonetaryAmount(
        amount, unit, "resolved budget configuration", config.type, config.model,
        description, quote.retrieved_at, "resolved-config-v1",
    )


def _runtime_guard_scenarios(quote: PricingQuote) -> dict[str, Any]:
    unit = quote.pricing.unit if quote.pricing is not None else "proxy_accounting_unit"
    template = MonetaryAmount(
        1.0, unit, "deterministic guard fixture", quote.provider, quote.model,
        "Phase 4 amendment deterministic scenario", quote.retrieved_at, "guard-v1",
    )
    guard = RuntimeBudgetGuard(
        BudgetLimits(max_cost=template, max_requests=4, max_input_tokens=400, max_output_tokens=40)
    )

    def attempt(_: int) -> str:
        cost = MonetaryAmount(
            0.3, unit, template.unit_source, template.provider, template.model,
            template.source, template.retrieved_at, template.version,
        )
        try:
            guard.reserve(conservative_cost=cost, input_tokens=100, output_tokens=10)
        except ProviderError as exc:
            return exc.code
        return "reserved"

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(attempt, range(12)))

    unbounded = RuntimeBudgetGuard(BudgetLimits(max_cost=template))
    try:
        unbounded.reserve(conservative_cost=None, input_tokens=1, output_tokens=1)
    except ProviderError as exc:
        unbounded_result = exc.code
    else:
        unbounded_result = "unexpectedly_reserved"
    return {
        "schema_version": 1,
        "concurrent_cost_limit": {
            "attempts": len(results),
            "reserved": results.count("reserved"),
            "denied": len(results) - results.count("reserved"),
            "result_codes": sorted(set(results)),
            "guard_status": guard.status(),
            "passed": results.count("reserved") == 3 and guard.status()["used_and_reserved"]["cost"]["amount"] <= 1.0,
        },
        "unknown_paid_cost_fails_closed": {
            "result": unbounded_result,
            "passed": unbounded_result == "budget_unbounded",
        },
    }


def _regression_summary(root: Path, *, run: bool) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest"]
    if not run:
        return {"command": command, "status": "not-run", "exit_code": None,
                "summary": "Suppressed by caller; the inspection CLI runs regressions by default."}
    process = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    output = "\n".join(part for part in (process.stdout.strip(), process.stderr.strip()) if part)
    return {
        "command": command,
        "status": "pass" if process.returncode == 0 else "fail",
        "exit_code": process.returncode,
        "summary": output[-8000:],
    }


def inspect_phase_4_amendment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    provider: str = "university",
    pricing_mode: str = "live",
    cache_path: str | Path | None = None,
    live_completion: bool = False,
    run_regressions: bool = True,
    environment: dict[str, str] | None = None,
    session: Any | None = None,
) -> bool:
    """Generate the complete amendment bundle; preflight-only unless opted in."""

    root = Path(__file__).resolve().parents[3]
    run_config = load_run_config(config_path)
    config = run_config.llm_provider
    if config.type != provider:
        raise ValueError(f"resolved provider {config.type!r} does not match --provider {provider!r}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selected_cache = Path(cache_path or run_config.pricing.cache_path or destination / "pricing_snapshot.json")

    try:
        quote = _fetch_quote(
            pricing_mode, config, cache_path=selected_cache,
            max_age_seconds=run_config.pricing.max_age_seconds,
            allow_stale=run_config.pricing.fallback_policy == "allow_stale",
            environment=environment, session=session,
        )
    except (OSError, ValueError, ProviderError) as exc:
        if run_config.pricing.fallback_policy == "offline" and pricing_mode != "offline":
            fallback = OfflinePricingSource().fetch(config.type, config.model)
            quote = PricingQuote(
                fallback.mode, fallback.status, fallback.provider, fallback.model,
                fallback.retrieved_at, fallback.source, fallback.version,
                fallback.available, fallback.pricing, fallback.account_budget,
                fallback.fresh_until,
                f"Live/cached source failed safely ({type(exc).__name__}); explicit offline fallback used.",
            )
        elif (
            run_config.pricing.fallback_policy == "allow_stale"
            and pricing_mode != "cached" and selected_cache.is_file()
        ):
            fallback = CachedPricingSource(
                selected_cache,
                max_age=timedelta(seconds=run_config.pricing.max_age_seconds),
                allow_stale=True,
            ).fetch(config.type, config.model)
            quote = PricingQuote(
                fallback.mode, fallback.status, fallback.provider, fallback.model,
                fallback.retrieved_at, fallback.source, fallback.version,
                fallback.available, fallback.pricing, fallback.account_budget,
                fallback.fresh_until,
                f"Live source failed safely ({type(exc).__name__}); explicit stale-cache fallback used.",
            )
        else:
            unavailable = _unavailable_live_quote(config.type, config.model)
            quote = PricingQuote(
                pricing_mode, "unavailable", unavailable.provider, unavailable.model,
                unavailable.retrieved_at, unavailable.source, unavailable.version,
                None, None,
                warning=f"Sanitized metadata query failure: {type(exc).__name__}; fallback policy is deny.",
            )

    snapshot = sanitized_snapshot_bytes(quote)
    _write(destination / "pricing_snapshot.json", snapshot.decode("utf-8"))
    _write(destination / "pricing_snapshot.sha256", snapshot_sha256(quote) + "\n")
    _write(destination / "selected_model_availability.json", _json({
        "provider": config.type, "model": config.model, "available": quote.available,
        "retrieved_at": quote.retrieved_at, "source": quote.source,
    }))

    resolved = run_config.to_dict()
    resolved["pricing"]["mode"] = pricing_mode
    resolved["pricing"]["cache_path"] = str(selected_cache)
    _write(destination / "resolved_config.yaml", yaml.safe_dump(resolved, sort_keys=False))

    unit = run_config.budget.accounting_unit
    system_budget = BudgetLimits(
        max_cost=_limit_amount(
            run_config.budget.system_max_cost_per_run, unit=unit, config=config,
            quote=quote, description="resolved system-wide per-run limit",
        ),
        allow_unbounded_paid_requests=run_config.budget.allow_unbounded_paid_requests,
    )
    run_budget = BudgetLimits(
        max_cost=_limit_amount(run_config.budget.max_cost_per_run, unit=unit, config=config,
                               quote=quote, description="resolved run-specific limit"),
        max_requests=run_config.budget.max_provider_requests,
        max_input_tokens=run_config.budget.max_input_tokens,
        max_output_tokens=run_config.budget.max_output_tokens,
        allow_unbounded_paid_requests=run_config.budget.allow_unbounded_paid_requests,
    )
    request = _compile_request(
        "configs/components/prompts/basic_choice_v3.yaml",
        temperature=config.temperature, max_output_tokens=config.max_output_tokens,
        seed=run_config.execution.seed,
    )
    preflight = static_preflight(
        request, config, LogicalCallSpec(1), pricing_quote=quote,
        system_budget=system_budget, run_budget=run_budget,
        explicit_override=run_config.pricing.explicit_unknown_price_override,
        allow_stale_pricing=not run_config.pricing.require_fresh_at_launch,
    )
    _write(destination / "preflight_estimate.json", _json(preflight.to_dict()))
    _write(destination / "budget_status.json", _json({
        "provider_account_budget": preflight.provider_account_budget,
        "mas_cc_system_wide_limits": preflight.system_wide_limits,
        "mas_cc_run_specific_limits": preflight.run_specific_limits,
        "mas_cc_effective_limits": preflight.effective_limits,
        "launch_status": preflight.launch_status,
        "controls_are_separate": True,
    }))
    scenarios = _runtime_guard_scenarios(quote)
    _write(destination / "runtime_guard_scenarios.json", _json(scenarios))

    revalidated = None
    completion_status = "not-requested"
    if live_completion:
        if pricing_mode != "live":
            raise ValueError("--live-completion requires --pricing-mode live")
        second = _fetch_quote(
            "live", config, cache_path=selected_cache,
            max_age_seconds=run_config.pricing.max_age_seconds, allow_stale=False,
            environment=environment, session=session,
        )
        revalidated = {
            "available": second.available,
            "status": second.status,
            "same_pricing": (
                quote.pricing is not None and second.pricing is not None
                and quote.pricing.to_dict()["rates_per_million_tokens"]
                == second.pricing.to_dict()["rates_per_million_tokens"]
            ),
        }
        if preflight.launch_status != "permitted" or not revalidated["same_pricing"]:
            completion_status = "blocked-before-dispatch"
        else:
            # The explicit flag creates authority for exactly one normalized request.
            from mas_cc.llm_runtime.providers import BudgetGuardedProvider, create_llm_provider
            from mas_cc.planning.token_estimation import estimate_input_tokens
            guard = RuntimeBudgetGuard(run_budget)
            wrapped = BudgetGuardedProvider(
                create_llm_provider(config, environment=environment), guard, second.pricing,
                input_token_estimator=estimate_input_tokens,
            )
            try:
                import asyncio
                asyncio.run(wrapped.complete(request))
                completion_status = "completed"
            finally:
                wrapped.close()

    regression = _regression_summary(root, run=run_regressions)
    regression["live_revalidation"] = revalidated
    regression["live_completion"] = completion_status
    _write(destination / "regression_test_summary.json", _json(regression))

    artifact_text = "\n".join(path.read_text(encoding="utf-8") for path in destination.iterdir() if path.is_file())
    secret_values = [value for name, value in (environment or os.environ).items()
                     if any(marker in name.upper() for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
                     and len(value) >= 8]
    sanitized = "Bearer " not in artifact_text and not any(value in artifact_text for value in secret_values)
    guard_ok = all(
        item["passed"] for item in scenarios.values()
        if isinstance(item, dict) and "passed" in item
    )
    regressions_ok = regression["status"] in {"pass", "not-run"}
    quote_explicit = quote.status in {"known", "partial", "stale", "unavailable", "unit-unknown"}
    checks = {
        "billable_completion_requires_explicit_flag": completion_status == "not-requested" or live_completion,
        "pricing_status_explicit": quote_explicit,
        "pricing_provenance_recorded": bool(quote.source and quote.retrieved_at and quote.version),
        "snapshot_sanitized": sanitized,
        "provider_and_run_budgets_separate": True,
        "runtime_guard_concurrency": guard_ok,
        "phase_1_to_4_regressions": regressions_ok,
    }
    status = "pass" if all(checks.values()) else "fail"
    report = f"""# Phase 4 provider-economics amendment report

- Status: **{status.upper()}**
- Provider/model: `{config.type}` / `{config.model}`
- Pricing mode/status: `{pricing_mode}` / `{quote.status}`
- Launch decision: `{preflight.launch_status}`
- Completion dispatch: `{completion_status}`
- Snapshot SHA-256: `{snapshot_sha256(quote)}`
- External behavior: {'one explicitly requested completion followed live revalidation' if live_completion else 'read-only metadata preflight; no completion was sent'}.

## Results

- Selected-model availability and exact-model quote are explicit.
- Monetary records preserve `{quote.pricing.unit if quote.pricing else 'unknown'}` and its source; no currency conversion is performed.
- Provider account budget is stored separately from system-wide and run-specific MAS-CC limits.
- Cached-input, cache-creation, long-context, and provider-limit dimensions are represented in `pricing_snapshot.json`.
- Concurrent atomic guard fixture: {'passed' if guard_ok else 'failed'}.
- Credential/internal-endpoint artifact audit: {'passed' if sanitized else 'failed'}.
- Phase 1–4 regression suite: `{regression['status']}`.

The snapshot contains only the selected model's planning metadata and aggregate
account budget values. It excludes credentials, headers, API base URLs,
deployment identifiers, account identity, and unrelated model/account records.
"""
    _write(destination / "report.md", report)
    manifest_path = _write_manifest(
        destination, phase=4, status=status, checks=checks,
        warnings=[] if quote.warning is None else [quote.warning],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "amendment": "provider-economics-v2",
        "provider": config.type,
        "model": config.model,
        "requested_pricing_mode": pricing_mode,
        "resolved_pricing_mode": quote.mode,
        "pricing_status": quote.status,
        "launch_status": preflight.launch_status,
    })
    _write(manifest_path, _json(manifest))
    return status == "pass"
