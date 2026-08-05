"""Unified provider smoke test and Phase 4 artifact writer."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
from pathlib import Path
from typing import Any

from mas_cc.config import LLMProviderConfig, PromptConfig, load_component_config
from mas_cc.llm_runtime.providers import (
    BudgetCeiling,
    CompletionRequest,
    ProviderError,
    create_llm_provider,
)
from mas_cc.planning import LogicalCallSpec, static_preflight
from mas_cc.llm_runtime.prompts import CompiledPrompt, RegexTokenCounter

from .inspect import _phase_3_bound_prompt, _write, _write_manifest


PROVIDER_CONFIGS = {
    "mock": Path("configs/components/llm_providers/mock.yaml"),
    "openai": Path("configs/components/llm_providers/openai.yaml"),
    "university": Path("configs/components/llm_providers/university.yaml"),
    "gemma_local": Path("configs/components/llm_providers/gemma_local.yaml"),
}


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _compile_request(
    prompt_path: str | Path,
    *,
    temperature: float,
    max_output_tokens: int,
    seed: int | None,
) -> CompletionRequest:
    _, request = _compile_prompt_and_request(
        prompt_path,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        seed=seed,
    )
    return request


def _compile_prompt_and_request(
    prompt_path: str | Path,
    *,
    temperature: float,
    max_output_tokens: int,
    seed: int | None,
) -> tuple[CompiledPrompt, CompletionRequest]:
    prompt = load_component_config(Path(prompt_path).resolve(), "prompt", environment={})
    if not isinstance(prompt, PromptConfig):
        raise ValueError("prompt component did not resolve to PromptConfig")
    instance = _phase_3_bound_prompt(prompt).compile(RegexTokenCounter())
    request = CompletionRequest(
        messages=instance.messages,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        seed=seed,
        metadata={
            "prompt_family": prompt.prompt_family,
            "prompt_version": prompt.prompt_version,
            "prompt_definition_hash": instance.definition_hash,
            "prompt_instance_hash": instance.instance_hash,
            "response_contract": prompt.response_contract,
        },
    )
    return instance, request


async def provider_smoke_test(
    provider_name: str,
    prompt_path: str | Path,
    output_dir: str | Path,
    *,
    provider_config_path: str | Path | None = None,
    logical_calls: int = 1,
    assumed_output_tokens: int = 16,
    temperature: float = 0.0,
    max_output_tokens: int = 16,
    seed: int | None = 1026,
    budget_usd: float | None = None,
) -> bool:
    """Run one normalized request and always emit a safe inspection bundle."""

    if provider_name not in PROVIDER_CONFIGS:
        raise ValueError(f"unknown provider {provider_name!r}")
    config_path = Path(provider_config_path or PROVIDER_CONFIGS[provider_name]).resolve()
    config = load_component_config(config_path, "llm_provider", environment={})
    if not isinstance(config, LLMProviderConfig):
        raise ValueError("provider component did not resolve to LLMProviderConfig")
    if config.type != provider_name:
        raise ValueError(
            f"provider config type {config.type!r} does not match {provider_name!r}"
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    compiled_prompt, request = _compile_prompt_and_request(
        prompt_path,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        seed=seed,
    )
    budget = None if budget_usd is None else BudgetCeiling(budget_usd)
    preflight = static_preflight(
        request,
        config,
        LogicalCallSpec(logical_calls),
        assumed_output_tokens=assumed_output_tokens,
        budget=budget,
    )
    _write(destination / "request.json", _json(request.to_dict()))
    _write(destination / "compiled_prompt.json", _json(compiled_prompt.to_dict()))
    _write(destination / "preflight_estimate.json", _json(preflight.to_dict()))
    _write(
        destination / "pricing_snapshot.json",
        _json(
            {
                "mode": preflight.pricing_mode,
                "status": preflight.pricing_status,
                "source": preflight.pricing_source,
                "version": preflight.pricing_version,
                "retrieved_at": preflight.pricing_retrieved_at,
                "fresh_until": preflight.pricing_fresh_until,
            }
        ),
    )
    _write(
        destination / "budget_status.json",
        _json(
            {
                "configured_budget_usd": budget_usd,
                "launch_status": preflight.launch_status,
                "within_budget": preflight.within_budget,
            }
        ),
    )
    normalized = [
        {"role": message.role.value, "content": message.content}
        for message in compiled_prompt.messages
    ]
    boundary_ok = request.wire_messages() == normalized
    _write(
        destination / "provider_boundary_diff.md",
        "# Provider boundary diff\n\n"
        f"- Compiled messages equal wire messages: **{str(boundary_ok).lower()}**\n"
        "- Prompt family, versions, hashes, block values, and response contracts remain "
        "local request metadata and are excluded from `wire_messages()`.\n"
        "- Provider adapters receive only `CompletionRequest`.\n",
    )

    provider = None
    response = None
    failure: ProviderError | None = None
    try:
        provider = create_llm_provider(config)
        response = await provider.complete(request)
    except ProviderError as exc:
        failure = exc
    finally:
        if provider is not None:
            provider.close()

    if response is not None:
        _write(destination / "normalized_response.json", _json(response.to_dict()))
        _write(
            destination / "raw_response_redacted.json",
            _json(response.redacted_raw_response()),
        )
        _write(destination / "usage.json", _json(response.usage.to_dict()))
        timing = io.StringIO(newline="")
        writer = csv.writer(timing, lineterminator="\n")
        writer.writerow(["provider", "model", "load_seconds", "inference_seconds", "total_seconds"])
        writer.writerow(
            [
                response.provider,
                response.model,
                response.load_seconds,
                response.inference_seconds,
                response.latency_seconds,
            ]
        )
        _write(destination / "timing.csv", timing.getvalue())
    else:
        assert failure is not None
        _write(destination / "provider_error.json", _json(failure.to_dict()))

    secret_values = [
        value
        for name, value in os.environ.items()
        if any(marker in name.upper() for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
        and len(value) >= 8
    ]
    artifact_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in destination.iterdir()
        if path.is_file()
    )
    secret_free = "Bearer " not in artifact_text and not any(
        value in artifact_text for value in secret_values
    )
    checks = {
        "request_is_normalized": bool(request.messages),
        "compiled_messages_equal_wire_messages": boundary_ok,
        "static_preflight_completed": preflight.logical_calls == logical_calls,
        "normalized_response_received": response is not None,
        "provider_identity_matches": response is not None and response.provider == provider_name,
        "artifacts_are_secret_free": secret_free,
    }
    status = "pass" if all(checks.values()) else "fail"
    warning = (
        "none"
        if failure is None
        else f"normalized provider failure: {failure.code}: {failure}"
    )
    report = f"""# Phase 4 provider inspection report

- Status: **{status.upper()}**
- Provider: `{provider_name}`
- Model: `{config.model}`
- Provider config: `{config_path}`
- Prompt: `{Path(prompt_path).resolve()}`
- Code paths exercised: provider-independent prompt compilation, normalized request construction, static token/call/cost/runtime preflight, lazy adapter creation, completion, usage normalization, raw-response redaction, and timing separation.
- Deviations or warnings: {warning}

## Results

- Static preflight completed: {'passed' if checks['static_preflight_completed'] else 'failed'}
- Normalized response received: {'passed' if checks['normalized_response_received'] else 'failed'}
- Provider identity normalized: {'passed' if checks['provider_identity_matches'] else 'failed'}
- Credential audit: {'passed' if secret_free else 'failed'}

The static token counter is an estimate. Live cost is unknown when the versioned
pricing catalog has no exact provider/model entry. Local Gemma timing records
one-time model loading separately from inference.
"""
    _write(destination / "report.md", report)
    _write_manifest(
        destination,
        phase=4,
        status=status,
        checks=checks,
        warnings=[] if failure is None else [warning],
    )
    return status == "pass"


def run_provider_smoke_test(*args: Any, **kwargs: Any) -> bool:
    return asyncio.run(provider_smoke_test(*args, **kwargs))
