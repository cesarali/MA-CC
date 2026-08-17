#!/usr/bin/env python3
"""Probe OpenAI-compatible model parameter contracts with explicit opt-in."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from mas_cc.llm_runtime.config import LLMProviderConfig
from mas_cc.llm_runtime.providers.adapters.university import UniversityProvider
from mas_cc.llm_runtime.providers.model_profiles import (
    ModelProfile,
    TemperatureRule,
    infer_model_family,
)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    variant: str
    status_code: int | None
    succeeded: bool
    error: str | None = None


def derive_temperature_rule(
    results: Mapping[str, ProbeResult],
) -> TemperatureRule | None:
    """Derive a rule from statuses, never from provider-specific error prose."""

    zero = results["temperature_0"].succeeded
    one = results["temperature_1"].succeeded
    omitted = results["temperature_omitted"].succeeded
    if zero:
        return TemperatureRule("any")
    if one:
        return TemperatureRule("fixed", 1.0)
    if omitted:
        return TemperatureRule("omit")
    return None


def _scrub(value: Any, secrets: Iterable[str]) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    text = re.sub(r"(?i)bearer\s+[a-z0-9._~+/=-]+", "Bearer <redacted>", text)
    return text[:2000]


def _response_error(response: Any, secrets: Iterable[str]) -> str | None:
    if 200 <= response.status_code < 300:
        return None
    try:
        body = response.json()
    except Exception:
        body = getattr(response, "text", "")
    return _scrub(body, secrets)


def _read_models(path: Path) -> tuple[str, ...]:
    models = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not models:
        raise ValueError(f"no models selected in {path}")
    return tuple(sorted(models))


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _provider(args: argparse.Namespace, model: str) -> UniversityProvider:
    return UniversityProvider(
        LLMProviderConfig(
            type="university",
            model=model,
            credentials_env=args.credentials_env,
            base_url_env=args.base_url_env,
            timeout_seconds=args.timeout_seconds,
            max_retries=0,
        )
    )


def _base_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
    }


def _compatible_temperature(rule: TemperatureRule | None) -> dict[str, float]:
    if rule is None or rule.mode == "omit":
        return {}
    if rule.mode == "fixed":
        return {"temperature": rule.value or 1.0}
    return {"temperature": 0.0}


def _probe_payloads(model: str) -> dict[str, dict[str, Any]]:
    base = _base_payload(model)
    return {
        "temperature_0": {**base, "temperature": 0.0},
        "temperature_1": {**base, "temperature": 1.0},
        "temperature_omitted": dict(base),
    }


def _run_payload(provider: UniversityProvider, payload: dict[str, Any]) -> ProbeResult:
    if provider._chat_url is None:  # Endpoint discovery must have run first.
        raise RuntimeError("chat endpoint is not initialized")
    session = provider._get_session()
    variant = str(payload.pop("_probe_variant"))
    try:
        response = session.post(
            provider._chat_url,
            headers={
                "Authorization": f"Bearer {provider._key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=provider._timeout,
        )
    except Exception as exc:
        return ProbeResult(variant, None, False, _scrub(exc, (provider._key,)))
    succeeded = 200 <= response.status_code < 300
    return ProbeResult(
        variant,
        response.status_code,
        succeeded,
        _response_error(response, (provider._key,)),
    )


def _probe_model(provider: UniversityProvider, model: str) -> tuple[ModelProfile, int]:
    results: dict[str, ProbeResult] = {}
    for variant, payload in _probe_payloads(model).items():
        results[variant] = _run_payload(
            provider, {"_probe_variant": variant, **payload}
        )

    temperature = derive_temperature_rule(results)
    compatible = _compatible_temperature(temperature)
    followups = {
        "seed": {**_base_payload(model), **compatible, "seed": 7},
        "system_messages": {
            **_base_payload(model),
            **compatible,
            "messages": [
                {"role": "system", "content": "Reply briefly."},
                {"role": "user", "content": "hi"},
            ],
        },
        "max_tokens": {**_base_payload(model), **compatible},
        "max_completion_tokens": {
            **{
                key: value
                for key, value in _base_payload(model).items()
                if key != "max_tokens"
            },
            **compatible,
            "max_completion_tokens": 16,
        },
    }
    for variant, payload in followups.items():
        results[variant] = _run_payload(
            provider, {"_probe_variant": variant, **payload}
        )

    max_tokens = results["max_tokens"].succeeded
    max_completion_tokens = results["max_completion_tokens"].succeeded
    if max_tokens:
        max_field = "max_tokens"
    elif max_completion_tokens:
        max_field = "max_completion_tokens"
    else:
        max_field = "max_tokens"
    supported = temperature is not None and (max_tokens or max_completion_tokens)
    if temperature is None:
        temperature = TemperatureRule("any")

    notes = tuple(
        f"{item.variant}: "
        + (
            f"HTTP {item.status_code}"
            if item.succeeded
            else f"failed ({'no HTTP response' if item.status_code is None else f'HTTP {item.status_code}'}): {item.error or 'no error body'}"
        )
        for item in results.values()
    )
    profile = ModelProfile(
        provider_type="university",
        model=model,
        family=infer_model_family(model),
        temperature=temperature,
        supports_seed=results["seed"].succeeded if supported else None,
        supports_system_messages=(
            results["system_messages"].succeeded if supported else None
        ),
        max_output_tokens_field=max_field,
        probed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        probe_source="probe",
        supported=supported,
        notes=notes,
    )
    return profile, sum(item.succeeded for item in results.values())


def _existing_profiles(path: Path) -> dict[tuple[str, str], ModelProfile]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (profile.provider_type, profile.model): profile
        for profile in (
            ModelProfile.from_mapping(item) for item in payload.get("profiles", [])
        )
    }


async def _list_models(args: argparse.Namespace) -> int:
    provider = _provider(args, "__model_parameter_probe__")
    try:
        models = await provider.discover_models()
    finally:
        provider.close()
    _atomic_text(args.list_models, "".join(f"{model}\n" for model in sorted(models)))
    print(f"Wrote {len(models)} advertised models to {args.list_models}")
    return 0


async def _probe_models(args: argparse.Namespace, models: tuple[str, ...]) -> int:
    provider = _provider(args, models[0])
    successful_calls = 0
    try:
        advertised = set(await provider.discover_models())
        selected = set(models)
        missing = sorted(selected - advertised)
        if missing:
            raise ValueError("selected models are not advertised: " + ", ".join(missing))
        existing = _existing_profiles(args.output)
        for model in models:
            print(f"Probing {model} ...", flush=True)
            profile, successes = await asyncio.to_thread(_probe_model, provider, model)
            existing[(profile.provider_type, profile.model)] = profile
            successful_calls += successes
    finally:
        provider.close()
    profiles = sorted(existing.values(), key=lambda item: (item.provider_type, item.model))
    payload = {
        "schema_version": 1,
        "profiles": [profile.to_dict() for profile in profiles],
    }
    _atomic_text(args.output, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(models)} profiles to {args.output}")
    print(f"Successful billable probe calls: {successful_calls}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("university",), default="university")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-models", type=Path, metavar="PATH")
    mode.add_argument("--models-file", type=Path, metavar="PATH")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confirm-live-calls", action="store_true")
    parser.add_argument("--credentials-env", default="POTSDAM_API_KEY")
    parser.add_argument("--base-url-env", default="BASE_POTSDAM_LLM_URL")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.models_file is not None and args.output is None:
        parser.error("--output is required with --models-file")
    models = () if args.models_file is None else _read_models(args.models_file)
    if not args.confirm_live_calls:
        if args.list_models is not None:
            print(f"Plan: make one live GET /models call and write {args.list_models}")
        else:
            print(
                f"Plan: probe {len(models)} selected model(s), 7 minimal calls each, "
                f"then update {args.output}:"
            )
            for model in models:
                print(f"  {model}")
        print("No network calls made. Re-run with --confirm-live-calls to execute.")
        return 0
    if args.list_models is not None:
        return asyncio.run(_list_models(args))
    return asyncio.run(_probe_models(args, models))


if __name__ == "__main__":
    sys.exit(main())
