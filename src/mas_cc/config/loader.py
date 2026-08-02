"""YAML composition, environment expansion, and strict config validation."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from mas_cc.core.exceptions import ConfigurationError
from mas_cc.core.validation import ValidationIssue, ValidationResult

from .models import (
    AnalysisConfig,
    BudgetConfig,
    ExecutionConfig,
    ExperimentConfig,
    GameConfig,
    LLMProviderConfig,
    LoggingConfig,
    PromptConfig,
    PricingConfig,
    RunConfig,
    StorageConfig,
)

ConfigSection = (
    LLMProviderConfig
    | PromptConfig
    | GameConfig
    | ExecutionConfig
    | LoggingConfig
    | StorageConfig
    | AnalysisConfig
    | ExperimentConfig
)

SUPPORTED_SCHEMA_VERSIONS = (1,)

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_SECRET_ENV_NAME = re.compile(r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)
_SECRET_FIELD = re.compile(
    r"(?:^|_)(?:api_key|access_token|auth_token|authorization|bearer|client_secret|"
    r"private_key|secret|password|credential|credentials)(?:$|_)",
    re.IGNORECASE,
)
_ENV_NAME_FIELDS = frozenset(
    {"credentials_env", "base_url_env", "api_key_env", "token_env", "project_env"}
)
_COMPONENT_SECTIONS = frozenset(
    {
        "llm_provider", "provider", "prompt", "game", "execution", "logging",
        "storage", "analysis", "experiment",
    }
)


def _issue(issues: list[ValidationIssue], field: str, message: str, value: Any = None) -> None:
    issues.append(ValidationIssue(field or "config", message, value))


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(
            [ValidationIssue("config", f"cannot read {path}: {exc}")],
            context="configuration loading",
        ) from exc
    try:
        raw = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            [ValidationIssue("config", f"invalid YAML in {path}: {exc}")],
            context="configuration loading",
        ) from exc
    if not isinstance(raw, Mapping):
        raise ConfigurationError(
            [ValidationIssue("config", "top-level YAML value must be a mapping", raw)],
            context="configuration loading",
        )
    return {str(key): value for key, value in raw.items()}


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


def _component_path(reference: str, owner: Path) -> Path:
    candidate = Path(reference)
    if candidate.is_absolute():
        return candidate.resolve()
    relative = (owner.parent / candidate).resolve()
    if relative.exists():
        return relative
    return candidate.resolve()


def _resolve_component(
    value: Any,
    *,
    section: str,
    owner: Path,
    stack: tuple[Path, ...],
) -> Any:
    reference: str | None = None
    overrides: Mapping[str, Any] = {}
    if isinstance(value, str):
        reference = value
    elif isinstance(value, Mapping) and "component" in value:
        unknown = set(value) - {"component", "overrides"}
        if unknown:
            field = f"{section}.{sorted(unknown)[0]}"
            raise ConfigurationError(
                [ValidationIssue(field, "unknown component reference field")],
                context="configuration resolution",
            )
        reference_value = value.get("component")
        if not isinstance(reference_value, str) or not reference_value.strip():
            raise ConfigurationError(
                [ValidationIssue(f"{section}.component", "must be a non-empty path")],
                context="configuration resolution",
            )
        reference = reference_value
        overrides_value = value.get("overrides", {})
        if not isinstance(overrides_value, Mapping):
            raise ConfigurationError(
                [ValidationIssue(f"{section}.overrides", "must be a mapping")],
                context="configuration resolution",
            )
        overrides = overrides_value
    if reference is None:
        return value

    path = _component_path(reference, owner)
    if path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigurationError(
            [ValidationIssue(f"{section}.component", f"component reference cycle: {cycle}")],
            context="configuration resolution",
        )
    component = _read_yaml(path)
    if "component" in component:
        component = _resolve_component(
            component,
            section=section,
            owner=path,
            stack=(*stack, path),
        )
    if not isinstance(component, Mapping):
        raise ConfigurationError(
            [ValidationIssue(f"{section}.component", "referenced component must be a mapping")],
            context="configuration resolution",
        )
    return _deep_merge(component, overrides)


def _resolve_components(raw: Mapping[str, Any], source: Path) -> dict[str, Any]:
    result = dict(raw)
    if "provider" in result and "llm_provider" in result:
        raise ConfigurationError(
            [ValidationIssue("provider", "cannot be used together with llm_provider")],
            context="configuration resolution",
        )
    if "provider" in result:
        result["llm_provider"] = result.pop("provider")
    for section in _COMPONENT_SECTIONS:
        canonical = "llm_provider" if section == "provider" else section
        if canonical in result:
            result[canonical] = _resolve_component(
                result[canonical], section=canonical, owner=source, stack=(source.resolve(),)
            )
    return result


def _expand_environment(
    value: Any,
    *,
    environment: Mapping[str, str],
    path: str,
    issues: list[ValidationIssue],
) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _expand_environment(
                item,
                environment=environment,
                path=f"{path}.{key}" if path else str(key),
                issues=issues,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _expand_environment(
                item,
                environment=environment,
                path=f"{path}[{index}]",
                issues=issues,
            )
            for index, item in enumerate(value)
        ]
    if not isinstance(value, str) or "${" not in value:
        return value

    field_name = path.rsplit(".", 1)[-1]
    if field_name in _ENV_NAME_FIELDS:
        _issue(
            issues,
            path,
            "must name an environment variable directly; secret values are never expanded",
        )
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if _SECRET_ENV_NAME.search(name):
            _issue(issues, path, f"secret environment variable {name!r} cannot be expanded")
            return match.group(0)
        if name in environment:
            return environment[name]
        if default is not None:
            return default
        _issue(issues, path, f"environment variable {name!r} is not set")
        return match.group(0)

    full_reference = _ENV_REFERENCE.fullmatch(value)
    expanded = _ENV_REFERENCE.sub(replace, value)
    if "${" in expanded and not _ENV_REFERENCE.search(expanded):
        _issue(issues, path, "contains a malformed environment-variable reference")
    if full_reference is not None and expanded != value:
        # YAML scalar parsing retains useful types for whole-value references:
        # ``${WORKERS}`` may become an integer while ``run-${NAME}`` stays text.
        parsed = yaml.safe_load(expanded)
        if not isinstance(parsed, (Mapping, list)):
            return parsed
    return expanded


def _validate_secret_fields(value: Any, *, path: str, issues: list[ValidationIssue]) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            child = f"{path}.{key}" if path else key
            if _SECRET_FIELD.search(key) and key not in _ENV_NAME_FIELDS:
                _issue(
                    issues,
                    child,
                    "inline secret fields are forbidden; use an *_env variable-name field",
                )
            if key in _ENV_NAME_FIELDS and item is not None:
                if not isinstance(item, str) or not _ENV_NAME.fullmatch(item):
                    _issue(issues, child, "must be an environment variable name, not a value")
            _validate_secret_fields(item, path=child, issues=issues)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_secret_fields(item, path=f"{path}[{index}]", issues=issues)


def _as_mapping(value: Any, path: str, issues: list[ValidationIssue]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _issue(issues, path, "must be a mapping", value)
        return {}
    return {str(key): item for key, item in value.items()}


def _unknown_fields(
    values: Mapping[str, Any], allowed: set[str], path: str, issues: list[ValidationIssue]
) -> None:
    for name in sorted(set(values) - allowed):
        _issue(issues, f"{path}.{name}" if path else name, "unknown field")


def _string(
    values: Mapping[str, Any],
    key: str,
    path: str,
    issues: list[ValidationIssue],
    *,
    default: str | None = None,
    required: bool = False,
) -> str | None:
    field = f"{path}.{key}" if path else key
    value = values.get(key, default)
    if value is None and not required:
        return None
    if not isinstance(value, str) or (required and not value.strip()):
        _issue(issues, field, "must be a non-empty string" if required else "must be a string", value)
        return default
    return value


def _integer(
    values: Mapping[str, Any],
    key: str,
    path: str,
    issues: list[ValidationIssue],
    *,
    default: int,
    minimum: int | None = None,
) -> int:
    field = f"{path}.{key}" if path else key
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        _issue(issues, field, "must be an integer", value)
        return default
    if minimum is not None and value < minimum:
        _issue(issues, field, f"must be at least {minimum}", value)
        return default
    return value


def _number(
    values: Mapping[str, Any],
    key: str,
    path: str,
    issues: list[ValidationIssue],
    *,
    default: float | None,
    minimum: float | None = None,
) -> float | None:
    field = f"{path}.{key}" if path else key
    value = values.get(key, default)
    if value is None and default is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _issue(issues, field, "must be a number", value)
        return default
    number = float(value)
    if minimum is not None and number < minimum:
        _issue(issues, field, f"must be at least {minimum}", value)
        return default
    return number


def _boolean(
    values: Mapping[str, Any], key: str, path: str, issues: list[ValidationIssue], *, default: bool
) -> bool:
    field = f"{path}.{key}" if path else key
    value = values.get(key, default)
    if not isinstance(value, bool):
        _issue(issues, field, "must be a boolean", value)
        return default
    return value


def _string_tuple(
    values: Mapping[str, Any],
    key: str,
    path: str,
    issues: list[ValidationIssue],
    *,
    required: bool = False,
) -> tuple[str, ...]:
    field = f"{path}.{key}" if path else key
    value = values.get(key, [])
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _issue(issues, field, "must be a list of strings", value)
        return ()
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            _issue(issues, f"{field}[{index}]", "must be a non-empty string", item)
        else:
            result.append(item)
    if required and not result:
        _issue(issues, field, "must contain at least one item", value)
    return tuple(result)


def _schema_version(values: Mapping[str, Any], path: str, issues: list[ValidationIssue]) -> int:
    version = _integer(values, "schema_version", path, issues, default=1, minimum=1)
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        _issue(
            issues,
            f"{path}.schema_version" if path else "schema_version",
            f"unsupported version {version}; supported versions: {SUPPORTED_SCHEMA_VERSIONS}",
            version,
        )
        return 1
    return version


def _parse_provider(raw: Any, issues: list[ValidationIssue]) -> LLMProviderConfig:
    path = "llm_provider"
    values = _as_mapping(raw, path, issues)
    allowed = {
        "schema_version", "type", "model", "credentials_env", "base_url_env",
        "timeout_seconds", "max_retries", "request_concurrency", "temperature",
        "max_output_tokens", "options",
    }
    _unknown_fields(values, allowed, path, issues)
    options = _as_mapping(values.get("options", {}), f"{path}.options", issues)
    credentials = _string(values, "credentials_env", path, issues)
    base_url = _string(values, "base_url_env", path, issues)
    return LLMProviderConfig(
        schema_version=_schema_version(values, path, issues),
        type=_string(values, "type", path, issues, default="invalid", required=True) or "invalid",
        model=_string(values, "model", path, issues, default="invalid", required=True) or "invalid",
        credentials_env=credentials,
        base_url_env=base_url,
        timeout_seconds=_number(values, "timeout_seconds", path, issues, default=60.0, minimum=0.001) or 60.0,
        max_retries=_integer(values, "max_retries", path, issues, default=2, minimum=0),
        request_concurrency=_integer(values, "request_concurrency", path, issues, default=1, minimum=1),
        temperature=_number(values, "temperature", path, issues, default=0.0, minimum=0.0) or 0.0,
        max_output_tokens=_integer(values, "max_output_tokens", path, issues, default=256, minimum=1),
        options=options,
    )


def _parse_prompt(raw: Any, issues: list[ValidationIssue]) -> PromptConfig:
    path = "prompt"
    values = _as_mapping(raw, path, issues)
    allowed = {
        "schema_version", "prompt_family", "prompt_version", "blocks",
        "response_contract", "message_mode", "block_separator", "options",
    }
    _unknown_fields(values, allowed, path, issues)
    response_contract = _as_mapping(
        values.get("response_contract", {}), f"{path}.response_contract", issues
    )
    raw_schema = values.get("schema_version", 1)
    if isinstance(raw_schema, bool) or not isinstance(raw_schema, int):
        _issue(issues, f"{path}.schema_version", "must be an integer", raw_schema)
        prompt_schema = 2
    elif raw_schema not in {1, 2}:
        _issue(
            issues,
            f"{path}.schema_version",
            "unsupported prompt schema; supported versions are 1 and 2",
            raw_schema,
        )
        prompt_schema = 2
    else:
        prompt_schema = raw_schema
    if prompt_schema == 1 and not response_contract:
        _issue(issues, f"{path}.response_contract", "must contain a response contract")
    elif response_contract and (
        not isinstance(response_contract.get("type"), str)
        or not response_contract["type"].strip()
    ):
        _issue(issues, f"{path}.response_contract.type", "must be a non-empty string")
    blocks = _string_tuple(
        values, "blocks", path, issues, required=prompt_schema == 1
    )
    if prompt_schema == 2 and "blocks" in values:
        _issue(
            issues,
            f"{path}.blocks",
            "schema version 2 uses authoritative registered FullPrompt order; remove blocks",
            list(blocks),
        )
    message_mode = _string(values, "message_mode", path, issues)
    if message_mode is not None and message_mode not in {
        "per_block", "merge_consecutive_roles"
    }:
        _issue(
            issues,
            f"{path}.message_mode",
            "must be per_block or merge_consecutive_roles",
            message_mode,
        )
    separator = values.get("block_separator")
    if separator is not None and not isinstance(separator, str):
        _issue(issues, f"{path}.block_separator", "must be a string", separator)
        separator = None
    return PromptConfig(
        schema_version=prompt_schema,
        prompt_family=_string(values, "prompt_family", path, issues, default="invalid", required=True) or "invalid",
        prompt_version=_integer(values, "prompt_version", path, issues, default=1, minimum=1),
        blocks=() if prompt_schema == 2 else blocks,
        response_contract=response_contract,
        message_mode=message_mode,
        block_separator=separator,
        options=_as_mapping(values.get("options", {}), f"{path}.options", issues),
    )


def _parse_game(raw: Any, issues: list[ValidationIssue]) -> GameConfig:
    path = "game"
    values = _as_mapping(raw, path, issues)
    allowed = {"schema_version", "type", "population_size", "horizon", "topology", "options"}
    _unknown_fields(values, allowed, path, issues)
    return GameConfig(
        schema_version=_schema_version(values, path, issues),
        type=_string(values, "type", path, issues, default="invalid", required=True) or "invalid",
        population_size=_integer(values, "population_size", path, issues, default=2, minimum=2),
        horizon=_integer(values, "horizon", path, issues, default=1, minimum=1),
        topology=_string(values, "topology", path, issues, default="complete", required=True) or "complete",
        options=_as_mapping(values.get("options", {}), f"{path}.options", issues),
    )


def _parse_pricing(raw: Any, issues: list[ValidationIssue]) -> PricingConfig:
    path = "pricing"
    values = _as_mapping(raw, path, issues)
    allowed = {"schema_version", "mode", "cache_path", "max_age_seconds",
               "require_fresh_at_launch", "fallback_policy", "explicit_unknown_price_override"}
    _unknown_fields(values, allowed, path, issues)
    mode = _string(values, "mode", path, issues, default="offline", required=True) or "offline"
    fallback = _string(values, "fallback_policy", path, issues, default="deny", required=True) or "deny"
    if mode not in {"live", "cached", "offline"}:
        _issue(issues, f"{path}.mode", "must be live, cached, or offline", mode)
    if fallback not in {"deny", "offline", "allow_stale"}:
        _issue(issues, f"{path}.fallback_policy", "must be deny, offline, or allow_stale", fallback)
    return PricingConfig(
        schema_version=_schema_version(values, path, issues), mode=mode,
        cache_path=_string(values, "cache_path", path, issues),
        max_age_seconds=_number(values, "max_age_seconds", path, issues, default=86400.0, minimum=0.0) or 0.0,
        require_fresh_at_launch=_boolean(values, "require_fresh_at_launch", path, issues, default=True),
        fallback_policy=fallback,
        explicit_unknown_price_override=_boolean(values, "explicit_unknown_price_override", path, issues, default=False),
    )


def _parse_budget(raw: Any, issues: list[ValidationIssue]) -> BudgetConfig:
    path = "budget"
    values = _as_mapping(raw, path, issues)
    allowed = {"schema_version", "accounting_unit", "system_max_cost_per_run", "max_cost_per_run",
               "max_provider_requests", "max_input_tokens", "max_output_tokens",
               "allow_unbounded_paid_requests"}
    _unknown_fields(values, allowed, path, issues)
    def optional_integer(name: str) -> int | None:
        if values.get(name) is None:
            return None
        return _integer(values, name, path, issues, default=0, minimum=0)
    return BudgetConfig(
        schema_version=_schema_version(values, path, issues),
        accounting_unit=_string(values, "accounting_unit", path, issues, default="unknown", required=True) or "unknown",
        system_max_cost_per_run=_number(values, "system_max_cost_per_run", path, issues, default=None, minimum=0.0),
        max_cost_per_run=_number(values, "max_cost_per_run", path, issues, default=None, minimum=0.0),
        max_provider_requests=optional_integer("max_provider_requests"),
        max_input_tokens=optional_integer("max_input_tokens"),
        max_output_tokens=optional_integer("max_output_tokens"),
        allow_unbounded_paid_requests=_boolean(values, "allow_unbounded_paid_requests", path, issues, default=False),
    )


def _parse_execution(raw: Any, issues: list[ValidationIssue]) -> ExecutionConfig:
    path = "execution"
    values = _as_mapping(raw, path, issues)
    _unknown_fields(
        values,
        {"schema_version", "seed", "repetitions", "parallelism", "fail_fast", "timeout_seconds"},
        path,
        issues,
    )
    return ExecutionConfig(
        schema_version=_schema_version(values, path, issues),
        seed=_integer(values, "seed", path, issues, default=0, minimum=0),
        repetitions=_integer(values, "repetitions", path, issues, default=1, minimum=1),
        parallelism=_integer(values, "parallelism", path, issues, default=1, minimum=1),
        fail_fast=_boolean(values, "fail_fast", path, issues, default=True),
        timeout_seconds=_number(values, "timeout_seconds", path, issues, default=None, minimum=0.001),
    )


def _parse_logging(raw: Any, issues: list[ValidationIssue]) -> LoggingConfig:
    path = "logging"
    values = _as_mapping(raw, path, issues)
    _unknown_fields(
        values, {"schema_version", "level", "console", "audit", "comet", "options"}, path, issues
    )
    level = (_string(values, "level", path, issues, default="INFO", required=True) or "INFO").upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        _issue(issues, f"{path}.level", "must be DEBUG, INFO, WARNING, ERROR, or CRITICAL", level)
        level = "INFO"
    return LoggingConfig(
        schema_version=_schema_version(values, path, issues),
        level=level,
        console=_boolean(values, "console", path, issues, default=True),
        audit=_boolean(values, "audit", path, issues, default=True),
        comet=_boolean(values, "comet", path, issues, default=False),
        options=_as_mapping(values.get("options", {}), f"{path}.options", issues),
    )


def _parse_storage(raw: Any, issues: list[ValidationIssue]) -> StorageConfig:
    path = "storage"
    values = _as_mapping(raw, path, issues)
    _unknown_fields(
        values,
        {"schema_version", "output_dir", "format", "checkpoints", "overwrite", "options"},
        path,
        issues,
    )
    return StorageConfig(
        schema_version=_schema_version(values, path, issues),
        output_dir=_string(values, "output_dir", path, issues, default="results", required=True) or "results",
        format=_string(values, "format", path, issues, default="jsonl", required=True) or "jsonl",
        checkpoints=_boolean(values, "checkpoints", path, issues, default=True),
        overwrite=_boolean(values, "overwrite", path, issues, default=False),
        options=_as_mapping(values.get("options", {}), f"{path}.options", issues),
    )


def _parse_analysis(raw: Any, issues: list[ValidationIssue]) -> AnalysisConfig:
    path = "analysis"
    values = _as_mapping(raw, path, issues)
    _unknown_fields(values, {"schema_version", "enabled", "estimators", "options"}, path, issues)
    return AnalysisConfig(
        schema_version=_schema_version(values, path, issues),
        enabled=_boolean(values, "enabled", path, issues, default=False),
        estimators=_string_tuple(values, "estimators", path, issues),
        options=_as_mapping(values.get("options", {}), f"{path}.options", issues),
    )


def _parse_experiment(raw: Any, issues: list[ValidationIssue]) -> ExperimentConfig:
    path = "experiment"
    values = _as_mapping(raw, path, issues)
    _unknown_fields(
        values, {"schema_version", "name", "description", "tags", "metadata"}, path, issues
    )
    return ExperimentConfig(
        schema_version=_schema_version(values, path, issues),
        name=_string(values, "name", path, issues, default="unnamed-experiment", required=True)
        or "unnamed-experiment",
        description=_string(values, "description", path, issues, default="") or "",
        tags=_string_tuple(values, "tags", path, issues),
        metadata=_as_mapping(values.get("metadata", {}), f"{path}.metadata", issues),
    )


def parse_run_config(raw: Mapping[str, Any]) -> RunConfig:
    """Validate a resolved mapping and return immutable typed models."""

    issues: list[ValidationIssue] = []
    values = {str(key): value for key, value in raw.items()}
    allowed = {
        "schema_version", "llm_provider", "prompt", "game", "execution",
        "logging", "storage", "analysis", "experiment", "pricing", "budget",
    }
    _unknown_fields(values, allowed, "", issues)
    schema_version = _schema_version(values, "", issues)
    for required in ("llm_provider", "prompt", "game"):
        if required not in values:
            _issue(issues, required, "required field is missing")

    _validate_secret_fields(values, path="", issues=issues)
    config = RunConfig(
        schema_version=schema_version,
        llm_provider=_parse_provider(values.get("llm_provider", {}), issues),
        prompt=_parse_prompt(values.get("prompt", {}), issues),
        game=_parse_game(values.get("game", {}), issues),
        execution=_parse_execution(values.get("execution", {}), issues),
        logging=_parse_logging(values.get("logging", {}), issues),
        storage=_parse_storage(values.get("storage", {}), issues),
        analysis=_parse_analysis(values.get("analysis", {}), issues),
        experiment=_parse_experiment(values.get("experiment", {}), issues),
        pricing=_parse_pricing(values.get("pricing", {}), issues),
        budget=_parse_budget(values.get("budget", {}), issues),
    )
    if issues:
        raise ConfigurationError(issues, context="configuration validation")
    return config


def validate_run_config(raw: Mapping[str, Any]) -> ValidationResult:
    """Return structured issues instead of raising for an already-resolved mapping."""

    try:
        parse_run_config(raw)
    except ConfigurationError as exc:
        return ValidationResult(tuple(issue for issue in exc.issues if isinstance(issue, ValidationIssue)))
    return ValidationResult.success()


class ConfigLoader:
    """Resolve component references and validate one versioned run YAML file."""

    def __init__(self, environment: Mapping[str, str] | None = None) -> None:
        self._environment = dict(os.environ if environment is None else environment)

    def load(self, path: str | Path) -> RunConfig:
        source = Path(path).resolve()
        raw = _read_yaml(source)
        resolved = _resolve_components(raw, source)
        issues: list[ValidationIssue] = []
        expanded = _expand_environment(
            resolved, environment=self._environment, path="", issues=issues
        )
        if issues:
            raise ConfigurationError(issues, context="environment resolution")
        return parse_run_config(expanded)

    def load_component(self, path: str | Path, component_type: str) -> ConfigSection:
        """Load and validate one reusable component independently of a run."""

        source = Path(path).resolve()
        raw = _read_yaml(source)
        issues: list[ValidationIssue] = []
        expanded = _expand_environment(
            raw, environment=self._environment, path=component_type, issues=issues
        )
        _validate_secret_fields(expanded, path=component_type, issues=issues)
        parsers = {
            "llm_provider": _parse_provider,
            "provider": _parse_provider,
            "prompt": _parse_prompt,
            "game": _parse_game,
            "execution": _parse_execution,
            "logging": _parse_logging,
            "storage": _parse_storage,
            "analysis": _parse_analysis,
            "experiment": _parse_experiment,
        }
        parser = parsers.get(component_type)
        if parser is None:
            _issue(
                issues,
                "component_type",
                f"must be one of: {', '.join(sorted(parsers))}",
                component_type,
            )
        if issues:
            raise ConfigurationError(issues, context="component validation")
        assert parser is not None
        # Parsers attach canonical field paths (for example ``llm_provider.model``)
        # so standalone and composed diagnostics use the same vocabulary.
        parser_issues: list[ValidationIssue] = []
        result = parser(expanded, parser_issues)
        if parser_issues:
            raise ConfigurationError(parser_issues, context="component validation")
        return result


def load_run_config(
    path: str | Path, *, environment: Mapping[str, str] | None = None
) -> RunConfig:
    """Load and fully resolve a run config without reading a ``.env`` file."""

    return ConfigLoader(environment=environment).load(path)


def load_component_config(
    path: str | Path,
    component_type: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> ConfigSection:
    """Load one versioned provider, prompt, game, or supporting component."""

    return ConfigLoader(environment=environment).load_component(path, component_type)


resolve_run_config = load_run_config
