"""Machine-readable JSON Schema for the Phase 2 run configuration."""

from __future__ import annotations

from typing import Any


def config_schema() -> dict[str, Any]:
    """Return JSON Schema draft 2020-12 for resolved schema version 1."""

    env_name = {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]*$"}
    options = {"type": "object", "additionalProperties": True}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://mas-cc.local/schemas/run-config-v1.json",
        "title": "MAS-CC resolved run configuration",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "llm_provider", "prompt", "game"],
        "properties": {
            "schema_version": {"const": 1},
            "llm_provider": {
                "type": "object",
                "additionalProperties": False,
                "required": ["schema_version", "type", "model"],
                "properties": {
                    "schema_version": {"const": 1},
                    "type": {"type": "string", "minLength": 1},
                    "model": {"type": "string", "minLength": 1},
                    "credentials_env": {"anyOf": [env_name, {"type": "null"}]},
                    "base_url_env": {"anyOf": [env_name, {"type": "null"}]},
                    "timeout_seconds": {"type": "number", "exclusiveMinimum": 0},
                    "max_retries": {"type": "integer", "minimum": 0},
                    "request_concurrency": {"type": "integer", "minimum": 1},
                    "temperature": {"type": "number", "minimum": 0},
                    "max_output_tokens": {"type": "integer", "minimum": 1},
                    "options": options,
                },
            },
            "prompt": {
                "type": "object",
                "additionalProperties": False,
                "required": ["schema_version", "prompt_family", "prompt_version"],
                "allOf": [
                    {
                        "if": {
                            "properties": {"schema_version": {"const": 1}},
                            "required": ["schema_version"],
                        },
                        "then": {"required": ["blocks", "response_contract"]},
                    },
                    {
                        "if": {
                            "properties": {"schema_version": {"const": 2}},
                            "required": ["schema_version"],
                        },
                        "then": {"not": {"required": ["blocks"]}},
                    },
                ],
                "properties": {
                    "schema_version": {"enum": [1, 2]},
                    "prompt_family": {"type": "string", "minLength": 1},
                    "prompt_version": {"type": "integer", "minimum": 1},
                    "blocks": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "response_contract": {
                        "type": "object",
                        "required": ["type"],
                        "properties": {"type": {"type": "string", "minLength": 1}},
                    },
                    "message_mode": {
                        "enum": ["per_block", "merge_consecutive_roles"]
                    },
                    "block_separator": {"type": "string"},
                    "options": options,
                },
            },
            "game": {
                "type": "object",
                "additionalProperties": False,
                "required": ["schema_version", "type", "population_size", "horizon"],
                "properties": {
                    "schema_version": {"const": 1},
                    "type": {"type": "string", "minLength": 1},
                    "population_size": {"type": "integer", "minimum": 2},
                    "horizon": {"type": "integer", "minimum": 1},
                    "topology": {"type": "string", "minLength": 1},
                    "options": options,
                },
            },
            "execution": _simple_object(
                {
                    "schema_version": {"const": 1},
                    "seed": {"type": "integer", "minimum": 0},
                    "repetitions": {"type": "integer", "minimum": 1},
                    "parallelism": {"type": "integer", "minimum": 1},
                    "fail_fast": {"type": "boolean"},
                    "timeout_seconds": {
                        "anyOf": [
                            {"type": "number", "exclusiveMinimum": 0},
                            {"type": "null"},
                        ]
                    },
                }
            ),
            "logging": _simple_object(
                {
                    "schema_version": {"const": 1},
                    "level": {"enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                    "console": {"type": "boolean"},
                    "audit": {"type": "boolean"},
                    "comet": {"type": "boolean"},
                    "options": options,
                }
            ),
            "storage": _simple_object(
                {
                    "schema_version": {"const": 1},
                    "output_dir": {"type": "string", "minLength": 1},
                    "format": {"type": "string", "minLength": 1},
                    "checkpoints": {"type": "boolean"},
                    "overwrite": {"type": "boolean"},
                    "options": options,
                }
            ),
            "analysis": _simple_object(
                {
                    "schema_version": {"const": 1},
                    "enabled": {"type": "boolean"},
                    "estimators": {"type": "array", "items": {"type": "string"}},
                    "options": options,
                }
            ),
            "control": _simple_object(
                {
                    "schema_version": {"const": 1},
                    "mechanism": {"type": "string", "minLength": 1},
                    "options": options,
                }
            ),
            "metrics": _simple_object(
                {
                    "schema_version": {"const": 1},
                    "enabled": {"type": "boolean"},
                    "comet_export": {"type": "array", "items": {"type": "string"}},
                    "available": options,
                }
            ),
            "aggregation": _simple_object(
                {
                    "schema_version": {"const": 1},
                    "forward_fill": {"enum": ["absorbing", "truncate", "none"]},
                    "relabel_by_winner": {"type": "boolean"},
                    "percentiles": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "integer", "minimum": 0, "maximum": 100},
                    },
                    "rolling_window": {"type": "integer", "minimum": 1},
                    "cell_metrics": {"type": "array", "items": {"type": "string"}},
                    "sweep_metrics": {"type": "array", "items": {"type": "string"}},
                    "horizons": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "integer", "minimum": 1},
                    },
                    "null_permutations": {"type": "integer", "minimum": 0},
                }
            ),
            "observability": _simple_object(
                {
                    "schema_version": {"const": 1},
                    "comet": _simple_object(
                        {
                            # One legal writer: workers write episode files, the
                            # master writes Comet, so there is one writer per
                            # experiment key and no step-counter race.
                            "writer": {"const": "master_only"},
                            "heartbeat_seconds": {"type": "number", "exclusiveMinimum": 0},
                            "progress_metrics": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "grid_image_every_n_episodes": {"type": "integer", "minimum": 1},
                            "sweep_experiment": {"type": "boolean"},
                            "cell_experiments": {"type": "boolean"},
                        }
                    ),
                }
            ),
            "experiment": _simple_object(
                {
                    "schema_version": {"const": 1},
                    "name": {"type": "string", "minLength": 1},
                    "description": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "metadata": options,
                }
            ),
            "pricing": _simple_object(
                {
                    "schema_version": {"const": 1},
                    "mode": {"enum": ["live", "cached", "offline"]},
                    "cache_path": {"type": ["string", "null"]},
                    "max_age_seconds": {"type": "number", "minimum": 0},
                    "require_fresh_at_launch": {"type": "boolean"},
                    "fallback_policy": {"enum": ["deny", "offline", "allow_stale"]},
                    "explicit_unknown_price_override": {"type": "boolean"},
                }
            ),
            "budget": _simple_object(
                {
                    "schema_version": {"const": 1},
                    "accounting_unit": {"type": "string", "minLength": 1},
                    "system_max_cost_per_run": {"type": ["number", "null"], "minimum": 0},
                    "max_cost_per_run": {"type": ["number", "null"], "minimum": 0},
                    "max_provider_requests": {"type": ["integer", "null"], "minimum": 0},
                    "max_input_tokens": {"type": ["integer", "null"], "minimum": 0},
                    "max_output_tokens": {"type": ["integer", "null"], "minimum": 0},
                    "allow_unbounded_paid_requests": {"type": "boolean"},
                }
            ),
        },
    }


def _simple_object(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "properties": properties}
