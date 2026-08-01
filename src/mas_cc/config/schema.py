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
                "required": [
                    "schema_version", "prompt_family", "prompt_version", "blocks",
                    "response_contract",
                ],
                "properties": {
                    "schema_version": {"const": 1},
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
            "experiment": _simple_object(
                {
                    "schema_version": {"const": 1},
                    "name": {"type": "string", "minLength": 1},
                    "description": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "metadata": options,
                }
            ),
        },
    }


def _simple_object(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "properties": properties}
