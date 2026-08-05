"""Narrow guard against logging or exporting credential-shaped fields.

Scoped to what the provider/prompt subsystem needs to keep out of Markdown
logs and other diagnostics: a recursive field-name check, independent of the
repository-wide run-config schema.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .exceptions import ConfigurationError
from .validation import ValidationIssue

_SECRET_FIELD = re.compile(
    r"(?:^|_)(?:api_key|access_token|auth_token|authorization|bearer|client_secret|"
    r"private_key|secret|password|credential|credentials)(?:$|_)",
    re.IGNORECASE,
)
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_NAME_FIELDS = frozenset(
    {"credentials_env", "base_url_env", "api_key_env", "token_env", "project_env"}
)


def _validate_secret_fields(value: Any, *, path: str, issues: list[ValidationIssue]) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            child = f"{path}.{key}" if path else key
            if _SECRET_FIELD.search(key) and key not in _ENV_NAME_FIELDS:
                issues.append(
                    ValidationIssue(
                        child, "inline secret fields are forbidden; use an *_env variable-name field", item
                    )
                )
            if key in _ENV_NAME_FIELDS and item is not None:
                if not isinstance(item, str) or not _ENV_NAME.fullmatch(item):
                    issues.append(
                        ValidationIssue(child, "must be an environment variable name, not a value", item)
                    )
            _validate_secret_fields(item, path=child, issues=issues)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_secret_fields(item, path=f"{path}[{index}]", issues=issues)


def assert_secret_free(values: Mapping[str, Any]) -> None:
    """Raise if a serialization mapping contains a secret-bearing field."""

    issues: list[ValidationIssue] = []
    _validate_secret_fields(values, path="", issues=issues)
    if issues:
        raise ConfigurationError(issues, context="secret audit")
