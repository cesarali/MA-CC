"""Canonical deterministic prompt fingerprints."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ._values import thaw


def canonical_json(value: Any) -> str:
    return json.dumps(
        thaw(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
