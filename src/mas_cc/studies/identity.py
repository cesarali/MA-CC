"""Versioned scientific identities for reusable study observations.

These identities deliberately exclude scheduler topology, paths, target repetition
counts, and analysis settings.  They are not replacements for execution-local run,
cell, or episode labels; those labels remain provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any, Mapping, Sequence


PROTOCOL_FINGERPRINT_VERSION = 1
CELL_KEY_VERSION = 1
EPISODE_KEY_VERSION = 1
SEED_CONTRACT_STABLE_CELL_V1 = "stable-cell-key-v1"
SEED_CONTRACT_LEGACY_GRID_V1 = "legacy-grid-index-v1"

# This registry is intentionally explicit and versioned.  Prefixes describe fields
# that cannot affect an observation and therefore may change during an extension.
PROTOCOL_EXCLUDED_PATHS: tuple[str, ...] = (
    "execution.repetitions",
    "execution.parallelism",
    "execution.fail_fast",
    "storage.output_dir",
    "storage.overwrite",
    "storage.wipe_and_recompute",
    "logging",
    "metrics",
    "aggregation",
    "budget",
    "pricing",
    "experiment.name",
    "experiment.description",
    # These are materialized by the runner from the prompt source. The stable
    # prompt family/version and response contract remain included. Persisted
    # scientific rows separately retain the compiled definition hash.
    "prompt.resolved_block_manifest",
    "prompt.definition_hash",
)


def _typed(value: Any) -> Any:
    """Return a JSON-safe value that retains scalar type distinctions."""

    if value is None:
        return {"type": "null", "value": None}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("scientific identity does not permit non-finite floats")
        return {"type": "float", "value": value.hex()}
    if isinstance(value, str):
        return {"type": "string", "value": value}
    if isinstance(value, Mapping):
        return {
            "type": "mapping",
            "value": [
                {"key": str(key), "value": _typed(item)}
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ],
        }
    if isinstance(value, (list, tuple)):
        return {"type": "sequence", "value": [_typed(item) for item in value]}
    raise TypeError(f"unsupported scientific identity value: {type(value).__name__}")


def _remove_path(payload: dict[str, Any], dotted: str) -> None:
    parts = dotted.split(".")
    current: Any = payload
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)


def protocol_payload(
    resolved_config: Mapping[str, Any], *, swept_paths: Sequence[str] = ()
) -> dict[str, Any]:
    """Build the versioned protocol payload for one resolved cell.

    Swept coordinates are removed because they are represented separately by the
    cell key.  Everything else is included unless the registry above explicitly
    classifies it as operational, target-size, analysis-only, or descriptive.
    """

    payload = deepcopy(dict(resolved_config))
    for path in (*PROTOCOL_EXCLUDED_PATHS, *tuple(swept_paths)):
        _remove_path(payload, path)
    return {
        "protocol_fingerprint_version": PROTOCOL_FINGERPRINT_VERSION,
        "excluded_paths": list(PROTOCOL_EXCLUDED_PATHS),
        "config": _typed(payload),
    }


def _hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def protocol_fingerprint(
    resolved_config: Mapping[str, Any], *, swept_paths: Sequence[str] = ()
) -> str:
    return _hash(protocol_payload(resolved_config, swept_paths=swept_paths))


def canonical_coordinates(coordinates: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "coordinate_encoding_version": 1,
        "coordinates": _typed(dict(coordinates)),
    }


def scientific_cell_key(
    protocol: str, coordinates: Mapping[str, Any]
) -> str:
    return _hash(
        {
            "cell_key_version": CELL_KEY_VERSION,
            "protocol_fingerprint": protocol,
            **canonical_coordinates(coordinates),
        }
    )


def episode_key(cell_key: str, repetition_index: int) -> str:
    if isinstance(repetition_index, bool) or repetition_index < 0:
        raise ValueError("repetition_index must be a non-negative integer")
    return _hash(
        {
            "episode_key_version": EPISODE_KEY_VERSION,
            "cell_key": cell_key,
            "repetition_index": int(repetition_index),
        }
    )


__all__ = [
    "CELL_KEY_VERSION",
    "EPISODE_KEY_VERSION",
    "PROTOCOL_EXCLUDED_PATHS",
    "PROTOCOL_FINGERPRINT_VERSION",
    "SEED_CONTRACT_LEGACY_GRID_V1",
    "SEED_CONTRACT_STABLE_CELL_V1",
    "canonical_coordinates",
    "episode_key",
    "protocol_fingerprint",
    "protocol_payload",
    "scientific_cell_key",
]
