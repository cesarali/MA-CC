"""Canonical scientific-table I/O.

New analysis products are portable CSV.  Parquet remains a read-only legacy
format so retained older archives can still be reaggregated.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

CANONICAL_TABLE_FORMAT = "csv"


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        normalized = sorted(value, key=str) if isinstance(value, set) else value
        return json.dumps(
            normalized, sort_keys=True, separators=(",", ":"), default=str
        )
    return value


def csv_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose genuinely nested cells are deterministic JSON."""

    result = frame.copy()
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].map(_csv_value)
    return result


def write_scientific_table(
    tables_dir: str | Path, name: str, frame: pd.DataFrame
) -> Path:
    """Write and verify one canonical CSV scientific table."""

    directory = Path(tables_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{name}.csv"
    safe = csv_safe(frame)
    serialized = safe.to_csv(index=False, float_format="%.17g", lineterminator="\n")
    destination.write_text(serialized, encoding="utf-8")

    # A writer regression must be discovered at creation time, not after an
    # archive has been handed off.  Parse the actual bytes and compare every
    # scalar/null after pandas' normal CSV inference.
    if len(safe.columns):
        restored = pd.read_csv(destination, dtype=str, keep_default_na=False)
        expected = pd.read_csv(StringIO(serialized), dtype=str, keep_default_na=False)
        assert list(restored.columns) == list(safe.columns)
        assert len(restored) == len(safe)
        assert restored.equals(expected)
    return destination


def retained_table_path(tables_dir: str | Path, name: str) -> Path | None:
    """Prefer canonical CSV, then accept a legacy Parquet table."""

    directory = Path(tables_dir)
    for suffix in (".csv", ".parquet"):
        candidate = directory / f"{name}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def read_scientific_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source, engine="pyarrow")
    raise ValueError(f"unsupported scientific table format: {source}")


__all__ = [
    "CANONICAL_TABLE_FORMAT",
    "csv_safe",
    "read_scientific_table",
    "retained_table_path",
    "write_scientific_table",
]
