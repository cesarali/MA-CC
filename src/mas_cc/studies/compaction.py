"""Migrate an existing standardized analysis handoff to lean Parquet tables."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .table_io import write_scientific_table


def _tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def compact_study_analysis(study_dir: str | Path) -> dict[str, Any]:
    """Replace retained CSV tables with compressed Parquet and rebuild the ZIP.

    The conversion is staged and verified before any CSV source is removed.
    Estimator values and scientific definitions are not recomputed.
    """

    root = Path(study_dir).expanduser().resolve()
    analysis = root / "analysis"
    tables = analysis / "tables"
    manifest_path = analysis / "analysis_manifest.json"
    if not tables.is_dir() or not manifest_path.is_file():
        raise ValueError(f"not an aggregated standardized study: {root}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    study_id = str(manifest.get("study_id") or root.name)
    csv_paths = sorted(tables.glob("*.csv"))
    before = _tree_size(analysis)
    staging = analysis / ".parquet-compaction.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for source in csv_paths:
            frame = pd.read_csv(source)
            write_scientific_table(staging, source.stem, frame)
        for generated in sorted(staging.glob("*.parquet")):
            generated.replace(tables / generated.name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    for source in csv_paths:
        source.unlink()

    retention = dict(manifest.get("retention_contract") or {})
    retention.update(
        {
            "canonical_table_format": "parquet",
            "csv_tables": False,
            "parquet_tables": True,
        }
    )
    manifest["retention_contract"] = retention
    manifest["tables"] = sorted(path.name for path in tables.glob("*.parquet"))
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    from .aggregation import _package

    archive = _package(analysis, study_id)
    after = _tree_size(analysis)
    return {
        "study_id": study_id,
        "analysis_dir": str(analysis),
        "archive": str(archive),
        "converted_tables": len(csv_paths),
        "before_bytes": before,
        "after_bytes": after,
    }


__all__ = ["compact_study_analysis"]
