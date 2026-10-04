#!/usr/bin/env python3
"""Report whether strict study aggregation completed or failed validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class AggregationProgress:
    study_dir: str
    status: str
    detail: str


def _mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def inspect_aggregation(study_dir: str | Path) -> AggregationProgress:
    root = Path(study_dir).expanduser().resolve()
    analysis = root / "analysis"
    validation_path = analysis / "validation.json"
    manifest_path = analysis / "analysis_manifest.json"

    if validation_path.is_file():
        validation = _mapping(validation_path)
        if validation.get("valid") is False:
            errors = validation.get("errors", [])
            detail = "; ".join(str(error) for error in errors) or str(validation_path)
            return AggregationProgress(str(root), "failed", detail)

    if not manifest_path.is_file():
        return AggregationProgress(str(root), "pending", "analysis manifest is absent")
    manifest = _mapping(manifest_path)
    if manifest.get("status") != "complete":
        return AggregationProgress(
            str(root), "failed", f"analysis manifest status={manifest.get('status')!r}"
        )

    study_id = str(manifest.get("study_id", "")).strip()
    archive = analysis / f"{study_id}_analysis.zip"
    if not study_id or not archive.is_file() or archive.stat().st_size == 0:
        return AggregationProgress(str(root), "pending", "analysis archive is absent")
    return AggregationProgress(str(root), "complete", str(archive))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    args = parser.parse_args()
    try:
        progress = inspect_aggregation(args.study_dir)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.format == "tsv":
        print(f"{progress.status}\t{progress.detail}")
    else:
        print(json.dumps(asdict(progress), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
