#!/usr/bin/env python3
"""Report seal and failed-episode progress for a prepared NERSC study."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class StudyProgress:
    study_dir: str
    expected_cells: int
    sealed_cells: int
    completed_episodes: int
    failed_episodes: int
    in_progress_episodes: int
    complete: bool


def _json_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def inspect_study(study_dir: str | Path) -> StudyProgress:
    root = Path(study_dir).expanduser().resolve()
    manifest = root / "execution_manifest.csv"
    if not manifest.is_file():
        raise ValueError(f"execution manifest is missing: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"execution manifest is empty: {manifest}")

    sealed_cells = 0
    completed_ids: set[str] = set()
    failed_ids: set[str] = set()
    in_progress_ids: set[str] = set()
    seen_outputs: set[Path] = set()
    for row in rows:
        output = Path(row["output_dir"]).expanduser().resolve()
        if output in seen_outputs:
            raise ValueError(f"duplicate shard output in execution manifest: {output}")
        seen_outputs.add(output)
        try:
            output.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"shard output escapes prepared study root: {output}") from exc

        seals = list(output.rglob("cell_complete.json")) if output.is_dir() else []
        if len(seals) > 1:
            raise ValueError(f"multiple cell seals found below shard output: {output}")
        if seals:
            seal = _json_mapping(seals[0])
            if seal.get("status") == "completed":
                sealed_cells += 1
                row_counts = seal.get("episode_row_counts", {})
                if isinstance(row_counts, Mapping):
                    completed_ids.update(
                        f"{output}:{episode_id}" for episode_id in row_counts
                    )
            continue

        if not output.is_dir():
            continue
        for episode_manifest in output.rglob(".resume/*/manifest.json"):
            body = _json_mapping(episode_manifest)
            episode_id = str(body.get("episode_id") or episode_manifest.parent.name)
            identity = f"{output}:{episode_id}"
            status = body.get("status")
            if status == "failed":
                failed_ids.add(identity)
            elif status == "completed":
                completed_ids.add(identity)
            else:
                in_progress_ids.add(identity)

    complete = sealed_cells == len(rows) and not failed_ids
    return StudyProgress(
        study_dir=str(root),
        expected_cells=len(rows),
        sealed_cells=sealed_cells,
        completed_episodes=len(completed_ids),
        failed_episodes=len(failed_ids),
        in_progress_episodes=len(in_progress_ids),
        complete=complete,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    args = parser.parse_args()
    try:
        progress = inspect_study(args.study_dir)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.format == "tsv":
        print(
            "\t".join(
                (
                    str(progress.expected_cells),
                    str(progress.sealed_cells),
                    str(progress.completed_episodes),
                    str(progress.failed_episodes),
                    str(progress.in_progress_episodes),
                    "true" if progress.complete else "false",
                )
            )
        )
    else:
        print(json.dumps(asdict(progress), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
