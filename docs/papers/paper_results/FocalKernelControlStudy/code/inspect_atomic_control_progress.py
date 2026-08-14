#!/usr/bin/env python3
"""Print a compact progress table for all atomic-control model workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def progress_rows(responses_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(responses_dir.rglob("PROGRESS.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["path"] = str(path.parent.relative_to(responses_dir))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = progress_rows(args.responses_dir.resolve())
    if not rows:
        print("No PROGRESS.json files found.")
        return
    headers = ("MODEL / SHARD", "DONE", "FAILED", "SKIPPED", "TOTAL", "PERCENT", "UPDATED")
    rendered = []
    for row in rows:
        label = f"{row['provider']}:{row['model']}"
        if row["num_shards"] > 1:
            label += f" [{row['shard_index']}/{row['num_shards']}]"
        rendered.append(
            (
                label,
                str(row["stored_completed_prompts"]),
                str(row["stored_failed_prompts"]),
                str(row["skipped_completed_prompts"]),
                str(row["total_prompts"]),
                f"{100 * row['fraction_processed']:.1f}%",
                str(row["updated_at"]),
            )
        )
    widths = [max(len(headers[index]), *(len(row[index]) for row in rendered)) for index in range(len(headers))]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rendered:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


if __name__ == "__main__":
    main()
