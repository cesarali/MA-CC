#!/usr/bin/env python3
"""Retry failed atomic-control prompts with conservative JSON-object recovery.

This wrapper deliberately reuses the frozen-dataset verification, provider,
locking, result schema, and concurrency machinery from the primary runner. It
changes only two things:

1. select prompt tuples listed in a first-pass ``failures.jsonl``;
2. accept exactly one unambiguous schema-valid JSON vote object when surrounded
   by prose or a Markdown fence.

Exact JSON remains accepted. Zero or multiple valid objects remain failures.
Recovery results are written to a separate output directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path[:0] = [str(SCRIPT_DIR), str(REPO_ROOT / "src")]

import run_atomic_control_calibration as primary  # noqa: E402
from atomic_control_common import atomic_write_json, read_jsonl  # noqa: E402


def valid_vote_object(value: Any, options: list[str]) -> str | None:
    if not isinstance(value, dict) or set(value) != {"vote"}:
        return None
    vote = value["vote"]
    return vote if isinstance(vote, str) and vote in options else None


def parse_vote_recovery(raw: str, options: list[str]) -> tuple[str | None, str | None]:
    """Accept exact JSON or one unambiguous embedded valid vote object."""

    try:
        exact = json.loads(raw)
    except json.JSONDecodeError:
        exact = None
    exact_vote = valid_vote_object(exact, options)
    if exact_vote is not None:
        return exact_vote, None

    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, str]] = []
    for start, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, length = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            continue
        vote = valid_vote_object(value, options)
        if vote is not None:
            candidates.append((start, start + length, vote))

    # Nested scans can rediscover the same span only if the payload contains
    # nested objects. Deduplicate spans before enforcing unambiguity.
    unique = {(start, end, vote) for start, end, vote in candidates}
    if len(unique) == 1:
        return next(iter(unique))[2], None
    if not unique:
        return None, "response contains no schema-valid JSON vote object"
    return None, "response contains multiple schema-valid JSON vote objects"


def failure_keys(path: Path) -> set[tuple[str, str]]:
    rows = read_jsonl(path)
    keys = {(str(row["bucket"]), str(row["state_id"])) for row in rows}
    if not keys:
        raise ValueError(f"failure manifest is empty: {path}")
    return keys


def build_parser() -> argparse.ArgumentParser:
    parser = primary.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--failed-from",
        type=Path,
        required=True,
        help="first-pass failures.jsonl whose prompt keys should be retried",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    failed_from = args.failed_from.resolve()
    selected = failure_keys(failed_from)
    original_load_items = primary.load_items

    def load_failed_items(input_dir: Path) -> list[dict[str, Any]]:
        return [
            row
            for row in original_load_items(input_dir)
            if (str(row["bucket"]), str(row["state_id"])) in selected
        ]

    primary.load_items = load_failed_items
    primary.parse_vote = parse_vote_recovery
    outcomes = asyncio.run(primary.run(args))
    atomic_write_json(
        args.output_dir.resolve() / "RECOVERY_MANIFEST.json",
        {
            "source_failures": str(failed_from),
            "selected_failed_prompts": len(selected),
            "parser": "exact_or_one_unambiguous_embedded_schema_valid_vote_object",
            "model": args.model,
            "provider": args.provider,
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "outcomes": outcomes,
        },
    )
    print(json.dumps(outcomes, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
