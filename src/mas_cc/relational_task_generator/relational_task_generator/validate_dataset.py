#!/usr/bin/env python3
"""Independently validate an existing generated dataset folder."""

from __future__ import annotations

import argparse
from pathlib import Path

from validation import validate_dataset_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a generated relational reasoning dataset."
    )
    parser.add_argument("dataset", type=Path, help="Folder containing manifest.json and task_*.json")
    parser.add_argument(
        "--skip-reproducibility-check",
        action="store_true",
        help="Skip regenerating every task from its stored seed/config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    errors = validate_dataset_directory(
        args.dataset,
        check_reproducibility=not args.skip_reproducibility_check,
    )
    if errors:
        print(f"INVALID: {len(errors)} error(s)")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"VALID: {args.dataset}")


if __name__ == "__main__":
    main()
