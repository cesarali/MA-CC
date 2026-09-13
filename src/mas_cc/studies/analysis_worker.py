"""Entry point used by the generic detached study-analysis SLURM launcher."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from .analysis_slurm import finalize, prepare, run_group


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("prepare", "group", "finalize"))
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    if args.role == "prepare":
        prepare(args.manifest)
    elif args.role == "group":
        raw_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw_index is None:
            raise ValueError("SLURM_ARRAY_TASK_ID is required for a group worker")
        run_group(args.manifest, int(raw_index))
    else:
        finalize(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
