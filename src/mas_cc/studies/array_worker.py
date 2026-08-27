"""Resolve one SLURM config-array row and replace this process with the runner."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .submission import array_task_command, resolve_array_entry
from .runtime import configure_study_provider_load_control


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) not in {1, 2}:
        print("usage: python -m mas_cc.studies.array_worker MANIFEST [INDEX]", file=sys.stderr)
        return 2
    raw_index = arguments[1] if len(arguments) == 2 else os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw_index is None:
        print("SLURM_ARRAY_TASK_ID is not set", file=sys.stderr)
        return 2
    try:
        configure_study_provider_load_control(arguments[0])
        entry = resolve_array_entry(Path(arguments[0]), int(raw_index))
        from mas_cc.cli.main import main as cli_main

        command = array_task_command(entry)
        return int(cli_main(command[3:]))
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
