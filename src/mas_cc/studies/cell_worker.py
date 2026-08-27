"""Run one original scientific grid cell selected by an execution manifest."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.config.grid import GridAxis, GridCell
from mas_cc.experiments import run_experiment_grid_sync

from .execution import read_execution_manifest
from .runtime import configure_study_provider_load_control


@dataclass(frozen=True)
class _GridCellShard:
    base: Any
    axes: tuple[GridAxis, ...]
    original_grid_id: str
    selected_cell: GridCell

    @property
    def cells(self) -> tuple[GridCell, ...]:
        return (self.selected_cell,)

    @property
    def grid_id(self) -> str:
        return hashlib.sha256(
            f"{self.original_grid_id}:cell:{self.selected_cell.index}".encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_id": self.grid_id,
            "original_grid_id": self.original_grid_id,
            "selected_cell_index": self.selected_cell.index,
            "selected_cell_id": self.selected_cell.cell_id,
            "axes": [axis.to_dict() for axis in self.axes],
            "cells": [self.selected_cell.to_dict()],
        }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) not in {1, 2}:
        print("usage: python -m mas_cc.studies.cell_worker MANIFEST [INDEX]", file=sys.stderr)
        return 2
    raw_index = args[1] if len(args) == 2 else os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw_index is None:
        print("SLURM_ARRAY_TASK_ID is not set", file=sys.stderr)
        return 2
    try:
        configure_study_provider_load_control(args[0])
        entries = read_execution_manifest(args[0])
        index = int(raw_index)
        if index < 0 or index >= len(entries):
            raise ValueError(f"SLURM array index {index} is outside 0..{len(entries) - 1}")
        entry = entries[index]
        load_dotenv(Path.cwd() / ".env", override=False)
        source = load_run_config_or_grid(entry.config_path)
        if not isinstance(source, GridSpec):
            raise ValueError(f"configuration is not a grid: {entry.config_path}")
        cell = source.cells[entry.cell_index]
        if cell.cell_id != entry.cell_id:
            raise ValueError("execution manifest cell identity no longer matches config")
        shard = _GridCellShard(source.base, source.axes, source.grid_id, cell)
        output = Path(entry.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "shard_definition.json").write_text(
            json.dumps(shard.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"[shard] config={entry.config_index} cell={cell.cell_id} "
            f"index={cell.index} output={output}", flush=True,
        )
        run_experiment_grid_sync(shard, output, show_progress=False)
        return 0
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
