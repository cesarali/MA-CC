#!/usr/bin/env python3
"""Run one original grid cell in its own process.

This is intentionally an orchestration wrapper, not an experiment-engine
change. It retains the original GridCell object (including its index and
cell_id), so the grid-cell and episode seed derivation is byte-for-byte the
same as in the unsplit grid. Provider-backed use must be rate-limited by the
caller because each process owns an independent provider concurrency limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.config.grid import GridAxis, GridCell
from mas_cc.experiments import run_experiment_grid_sync


@dataclass(frozen=True)
class GridCellShard:
    """Grid-compatible view containing one cell from the original grid."""

    base: Any
    axes: tuple[GridAxis, ...]
    original_grid_id: str
    selected_cell: GridCell

    @property
    def cells(self) -> tuple[GridCell, ...]:
        return (self.selected_cell,)

    @property
    def grid_id(self) -> str:
        payload = f"{self.original_grid_id}:cell:{self.selected_cell.index}".encode()
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_id": self.grid_id,
            "original_grid_id": self.original_grid_id,
            "selected_cell_index": self.selected_cell.index,
            "selected_cell_id": self.selected_cell.cell_id,
            "axes": [axis.to_dict() for axis in self.axes],
            "cells": [self.selected_cell.to_dict()],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    load_dotenv(Path.cwd() / ".env")
    source = load_run_config_or_grid(args.config)
    if not isinstance(source, GridSpec):
        raise SystemExit(f"configuration is not a grid: {args.config}")
    if args.cell_index < 0 or args.cell_index >= len(source.cells):
        raise SystemExit(
            f"cell index {args.cell_index} outside 0..{len(source.cells) - 1}"
        )

    cell = source.cells[args.cell_index]
    shard = GridCellShard(source.base, source.axes, source.grid_id, cell)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "shard_definition.json").write_text(
        json.dumps(shard.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[shard] original_grid={source.grid_id} "
        f"cell={cell.cell_id} index={cell.index} overrides={dict(cell.overrides)}",
        flush=True,
    )
    run_experiment_grid_sync(shard, args.output_dir, show_progress=False)


if __name__ == "__main__":
    main()
