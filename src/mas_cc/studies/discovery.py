"""Discover normal and sharded run trees from a study submission manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from mas_cc.config import GridSpec, load_run_config_or_grid

from .identity import protocol_fingerprint, scientific_cell_key
from .submission import SubmissionEntry


@dataclass(frozen=True, slots=True)
class DiscoveredRun:
    entry: SubmissionEntry
    path: Path
    manifest: Mapping[str, Any]
    run_id: str
    game_type: str
    resolved_config: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class DiscoveredCell:
    run: DiscoveredRun
    path: Path
    local_cell_id: str
    resolved_config: Mapping[str, Any]
    overrides: Mapping[str, Any]

    @property
    def source_key(self) -> str:
        return f"config-{self.run.entry.array_index:04d}"

    @property
    def cell_key(self) -> str:
        return self.run.entry.scientific_cell_key or f"{self.source_key}/{self.local_cell_id}"


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read resolved config {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"resolved config must contain a mapping: {path}")
    return value


def discover_runs(entries: tuple[SubmissionEntry, ...]) -> tuple[DiscoveredRun, ...]:
    """Locate actual experiment roots without using directory order as identity."""

    found: list[DiscoveredRun] = []
    for entry in entries:
        output = Path(entry.output_dir)
        candidates: list[Path] = []
        if output.is_dir():
            if (output / "manifest.json").is_file():
                candidates.append(output)
            candidates.extend(path.parent for path in output.rglob("manifest.json"))
        unique = sorted(set(candidates), key=lambda path: str(path))
        scientific_roots: list[Path] = []
        for root in unique:
            manifest = _json(root / "manifest.json")
            if not all(key in manifest for key in ("run_id", "experiment_name", "game_type")):
                continue
            if not any(
                (root / name).is_file()
                for name in ("resolved_config.yaml", "resolved_base_config.yaml")
            ):
                continue
            scientific_roots.append(root)
        # A copied shard may not retain a run manifest at the wrapper root. Its
        # children do, and are deliberately all returned for canonical merging.
        for root in scientific_roots:
            manifest = _json(root / "manifest.json")
            config_path = (
                root / "resolved_base_config.yaml"
                if (root / "resolved_base_config.yaml").is_file()
                else root / "resolved_config.yaml"
            )
            found.append(
                DiscoveredRun(
                    entry=entry,
                    path=root.resolve(),
                    manifest=manifest,
                    run_id=str(manifest["run_id"]),
                    game_type=str(manifest["game_type"]),
                    resolved_config=_yaml(config_path),
                )
            )
    return tuple(found)


def discover_cells(runs: tuple[DiscoveredRun, ...]) -> tuple[DiscoveredCell, ...]:
    cells: list[DiscoveredCell] = []
    seen: set[tuple[int, str, str]] = set()
    for run in runs:
        cells_root = run.path / "cells"
        paths = (
            sorted((path for path in cells_root.iterdir() if path.is_dir()), key=lambda path: path.name)
            if cells_root.is_dir()
            else [run.path]
        )
        for path in paths:
            config_path = path / "resolved_config.yaml"
            resolved = _yaml(config_path) if config_path.is_file() else run.resolved_config
            override_path = path / "overrides.json"
            raw_overrides = _json(override_path) if override_path.is_file() else {}
            overrides = raw_overrides.get("overrides", raw_overrides)
            if not isinstance(overrides, Mapping):
                raise ValueError(f"cell overrides must be a mapping: {override_path}")
            local_id = str(raw_overrides.get("cell_id", path.name if path != run.path else "run"))
            scientific_key = run.entry.scientific_cell_key
            if scientific_key == "AUTO":
                source = load_run_config_or_grid(run.entry.config_path)
                swept_paths = (
                    tuple(axis.path for axis in source.axes)
                    if isinstance(source, GridSpec)
                    else ()
                )
                fingerprint = protocol_fingerprint(
                    resolved, swept_paths=swept_paths
                )
                scientific_key = scientific_cell_key(fingerprint, overrides)
            identity = (run.entry.array_index, run.run_id, local_id)
            if identity in seen:
                # A wrapper may contain both a copied run tree and a link to it.
                # Keep duplicates visible to validation by retaining both.
                local_id = f"{local_id}@{len(cells)}"
            seen.add(identity)
            cells.append(
                DiscoveredCell(
                    run=DiscoveredRun(
                        entry=replace(run.entry, scientific_cell_key=scientific_key),
                        path=run.path,
                        manifest=run.manifest,
                        run_id=run.run_id,
                        game_type=run.game_type,
                        resolved_config=run.resolved_config,
                    ),
                    path=path.resolve(),
                    local_cell_id=local_id,
                    resolved_config=resolved,
                    overrides=dict(overrides),
                )
            )
    return tuple(cells)


__all__ = ["DiscoveredCell", "DiscoveredRun", "discover_cells", "discover_runs"]
