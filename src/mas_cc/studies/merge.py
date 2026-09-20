"""Create an independent canonical study from complementary completed studies."""

from __future__ import annotations

import json
import re
import tempfile
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.storage import canonical_hash, file_sha256
from .canonical import TABLE_SCHEMAS, build_canonical_tables
from .identity import episode_key, protocol_fingerprint
from .submission import (
    SubmissionEntry,
    read_submission_manifest,
    write_submission_manifest,
)
from .table_io import read_scientific_table, retained_table_path, write_scientific_table
from .validation import paired_initialization_diagnostics


CORE = ("cells", "episodes", "rounds", "micro_slots")


def _cell_identity(config: dict[str, Any]) -> str:
    payload = deepcopy(config)
    experiment = payload.get("experiment", {})
    experiment.pop("tags", None)
    metadata = experiment.get("metadata", {})
    # These duplicate the descriptive study name and target sample count.
    metadata.pop("study", None)
    metadata.pop("repetitions", None)
    return protocol_fingerprint(payload)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _inputs(root: Path):
    # Use the same lineage-aware discovery as detached aggregation.
    from .analysis_slurm import _entries
    from .extension import consolidate_extension_tables
    from .validation import validate_study

    analysis = root / "analysis" if (root / "analysis/tables").is_dir() else root
    submitted = (root / "study_manifest.json").is_file()
    if submitted:
        manifest = _read(root / "study_manifest.json")
        entries, runs, cells, target = _entries(root)
    else:
        manifest = _read(analysis / "provenance/study_manifest.json")
        entries = read_submission_manifest(
            analysis / "provenance/submission_manifest.csv"
        )
        runs, cells, target = (), (), None
    configs = {}
    if cells:
        tables, _ = build_canonical_tables(str(manifest["study_id"]), cells)
        if target is not None:
            tables, validation = consolidate_extension_tables(tables, target)
        else:
            validation = validate_study(entries, runs, cells, tables)
        configs = {cell.cell_key: dict(cell.resolved_config) for cell in cells}
    else:
        paths = {name: retained_table_path(analysis / "tables", name) for name in CORE}
        if any(path is None for path in paths.values()):
            raise ValueError(f"no complete canonical observations or run trees: {root}")
        tables = {name: read_scientific_table(path) for name, path in paths.items()}
        for name in TABLE_SCHEMAS:
            if name not in tables:
                path = retained_table_path(analysis / "tables", name)
                if path is not None:
                    tables[name] = read_scientific_table(path)
        validation = _read(analysis / "validation.json")
        resolved = {}
        for entry in entries:
            matches = list(
                (analysis / "provenance").glob(f"config-{entry.array_index:04d}-*.y*ml")
            )
            original = Path(entry.config_path)
            path = (
                original
                if original.is_file()
                else (matches[0] if len(matches) == 1 else original)
            )
            if not path.is_file():
                raise ValueError(
                    f"missing config snapshot for index {entry.array_index}: {root}"
                )
            if file_sha256(path) != entry.config_hash:
                raise ValueError(f"config snapshot changed since submission: {path}")
            source = load_run_config_or_grid(path)
            if isinstance(source, GridSpec):
                for cell in source.cells:
                    resolved[(entry.array_index, cell.cell_id)] = cell.config.to_dict()
            else:
                resolved[(entry.array_index, "run")] = source.to_dict()
        for row in tables["cells"].to_dict(orient="records"):
            key = (int(row["source_config_index"]), str(row["source_cell_id"]))
            if key not in resolved:
                raise ValueError(f"cannot recover resolved cell config {key}: {root}")
            configs[str(row["cell_id"])] = resolved[key]
    if not validation.get("valid") or not validation.get("complete"):
        raise ValueError(f"merge requires a valid, complete source study: {root}")
    cell_ids = set(tables["cells"]["cell_id"].astype(str))
    for name, frame in tables.items():
        if not set(frame["cell_id"].astype(str)).issubset(cell_ids):
            raise ValueError(f"orphan cell references in {name}: {root}")
    for row in tables["cells"].to_dict(orient="records"):
        if int(row["completed_episodes"]) != int(row["expected_episodes"]):
            raise ValueError(f"incomplete canonical cell: {root}")
    episodes = tables["episodes"]
    if set(episodes["cell_id"].astype(str)) != cell_ids:
        raise ValueError(f"cells without canonical episodes: {root}")
    if episodes.duplicated(["cell_id", "episode_id"]).any():
        raise ValueError(f"duplicate canonical episodes: {root}")
    for cell_id, group in episodes.groupby("cell_id"):
        expected = int(
            tables["cells"].set_index("cell_id").loc[cell_id, "expected_episodes"]
        )
        if (
            len(group) != expected
            or not group["status"].isin(["completed", "skipped_resumed"]).all()
        ):
            raise ValueError(f"incomplete canonical episodes: {root}")
    return manifest, tables, configs, validation


def merge_studies(
    study_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    name: str | None = None,
    analysis_recipe: str | Path | None = None,
    overlap: str = "error",
) -> dict[str, Any]:
    """Snapshot observations without touching sources or executing estimators.

    Overlapping protocols/cells are rejected unless the user explicitly chooses
    keep-first. Never pool or average precomputed estimates.
    """
    roots = [Path(path).expanduser().resolve() for path in study_dirs]
    destination = Path(output_dir).expanduser().resolve()
    if len(roots) < 2 or len(set(roots)) != len(roots):
        raise ValueError("provide at least two distinct source studies")
    if destination.exists():
        raise ValueError(f"merge destination already exists: {destination}")
    if any(destination.is_relative_to(root) for root in roots):
        raise ValueError("merge destination must be outside the source studies")
    if overlap not in {"error", "keep-first"}:
        raise ValueError("overlap must be error or keep-first")
    study_id = name or destination.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", study_id):
        raise ValueError("study name must be a safe file label")
    inputs = [_inputs(root) for root in roots]
    from .aggregation import _recipe

    recipes = []
    for root, (manifest, _, _, _) in zip(roots, inputs):
        analysis = root / "analysis" if (root / "analysis/tables").is_dir() else root
        recipe_path = analysis / "analysis_recipe.yaml"
        if recipe_path.is_file():
            recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8")) or {}
        else:
            recipe, _ = _recipe(manifest)
        recipes.append(recipe)
    if analysis_recipe is not None:
        recipe = yaml.safe_load(Path(analysis_recipe).read_text(encoding="utf-8")) or {}
    else:
        if any(item != recipes[0] for item in recipes[1:]):
            raise ValueError(
                "source analysis recipes differ; select --analysis-recipe explicitly"
            )
        recipe = recipes[0]
    if not isinstance(recipe, dict):
        raise ValueError("analysis recipe must be a mapping")

    frames: dict[str, list[pd.DataFrame]] = {}
    configs = []
    seen = {}
    skipped = []
    for source_index, (root, (manifest, tables, resolved, _)) in enumerate(
        zip(roots, inputs)
    ):
        if tables["cells"]["cell_id"].duplicated().any():
            raise ValueError(f"duplicate canonical cell IDs in {root}")
        for cell in tables["cells"].to_dict(orient="records"):
            old_id = str(cell["cell_id"])
            config = resolved[old_id]
            physical_key = _cell_identity(config)
            if physical_key in seen:
                if overlap == "error":
                    raise ValueError(
                        f"overlapping scientific cell {old_id} in {root} and {seen[physical_key]}; "
                        "use --overlap keep-first only to intentionally discard the later cell"
                    )
                skipped.append(
                    {
                        "source": str(root),
                        "cell_id": old_id,
                        "kept_source": seen[physical_key],
                    }
                )
                continue
            seen[physical_key] = str(root)
            index = len(configs)
            new_id = f"config-{index:04d}/run"
            configs.append(config)
            for table_name, frame in tables.items():
                part = frame[frame["cell_id"].astype(str) == old_id].copy()
                part["merge_source_study_id"] = str(manifest["study_id"])
                part["merge_source_study_path"] = str(root)
                part["merge_source_cell_id"] = old_id
                part["merge_source_index"] = source_index
                if "source_config_index" in part:
                    part["merge_source_config_index"] = part["source_config_index"]
                    part["source_config_index"] = index
                part["study_id"] = study_id
                part["cell_id"] = new_id
                if "cell_key" in part:
                    part["merge_source_cell_key"] = part["cell_key"]
                    part["cell_key"] = new_id
                if "episode_key" in part:
                    part["merge_source_episode_key"] = part["episode_key"]
                    part["episode_key"] = part["repetition_index"].map(
                        lambda value: episode_key(new_id, int(value))
                    )
                if "source_cell_id" in part:
                    part["source_cell_id"] = "run"
                if "source_run_id" in part:
                    part["source_run_id"] = part["source_run_id"].map(
                        lambda value: f"source-{source_index:04d}/{value}"
                    )
                frames.setdefault(table_name, []).append(part)
    combined = {
        key: pd.concat(value, ignore_index=True) for key, value in frames.items()
    }
    _, _, paired = paired_initialization_diagnostics(combined)
    if paired["required"] and not paired["paired_initialization_pass"]:
        raise ValueError(
            "merged paired initialization is inconsistent: "
            + "; ".join(paired["errors"])
        )
    schemas = (
        combined["episodes"]
        .get("scientific_schema_version", pd.Series(dtype=str))
        .dropna()
        .unique()
    )
    if len(schemas) > 1:
        raise ValueError("source scientific schema versions differ")
    counts = {
        "expected_configs": len(configs),
        "found_configs": len(configs),
        "expected_cells": len(configs),
        "found_cells": len(configs),
        "expected_episodes": int(combined["cells"]["expected_episodes"].sum()),
        "completed_episodes": int(combined["cells"]["completed_episodes"].sum()),
        "round_rows": len(combined["rounds"]),
        "micro_slot_rows": len(combined["micro_slots"]),
        "sealed_cells": int(combined["cells"]["sealed"].sum()),
        "incomplete_cells": 0,
        "failed_episodes": 0,
        "aborted_episodes": 0,
        "duplicate_run_identities": 0,
        "duplicate_cell_identities": 0,
        "duplicate_episode_identities": 0,
        "missing_scientific_events": 0,
        "artifact_hash_mismatches": 0,
        "config_mismatches": 0,
    }
    validation = {
        "schema_version": 1,
        "valid": True,
        "complete": True,
        "errors": [],
        "warnings": ["canonical snapshot merged from independently validated sources"],
        "counts": counts,
        "paired_initialization": paired,
    }
    provenance = {
        "schema_version": 1,
        "study_id": study_id,
        "overlap_policy": overlap,
        "sources": [
            {"path": str(root), "study_id": item[0]["study_id"], "validation": item[3]}
            for root, item in zip(roots, inputs)
        ],
        "skipped_cells": skipped,
        "counts": counts,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".study-merge-", dir=destination.parent
    ) as temporary:
        stage = Path(temporary) / "study"
        (stage / "configs").mkdir(parents=True)
        entries = []
        for index, config in enumerate(configs):
            local = stage / "configs" / f"cell-{index:04d}.yaml"
            local.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            entries.append(
                SubmissionEntry(
                    index,
                    str(destination / "configs" / local.name),
                    file_sha256(local),
                    canonical_hash(config),
                    str(destination / "unavailable-runs" / local.stem),
                    1,
                    int(config["execution"]["repetitions"]),
                    int(config["execution"]["seed"]),
                    "",
                )
            )
        write_submission_manifest(stage / "submission_manifest.csv", entries)
        (stage / "analysis.yaml").write_text(
            yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8"
        )
        for key, frame in combined.items():
            write_scientific_table(stage / "analysis/tables", key, frame)
        manifest = {
            "schema_version": 1,
            "study_id": study_id,
            "config_dir": str(destination / "configs"),
            "analysis_recipe": str(destination / "analysis.yaml"),
            "execution": {},
            "configs": [asdict(entry) for entry in entries],
            "expected_config_count": len(entries),
            "expected_cell_count": len(entries),
            "expected_episode_count": counts["expected_episodes"],
        }
        for relative, value in (
            ("study_manifest.json", manifest),
            ("merge_manifest.json", provenance),
            ("analysis/validation.json", validation),
        ):
            (stage / relative).write_text(
                json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8"
            )
        (stage / "analysis/analysis_manifest.json").write_text(
            json.dumps(
                {
                    "study_id": study_id,
                    "scientific_input_identity": canonical_hash(provenance),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        stage.rename(destination)
    return {
        "study_id": study_id,
        "study_dir": str(destination),
        **counts,
        "skipped_cells": len(skipped),
    }
