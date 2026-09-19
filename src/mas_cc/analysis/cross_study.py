"""Combine several published MA-CC analysis packages into one cross-study view.

A study package (``<study>/analysis/``) carries its canonical tables
(``cells``, ``episodes``, ``rounds``, ``micro_slots``), the recipe that produced
it, and the per-study aggregated tables. When two packages were run on the same
paired initializations (the same seeds, hence the same
``initialization_artifact_hash`` per repetition), their cells belong to one
design and can be aggregated *jointly*: one paired block bootstrap across every
budget, not a concatenation of per-study intervals.

This module does exactly that and nothing new numerically:

1. loads each package's canonical ``cells`` + ``rounds``;
2. reports how the packages pair (shared initialization blocks per pair);
3. rebuilds round events with the finalizer's own
   ``_round_events_from_canonical`` and runs the finalizer's own
   ``derive_study_control_aggregates`` over the union with a
   ``PairedBootstrap`` built on the union of rounds - so a block drawn in a
   replicate is drawn for every budget it appears in;
4. concatenates the per-study descriptive tables with a ``source_package``
   column for plotting budget-response curves across the full sweep;
5. writes a package with a manifest that names every input package and hash.

    python -m mas_cc.analysis.cross_study \
        --package b9b15=<study_a>/analysis --package potsdam=<study_b>/analysis \
        --output <dir> [--resamples 1000] [--workers 8]

``provider_calls`` is 0 by construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

VERSION = "cross_study_v1"
CANONICAL = ("cells", "rounds")
DESCRIPTIVE_TABLES = (
    "study_aggregated_metrics", "state_local_aggregated_metrics",
    "causal_response_aggregated_metrics", "causal_state_local_aggregated_metrics",
    "epistemic_aggregated_metrics", "blackboard_calibration_estimates",
    "blackboard_diagnostics", "rho_b_summary", "cell_summary", "primary_estimates",
    "efficiencies", "susceptibility",
)
BLOCK_COLUMNS = ("physical_initial_state_hash", "initialization_artifact_hash")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Package:
    name: str
    root: Path
    cells: pd.DataFrame
    rounds: pd.DataFrame
    recipe: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def tables_dir(self) -> Path:
        return self.root / "tables" if (self.root / "tables").is_dir() else self.root

    def table(self, name: str) -> pd.DataFrame | None:
        path = self.tables_dir / f"{name}.parquet"
        return pd.read_parquet(path) if path.is_file() else None

    def blocks(self) -> pd.Series:
        """One block identity per episode, the same rule ``PairedBootstrap`` applies."""
        rounds = self.rounds
        if "episode_complete" in rounds:
            rounds = rounds[rounds["episode_complete"].fillna(False)]
        keep = ["cell_id", "episode_id", *[c for c in BLOCK_COLUMNS if c in rounds.columns]]
        frame = rounds[keep].drop_duplicates(["cell_id", "episode_id"]).copy()
        block = pd.Series([None] * len(frame), index=frame.index, dtype=object)
        for column in BLOCK_COLUMNS:
            if column in frame:
                values = frame[column].where(frame[column].notna() & (frame[column].astype(str) != ""))
                block = block.where(block.notna(), values)
        fallback = frame.apply(lambda r: json.dumps([str(r["cell_id"]), str(r["episode_id"])]), axis=1)
        return block.where(block.notna(), fallback).astype(str)


def load_package(spec: str) -> Package:
    """``name=path`` or ``path``; path is an analysis dir (with ``tables/``) or a tables dir."""
    name, _, raw = spec.rpartition("=") if "=" in spec else ("", "", spec)
    root = Path(raw).expanduser().resolve()
    tables = root / "tables" if (root / "tables").is_dir() else root
    if not (tables / "rounds.parquet").is_file():
        raise ValueError(f"not an analysis package (no tables/rounds.parquet): {root}")
    recipe: dict[str, Any] = {}
    recipe_path = next((p for p in (root / "analysis_recipe.yaml", root.parent / "analysis_recipe.yaml") if p.is_file()), None)
    if recipe_path is not None:
        import yaml
        recipe = dict(yaml.safe_load(recipe_path.read_text(encoding="utf-8")) or {})
    manifest_path = root / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    cells = pd.read_parquet(tables / "cells.parquet")
    package_name = name or str(cells["study_id"].iloc[0]) if "study_id" in cells and len(cells) else name or root.parent.name
    return Package(package_name, root, cells, pd.read_parquet(tables / "rounds.parquet"), recipe, manifest)


def pairing_report(packages: Sequence[Package]) -> dict[str, Any]:
    """How many initialization blocks each package has and how many each pair shares."""
    blocks = {p.name: set(p.blocks()) for p in packages}
    pairs = []
    names = [p.name for p in packages]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = len(blocks[a] & blocks[b])
            pairs.append({"a": a, "b": b, "shared_blocks": shared,
                          "jaccard": shared / max(1, len(blocks[a] | blocks[b]))})
    return {
        "packages": [{"name": p.name, "cells": int(p.cells["cell_id"].nunique()), "blocks": len(blocks[p.name]),
                      "episodes": int(p.rounds[["cell_id", "episode_id"]].drop_duplicates().shape[0])} for p in packages],
        "pairs": pairs,
        "fully_paired": all(pair["shared_blocks"] > 0 and pair["jaccard"] == 1.0 for pair in pairs) if pairs else True,
    }


def union_canonical(packages: Sequence[Package]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Union of cells and rounds with a ``source_package`` column; cell ids must be distinct."""
    cells = pd.concat([p.cells.assign(source_package=p.name) for p in packages], ignore_index=True)
    if cells["cell_id"].duplicated().any():
        dupes = sorted(cells.loc[cells["cell_id"].duplicated(), "cell_id"].astype(str))[:5]
        raise ValueError(f"cell ids collide across packages: {dupes}")
    rounds = pd.concat([p.rounds.assign(source_package=p.name) for p in packages], ignore_index=True)
    return cells, rounds


@dataclass
class CombinedOutputs:
    study_metrics: pd.DataFrame
    state_local_metrics: pd.DataFrame
    stability: pd.DataFrame
    state_local_reconstruction: pd.DataFrame
    descriptive: dict[str, pd.DataFrame]
    pairing: dict[str, Any]
    manifest: dict[str, Any]


def combine(
    packages: Sequence[Package],
    *,
    recipe: Mapping[str, Any] | None = None,
    resampling: Mapping[str, Any] | None = None,
    workers: int = 1,
    events_builder: Callable[[pd.DataFrame], list] | None = None,
) -> CombinedOutputs:
    """Joint paired aggregation over every package plus concatenated descriptive tables."""
    if not packages:
        raise ValueError("at least one package is required")
    recipe = dict(recipe if recipe is not None else packages[0].recipe)
    if not recipe.get("derived_study_aggregates", {}).get("enabled"):
        raise ValueError("the recipe must enable derived_study_aggregates")
    resampling = dict(resampling if resampling is not None else recipe.get("resampling", {}))
    for key, default in (("bootstrap_resamples", 1000), ("null_permutations", 1000), ("confidence", 0.95), ("seed", 1)):
        resampling.setdefault(key, default)
    pairing = pairing_report(packages)
    cells, rounds = union_canonical(packages)

    from mas_cc.studies.aggregation import _round_events_from_canonical
    from mas_cc.studies.derived_aggregation import derive_study_control_aggregates
    from mas_cc.studies.weighted_summaries import PairedBootstrap

    started = time.time()
    events = (events_builder or _round_events_from_canonical)(rounds)
    plan = None
    if recipe["derived_study_aggregates"].get("bootstrap", {}).get("unit") == "shared_initialization_block":
        plan = PairedBootstrap(rounds, resamples=int(resampling["bootstrap_resamples"]), seed=int(resampling["seed"]))
    analysis_hash = hashlib.sha256(json.dumps({
        "estimator": VERSION,
        "packages": [{"name": p.name, "analysis_hash": p.manifest.get("analysis_hash"),
                      "cells": sorted(p.cells["cell_id"].astype(str))} for p in packages],
        "recipe": recipe.get("derived_study_aggregates"), "resampling": resampling,
    }, sort_keys=True, default=str).encode()).hexdigest()
    outputs = derive_study_control_aggregates(
        events, cells, recipe, resampling, analysis_hash, bootstrap_plan=plan, workers=workers)
    for frame in (outputs.study_metrics, outputs.state_local_metrics, outputs.stability, outputs.state_local_reconstruction):
        if not frame.empty:
            frame["combined_from_packages"] = ",".join(p.name for p in packages)
    descriptive: dict[str, pd.DataFrame] = {}
    for name in DESCRIPTIVE_TABLES:
        parts = [(p.name, p.table(name)) for p in packages]
        if all(frame is not None for _, frame in parts):
            descriptive[name] = pd.concat(
                [frame.assign(source_package=source) for source, frame in parts], ignore_index=True, sort=False)
    manifest = {
        "estimator_version": VERSION,
        "analysis_hash": analysis_hash,
        "packages": [{"name": p.name, "root": str(p.root), "analysis_hash": p.manifest.get("analysis_hash"),
                      "cells": int(p.cells["cell_id"].nunique()),
                      "rounds_sha256": _sha256(p.tables_dir / "rounds.parquet"),
                      "cells_sha256": _sha256(p.tables_dir / "cells.parquet")} for p in packages],
        "pairing": pairing,
        "resampling": resampling,
        "bootstrap_unit": "shared_initialization_block" if plan is not None else "episode",
        "events": len(events),
        "cells": int(cells["cell_id"].nunique()),
        "workers": workers,
        "seconds": round(time.time() - started, 1),
        "provider_calls": 0,
        "descriptive_tables": sorted(descriptive),
    }
    if plan is not None:
        manifest["bootstrap_diagnostics"] = plan.diagnostics(cells["cell_id"])
    return CombinedOutputs(outputs.study_metrics, outputs.state_local_metrics, outputs.stability,
                           outputs.state_local_reconstruction, descriptive, pairing, manifest)


def _first(frame: pd.DataFrame, *names: str) -> str | None:
    return next((n for n in names if n in frame.columns), None)


def budget_response_plots(descriptive: Mapping[str, pd.DataFrame], combined: pd.DataFrame, output: Path) -> list[str]:
    """Estimate vs intervention budget, one panel per metric, series by rho where present."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[str] = []
    output.mkdir(parents=True, exist_ok=True)
    sources = {"combined_study_aggregated_metrics": combined, **descriptive}
    for source_name in ("combined_study_aggregated_metrics", "study_aggregated_metrics",
                        "causal_response_aggregated_metrics", "epistemic_aggregated_metrics", "rho_b_summary"):
        frame = sources.get(source_name)
        if frame is None or frame.empty:
            continue
        x = _first(frame, "intervention_budget", "b_budget", "b")
        y = _first(frame, "estimate", "value", "mean")
        metric = _first(frame, "metric")
        if x is None or y is None:
            continue
        series = _first(frame, "epistemic_persistence", "rho")
        lo, hi = _first(frame, "ci_low", "ci_lower"), _first(frame, "ci_high", "ci_upper")
        metrics = sorted(frame[metric].dropna().unique()) if metric else [None]
        for name in metrics:
            sub = frame if name is None else frame[frame[metric] == name]
            if "lag" in sub.columns:
                sub = sub[(sub["lag"] == 1) | sub["lag"].isna()]
            sub = sub.dropna(subset=[x, y])
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            groups = sub.groupby(series, dropna=False) if series else [(None, sub)]
            for key, group in groups:
                group = group.sort_values(x)
                if lo and hi and group[lo].notna().any():
                    err = [(group[y] - group[lo]).clip(lower=0), (group[hi] - group[y]).clip(lower=0)]
                    ax.errorbar(group[x], group[y], yerr=err, marker="o", capsize=3,
                                label=None if key is None else f"{series}={key}")
                else:
                    ax.plot(group[x], group[y], marker="o", label=None if key is None else f"{series}={key}")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(f"{source_name}: {name}" if name else source_name, fontsize=9)
            if series:
                ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            filename = f"{source_name}__{name or 'value'}_by_budget.png".replace("/", "_")
            fig.savefig(output / filename, dpi=120)
            plt.close(fig)
            written.append(filename)
    return written


def write_outputs(result: CombinedOutputs, output: Path) -> None:
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for name, frame in (("combined_study_aggregated_metrics", result.study_metrics),
                        ("combined_state_local_aggregated_metrics", result.state_local_metrics),
                        ("combined_sample_size_stability", result.stability),
                        ("combined_state_local_reconstruction", result.state_local_reconstruction)):
        if not frame.empty:
            frame.to_parquet(tables / f"{name}.parquet", index=False)
    for name, frame in result.descriptive.items():
        frame.to_parquet(tables / f"all_packages_{name}.parquet", index=False)
    plots = budget_response_plots(result.descriptive, result.study_metrics, output / "plots")
    manifest = {**result.manifest, "plots": plots, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (output / "cross_study_manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", action="append", required=True, help="name=path to an analysis package; repeatable")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resamples", type=int, default=None)
    parser.add_argument("--null-permutations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    args = parser.parse_args(argv)
    packages = [load_package(spec) for spec in args.package]
    resampling = dict(packages[0].recipe.get("resampling", {}))
    if args.resamples is not None:
        resampling["bootstrap_resamples"] = args.resamples
    if args.null_permutations is not None:
        resampling["null_permutations"] = args.null_permutations
    if args.seed is not None:
        resampling["seed"] = args.seed
    result = combine(packages, resampling=resampling, workers=args.workers)
    write_outputs(result, args.output)
    print(json.dumps({k: result.manifest[k] for k in ("cells", "events", "seconds", "bootstrap_unit", "pairing")}, default=str), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
