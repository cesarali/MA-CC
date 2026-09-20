"""Relocate a frozen aggregation-input bundle onto another cluster and verify it.

A frozen bundle (see docs/handoff/18092026_*_aggregation_handoff.md) is a data
handoff: canonical Parquet inputs, completed information/support groups, the
analysis recipe, the scientific configs and the original execution manifest whose
absolute paths point at the cluster that produced it. Before `analysis_worker
finalize` can run elsewhere, every path in that manifest (and in the study
manifests) must point at a study root on the new cluster, and every hash must
still verify. Doing that by hand takes an afternoon and was done twice on
2026-09-19; this module makes it one command:

    python -m mas_cc.studies.relocate --bundle <extracted bundle dir> \
        --study-root /shared/home/<user>/agg/<study>/<study_id>

Two bundle layouts are recognised:

* ``frozen_generation/{execution_manifest.json,input/,groups/,progress.json,
  prepare_metrics.json}`` + ``study/{study_manifest.json,submission_manifest.csv}``
  + ``config/{analysis.yaml,<configs>.yaml}`` (the ICLR false-arm bundle);
* flat ``provenance/execution_manifest.original_paths.json`` +
  ``provenance/{study_manifest.json,submission_manifest.csv,study_lineage.json}``
  + ``input/`` + ``groups/`` + ``configs/`` + ``analysis_recipe.yaml`` (the
  checkpoint-ensemble bundle).

The generation workspace is recreated exactly where the manifest expects it
(``analysis/.work/<generation_id>`` or ``analysis-runs/<name>``), because
``finalize`` derives ``input/``, ``groups/`` and ``final/`` from the manifest's
own location. Nothing is written outside ``--study-root``. Provenance copies of
the configs keep their historical paths where the bundle documents them as such.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def detect_layout(bundle: Path) -> dict[str, Path]:
    """Return the bundle's manifest, inputs, groups, study-manifest and config paths."""
    frozen = bundle / "frozen_generation"
    if (frozen / "execution_manifest.json").is_file():
        return {
            "manifest": frozen / "execution_manifest.json",
            "input": frozen / "input",
            "groups": frozen / "groups",
            "extras": frozen,
            "study": bundle / "study",
            "config": bundle / "config",
            "recipe": None,
        }
    provenance = bundle / "provenance"
    if (provenance / "execution_manifest.original_paths.json").is_file():
        return {
            "manifest": provenance / "execution_manifest.original_paths.json",
            "input": bundle / "input",
            "groups": bundle / "groups",
            "extras": None,
            "study": provenance,
            "config": bundle / "configs",
            "recipe": bundle / "analysis_recipe.yaml",
        }
    raise ValueError(f"unrecognised frozen bundle layout: {bundle}")


def relocate(bundle: Path, study_root: Path, *, config_dir_name: str = "config_snapshot") -> Path:
    """Build the study root, rewrite paths, copy data and return the new manifest path."""
    bundle = bundle.expanduser().resolve()
    study_root = study_root.expanduser().resolve()
    layout = detect_layout(bundle)
    manifest = _read_json(layout["manifest"])
    old_study = str(manifest["study_dir"]).rstrip("/")
    recipe_path = str(manifest["analysis_recipe_path"])
    old_config = os.path.dirname(recipe_path)
    new_config = str(study_root / config_dir_name)
    if not old_study or not old_config:
        raise ValueError("manifest lacks study_dir or analysis_recipe_path")

    def rewrite(text: str) -> str:
        return text.replace(old_study, str(study_root)).replace(old_config, new_config)

    relocated = json.loads(rewrite(json.dumps(manifest)))
    inputs = list(relocated["canonical_inputs"].values())
    if not inputs:
        raise ValueError("manifest has no canonical inputs")
    generation = Path(os.path.dirname(os.path.dirname(inputs[0]["path"])))
    if study_root not in generation.parents:
        raise ValueError(f"generation directory {generation} is not under {study_root}")

    if study_root.exists() and any(study_root.iterdir()):
        raise ValueError(f"study root is not empty: {study_root}")
    generation.mkdir(parents=True, exist_ok=True)
    (study_root / "analysis").mkdir(exist_ok=True)
    (study_root / "logs").mkdir(exist_ok=True)
    Path(new_config).mkdir(exist_ok=True)

    shutil.copytree(layout["input"], generation / "input", dirs_exist_ok=True)
    shutil.copytree(layout["groups"], generation / "groups", dirs_exist_ok=True)
    if layout["extras"] is not None:
        for name in ("prepare_metrics.json",):
            source = layout["extras"] / name
            if source.is_file():
                shutil.copy(source, generation / name)
    for name in ("study_manifest.json", "submission_manifest.csv"):
        source = layout["study"] / name
        if source.is_file():
            with open(source, encoding="utf-8") as handle:
                text = handle.read()
            with open(study_root / name, "w", encoding="utf-8") as handle:
                handle.write(rewrite(text))
    # ``study_lineage.json`` switches ``_aggregate_study_local`` into lineage
    # mode, which then requires ``extensions/extension-*/target_manifest.json``.
    # Frozen bundles carry the lineage file but not the extensions tree, so a
    # relocated root with the file and without the tree fails at finalize with
    # "study lineage has no target manifest" (measured on Cygnus, job 43,
    # 2026-09-19). Keep the file only when the bundle also ships the tree;
    # otherwise leave it out and let the finalizer read the submission manifest.
    lineage = layout["study"] / "study_lineage.json"
    extensions = bundle / "extensions"
    if lineage.is_file() and extensions.is_dir():
        with open(lineage, encoding="utf-8") as handle:
            text = handle.read()
        with open(study_root / "study_lineage.json", "w", encoding="utf-8") as handle:
            handle.write(rewrite(text))
        shutil.copytree(extensions, study_root / "extensions", dirs_exist_ok=True)
    elif lineage.is_file():
        shutil.copy(lineage, study_root / "study_lineage.omitted.json")
    for source in sorted(Path(layout["config"]).glob("*.yaml")):
        shutil.copy(source, Path(new_config) / source.name)
    if layout["recipe"] is not None:
        shutil.copy(layout["recipe"], Path(new_config) / os.path.basename(recipe_path))

    manifest_path = generation / "execution_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(relocated, handle, indent=1, sort_keys=True)
        handle.write("\n")
    progress_path = Path(str(relocated["progress_path"]))
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    if not progress_path.is_file():
        with open(progress_path, "w", encoding="utf-8") as handle:
            json.dump(
                {"generation_id": relocated["generation_id"], "stage": "relocated", "published": False},
                handle,
                indent=1,
            )
            handle.write("\n")
    placeholder = study_root / "analysis" / "README.txt"
    if not any((study_root / "analysis").iterdir()):
        placeholder.write_text(
            "placeholder created by mas_cc.studies.relocate so finalize can publish over it\n",
            encoding="utf-8",
        )
    return manifest_path


def verify(manifest_path: Path) -> dict[str, Any]:
    """Re-verify every hash the finalizer will check; raise on the first mismatch."""
    manifest = _read_json(manifest_path)
    problems: list[str] = []
    for name, item in manifest["canonical_inputs"].items():
        path = Path(str(item["path"]))
        if not path.is_file() or _sha256(path) != item["sha256"]:
            problems.append(f"canonical input {name}")
    for item in manifest.get("config_inputs", []):
        path = Path(str(item["path"]))
        if not path.is_file() or _sha256(path) != item["sha256"]:
            problems.append(f"config {path.name}")
    recipe = Path(str(manifest["analysis_recipe_path"]))
    if not recipe.is_file() or _sha256(recipe) != manifest["analysis_recipe_hash"]:
        problems.append("analysis recipe")
    from .analysis_slurm import _valid_group

    valid_groups = sum(_valid_group(group) for group in manifest["groups"])
    if valid_groups != len(manifest["groups"]):
        problems.append(f"groups valid {valid_groups}/{len(manifest['groups'])}")
    leftover = json.dumps(manifest).count("ojedamarin")
    study_root = Path(str(manifest["study_dir"]))
    if (study_root / "study_lineage.json").is_file() and not list(
        (study_root / "extensions").glob("extension-*/target_manifest.json")
    ):
        problems.append("study_lineage.json present without extensions/*/target_manifest.json")
    summary = {
        "generation_id": manifest["generation_id"],
        "study_dir": manifest["study_dir"],
        "allow_incomplete": manifest["allow_incomplete"],
        "groups": len(manifest["groups"]),
        "valid_groups": valid_groups,
        "problems": problems,
        "foreign_path_references": leftover,
        "lineage_omitted": (study_root / "study_lineage.omitted.json").is_file(),
    }
    if problems:
        raise ValueError("relocation verification failed: " + "; ".join(problems))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True, type=Path, help="extracted frozen bundle directory")
    parser.add_argument("--study-root", required=True, type=Path, help="new study root (must be empty or absent)")
    parser.add_argument("--config-dir-name", default="config_snapshot")
    args = parser.parse_args(argv)
    manifest_path = relocate(args.bundle, args.study_root, config_dir_name=args.config_dir_name)
    summary = verify(manifest_path)
    print(json.dumps({"manifest": str(manifest_path), **summary}, indent=1))
    print(
        "next: python -m mas_cc.studies.analysis_worker finalize "
        f"{manifest_path}   (inside an allocation; use MA_CC_ANALYSIS_LAUNCHER for the Slurm graph)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
