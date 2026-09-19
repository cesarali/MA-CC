"""Frozen-bundle relocation verifies every hash; the analysis launcher is overridable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mas_cc.studies import analysis_slurm
from mas_cc.studies import relocate as relocate_module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_bundle(tmp_path: Path, layout: str) -> Path:
    """A minimal frozen bundle with one group, two canonical inputs, one config, one recipe."""
    old_study = "/work/someone/MA-CC/results/studies/demo_study"
    old_cfg = "/home/someone/MA-CC/configs/runs/demo_study"
    gen_id = "abc123"
    gen_dir = f"{old_study}/analysis/.work/{gen_id}"
    bundle = tmp_path / f"bundle-{layout}"
    if layout == "frozen_generation":
        inputs_dir = bundle / "frozen_generation" / "input"
        groups_dir = bundle / "frozen_generation" / "groups"
        study_dir = bundle / "study"
        config_dir = bundle / "config"
        manifest_path = bundle / "frozen_generation" / "execution_manifest.json"
        recipe_file = config_dir / "analysis.yaml"
    else:
        inputs_dir = bundle / "input"
        groups_dir = bundle / "groups"
        study_dir = bundle / "provenance"
        config_dir = bundle / "configs"
        manifest_path = bundle / "provenance" / "execution_manifest.original_paths.json"
        recipe_file = bundle / "analysis_recipe.yaml"
    for directory in (inputs_dir, groups_dir, study_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (inputs_dir / "cells.parquet").write_bytes(b"cells")
    (inputs_dir / "rounds.parquet").write_bytes(b"rounds")
    info = groups_dir / "g1.information.parquet"
    support = groups_dir / "g1.support.parquet"
    info.write_bytes(b"info")
    support.write_bytes(b"support")
    (groups_dir / "g1.complete.json").write_text(json.dumps({
        "cell_id": "cell-a", "group_hash": "g1",
        "information_sha256": _sha(info), "support_sha256": _sha(support),
    }))
    recipe_file.parent.mkdir(parents=True, exist_ok=True)
    recipe_file.write_text("estimators: []\n")
    config = config_dir / "demo.yaml"
    config.write_text("experiment: demo\n")
    (study_dir / "study_manifest.json").write_text(json.dumps({"config_dir": old_cfg, "study_dir": old_study}))
    (study_dir / "study_lineage.json").write_text(json.dumps({
        "study_id": "demo_study", "legacy_root_layout": True, "latest_extension_index": 0,
    }))
    (study_dir / "submission_manifest.csv").write_text(f"array_index,config_path,output_dir\n0,{old_cfg}/demo.yaml,{old_study}/runs\n")
    manifest = {
        "schema_version": 1,
        "generation_id": gen_id,
        "study_id": "demo_study",
        "study_dir": old_study,
        "allow_incomplete": False,
        "analysis_recipe_path": f"{old_cfg}/analysis.yaml",
        "analysis_recipe_hash": _sha(recipe_file),
        "analysis_hash": "deadbeef",
        "config_inputs": [{"path": f"{old_cfg}/demo.yaml", "sha256": _sha(config)}],
        "canonical_inputs": {
            "cells": {"path": f"{gen_dir}/input/cells.parquet", "sha256": _sha(inputs_dir / "cells.parquet")},
            "rounds": {"path": f"{gen_dir}/input/rounds.parquet", "sha256": _sha(inputs_dir / "rounds.parquet")},
        },
        "groups": [{
            "cell_id": "cell-a", "group_hash": "g1", "group_index": 0,
            "completion_path": f"{gen_dir}/groups/g1.complete.json",
            "information_path": f"{gen_dir}/groups/g1.information.parquet",
            "support_path": f"{gen_dir}/groups/g1.support.parquet",
        }],
        "progress_path": f"{gen_dir}/progress.json",
        "resources": {},
    }
    manifest_path.write_text(json.dumps(manifest))
    return bundle


@pytest.mark.parametrize("layout", ["frozen_generation", "provenance"])
def test_relocate_rewrites_paths_and_verifies(tmp_path, layout):
    bundle = _make_bundle(tmp_path, layout)
    root = tmp_path / "new-root" / "demo_study"
    manifest_path = relocate_module.relocate(bundle, root)
    assert manifest_path == root / "analysis" / ".work" / "abc123" / "execution_manifest.json"
    summary = relocate_module.verify(manifest_path)
    assert summary["valid_groups"] == 1 and not summary["problems"]
    relocated = json.loads(manifest_path.read_text())
    assert relocated["study_dir"] == str(root)
    assert relocated["analysis_recipe_path"] == str(root / "config_snapshot" / "analysis.yaml")
    assert "someone" not in json.dumps(relocated)
    assert "someone" not in (root / "submission_manifest.csv").read_text()
    assert (root / "analysis" / ".work" / "abc123" / "progress.json").is_file()
    # A lineage file without an extensions tree would push finalize into lineage
    # mode and fail; the bundle had no tree, so the file must be left out.
    assert not (root / "study_lineage.json").exists()
    assert (root / "study_lineage.omitted.json").is_file()
    assert summary["lineage_omitted"] is True


def test_relocate_refuses_nonempty_root(tmp_path):
    bundle = _make_bundle(tmp_path, "provenance")
    root = tmp_path / "occupied"
    root.mkdir()
    (root / "something").write_text("x")
    with pytest.raises(ValueError):
        relocate_module.relocate(bundle, root)


def test_launcher_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv("MA_CC_ANALYSIS_LAUNCHER", raising=False)
    assert analysis_slurm.launcher_path() == analysis_slurm.DEFAULT_LAUNCHER
    custom = tmp_path / "run_study_analysis.job"
    custom.write_text("#!/usr/bin/env bash\n")
    monkeypatch.setenv("MA_CC_ANALYSIS_LAUNCHER", str(custom))
    assert analysis_slurm.launcher_path() == custom.resolve()
