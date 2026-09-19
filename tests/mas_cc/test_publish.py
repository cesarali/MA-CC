"""Publish node: package + analysis mirror to an rclone target, verified by reading the remote back."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from mas_cc.cli.main import main
from mas_cc.studies import publish as publish_module

rclone_binary = shutil.which("rclone") or (str(Path.home() / "bin" / "rclone") if (Path.home() / "bin" / "rclone").exists() else None)
needs_rclone = pytest.mark.skipif(rclone_binary is None, reason="rclone binary not available")


def _study(tmp_path: Path, *, layout: str = "analysis") -> Path:
    study = tmp_path / "demo_study"
    analysis = study / "analysis" if layout == "analysis" else study / "analysis-runs" / "run-1" / "output"
    (analysis / "tables").mkdir(parents=True)
    (analysis / "tables" / "cells.parquet").write_bytes(b"x" * 1000)
    (analysis / "analysis_manifest.json").write_text(json.dumps({"study_id": "demo_study"}))
    with zipfile.ZipFile(analysis / "demo_study_analysis.zip", "w") as archive:
        archive.writestr("tables/cells.parquet", b"x" * 1000)
    return study


def test_find_analysis_dir_prefers_the_layout_that_holds_a_package(tmp_path):
    plain = _study(tmp_path / "a")
    assert publish_module.find_analysis_dir(plain) == plain / "analysis"
    checkpoint = _study(tmp_path / "b", layout="analysis-runs")
    assert publish_module.find_analysis_dir(checkpoint) == checkpoint / "analysis-runs" / "run-1" / "output"
    with pytest.raises(publish_module.PublishError):
        publish_module.find_analysis_dir(tmp_path / "nowhere")


def test_dry_run_resolves_targets_without_a_binary(tmp_path):
    study = _study(tmp_path)
    receipt = publish_module.publish(study, remote="bucket:name/", prefix="/ctodie/", dry_run=True)
    assert receipt["status"] == "dry_run" and receipt["verified"] is None
    assert receipt["package_target"] == "bucket:name/aggregation_results/demo_study_analysis.zip"
    assert receipt["analysis_target"] == "bucket:name/ctodie/demo_study/analysis"
    assert receipt["local_files"] == 3 and receipt["local_bytes"] > 1000
    assert not (study / publish_module.RECEIPT).exists()


@needs_rclone
def test_publish_to_a_local_rclone_remote_writes_a_verified_receipt(tmp_path):
    study = _study(tmp_path)
    bucket = tmp_path / "bucket"
    receipt = publish_module.publish(study, remote=str(bucket), prefix="ctodie", rclone=publish_module.Rclone(rclone_binary))
    assert receipt["status"] == "ok" and receipt["verified"] is True
    assert (bucket / "aggregation_results" / "demo_study_analysis.zip").stat().st_size == receipt["package_bytes"]
    mirrored = sorted(p.relative_to(bucket / "ctodie" / "demo_study" / "analysis") for p in (bucket / "ctodie" / "demo_study" / "analysis").rglob("*") if p.is_file())
    assert [str(p) for p in mirrored] == ["analysis_manifest.json", "demo_study_analysis.zip", "tables/cells.parquet"]
    assert receipt["remote_files"] == 3 and receipt["remote_bytes"] == receipt["local_bytes"]
    on_disk = json.loads((study / publish_module.RECEIPT).read_text())
    assert on_disk["verified"] is True
    assert (bucket / "ctodie" / "demo_study" / publish_module.RECEIPT).exists()


@needs_rclone
def test_cli_publish_uses_environment_defaults(tmp_path, monkeypatch, capsys):
    study = _study(tmp_path)
    bucket = tmp_path / "bucket"
    monkeypatch.setenv("MA_CC_PUBLISH_REMOTE", str(bucket))
    monkeypatch.setenv("MA_CC_PUBLISH_PREFIX", "who")
    monkeypatch.setenv("MA_CC_RCLONE", rclone_binary)
    assert main(["study", "publish", "--study-dir", str(study)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "ok" and printed["analysis_target"].endswith("/who/demo_study/analysis")


def test_cli_publish_without_a_remote_is_an_error(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("MA_CC_PUBLISH_REMOTE", raising=False)
    assert main(["study", "publish", "--study-dir", str(_study(tmp_path))]) == 2
    assert "MA_CC_PUBLISH_REMOTE" in capsys.readouterr().err
