"""study publish --with-dashboard: bundle mirrored, verified by read-back, listed in the catalog, servable."""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from mas_cc.blackboard_dashboard import PublishedStudyReader
from mas_cc.blackboard_dashboard.store import open_store
from mas_cc.blackboard_dashboard.study_data import BlackboardStudyReader
from mas_cc.cli.main import main
from mas_cc.studies import publish as publish_module

from test_blackboard_study_dashboard import _study

rclone_binary = shutil.which("rclone") or (str(Path.home() / "bin" / "rclone") if (Path.home() / "bin" / "rclone").exists() else None)
needs_rclone = pytest.mark.skipif(rclone_binary is None, reason="rclone binary not available")


def _finished_study(tmp_path: Path, name: str = "source") -> Path:
    (tmp_path / name).mkdir()
    study = _study(tmp_path / name)
    analysis = study / "analysis"
    (analysis / "tables").mkdir(parents=True, exist_ok=True)
    (analysis / "tables" / "cells.parquet").write_bytes(b"x" * 100)
    with zipfile.ZipFile(analysis / f"{study.name}_analysis.zip", "w") as archive:
        archive.writestr("tables/cells.parquet", b"x" * 100)
    return study


def test_dry_run_names_the_dashboard_target_and_transfers_nothing(tmp_path: Path):
    study = _finished_study(tmp_path)
    receipt = publish_module.publish(study, remote="bucket:name", prefix="ctodie", dry_run=True, with_dashboard=True)
    assert receipt["dashboard_target"] == f"bucket:name/ctodie/{study.name}/dashboard"
    assert receipt["status"] == "dry_run" and "dashboard" not in receipt


@needs_rclone
def test_published_bundle_is_verified_catalogued_and_servable(tmp_path: Path):
    study = _finished_study(tmp_path)
    bucket = tmp_path / "bucket"
    receipt = publish_module.publish(study, remote=str(bucket), prefix="ctodie", with_dashboard=True,
                                     rclone=publish_module.Rclone(rclone_binary), bundle_dir=tmp_path)
    dashboard = receipt["dashboard"]
    assert receipt["verified"] is True and dashboard["verified"] is True
    assert dashboard["objects"] == dashboard["remote_objects"] and dashboard["bytes"] == dashboard["remote_bytes"]
    assert not list(tmp_path.glob("dashboard-bundle-*"))  # the scratch bundle is removed
    catalog = json.loads((bucket / "ctodie" / "catalog.json").read_text())
    assert catalog["schema_version"] == 1
    assert [(row["study"], row["dashboard"], row["cells"]) for row in catalog["studies"]] == [
        (study.name, f"{study.name}/dashboard", 2)]
    served = PublishedStudyReader(open_store(bucket / "ctodie" / study.name / "dashboard", cache_dir=tmp_path / "cache"))
    live = BlackboardStudyReader(study, scheduler=False)
    assert [c["qualified_id"] for c in served.study()["cells"]] == [c["qualified_id"] for c in live.study()["cells"]]
    assert served.episode_reader("config-0000~cell-0000~episode-0000").timeline()["available_cursors"]
    assert json.loads((study / publish_module.RECEIPT).read_text())["dashboard"]["verified"] is True


@needs_rclone
def test_catalog_keeps_other_studies_and_replaces_a_republished_one(tmp_path: Path):
    bucket = tmp_path / "bucket"
    client = publish_module.Rclone(rclone_binary)
    first = _finished_study(tmp_path, "one")
    second = _finished_study(tmp_path, "two")
    publish_module.publish(first, remote=str(bucket), prefix="ctodie", with_dashboard=True, rclone=client, study_name="alpha")
    publish_module.publish(second, remote=str(bucket), prefix="ctodie", with_dashboard=True, rclone=client, study_name="beta")
    publish_module.publish(first, remote=str(bucket), prefix="ctodie", with_dashboard=True, rclone=client, study_name="alpha")
    catalog = json.loads((bucket / "ctodie" / "catalog.json").read_text())
    assert [row["study"] for row in catalog["studies"]] == ["alpha", "beta"]


@needs_rclone
def test_a_corrupt_catalog_is_not_overwritten(tmp_path: Path):
    study = _finished_study(tmp_path)
    bucket = tmp_path / "bucket"
    (bucket / "ctodie").mkdir(parents=True)
    (bucket / "ctodie" / "catalog.json").write_text("{not json")
    with pytest.raises(publish_module.PublishError, match="not valid JSON"):
        publish_module.publish(study, remote=str(bucket), prefix="ctodie", with_dashboard=True,
                               rclone=publish_module.Rclone(rclone_binary))
    assert (bucket / "ctodie" / "catalog.json").read_text() == "{not json"


@needs_rclone
def test_cli_flag(tmp_path: Path, capsys):
    study = _finished_study(tmp_path)
    bucket = tmp_path / "bucket"
    code = main(["study", "publish", "--study-dir", str(study), "--remote", str(bucket), "--prefix", "ctodie",
                 "--with-dashboard", "--rclone", rclone_binary])
    printed = json.loads(capsys.readouterr().out)
    assert code == 0 and printed["dashboard"]["verified"] is True and printed["dashboard_target"].endswith("/dashboard")
