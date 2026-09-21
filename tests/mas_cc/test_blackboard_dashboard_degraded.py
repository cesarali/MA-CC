"""A study must open even when part of it is unreadable, and say which part."""
from __future__ import annotations

import json
import os
from pathlib import Path

from mas_cc.blackboard_dashboard.study_data import BlackboardStudyReader

from test_blackboard_study_dashboard import _study


def _with_extension(tmp_path: Path, mode: int) -> Path:
    (tmp_path / "s").mkdir()
    study = _study(tmp_path / "s")
    extension = study / "extensions" / "extension-0001"
    extension.mkdir(parents=True)
    manifest = extension / "target_manifest.json"
    manifest.write_text(json.dumps({"extension_index": 1, "cells": []}))
    os.chmod(manifest, mode)
    return study


def test_an_unreadable_extension_manifest_degrades_instead_of_failing(tmp_path: Path):
    """Studies write target_manifest.json 0600 while the rest is 0664, so a dashboard running as
    another uid over a read-only mount of someone else's results could not open the study at all."""
    study = _with_extension(tmp_path, 0o000)
    payload = BlackboardStudyReader(study, scheduler=False).study()
    assert len(payload["cells"]) == 2
    assert payload["degraded"] and "extension manifest unreadable" in payload["degraded"][0]
    # the reason names the file, never the server path it sat in
    assert str(tmp_path) not in payload["degraded"][0]
    assert "'" not in payload["degraded"][0] and '"' not in payload["degraded"][0]


def test_a_readable_study_reports_nothing_degraded(tmp_path: Path):
    study = _with_extension(tmp_path, 0o644)
    assert BlackboardStudyReader(study, scheduler=False).study()["degraded"] == []


def test_a_plain_study_reports_nothing_degraded(tmp_path: Path):
    assert BlackboardStudyReader(_study(tmp_path), scheduler=False).study()["degraded"] == []


def test_the_study_view_shows_a_partial_study_notice():
    """A study missing a part it could not read must say so: the counts beside it are derived from
    what was readable, so silence would present an incomplete study as a whole one."""
    from importlib.resources import files

    assets = files("mas_cc.blackboard_dashboard.assets")
    script = assets.joinpath("app.js").read_text(encoding="utf-8")
    page = assets.joinpath("index.html").read_text(encoding="utf-8")
    assert '<div id="study-degraded" hidden></div>' in page
    assert "const degraded = study.degraded || [];" in script
    assert "notice.hidden = !degraded.length;" in script
    assert "degraded.map(esc)" in script          # reasons come from disk; escape them
    assert "Partial study." in script
