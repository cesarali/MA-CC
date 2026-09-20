"""Liveness, re-discovery, cache bounds and thread-safety of the dashboard readers."""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

import yaml

from mas_cc.blackboard_dashboard import study_data
from mas_cc.blackboard_dashboard.data import BlackboardRunReader
from mas_cc.blackboard_dashboard.study_data import BlackboardStudyReader

from test_blackboard_study_dashboard import _direct_grid, _study

EPISODE = "config-0000~cell-0000~episode-0000"


def _activity_file(reader: BlackboardStudyReader) -> Path:
    paths = reader.resolved_paths("config-0000~cell-0000")
    return paths.round_records_root / "cell-0000-0000" / "round_trajectory.jsonl"


def _make_incomplete(reader: BlackboardStudyReader) -> None:
    """Turn the fixture's completed episode into one that is still being written."""
    paths = reader.resolved_paths("config-0000~cell-0000")
    manifest = paths.full_episodes_root / "cell-0000-0000" / "manifest.json"
    record = json.loads(manifest.read_text())
    record["status"] = "running"
    record.pop("finished_at", None)
    manifest.write_text(json.dumps(record))


def test_liveness_is_a_function_of_file_age_not_of_who_asked_first(tmp_path: Path):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    _make_incomplete(reader)
    reader = BlackboardStudyReader(reader.study_dir, scheduler=False)
    activity = _activity_file(reader)

    now = time.time()
    os.utime(activity, (now, now))
    first = reader.episode_status(EPISODE)["activity_status"]
    second = reader.episode_status(EPISODE)["activity_status"]
    other = BlackboardStudyReader(reader.study_dir, scheduler=False).episode_status(EPISODE)["activity_status"]
    assert first == second == other == "advancing"

    stale = now - 10 * study_data.ACTIVITY_WINDOW_SECONDS
    os.utime(activity, (stale, stale))
    assert reader.episode_status(EPISODE)["activity_status"] == "started_unchanged"


def test_completed_episode_is_never_reported_as_advancing(tmp_path: Path):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    now = time.time()
    os.utime(_activity_file(reader), (now, now))
    status = reader.episode_status(EPISODE)
    assert status["durable_status"] == "completed"
    assert status["activity_status"] == "started_unchanged"


def test_cell_created_after_start_becomes_visible(tmp_path: Path):
    root = _direct_grid(tmp_path)
    reader = BlackboardStudyReader(root, scheduler=False)
    assert len(reader.study()["cells"]) == 4
    source, late = root / "cells" / "cell-0003", root / "cells" / "cell-0004"
    shutil.copytree(source, late)
    overrides = json.loads((late / "overrides.json").read_text())
    overrides.update(cell_id="cell-0004", index=4)
    (late / "overrides.json").write_text(json.dumps(overrides))
    config = yaml.safe_load((late / "resolved_config.yaml").read_text())
    (late / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    payload = reader.study()
    assert len(payload["cells"]) == 5
    assert reader.cell("config-0000~cell-0004")["cell_id"] == "cell-0004"


def test_row_cache_is_bounded_by_bytes(tmp_path: Path, monkeypatch):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    monkeypatch.setattr(reader, "_jsonl_cache_limit_bytes", 600)
    files = []
    for index in range(6):
        path = tmp_path / f"rows-{index}.jsonl"
        path.write_text("".join(json.dumps({"event": {"round_index": r, "pad": "x" * 40}}) + "\n" for r in range(4)))
        files.append(path)
        reader._rows(path)
    assert reader._jsonl_cache_bytes <= 600
    assert files[-1] in reader._jsonl_cache and files[0] not in reader._jsonl_cache
    assert reader._rows(files[0])[0]["round_index"] == 0


def test_run_reader_publishes_the_payload_before_its_signature(tmp_path: Path):
    """A concurrent reader compares signatures first; the payload must already be the new one."""
    study = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    paths = study.resolved_paths("config-0000~cell-0000")
    order: list[str] = []

    class Probe(BlackboardRunReader):
        def __setattr__(self, name, value):
            if name in {"_cache", "_cache_signature"} and value is not None:
                order.append(name)
            super().__setattr__(name, value)

    reader = Probe(paths.run_root, "cell-0000-0000")
    assert reader._load()["trajectory"]
    assert order == ["_cache", "_cache_signature"]


def test_run_reader_reloads_are_serialised(tmp_path: Path, monkeypatch):
    study = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    paths = study.resolved_paths("config-0000~cell-0000")
    reader = BlackboardRunReader(paths.run_root, "cell-0000-0000")
    active, peak, guard = [0], [0], threading.Lock()
    original = study_data._safe_json

    def slow_json(path, **kwargs):
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with guard:
            active[0] -= 1
        return original(path, **kwargs)

    monkeypatch.setattr("mas_cc.blackboard_dashboard.data._safe_json", slow_json)
    threads = [threading.Thread(target=reader._load) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert peak[0] == 1
