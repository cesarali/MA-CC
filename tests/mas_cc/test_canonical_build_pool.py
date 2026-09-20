"""Canonical tables built per cell in a pool equal the serial build, table for table and counter for counter."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from mas_cc.studies.canonical import build_canonical_tables


def _cell(root, index: int, *, rounds: int):
    cell_path = root / f"cell-{index:04d}"
    episode = f"cell-{index:04d}-0000"
    resume = cell_path / ".resume" / episode
    records = cell_path / "round_records" / episode
    resume.mkdir(parents=True)
    records.mkdir(parents=True)
    (resume / "manifest.json").write_text(json.dumps({
        "episode_id": episode, "cell_id": f"cell-{index:04d}", "seed": 7 + index, "status": "failed",
        "error_type": "RelationalDecisionFailed", "scientific_schema_version": 1}), encoding="utf-8")
    (resume / "failure_checkpoint.json").write_text(json.dumps({
        "schema_version": 1, "created_at": "2026-09-10T00:00:00Z",
        "runtime": {"schema_version": 1, "interruption_type": "validation_exhausted",
                    "failed_call": {"stage": "focal_update", "agent_id": "agent_003"}}}), encoding="utf-8")
    (records / "round_trajectory.jsonl").write_text(
        "".join(json.dumps({"episode_id": episode, "round_index": k, "target_share": 0.25 * index + 0.1 * k}) + "\n"
                for k in range(1, rounds + 1)), encoding="utf-8")
    (records / "micro_slot_trajectory.jsonl").write_text(
        "".join(json.dumps({"episode_id": episode, "round_index": k, "within_round_index": j, "target_share": 0.5}) + "\n"
                for k in range(1, rounds + 1) for j in range(3)), encoding="utf-8")
    entry = SimpleNamespace(source_extension_index=0, source_submission_attempt=0, array_index=index, config_hash=f"hash-{index}")
    run = SimpleNamespace(entry=entry, run_id=f"run-{index}", path=root)
    return SimpleNamespace(run=run, path=cell_path, cell_key=f"config-{index:04d}/cell-{index:04d}", local_cell_id=f"cell-{index:04d}",
                           overrides={}, resolved_config={"execution": {"repetitions": 1},
                                                           "game": {"type": "relational_imitation_round_feedback"}})


def test_pooled_canonical_build_equals_serial(tmp_path):
    cells = tuple(_cell(tmp_path, index, rounds=2 + index) for index in range(3))
    serial, serial_meta = build_canonical_tables("study", cells, workers=1)
    pooled, pooled_meta = build_canonical_tables("study", cells, workers=3)
    assert set(serial) == set(pooled)
    for name in serial:
        pd.testing.assert_frame_equal(serial[name], pooled[name], check_exact=True)
    assert serial_meta["record_selection"] == pooled_meta["record_selection"]
    assert list(serial_meta["scientific_frames"]) == list(pooled_meta["scientific_frames"]) == [cell.cell_key for cell in cells]
    assert len(serial["available_round_prefixes"]) == 2 + 3 + 4
    assert list(serial["cells"]["cell_id"]) == [cell.cell_key for cell in cells]
