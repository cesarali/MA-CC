from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = ROOT / "scripts/experiment_design/focal_kernel_study"
sys.path.insert(0, str(TOOL_DIR))

import analyze_atomic_control_calibration as analysis  # noqa: E402
import generate_atomic_control_calibration as generator  # noqa: E402
from atomic_control_common import BUCKETS, stable_shard, verify_frozen_dataset  # noqa: E402
from run_atomic_control_calibration import parse_vote  # noqa: E402


def test_generator_builds_paired_mixed_evidence_dataset(tmp_path: Path) -> None:
    destination = tmp_path / "atomic_control_calibration"
    summary = generator.generate(destination)

    assert summary["prompts"] == 600
    states = [json.loads(line) for line in (destination / "base_states.jsonl").read_text().splitlines()]
    assert len(states) == 100
    assert sum(row["control_alignment"] == "truth" for row in states) == 50
    assert sum(row["history"] != "No previous interaction." for row in states) == 50
    assert all(len(row["paraphrase_ids"]) == 1 for row in states)
    assert all(sum(trace.endswith(":source") for trace in row["fact_trace_ids"]) == 1 for row in states)
    assert all(len(list((destination / bucket / "prompts").glob("*.md"))) == 100 for bucket in BUCKETS)
    assert verify_frozen_dataset(destination / "frozen_prompts")["dataset_hash"] == summary["dataset_hash"]


def test_frozen_dataset_verification_detects_prompt_modification(tmp_path: Path) -> None:
    destination = tmp_path / "atomic_control_calibration"
    generator.generate(destination)
    prompt = destination / "frozen_prompts/bucket_01_anonymous/prompts/state_0001.md"
    prompt.write_text(prompt.read_text() + "modified\n")
    with pytest.raises(ValueError, match="modified"):
        verify_frozen_dataset(destination / "frozen_prompts")


def test_vote_parser_is_strict_and_sharding_is_stable() -> None:
    options = ["West City", "East Town", "North Hill"]
    assert parse_vote('{"vote":"West City"}', options) == ("West City", None)
    assert parse_vote('```json\n{"vote":"West City"}\n```', options)[0] is None
    assert parse_vote('{"vote":"Nowhere"}', options)[0] is None
    assert stable_shard("bucket_01_anonymous", "state_0001", 7) == stable_shard(
        "bucket_01_anonymous", "state_0001", 7
    )


def test_analysis_metrics_distinguish_adoption_truth_and_stay() -> None:
    rows = [
        {
            "provider": "mock",
            "model": "fixture",
            "bucket": "bucket_01_anonymous",
            "state_id": "state_0001",
            "task_id": "1",
            "current_vote": "B",
            "control_target": "A",
            "control_alignment": "truth",
            "correct_answer": "A",
            "vote_after": "A",
        },
        {
            "provider": "mock",
            "model": "fixture",
            "bucket": "bucket_01_anonymous",
            "state_id": "state_0002",
            "task_id": "1",
            "current_vote": "A",
            "control_target": "B",
            "control_alignment": "incorrect",
            "correct_answer": "A",
            "vote_after": "A",
        },
    ]
    metrics = analysis.compute_metrics(rows, repetitions=2000, seed=1)[0]
    assert metrics["control_target_adoption_rate"] == 0.5
    assert metrics["truth_rate"] == 1.0
    assert metrics["stay_rate"] == 0.5
    assert metrics["aligned_target_adoption_rate"] == 1.0
    assert metrics["adversarial_target_adoption_rate"] == 0.0
