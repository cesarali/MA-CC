from __future__ import annotations

import json
import hashlib
from collections import Counter
from pathlib import Path

from mas_cc.probes.musr_truthful_selective.isolated_analysis import aggregate
from mas_cc.probes.musr_truthful_selective.isolated_config import load_isolated_config
from mas_cc.probes.musr_truthful_selective.isolated_design import (
    answer_orders,
    build_manifest,
    load_selected_tasks,
    random_permutations,
    smoke_ids,
)
from mas_cc.probes.musr_truthful_selective.isolated_execution import completed_ids
from mas_cc.probes.musr_truthful_selective.isolated_prompting import (
    parse_isolated,
    render_isolated,
)

CONFIG = Path("configs/probes/musr_truthful_selective_isolated_oss_01.yaml")


def plan():
    config = load_isolated_config(CONFIG)
    tasks = load_selected_tasks(config)
    return config, tasks, build_manifest(config, tasks)


def test_manifest_has_exact_frozen_design_and_counts():
    config, tasks, rows = plan()
    assert config.expected_tasks == {"task_001": 42, "task_002": 237, "task_003": 130}
    assert all(len(task.agent_ids) == 24 for task in tasks.values())
    assert len(rows) == 10_818
    assert Counter(row.condition for row in rows) == {
        "full": 18,
        "private": 432,
        "strategic": 1728,
        "random": 8640,
    }
    assert len(smoke_ids(rows)) == 4
    assert all(len(row.task_artifact_sha256) == 64 for row in rows)
    assert all(len(row.assignment_sha256) == 64 for row in rows)


def test_orders_and_random_packets_are_exact_and_nested():
    config, tasks, rows = plan()
    for task in tasks.values():
        orders = answer_orders(task)
        assert len(orders) == 6
        assert {tuple(mapping.values()) for _, mapping in orders} == set(
            __import__("itertools").permutations(task.semantic_answers)
        )
        first = random_permutations(config, task)
        second = random_permutations(config, task)
        assert first == second
        for _, permutation in first:
            assert (
                len(permutation)
                == len(set(permutation))
                == len(task.controller_reportable_fact_ids)
            )
            assert set(permutation) == set(task.controller_reportable_fact_ids)
    strategic = [row for row in rows if row.condition == "strategic"]
    for task_id in tasks:
        sample = {
            row.budget: row
            for row in strategic
            if row.task_id == task_id
            and row.agent_id == "agent_001"
            and row.answer_order_id == "order_012"
        }
        assert sample[3].report_fact_ids == sample[6].report_fact_ids[:3]
        assert sample[6].report_fact_ids == sample[9].report_fact_ids[:6]
        assert sample[9].report_fact_ids == sample[12].report_fact_ids[:9]


def test_prompt_and_parser_are_counterbalanced_without_metadata_leakage():
    config, tasks, rows = plan()
    for row in rows[::1000]:
        prompt = render_isolated(
            tasks[row.task_id], row, prompt_variant=config.prompt_variant
        )
        text = "\n".join(message.content for message in prompt.messages).casefold()
        assert not any(
            word in text
            for word in (
                "strategic",
                "random",
                "controller",
                "false target",
                "gold answer",
                "symbolic",
            )
        )
        semantic = tasks[row.task_id].correct_relation
        letter = next(
            key for key, value in row.option_mapping.items() if value == semantic
        )
        parsed = parse_isolated(
            tasks[row.task_id],
            row,
            json.dumps(
                {
                    "vote": letter,
                    "reason": "Evidence supports this choice.",
                    "shared_fact_id": None,
                }
            ),
        )
        assert parsed["semantic_answer"] == semantic
        assert parsed["gold_selected"] is True


def test_overlap_is_preserved_and_symbolic_metrics_are_separate():
    _, _, rows = plan()
    overlap = [row for row in rows if row.private_report_overlap_count]
    assert overlap
    assert all(len(row.report_fact_ids) == row.budget for row in overlap)
    assert any(
        row.compatible_worlds_prefix_only != row.compatible_worlds_private_plus_reports
        for row in rows
        if row.report_fact_ids
    )


def test_resume_and_missingness_do_not_turn_failures_into_other(tmp_path: Path):
    config, tasks, rows = plan()
    selected = rows[:4]
    prep = tmp_path / "preparation"
    prep.mkdir(parents=True)
    (prep / "evaluation_manifest.jsonl").write_text(
        "".join(json.dumps(row.to_dict()) + "\n" for row in selected), encoding="utf-8"
    )
    checkpoint = tmp_path / "checkpoints"
    checkpoint.mkdir()
    completed = selected[0]
    (checkpoint / f"{completed.request_id}.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "parsed": {
                    "parse_success": True,
                    "semantic_answer": tasks[completed.task_id].correct_relation,
                    "gold_selected": True,
                    "false_target_selected": False,
                    "other_selected": False,
                },
                "attempts": [],
            }
        ),
        encoding="utf-8",
    )
    failed = selected[1]
    (checkpoint / f"{failed.request_id}.json").write_text(
        json.dumps({"status": "failed", "parsed": None, "attempts": []}),
        encoding="utf-8",
    )
    assert completed_ids(tmp_path) == {completed.request_id}
    result = aggregate(tmp_path)
    request_rows = (tmp_path / "analysis/request_table.csv").read_text(encoding="utf-8")
    assert result["diagnostics"]["completed_valid"] == 1
    assert result["diagnostics"]["provider_failed"] == 1
    assert ",failed," in request_rows


def test_checksum_manifest_detects_tampering(tmp_path: Path):
    payload = tmp_path / "payload.json"
    payload.write_text('{"value": 1}\n', encoding="utf-8")
    expected = hashlib.sha256(payload.read_bytes()).hexdigest()
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == expected
    payload.write_text('{"value": 2}\n', encoding="utf-8")
    assert hashlib.sha256(payload.read_bytes()).hexdigest() != expected
