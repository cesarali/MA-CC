from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.control import create_control
from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import RoundEvent
from mas_cc.games.relational_reasoning.data import load_relational_task
from mas_cc.games.relational_reasoning.imitation_round_feedback.runtime import (
    sample_controlled_positions,
)
from mas_cc.studies.episode_endpoints import relational_false_takeover_tables
from mas_cc.studies.manifest import discover_study
from mas_cc.studies.preflight import validate_study_preflight_contract
from mas_cc.studies.submission import build_submission_entries


ROOT = Path("configs/runs/relational_reasoning/population_study_09b")
DATASET = Path(
    "src/mas_cc/relational_task_generator/relational_task_generator/datasets/"
    "n12_L3_r03_k3"
)


def test_study09b_is_exactly_the_eight_episode_false_takeover_design():
    spec = discover_study(ROOT)
    report = validate_study_preflight_contract(spec)
    entries = build_submission_entries(spec, "/tmp/test-study09b", git_commit="test")

    assert report["status"] == "permitted"
    assert report["population_size"] == [12]
    assert report["q_values"] == [2, 3]
    assert report["L_values"] == [3]
    assert report["support_redundancy"] == [3]
    assert report["sensor_size"] == [6]
    assert report["b_values"] == [9, 12]
    assert report["receiver_dispositions"] == ["naive"]
    assert report["evidence_strategies"] == ["strategic"]
    assert report["message_modes"] == ["recommendation_plus_fact"]
    assert report["beta"] == [4.0]
    assert report["theta"] == [0.75]
    assert report["schedule"] == ["soft"]
    assert report["resolved_regimes"] == [[2, 9], [2, 12], [3, 9], [3, 12]]
    assert report["matched_revised_theory_applicable"] is False
    assert [entry.expected_cell_count for entry in entries] == [4, 4]
    assert [entry.expected_episode_count for entry in entries] == [4, 4]


def test_study09b_tasks_and_strategic_evidence_are_frozen_true_and_semantic():
    manifest = json.loads((DATASET / "manifest.json").read_text())
    assert manifest["num_tasks"] == 2
    assert manifest["config"] == {
        "population_size": 12,
        "reasoning_depth": 3,
        "support_redundancy": 3,
        "distractors": 2,
        "distractor_redundancy": 1,
        "num_options": 3,
        "no_single_agent_solution": True,
    }
    spec = discover_study(ROOT)
    for path in spec.configs:
        source = load_run_config_or_grid(path)
        assert isinstance(source, GridSpec)
        assert [(axis.path, list(axis.values)) for axis in source.axes] == [
            ("game.options.social_group_size", [2, 3]),
            ("control.options.intervention_budget", [9, 12]),
        ]
        for cell in source.cells:
            config = cell.config
            task = load_relational_task(
                DATASET,
                str(config.game.options["task_id"]),
                population_size=12,
            )
            control = create_control(config.control)
            target = control.resolved_target_for_task(task, config.execution.seed)
            fact_id = control.resolve_fact_id(task, config.execution.seed)
            assert target in task.semantic_answers
            assert target != task.correct_relation
            assert fact_id in task.facts
            assert config.experiment.metadata["ground_truth"] == task.correct_relation
            assert config.experiment.metadata["controller_target"] == target
            assert config.experiment.metadata["controller_target_is_truth"] is False


def test_study09b_hard_contract_rejects_q1(tmp_path):
    copied = tmp_path / "population_study_09b"
    shutil.copytree(ROOT, copied)
    path = copied / "study09b_task0001_false_takeover.yaml"
    path.write_text(
        path.read_text().replace(
            "game.options.social_group_size: [2, 3]",
            "game.options.social_group_size: [1, 2]",
        )
    )
    with pytest.raises(ValueError, match="q values must be \[2, 3\]"):
        validate_study_preflight_contract(discover_study(copied))


class _LastNine:
    def sample(self, population, k):
        return list(population)[-k:]


@pytest.mark.parametrize("budget", [9, 12])
def test_study09b_advocacy_budget_controls_exactly_b_of_twelve_positions(budget):
    positions = sample_controlled_positions(12, budget, _LastNine())
    assert len(positions) == budget
    assert len(set(positions)) == budget


def _round(
    *,
    index: int,
    before: tuple[int, int, int],
    after: tuple[int, int, int],
) -> RoundEvent:
    return RoundEvent(
        cell_id="config-0000/cell-0000",
        episode_id="episode-0000",
        round_index=index,
        event={
            "task_id": "task_0001",
            "possible_answers": ["NORTH", "SOUTHEAST", "WEST"],
            "correct_answer": "SOUTHEAST",
            "controller_target": "NORTH",
            "analysis_target": "NORTH",
            "social_group_size": 2,
            "intervention_budget": 9,
            "occupation_counts_before": before,
            "occupation_counts_after": after,
        },
    )


def test_false_takeover_endpoint_uses_semantics_and_never_counts_ties_as_wins():
    rounds = [
        _round(index=0, before=(2, 8, 2), after=(7, 3, 2)),
        _round(index=1, before=(7, 3, 2), after=(5, 5, 2)),
    ]
    cells = pd.DataFrame([{"cell_id": "config-0000/cell-0000"}])
    episodes, summary = relational_false_takeover_tables(rounds, cells)
    row = episodes.iloc[0]

    assert row["initial_false_target_share"] == pytest.approx(2 / 12)
    assert row["max_false_target_share"] == pytest.approx(7 / 12)
    assert row["first_false_majority_round"] == 0
    assert bool(row["final_is_tie"]) is True
    assert bool(row["false_target_is_final_winner"]) is False
    assert row["takeover_classification"] == "TRANSIENT_FALSE_MAJORITY"
    assert summary.iloc[0]["false_wins"] == 0
    assert summary.iloc[0]["ties"] == 1
