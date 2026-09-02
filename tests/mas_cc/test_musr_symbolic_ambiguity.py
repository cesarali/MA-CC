from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from pathlib import Path

import pytest

from mas_cc.musr_team_allocation_generator.ambiguity import (
    TeamAllocationCompletionIndex,
    choose_private_views,
    exact_private_view_metrics,
)
from mas_cc.musr_team_allocation_generator.latent_problem import (
    LATENT_VALUE_SUPPORT,
    latent_facts,
    latent_values,
    problem_from_latent_values,
)
from mas_cc.probes.musr_prompt_solvability.prompting import render
from mas_cc.probes.musr_symbolic_ambiguity.config import load_config
from mas_cc.probes.musr_symbolic_ambiguity.design import call_plan
from mas_cc.probes.musr_symbolic_ambiguity.runner import load_tasks

CONFIG = Path("configs/probes/musr_symbolic_ambiguity_calibration_01.yaml")
RESULT = Path("results/studies/musr_symbolic_ambiguity_calibration_01")


def test_exact_completion_enumeration_and_tie_policy_on_tiny_case():
    result = exact_private_view_metrics(
        latent_count=2,
        visible_indices=(0,),
        visible_values=(1,),
        support=(0, 1),
        score_function=lambda z: (z[0], z[1], 0),
        min_score_margin=0,
    )

    assert result.valid_completion_count == 1
    assert result.invalid_completion_count == 1
    assert result.probabilities == pytest.approx((1.0, 0.0, 0.0))
    assert sum(result.probabilities) == pytest.approx(1.0)


def test_nonuniform_completion_prior_is_respected():
    result = exact_private_view_metrics(
        latent_count=1,
        visible_indices=(),
        visible_values=(),
        support=(0, 1),
        priors={0: 0.8, 1: 0.2},
        score_function=lambda z: (1, 0, 0) if z[0] == 0 else (0, 1, 0),
        min_score_margin=1,
    )

    assert result.probabilities == pytest.approx((0.8, 0.2, 0.0))


@pytest.fixture(scope="module")
def completion_index() -> TeamAllocationCompletionIndex:
    return TeamAllocationCompletionIndex(min_score_margin=1)


def test_actual_semantic_gold_and_full_observation_certainty(completion_index):
    problem = problem_from_latent_values((3, 1, 1, 3, 1, 3, 1, 1, 3))
    scores = problem.candidate_scores
    assert problem.gold_index == scores.index(max(scores))
    assert len([score for score in scores if score == max(scores)]) == 1

    metrics = completion_index.metrics(latent_values(problem), range(9))
    expected = [0.0, 0.0, 0.0]
    expected[problem.gold_index] = 1.0
    assert metrics.probabilities == pytest.approx(expected)
    assert metrics.max_predictability == pytest.approx(1.0)
    assert metrics.normalized_entropy == pytest.approx(0.0)


def test_all_k_scans_are_deterministic_and_bounded(completion_index):
    problem = problem_from_latent_values((3, 2, 1, 3, 2, 3, 1, 2, 3))
    for k, expected_count in ((2, 36), (3, 84), (4, 126)):
        first = completion_index.scan(problem, k)
        second = completion_index.scan(problem, k)
        assert first == second
        assert len(first) == expected_count
        assert all(sum(row.probabilities) == pytest.approx(1.0) for row in first)
        assert all(1 / 3 <= row.max_predictability <= 1 for row in first)
        assert all(0 <= row.normalized_entropy <= 1 for row in first)


def test_private_assignment_satisfies_ambiguity_and_population_coverage(
    completion_index,
):
    qualifying = None
    for vector in itertools.product(LATENT_VALUE_SUPPORT, repeat=9):
        problem = problem_from_latent_values(vector)
        if problem.margin_to_second_best < 1:
            continue
        try:
            qualifying = (
                problem,
                choose_private_views(
                    problem,
                    completion_index,
                    breadth=4,
                    population_size=12,
                    max_predictability=0.45,
                    min_normalized_entropy=0.90,
                    seed=7,
                ),
            )
            break
        except ValueError:
            continue
    assert qualifying is not None
    _, views = qualifying
    holders = Counter(index for view in views for index in view.visible_indices)
    assert len(views) == 12
    assert set(holders) == set(range(9))
    assert min(holders.values()) >= 2
    assert all(len(view.visible_indices) == 4 for view in views)
    assert all(view.max_predictability <= 0.45 for view in views)
    assert all(view.normalized_entropy >= 0.90 for view in views)


def test_latent_order_is_exactly_six_skills_then_three_cooperations():
    problem = problem_from_latent_values((1, 2, 3, 1, 2, 3, 1, 2, 3))
    facts = latent_facts(problem)
    assert [fact.kind for fact in facts] == ["skill"] * 6 + ["cooperation"] * 3
    assert latent_values(problem) == (1, 2, 3, 1, 2, 3, 1, 2, 3)


def test_answer_letter_permutations_do_not_change_semantic_gold():
    problem = problem_from_latent_values((3, 1, 1, 3, 1, 3, 1, 1, 3))
    semantic = f"ALLOCATION_{problem.gold_index}"
    for permutation in itertools.permutations(
        ("ALLOCATION_0", "ALLOCATION_1", "ALLOCATION_2")
    ):
        mapping = dict(zip("ABC", permutation, strict=True))
        selected_letter = next(key for key, value in mapping.items() if value == semantic)
        assert mapping[selected_letter] == semantic


def test_entropy_formula_matches_three_choice_normalization():
    result = exact_private_view_metrics(
        latent_count=1,
        visible_indices=(),
        visible_values=(),
        support=(0, 1, 2),
        score_function=lambda z: tuple(int(index == z[0]) for index in range(3)),
        min_score_margin=1,
    )
    assert result.max_predictability == pytest.approx(1 / 3)
    assert result.normalized_entropy == pytest.approx(
        -3 * (1 / 3) * math.log(1 / 3) / math.log(3)
    )


def test_calibration_config_freezes_models_thresholds_and_sample_sizes():
    config = load_config(CONFIG)
    assert config.generation_provider.model == "microsoft/gpt-5.6-terra"
    assert config.behavioral_provider.model == "gwdg/openai-gpt-oss-120b"
    assert config.candidate_worlds == 10_000
    assert config.private_breadth_candidates == (2, 3, 4)
    assert config.nominal_generation_calls == 54
    assert config.maximum_generation_calls == 162
    assert config.behavioral_calls == 336


def test_frozen_task_pack_has_exact_balance_and_valid_private_views():
    frozen = json.loads((RESULT / "symbolic_scan/frozen_selection.json").read_text())
    rule = frozen["construction_rule"]
    assert rule == {
        "available_by_gold": {
            "ALLOCATION_0": 133,
            "ALLOCATION_1": 130,
            "ALLOCATION_2": 125,
        },
        "balance_rule": "exactly 2 tasks per ALLOCATION ID",
        "criterion": "preferred",
        "max_predictability": 0.45,
        "min_normalized_entropy": 0.9,
        "min_score_margin": 2,
        "private_breadth": 4,
        "selection_rule": (
            "preferred before fallback; largest feasible breadth; "
            "then largest feasible score margin"
        ),
    }
    assert Counter(row["gold_answer"] for row in frozen["selected_worlds"]) == {
        "ALLOCATION_0": 2,
        "ALLOCATION_1": 2,
        "ALLOCATION_2": 2,
    }
    for task in frozen["selected_worlds"]:
        assert task["score_margin"] >= 2
        assert len(task["private_views"]) == 12
        assert all(len(row["visible_indices"]) == 4 for row in task["private_views"])
        assert all(row["max_predictability"] <= 0.45 for row in task["private_views"])
        assert all(row["normalized_entropy"] >= 0.9 for row in task["private_views"])
        holders = Counter(
            index for row in task["private_views"] for index in row["visible_indices"]
        )
        assert set(holders) == set(range(9))
        assert min(holders.values()) >= 2


def test_generated_assignments_cover_cards_without_increasing_latent_breadth():
    for task_dir in sorted((RESULT / "accepted_tasks").glob("task_*")):
        base = json.loads((task_dir / "base_task.json").read_text())
        distribution = json.loads((task_dir / "distribution_N12.json").read_text())
        evidence_ids = {row["evidence_id"] for row in base["evidence"]}
        union = {
            evidence_id
            for held in distribution["agent_evidence_ids"].values()
            for evidence_id in held
        }
        assert len(base["evidence"]) == 27
        assert union == evidence_ids
        assert all(len(held) == 6 for held in distribution["agent_evidence_ids"].values())
        assert all(row["distinct_latent_facts"] == 4 for row in distribution["agent_diagnostics"])
        assert distribution["no_single_agent_violations"] == 0


def test_behavioral_plan_is_p2_f9_and_never_leaks_hidden_matrices():
    config = load_config(CONFIG)
    tasks = load_tasks(config, RESULT)
    specs = call_plan(
        tasks,
        private_repetitions=config.private_repetitions,
        endpoint_repetitions=config.endpoint_repetitions,
        seed=config.seed,
    )
    assert len(specs) == 336
    assert Counter(spec.packet_variant for spec in specs) == {
        "Zero": 60,
        "Private": 216,
        "F9": 60,
    }
    forbidden = (
        "skill_matrix",
        "cooperation_matrix",
        "candidate_scores",
        "gold_answer",
        "max_predictability",
        "normalized_entropy",
    )
    for spec in specs:
        prompt = "\n".join(
            message.content for message in render(tasks[spec.task_id], spec).messages
        )
        assert spec.prompt_variant == "P2"
        assert all(term not in prompt for term in forbidden)
