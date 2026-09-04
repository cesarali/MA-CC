from __future__ import annotations

import json
from pathlib import Path

import pytest

from mas_cc.games.relational_reasoning.data import load_musr_team_allocation_task
from mas_cc.musr_team_allocation_generator.ambiguity import (
    TeamAllocationCompletionIndex,
)
from mas_cc.musr_team_allocation_generator.latent_problem import (
    problem_from_latent_values,
)
from mas_cc.musr_team_allocation_generator.selective_design import (
    SelectiveThresholds,
    build_selective_design,
)
from mas_cc.musr_team_allocation_generator.symbolic_facts import canonical_fact_catalog
from mas_cc.probes.musr_truthful_selective.config import load_config
from mas_cc.probes.musr_truthful_selective.design import CallSpec
from mas_cc.probes.musr_truthful_selective.diversity import build_diversity_audit
from mas_cc.probes.musr_truthful_selective.generation_validation import (
    deterministic_checks,
    parse_semantic_audit,
)
from mas_cc.probes.musr_truthful_selective.prompting import parse, render
from mas_cc.probes.musr_truthful_selective.symbolic import write_design_artifacts
from mas_cc.games.relational_reasoning.imitation_round_feedback.controller import (
    RECOMMENDATION_ONLY,
    TIMING_DAWN_ONLY,
    TRUTHFUL_STRATEGIC_REPORT,
    RelationalRoundBudgetedControl,
)

CONFIG = Path(
    "configs/runs/relational_reasoning/blackboard_game/"
    "task_calibration_truthful_selective_01/calibration.yaml"
)
VECTOR = (2, 2, 2, 2, 2, 2, 1, 3, 1)


@pytest.fixture(scope="module")
def design():
    return build_selective_design(
        problem_from_latent_values(VECTOR),
        TeamAllocationCompletionIndex(),
        SelectiveThresholds(subset_samples=12),
        seed=7,
        false_target_index=2,
    )


def test_canonical_catalog_has_unique_logical_propositions():
    facts = canonical_fact_catalog(problem_from_latent_values(VECTOR))
    assert len(facts) == 99
    assert len({fact.fact_id for fact in facts}) == len(facts)
    assert len({fact.logical_signature for fact in facts}) == len(facts)


def test_symbolic_design_satisfies_required_response_profile(design):
    assert design.gold_target == "ALLOCATION_1"
    assert design.false_target == "ALLOCATION_2"
    assert len(design.controller_facts) >= 24
    assert len(design.private_assignment) == 24
    assert not (
        {fact.fact_id for fact in design.controller_facts}
        & {fact.fact_id for fact in design.decisive_facts}
    )
    zero_false = design.profiles["ZERO"]["posterior_vector"][2]
    for budget in (3, 6, 12, 24):
        facts = design.selected_controller[budget]
        assert len(facts) == len({fact.fact_id for fact in facts}) == budget
        profile = design.profiles[f"CONTROLLER_b{budget:02d}"]
        assert zero_false < profile["posterior_vector"][2] <= 0.70
        assert profile["posterior_vector"][1] > 0
        assert design.profiles[f"CONTROLLER_b{budget:02d}+DECISIVE"][
            "posterior_vector"
        ][1] == pytest.approx(1)
    assert design.profiles["DECISIVE"]["posterior_vector"][1] == pytest.approx(1)
    assert design.profiles["FULL"]["posterior_vector"][1] == pytest.approx(1)
    assert all(
        row["fraction_eliminate_truth"] == 0 for row in design.robustness.values()
    )


def test_individual_controller_audit_preserves_both_targets(design):
    for row in design.individual_controller_audit:
        posterior = row["posterior_vector"]
        assert posterior[design.problem.gold_index] > 0
        assert posterior[design.false_target_index] > 0
        assert posterior[design.false_target_index] < 1
        assert row["canonical_fact_text"]
        assert row["exact_provenance"]["latent_indices"]


def test_config_freezes_models_population_and_call_counts():
    config = load_config(CONFIG)
    assert config.generation_provider.model == "microsoft/gpt-5.6-terra"
    assert config.behavioral_provider.model == "gwdg/openai-gpt-oss-120b"
    assert config.candidate_worlds == 10_000
    assert config.symbolic.population_size == 24
    assert config.symbolic.controller_budgets == (3, 6, 12, 24)
    assert config.behavioral_calls == 1155


def test_v2_loader_and_production_prompt_are_isolated(tmp_path: Path, design):
    task = write_design_artifacts(
        tmp_path,
        design,
        task_id="task_001",
        candidate_id=1,
        seed=7,
    )
    loaded = load_musr_team_allocation_task(tmp_path, "task_001", population_size=24)
    assert loaded.correct_relation == task["gold_target"]
    assert loaded.controller_target == task["false_target"]
    assert len(loaded.controller_reportable_fact_ids) >= 24
    spec = CallSpec(
        "test",
        "task_001",
        "C3",
        0,
        tuple(
            json.loads((tmp_path / "task_001/controller/selected_C3.json").read_text())
        ),
        {"A": "ALLOCATION_2", "B": "ALLOCATION_0", "C": "ALLOCATION_1"},
        9,
    )
    prompt = render(loaded, spec)
    text = "\n".join(message.content for message in prompt.messages).casefold()
    assert "blackboard" not in text
    assert "controller" not in text
    assert "previous" not in text
    parsed = parse(
        loaded,
        spec,
        '{"vote":"A","reason":"Evidence favors it.","shared_fact_id":null}',
    )
    assert parsed["parsed_semantic_answer"] == "ALLOCATION_2"
    assert parsed["false_target_selected"] is True


def test_generation_audit_rejects_hidden_numbers_and_certainty(design):
    fact = design.facts[0]
    checks = deterministic_checks(
        design.problem,
        fact,
        ("This definitely proves skill level 3.", "The score is 9 points."),
    )
    assert checks["no_score_leakage"] is False
    assert checks["no_explicit_numeric_level"] is False
    assert checks["no_certainty_language"] is False
    parsed = parse_semantic_audit(
        json.dumps(
            {
                "faithfulness": "PASS",
                "polarity_preserved": "PASS",
                "no_strengthening": "PASS",
                "no_unsupported_implication": "PASS",
                "no_hidden_state_leakage": "PASS",
                "coherent": "PASS",
                "reason": "The event supports only the intended comparison.",
            }
        )
    )
    assert parsed["passed"] is True


def test_diversity_audit_detects_unique_signatures(tmp_path: Path, design):
    write_design_artifacts(tmp_path, design, task_id="task_001", candidate_id=1, seed=7)
    audit = build_diversity_audit(tmp_path / "task_001")
    assert audit["controller_fact_ids"] >= 24
    assert audit["distinct_logical_signatures"] == audit["controller_fact_ids"]
    assert audit["duplicate_logical_signatures"] == 0
    assert len(audit["marginal_controller_order"]) == audit["controller_fact_ids"]


def test_runtime_controller_reproduces_frozen_budget_sets(tmp_path: Path, design):
    write_design_artifacts(tmp_path, design, task_id="task_001", candidate_id=1, seed=7)
    from mas_cc.probes.musr_truthful_selective.symbolic import repair_controller_ranking

    repair_controller_ranking(tmp_path / "task_001")
    task = load_musr_team_allocation_task(tmp_path, "task_001", population_size=24)
    for budget in (3, 6, 12, 24):
        control = RelationalRoundBudgetedControl(
            target=task.controller_target or "ALLOCATION_2",
            intervention_budget=budget,
            message_mode=RECOMMENDATION_ONLY,
            controller_actuation_mode=TRUTHFUL_STRATEGIC_REPORT,
            controller_timing=TIMING_DAWN_ONLY,
        )
        actual = tuple(
            row.fact_id
            for row in control.select_truthful_reports(
                task,
                episode_seed=0,
                round_index=0,
                live_fact_counts={},
                selected_rounds={},
            )
        )
        expected = tuple(
            json.loads(
                (tmp_path / f"task_001/controller/selected_C{budget}.json").read_text()
            )
        )
        assert actual == expected
