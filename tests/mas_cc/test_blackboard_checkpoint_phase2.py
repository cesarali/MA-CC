"""Phase-2 configuration, validation, and paired-analysis contracts."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mas_cc.analysis.checkpoint_ensemble import (
    _folds,
    assigned_policy_information,
    cross_fitted_classifier_score,
    endpoint_table,
    paired_response,
    prepare_checkpoint_ensemble_inputs,
    validate_checkpoint_ensemble,
)
from mas_cc.config import CheckpointEnsembleConfig, ConfigLoader, load_run_config, parse_run_config
from mas_cc.games import create_game
from mas_cc.games.relational_reasoning.imitation_round_feedback import (
    run_checkpoint_parent_bundle,
)
from mas_cc.llm_runtime.exceptions import ConfigurationError
from mas_cc.llm_runtime.providers.adapters.mock import MockLLMProvider
from mas_cc.planning.game_preflight import call_plan_for_run

STUDY = Path(
    "configs/runs/relational_reasoning/blackboard_game/"
    "blackboard_checkpoint_ensemble_01"
)
POLICIES = ("none", "always_truth", "always_false", "sensing_truth", "sensing_false")


def _rounds(parents: int = 6, horizons: int = 2) -> pd.DataFrame:
    rows = []
    for parent_index in range(parents):
        parent = f"cell:p{parent_index:03d}"
        checkpoint_hash = f"hash-{parent_index}"
        n0_truth = 3 + parent_index % 3
        rows.append(
            {
                "parent_id": parent,
                "checkpoint_id": f"checkpoint-{parent_index}",
                "checkpoint_hash": checkpoint_hash,
                "branch_policy": "checkpoint",
                "posting_budget": None,
                "copy_id": 1,
                "absolute_round": 2,
                "post_branch_horizon": 0,
                "q": 3,
                "rho": 0.7,
                "N": 10,
                "L": 2,
                "M": horizons,
                "correct_answer_semantic_id": "T",
                "false_target_semantic_id": "F",
                "preparation_seed": parent_index,
                "continuation_seed": None,
                "continuation_stream_identity": None,
                "branch_status": "checkpoint_sealed",
                "option_counts": {"T": n0_truth, "F": 10 - n0_truth},
            }
        )
        for policy in POLICIES:
            budgets = (None,) if policy == "none" else (3, 12)
            for budget in budgets:
                target = None if policy == "none" else "T" if policy.endswith("truth") else "F"
                for horizon in range(1, horizons + 1):
                    truth = n0_truth
                    if policy.endswith("truth"):
                        truth = min(10, truth + horizon)
                    elif policy.endswith("false"):
                        truth = max(0, truth - horizon)
                    rows.append(
                        {
                            "parent_id": parent,
                            "checkpoint_id": f"checkpoint-{parent_index}",
                            "checkpoint_hash": checkpoint_hash,
                            "branch_policy": policy,
                            "posting_budget": budget,
                            "copy_id": 1,
                            "absolute_round": 2 + horizon,
                            "post_branch_horizon": horizon,
                            "q": 3,
                            "rho": 0.7,
                            "N": 10,
                            "L": 2,
                            "M": horizons,
                            "correct_answer_semantic_id": "T",
                            "false_target_semantic_id": "F",
                            "controller_target_semantic_id": target,
                            "preparation_seed": parent_index,
                            "continuation_seed": 1000 + parent_index * 100 + len(rows),
                            "continuation_stream_identity": {
                                "parent": parent, "policy": policy, "budget": budget
                            },
                            "branch_status": "in_progress",
                            "possible_answers": ["T", "F"],
                            "occupation_counts_before": [n0_truth, 10 - n0_truth],
                            "occupation_counts_after": [truth, 10 - truth],
                            "controller_sampled_U": int(policy != "none"),
                            "actual_controller_posts": int(policy != "none"),
                            "total_eligible_board_message_reads": 10,
                            "controller_unique_readers": int(policy != "none"),
                            "retry_attempts_this_round": 0,
                        }
                    )
    return pd.DataFrame(rows)


def test_four_fixed_configs_resolve_and_preflight_exact_ensemble_demand():
    found = set()
    for path in sorted(STUDY.glob("q*_rho*.yaml")):
        config = ConfigLoader(environment={}).load(path)
        found.add(
            (
                int(config.game.options["social_group_size"]),
                float(config.game.options["epistemic_persistence"]),
            )
        )
        plan = call_plan_for_run(create_game(config.game), config)
        assert config.execution.repetitions == config.ensemble.parent_count == 120
        assert config.ensemble.branches_per_parent_copy == 9
        assert plan.metadata["focal_updates_per_parent"] == 2208
        assert plan.provider_requests.lower == 2312
    assert found == {(3, 0.7), (3, 1.0), (12, 0.7), (12, 1.0)}


def test_ensemble_cross_field_validation_is_strict():
    config = ConfigLoader(environment={}).load(STUDY / "q3_rho070.yaml")
    raw = config.to_dict()
    raw["execution"]["repetitions"] = 119
    with pytest.raises(ConfigurationError, match="ensemble.parent_count"):
        parse_run_config(raw)


def test_strict_coverage_endpoint_and_parent_paired_effects():
    rounds = _rounds()
    validation = validate_checkpoint_ensemble(
        rounds,
        expected_policies=POLICIES,
        posting_budgets=(3, 12),
        continuation_copies=1,
        continuation_rounds=2,
        expected_parents=6,
    )
    assert validation["complete"]
    endpoints = endpoint_table(rounds)
    paired, effects = paired_response(
        endpoints, horizons=(1, 2), bootstrap_resamples=50, seed=7
    )
    truth = effects[
        (effects["branch_policy"] == "always_truth")
        & (effects["posting_budget"] == 3)
        & (effects["target_semantics"] == "truth")
        & (effects["post_branch_horizon"] == 1)
    ].iloc[0]
    assert truth["effective_K"] == 6
    assert truth["estimate_fraction"] == pytest.approx(0.1)
    assert paired["baseline_count"].notna().all()

    missing = rounds[
        ~(
            (rounds["parent_id"] == "cell:p000")
            & (rounds["branch_policy"] == "always_truth")
            & (rounds["posting_budget"] == 3)
        )
    ]
    invalid = validate_checkpoint_ensemble(
        missing,
        expected_policies=POLICIES,
        posting_budgets=(3, 12),
        continuation_copies=1,
        continuation_rounds=2,
    )
    assert not invalid["complete"]
    assert any("branch coverage differs" in error for error in invalid["errors"])


def test_partial_checkpoint_analysis_recovers_only_complete_paths_and_qualifies_parents():
    completed = _rounds(parents=1, horizons=2).assign(cell_key="cell-a")
    interrupted = _rounds(parents=1, horizons=2).assign(cell_key="cell-b")
    missing_path = (
        interrupted["branch_policy"].eq("sensing_false")
        & interrupted["posting_budget"].eq(12)
    )
    interrupted = interrupted[~missing_path].copy()
    duplicate = interrupted[
        interrupted["branch_policy"].eq("always_truth")
        & interrupted["posting_budget"].eq(3)
        & interrupted["post_branch_horizon"].eq(1)
    ].copy()
    duplicate["actual_controller_posts"] = 99
    interrupted = pd.concat([interrupted, duplicate], ignore_index=True)

    recovered, micro, diagnostics = prepare_checkpoint_ensemble_inputs(
        completed,
        available_round_prefixes=interrupted,
        continuation_rounds=2,
        expected_policies=POLICIES,
        posting_budgets=(3, 12),
    )

    assert micro.empty
    assert diagnostics["parents_observed"] == 2
    assert diagnostics["parents_with_complete_branch_coverage"] == 1
    assert diagnostics["complete_paths"] == 17
    assert diagnostics["excluded_incomplete_paths"] == 1
    assert recovered["parent_id"].nunique() == 2
    assert not (
        recovered["parent_id"].str.startswith("cell-b::")
        & recovered["branch_policy"].eq("sensing_false")
        & recovered["posting_budget"].eq(12)
    ).any()
    latest = recovered[
        recovered["parent_id"].str.startswith("cell-b::")
        & recovered["branch_policy"].eq("always_truth")
        & recovered["posting_budget"].eq(3)
        & recovered["post_branch_horizon"].eq(1)
    ].iloc[0]
    assert latest["actual_controller_posts"] == 99


def test_existing_cmi_adapter_detects_assigned_policy_information():
    endpoints = endpoint_table(_rounds(parents=20))
    paired, _ = paired_response(endpoints, horizons=(2,), bootstrap_resamples=0)
    estimates = assigned_policy_information(paired, population_size=10, smoothing=(0, 1))
    row = estimates[
        (estimates["branch_policy"] == "always_truth")
        & (estimates["posting_budget"] == 3)
        & (estimates["target_semantics"] == "truth")
        & (estimates["estimator"] == "frequency_unsmoothed")
    ].iloc[0]
    assert row["estimate_bits"] > 0.5


def test_grouped_classifier_folds_never_split_a_parent():
    groups = np.repeat(np.array([f"p{i}" for i in range(10)]), 2)
    for train, test in _folds(groups, 5, 11):
        assert set(groups[train]).isdisjoint(set(groups[test]))

    endpoints = endpoint_table(_rounds(parents=10))
    paired, _ = paired_response(endpoints, horizons=(2,), bootstrap_resamples=0)
    group = paired[
        (paired["branch_policy"] == "always_truth")
        & (paired["posting_budget"] == 3)
        & (paired["target_semantics"] == "truth")
    ]
    result = cross_fitted_classifier_score(group, population_size=10, seed=5)
    assert result["outer_folds"] == 5
    assert result["parents"] == 10
    assert result["probability_clip"] == [1e-6, 1 - 1e-6]


def test_classifier_progress_counts_refits_without_changing_results(monkeypatch):
    import mas_cc.analysis.checkpoint_ensemble as module

    paired, _ = paired_response(
        endpoint_table(_rounds(parents=2)), horizons=(2,), bootstrap_resamples=0
    )
    paired = paired[
        paired["branch_policy"].eq("always_truth")
        & paired["posting_budget"].eq(3)
        & paired["target_semantics"].eq("truth")
    ]
    monkeypatch.setattr(
        module, "cross_fitted_classifier_score",
        lambda *args, **kwargs: {"estimate_bits": float(kwargs["seed"])},
    )
    updates = []
    actual = module.classifier_analysis(
        paired, population_size=10, seed=7, repeated_splits=2,
        label_swap_permutations=3, progress_callback=updates.append,
    )
    expected = module.classifier_analysis(
        paired, population_size=10, seed=7, repeated_splits=2,
        label_swap_permutations=3,
    )
    for left, right in zip(actual, expected):
        pd.testing.assert_frame_equal(left, right)
    assert updates[0]["completed_classifier_refits"] == 0
    assert updates[-1]["completed_classifier_refits"] == 5
    assert updates[-1]["total_classifier_refits"] == 5
    assert updates[-1]["completed_permutations"] == 3
    assert updates[-1]["completed_comparisons"] == 1
    assert updates[-1]["classifier_eta_seconds"] == 0


def test_classifier_parallel_permutations_match_serial():
    from mas_cc.analysis.checkpoint_ensemble import classifier_analysis

    paired, _ = paired_response(
        endpoint_table(_rounds(parents=6)), horizons=(2,), bootstrap_resamples=0
    )
    paired = paired[
        paired["branch_policy"].eq("always_truth")
        & paired["posting_budget"].eq(3)
        & paired["target_semantics"].eq("truth")
    ]
    settings = dict(population_size=10, seed=17, repeated_splits=1,
                    label_swap_permutations=3)
    serial = classifier_analysis(paired, workers=1, **settings)
    parallel = classifier_analysis(paired, workers=2, **settings)
    for left, right in zip(serial, parallel):
        pd.testing.assert_frame_equal(left, right)


def test_downscaled_orchestrator_bundle_path_seals_all_children(tmp_path):
    config = load_run_config(
        "configs/runs/relational_reasoning/misselaneous/"
        "relational_blackboard_no_control_smoke.yaml",
        environment={},
    )
    game = create_game(config.game)
    initial = game.initialize(config.game, config.execution.seed)
    false_target = next(
        answer for answer in initial.possible_answers if answer != initial.correct_answer
    )
    config = replace(
        config,
        ensemble=CheckpointEnsembleConfig(
            enabled=True,
            parent_count=1,
            preparation_rounds=1,
            continuation_rounds=1,
            continuation_copies=1,
            posting_budgets=(1, 2),
            false_target=false_target,
        ),
    )

    class Observer:
        retain_result_history = True
        episode_label = "episode-000001"

        def __init__(self) -> None:
            self.recorder = SimpleNamespace(
                output_dir=tmp_path, scientific_identity=None
            )
            self.rounds = []

        def record_round_trajectory(self, *, record):
            self.rounds.append(record)

    observer = Observer()
    result = asyncio.run(
        run_checkpoint_parent_bundle(
            game=game,
            config=config,
            provider=MockLLMProvider(config.llm_provider),
            observer=observer,
        )
    )
    assert result.termination_reason == "checkpoint_parent_bundle_complete"
    assert len(result.bundle["completed_branches"]) == 9
    assert sum(
        isinstance(row, dict) and row.get("branch_policy") == "checkpoint"
        for row in observer.rounds
    ) == 1
