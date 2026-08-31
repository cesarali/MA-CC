import asyncio
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.control import create_control
from mas_cc.core import AgentId
from mas_cc.games import create_game
from mas_cc.games.protocols import Action
from mas_cc.games.relational_reasoning.data import load_relational_task
from mas_cc.games.relational_reasoning.imitation_round_feedback.initialization import (
    artifact_from_actions,
    initialization_artifact_path,
    physical_initial_state_projection,
    read_initialization_artifact,
    write_initialization_artifact,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.runtime import (
    build_social_sources,
    run_relational_imitation_round_feedback_game,
)
from mas_cc.llm_runtime.providers.adapters.mock import MockLLMProvider
from mas_cc.studies.aggregation import (
    _phi_conditioning_comparison,
    _rho_aggregated_state_local_maps,
    _state_local_phase_tables,
)
from mas_cc.studies.initialization import build_initialization_plan
from mas_cc.studies.manifest import discover_study
from mas_cc.studies.preflight import validate_study_preflight_contract
from mas_cc.studies.submission import build_submission_entries
from mas_cc.studies.validation import paired_initialization_diagnostics


ROOT = Path("configs/runs/relational_reasoning")
FALSE = (
    ROOT / "population_study_09j" / "study09j_task0002_n24_q1_l2_false_persistence.yaml"
)
TRUTH = (
    ROOT / "population_study_09k" / "study09k_task0002_n24_q1_l2_truth_persistence.yaml"
)
RHO = [0.75, 0.85, 1.0]
BUDGETS = [6, 8, 12, 16, 18, 24]


def _grid(path: Path) -> GridSpec:
    source = load_run_config_or_grid(path)
    assert isinstance(source, GridSpec)
    return source


def test_n24_q1_l2_vigilant_strategic_contracts_are_exact():
    for study, semantics in (
        ("population_study_09j", "false only"),
        ("population_study_09k", "truth only"),
    ):
        spec = discover_study(ROOT / study)
        report = validate_study_preflight_contract(spec)
        entry = build_submission_entries(spec, f"/tmp/{study}", git_commit="test")[0]
        assert report["status"] == "permitted"
        assert report["population_size"] == [24]
        assert report["q_values"] == [1]
        assert report["L_values"] == [2]
        assert report["support_redundancy"] == [6]
        assert report["sensor_size"] == [12]
        assert report["rho_values"] == RHO
        assert report["b_values"] == BUDGETS
        assert report["receiver_dispositions"] == ["vigilant"]
        assert report["evidence_strategies"] == ["strategic"]
        assert report["initialization_modes"] == ["paired_local_vote"]
        assert report["target_semantics"] == [semantics]
        assert report["total_cells"] == 18
        assert report["total_episodes"] == 360
        assert entry.expected_cell_count == 18
        assert entry.expected_episode_count == 360


def test_dataset_and_targets_are_frozen_real_and_matched():
    dataset = Path(
        "src/mas_cc/relational_task_generator/relational_task_generator/datasets/pop24_L2_r06"
    )
    task = load_relational_task(dataset, "task_0002", population_size=24)
    assert task.correct_relation == "SOUTHWEST"
    assert len(task.supporting_fact_ids) == 2
    for path, target in ((FALSE, "SOUTH"), (TRUTH, "SOUTHWEST")):
        config = _grid(path).base
        control = create_control(config.control)
        assert control.resolved_target_for_task(task, config.execution.seed) == target
        assert control.resolve_fact_id(task, config.execution.seed) in task.facts


def test_false_truth_and_all_cells_share_initialization_compatibility(tmp_path):
    plan = build_initialization_plan([FALSE, TRUTH], tmp_path)
    assert len(plan) == 20
    assert len({entry.episode_seed for entry in plan}) == 20
    for source in (_grid(FALSE), _grid(TRUTH)):
        assert [(axis.path, list(axis.values)) for axis in source.axes] == [
            ("game.options.epistemic_persistence", RHO),
            ("control.options.intervention_budget", BUDGETS),
        ]


def _initial_actions(game, config, seed, offset=0):
    state = game.initialize(config.game, seed)
    actions = []
    for index, request in enumerate(game.initial_vote_requests(state, config.game)):
        known = state.relational_agent(request.agent_id).active_fact_ids
        actions.append(
            Action(
                request.agent_id,
                state.possible_answers[(index + offset) % len(state.possible_answers)],
                request.stage,
                {
                    "kind": "relational_ballot",
                    "reason": f"natural reason {index + offset}",
                    "shared_fact_id": known[0] if known else None,
                    "shared_fact_present": True,
                    "resolved": True,
                },
            )
        )
    return tuple(actions)


def test_artifact_replay_pairs_full_state_and_sends_no_initialization_calls(tmp_path):
    source = _grid(FALSE)
    config = source.cells[0].config
    initialization = {
        **config.game.options["initialization"],
        "artifact_dir": str(tmp_path),
    }
    config = replace(
        config,
        game=replace(
            config.game,
            options={
                **config.game.options,
                "initialization": initialization,
                "rounds": 1,
            },
            horizon=1,
        ),
    )
    seed = config.execution.seed
    game = create_game(config.game)
    actions = _initial_actions(game, config, seed)
    artifact = artifact_from_actions(game, config, seed, actions, repetition_index=0)
    write_initialization_artifact(initialization_artifact_path(config, seed), artifact)
    calls = 0

    def response(request):
        nonlocal calls
        calls += 1
        return '{"vote":"A","reason":"update","shared_fact_id":"none"}'

    result = asyncio.run(
        run_relational_imitation_round_feedback_game(
            game,
            config,
            MockLLMProvider(config.llm_provider, response_factory=response),
            control=create_control(config.control),
        )
    )
    assert result.initial_decisions == ()
    assert calls == 24  # one round of updates, not 24 extra local-vote calls
    assert (
        physical_initial_state_projection(result.initial_state)
        == artifact["physical_initial_state"]
    )
    assert all(
        row.event["initialization_artifact_hash"] == artifact["artifact_hash"]
        for row in result.rounds
    )


def test_several_repetitions_vary_naturally_but_pair_across_all_conditions(tmp_path):
    false, truth = _grid(FALSE), _grid(TRUTH)
    selected = [
        false.cells[0],
        false.cells[5],
        false.cells[-1],
        truth.cells[0],
        truth.cells[-1],
    ]
    plan = build_initialization_plan([FALSE, TRUTH], tmp_path)[:3]
    repetition_hashes = []
    for repetition, entry in enumerate(plan):
        base = false.base
        initialization = {
            **base.game.options["initialization"],
            "artifact_dir": str(tmp_path),
        }
        config = replace(
            base,
            game=replace(
                base.game,
                options={**base.game.options, "initialization": initialization},
            ),
            execution=replace(base.execution, seed=entry.episode_seed),
        )
        game = create_game(config.game)
        actions = _initial_actions(game, config, entry.episode_seed, offset=repetition)
        artifact = artifact_from_actions(
            game, config, entry.episode_seed, actions, repetition_index=repetition
        )
        write_initialization_artifact(
            initialization_artifact_path(config, entry.episode_seed), artifact
        )
        repetition_hashes.append(artifact["physical_initial_state_hash"])
        for cell in selected:
            cell_config = replace(
                cell.config,
                game=replace(
                    cell.config.game,
                    options={
                        **cell.config.game.options,
                        "initialization": initialization,
                    },
                ),
                execution=replace(cell.config.execution, seed=entry.episode_seed),
            )
            cell_game = create_game(cell_config.game)
            loaded, _, state = read_initialization_artifact(
                initialization_artifact_path(cell_config, entry.episode_seed),
                cell_game,
                cell_config,
                entry.episode_seed,
            )
            assert (
                loaded["physical_initial_state_hash"]
                == artifact["physical_initial_state_hash"]
            )
            assert (
                physical_initial_state_projection(state)
                == artifact["physical_initial_state"]
            )
    assert len(set(repetition_hashes)) == 3


def test_q1_production_control_replaces_the_only_peer():
    config = _grid(FALSE).cells[0].config
    game = create_game(config.game)
    state = game.initialize(config.game, config.execution.seed)
    sources = build_social_sources(
        state,
        (state.agents[1].agent_id,),
        replaced_peer_slot=0,
        controller_target="SOUTH",
        population_size=24,
        controller_fact_id=None,
    )
    assert len(sources) == 1
    assert sources[0]["source_type"] == "control"


def test_analysis_recipe_requests_eight_bin_phi_and_rho_outputs():
    for study in ("population_study_09j", "population_study_09k"):
        recipe = yaml.safe_load((ROOT / study / "analysis.yaml").read_text())
        assert recipe["theoretical_reference"] == "none"
        assert recipe["state_local"] == ["x"]
        assert recipe["state_local_x_bins"] == 8
        assert recipe["rho_aggregated_descriptive"] is True
        assert "round_phi_target_actuation_cmi" in recipe["estimators"]
        assert "eta_th" in recipe["derived"]
        assert "eta_th_state_local" not in recipe["derived"]


def test_phi_comparison_contains_raw_null_adjusted_and_support_values():
    rows = []
    for metric, estimate, null in (
        ("round_target_actuation_cmi", 0.2, 0.05),
        ("round_phi_target_actuation_cmi", 0.3, 0.1),
    ):
        rows.append(
            {
                "study_id": "s",
                "source_run_id": "r",
                "cell_id": "c",
                "metric": metric,
                "estimate": estimate,
                "null_mean": null,
                "ci_low": estimate - 0.1,
                "ci_high": estimate + 0.1,
                "action_entropy_ceiling_bits": 0.8,
                "dual_action_support_fraction": 0.6,
                "support_status": "adequate",
                "n_observations": 600,
            }
        )
    result = _phi_conditioning_comparison(pd.DataFrame(rows)).iloc[0]
    assert result["Delta_T_phi"] == pytest.approx(0.1)
    assert result["T_pi_null_adjusted"] == pytest.approx(0.15)
    assert result["T_pi_phi_null_adjusted"] == pytest.approx(0.2)
    assert result["T_pi_phi_dual_action_support_fraction"] == 0.6


def test_initialization_audit_fails_on_any_unpaired_repetition():
    rows = []
    for repetition in (0, 1):
        for cell in ("a", "b"):
            physical = (
                f"state-{repetition}"
                if not (repetition == 1 and cell == "b")
                else "bad"
            )
            rows.append(
                {
                    "cell_id": cell,
                    "episode_id": f"{cell}-{repetition}",
                    "round_index": 0,
                    "target_count_before": repetition + 2,
                    "N": 24,
                    "initialization_repetition": repetition,
                    "physical_initial_state_hash": physical,
                    "initial_task_id": "task_0002",
                    "initial_vote_vector": ["SOUTH"],
                    "initial_active_fact_ids_by_agent": [["f1"]],
                    "initial_known_fact_ids_by_agent": [["f1"]],
                }
            )
    _, audit, report = paired_initialization_diagnostics(
        {"rounds": pd.DataFrame(rows), "cells": pd.DataFrame({"cell_id": ["a", "b"]})}
    )
    assert report["paired_initialization_pass"] is False
    assert not bool(
        audit.loc[
            audit["initialization_repetition"] == 1, "paired_initialization_pass"
        ].iloc[0]
    )


def test_rho_aggregation_is_observation_weighted():
    phase = pd.DataFrame(
        [
            {
                "social_group_size": 1,
                "controller_evidence_strategy": "strategic",
                "receiver_epistemic_disposition": "vigilant",
                "target_semantics": "false",
                "intervention_budget": 6,
                "target_fraction_bin_index": 0,
                "target_fraction_bin_lower": 0.0,
                "target_fraction_bin_upper": 0.125,
                "target_fraction_bin_center": 0.0625,
                "target_fraction_bin_count": 8,
                "metric": "T_pi",
                "estimate": 1.0,
                "n_observations": 1,
                "epistemic_persistence": 0.75,
                "phase_status": "adequate",
            },
            {
                "social_group_size": 1,
                "controller_evidence_strategy": "strategic",
                "receiver_epistemic_disposition": "vigilant",
                "target_semantics": "false",
                "intervention_budget": 6,
                "target_fraction_bin_index": 0,
                "target_fraction_bin_lower": 0.0,
                "target_fraction_bin_upper": 0.125,
                "target_fraction_bin_center": 0.0625,
                "target_fraction_bin_count": 8,
                "metric": "T_pi",
                "estimate": 0.0,
                "n_observations": 3,
                "epistemic_persistence": 0.85,
                "phase_status": "adequate",
            },
        ]
    )
    occupancy = phase.drop(columns=["metric", "estimate"]).assign(n_episodes=1)
    maps, _ = _rho_aggregated_state_local_maps(phase, occupancy)
    assert maps.iloc[0]["estimate"] == 0.25
    assert maps.iloc[0]["aggregation_weight"] == "n_observations"
