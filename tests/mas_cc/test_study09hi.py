from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.control import create_control
from mas_cc.games import create_game
from mas_cc.games.relational_reasoning.data import load_relational_task
from mas_cc.games.relational_reasoning.imitation_round_feedback.runtime import (
    build_social_sources,
)
from mas_cc.studies.manifest import discover_study
from mas_cc.studies.execution import build_cell_execution_entries, plan_cell_execution
from mas_cc.studies.aggregation import (
    _factorial_contrasts,
    _render_plots,
    _rho_aggregated_descriptive_summary,
    _state_local_phase_tables,
)
from mas_cc.studies.preflight import (
    run_study_preflight,
    validate_study_preflight_contract,
)
from mas_cc.studies.submission import build_submission_entries


ROOT = Path("configs/runs/relational_reasoning")
DATASET = Path(
    "src/mas_cc/relational_task_generator/relational_task_generator/datasets/"
    "n12_L3_r03_k3"
)
RHO = [0.8, 0.85]
BUDGETS = [3, 4, 6, 8, 9, 12]


def _source(study: str, filename: str) -> GridSpec:
    loaded = load_run_config_or_grid(ROOT / study / filename)
    assert isinstance(loaded, GridSpec)
    return loaded


def test_high_statistics_persistence_contracts_are_exact_and_separate():
    for study, semantics in (
        ("population_study_09h", "false only"),
        ("population_study_09i", "truth only"),
    ):
        spec = discover_study(ROOT / study)
        report = validate_study_preflight_contract(spec)
        entries = build_submission_entries(spec, f"/tmp/{study}", git_commit="test")
        assert report["status"] == "permitted"
        assert report["population_size"] == [12]
        assert report["rounds"] == [30]
        assert report["q_values"] == [1, 2]
        assert report["L_values"] == [3]
        assert report["sensor_size"] == [6]
        assert report["rho_values"] == RHO
        assert report["b_values"] == BUDGETS
        assert report["evidence_strategies"] == ["strategic"]
        assert report["receiver_dispositions"] == ["naive"]
        assert report["repetitions"] == [15]
        assert report["target_semantics"] == [semantics]
        assert report["total_cells"] == 24
        assert report["total_episodes"] == 360
        assert entries[0].expected_cell_count == 24
        assert entries[0].expected_episode_count == 360


def test_high_statistics_grid_has_no_additional_scientific_axes():
    false = _source(
        "population_study_09h",
        "study09h_task0002_false_high_statistics_persistence.yaml",
    )
    assert [(axis.path, list(axis.values)) for axis in false.axes] == [
        ("game.options.social_group_size", [1, 2]),
        ("game.options.epistemic_persistence", RHO),
        ("control.options.intervention_budget", BUDGETS),
    ]
    for cell in false.cells:
        config = cell.config
        assert config.game.options["receiver_epistemic_disposition"] == "naive"
        assert config.control.options["message_mode"] == "recommendation_plus_fact"
        assert config.control.options["advocacy_schedule"] == "soft"
        assert config.control.options["beta"] == 4.0
        assert config.control.options["threshold"] == 0.75
        assert config.control.options["controller_evidence_strategy"] == "strategic"
        assert config.execution.repetitions == 15


def test_study09h_deepinfra_deepseek_preserves_design_and_caps_provider_load():
    source_root = ROOT / "population_study_09h"
    variant_root = ROOT / "population_study_09h_deepinfra_deepseek"
    source = _source(
        "population_study_09h",
        "study09h_task0002_false_high_statistics_persistence.yaml",
    )
    variant = _source(
        "population_study_09h_deepinfra_deepseek",
        "study09h_task0002_false_high_statistics_persistence_deepinfra_deepseek.yaml",
    )
    spec = discover_study(variant_root)
    report = validate_study_preflight_contract(spec)
    entries = build_submission_entries(spec, "/tmp/study09h-deepinfra", git_commit="test")
    shards = build_cell_execution_entries(spec, entries)
    plan = plan_cell_execution(spec, len(shards))

    assert report["status"] == "permitted"
    assert report["total_cells"] == 24
    assert report["total_episodes"] == 360
    assert source.axes == variant.axes
    for field in (
        "prompt",
        "game",
        "control",
        "execution",
        "logging",
        "storage",
        "analysis",
        "metrics",
        "aggregation",
        "observability",
    ):
        assert getattr(source.base, field) == getattr(variant.base, field)
    assert variant.base.llm_provider.type == "deepinfra"
    assert variant.base.llm_provider.model == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert variant.base.llm_provider.credentials_env == "DEEPINFRA_API_KEY"
    assert variant.base.llm_provider.request_concurrency == 10
    assert variant.base.llm_provider.max_output_tokens == 4096
    assert variant.base.pricing.mode == "offline"
    assert variant.base.budget.accounting_unit == "USD"
    assert len(shards) == 24
    assert plan.array_throttle == 24
    assert plan.total_episode_slots == 240
    assert plan.total_request_concurrency == 240
    assert plan.estimated_rpm == 1200
    assert plan.assumed_latency_seconds == 12.0
    assert plan.provider_load_control["initial_concurrency"] == 128
    assert plan.provider_load_control["maximum_concurrency"] == 200
    assert plan.provider_load_control["target_rpm"] == 1200
    assert plan.cpus_per_task == 10
    assert plan.time_limit == "04:00:00"
    assert (source_root / "analysis.yaml").read_bytes() == (
        variant_root / "analysis.yaml"
    ).read_bytes()
    source_study = (variant_root / "study.yaml").read_text()
    assert "partition: all" in source_study
    assert "qos: normal" in source_study
    assert "/pscratch/" not in source_study


def test_high_statistics_study_preflight_renders_persistence_fact_audit(tmp_path):
    result = run_study_preflight(
        ROOT / "population_study_09h_deepinfra_deepseek", tmp_path
    )

    report = (result.output_dir / "report.md").read_text()
    assert result.design["status"] == "permitted"
    assert "true strategic fact `" in report
    assert "fingerprint `" in report


def test_truth_and_false_families_match_except_target_identity_and_provenance():
    false = _source(
        "population_study_09h",
        "study09h_task0002_false_high_statistics_persistence.yaml",
    )
    truth = _source(
        "population_study_09i",
        "study09i_task0002_truth_high_statistics_persistence.yaml",
    )
    assert [(axis.path, axis.values) for axis in false.axes] == [
        (axis.path, axis.values) for axis in truth.axes
    ]
    assert false.base.llm_provider == truth.base.llm_provider
    assert false.base.prompt == truth.base.prompt
    assert false.base.game == truth.base.game
    assert false.base.execution == truth.base.execution
    assert false.base.storage == truth.base.storage
    assert false.base.control.options["target"] == "NORTHWEST"
    assert truth.base.control.options["target"] == "NORTH"


def test_strategic_evidence_policy_discloses_only_a_real_task_fact():
    source = _source(
        "population_study_09h",
        "study09h_task0002_false_high_statistics_persistence.yaml",
    )
    task = load_relational_task(DATASET, "task_0002", population_size=12)
    fact_ids = set(task.facts)
    fact_id = create_control(source.cells[0].config.control).resolve_fact_id(
        task, source.cells[0].config.execution.seed
    )
    assert fact_id in fact_ids


def test_q1_and_q2_use_one_production_controller_slot():
    source = _source(
        "population_study_09h",
        "study09h_task0002_false_high_statistics_persistence.yaml",
    )
    for q in (1, 2):
        cell = next(
            cell
            for cell in source.cells
            if cell.overrides["game.options.social_group_size"] == q
        )
        game = create_game(cell.config.game)
        state = game.initialize(cell.config.game, cell.config.execution.seed)
        peers = tuple(agent.agent_id for agent in state.agents[1 : q + 1])
        sources = build_social_sources(
            state,
            peers,
            replaced_peer_slot=0,
            controller_target="NORTHWEST",
            population_size=12,
            controller_fact_id=None,
        )
        assert len(sources) == q
        assert sum(item["source_type"] == "control" for item in sources) == 1
        assert sum(item["source_type"] == "ordinary" for item in sources) == q - 1


def test_analysis_is_csv_empirical_binned_and_support_explicit():
    for study in ("population_study_09h", "population_study_09i"):
        recipe = yaml.safe_load((ROOT / study / "analysis.yaml").read_text())
        assert recipe["theoretical_reference"] == "none"
        assert recipe["state_local"] == ["x"]
        assert recipe["state_local_x_bins"] == 8
        assert recipe["rho_aggregated_descriptive"] is True
        assert "eta_th" in recipe["derived"]
        assert "factorial_contrasts" in recipe["derived"]
        assert "eta_th_state_local" not in recipe["derived"]
        state_plots = [
            value
            for value in recipe["plots"].values()
            if value.get("source") == "state_local_phase_maps"
        ]
        assert state_plots
        assert all(value["status_column"] == "phase_status" for value in state_plots)
        assert all(value["color_scale"] == "independent" for value in state_plots)
        occupancy = [
            value
            for value in recipe["plots"].values()
            if value.get("source") == "state_occupancy_binned"
        ]
        assert len(occupancy) == 2
        comparison_plots = {
            name
            for name in recipe["plots"]
            if name.startswith(("q_effect_", "evidence_effect_"))
        }
        assert len(comparison_plots) == 4
        assert all(name.startswith("q_effect_") for name in comparison_plots)
        assert all(
            plot.get("filters", {}).get("controller_evidence_strategy") != "neutral"
            for plot in recipe["plots"].values()
        )


def test_direct_factorial_comparisons_are_matched_within_budget_and_rho():
    rows = []
    for q in (1, 2):
        for rho in RHO:
            for budget in (3, 4):
                rows.append(
                    {
                        "study_id": "study09h",
                        "cell_id": f"{q}-strategic-{rho}-{budget}",
                        "source_run_id": "run",
                        "task_id": "task_0002",
                        "receiver_epistemic_disposition": "naive",
                        "controller_evidence_strategy": "strategic",
                        "target_semantics": "false",
                        "social_group_size": q,
                        "epistemic_persistence": rho,
                        "intervention_budget": budget,
                        "metric": "round_target_actuation_cmi",
                        "estimate": q + rho + budget,
                    }
                )
    contrasts = _factorial_contrasts(
        pd.DataFrame(rows), pd.DataFrame(), ["factorial_contrasts"], "hash"
    )
    q_effect = contrasts[
        contrasts["metric"] == "delta_q_round_target_actuation_cmi"
    ]
    assert len(q_effect) == 2 * 2
    assert set(q_effect["estimate"]) == {1.0}
    assert set(q_effect["epistemic_persistence"]) == set(RHO)


def test_rho_descriptive_summary_records_the_focused_values_explicitly():
    primary = pd.DataFrame(
        [
            {
                "cell_id": f"cell-{rho}",
                "metric": "round_target_actuation_cmi",
                "estimate": rho,
                "n_episodes": 15,
                "social_group_size": 1,
                "controller_evidence_strategy": "neutral",
                "target_semantics": "false",
                "intervention_budget": 3,
                "epistemic_persistence": rho,
            }
            for rho in RHO
        ]
    )
    summary = _rho_aggregated_descriptive_summary(
        primary, pd.DataFrame(), pd.DataFrame()
    )
    assert len(summary) == 1
    assert summary.iloc[0]["n_rho"] == 2
    assert summary.iloc[0]["rho_values_json"] == "[0.8, 0.85]"


def test_phase_table_keeps_structural_occupancy_and_support_absence_distinct():
    spec = discover_study(ROOT / "population_study_09h")
    entries = build_submission_entries(spec, "/tmp/study09h", git_commit="test")
    present_cell = "config-0000/cell-0000"
    cells = pd.DataFrame([{"cell_id": present_cell}])
    events = [
        SimpleNamespace(
            cell_id=present_cell,
            episode_id="episode-0",
            N_k=(1, 11),
            event={"N": 12, "target_count_before": 1},
        )
    ]
    grouping = {
        "cell_id": present_cell,
        "resolution": "x",
        "target_fraction_bin_index": 0,
    }
    primary = pd.DataFrame(
        [
            {
                **grouping,
                "metric": "round_target_susceptibility",
                "estimate": 0.1,
                "units": "target_fraction",
                "support_status": "adequate",
            },
            {
                **grouping,
                "metric": "round_target_actuation_cmi",
                "estimate": float("nan"),
                "units": "bits",
                "support_status": "unsupported",
            },
        ]
    )
    phase, occupancy = _state_local_phase_tables(
        entries, cells, primary, pd.DataFrame(), events, bins=8
    )

    present = phase[
        (phase["cell_id"] == present_cell) & (phase["target_fraction_bin_index"] == 0)
    ].set_index("metric")
    assert present.loc["chi", "phase_status"] == "adequate"
    assert present.loc["T_pi", "phase_status"] == "insufficient_estimator_support"
    assert present.loc["eta_IR", "phase_status"] == "insufficient_estimator_support"
    unvisited = phase[
        (phase["cell_id"] == present_cell) & (phase["target_fraction_bin_index"] == 1)
    ]
    assert set(unvisited["phase_status"]) == {"state_not_visited"}
    missing = phase[phase["cell_id"] == "config-0000/cell-0001"]
    assert set(missing["phase_status"]) == {"structural_cell_not_run"}
    assert {
        "visited",
        "state_not_visited",
        "structural_cell_not_run",
    } <= set(occupancy["phase_status"])


def test_phase_plot_renders_independent_panels_and_status_legend(tmp_path):
    rows = []
    for rho in (0.7, 0.8):
        for budget, status, estimate in (
            (3, "adequate", 0.1 + rho),
            (4, "state_not_visited", float("nan")),
            (6, "insufficient_estimator_support", float("nan")),
        ):
            rows.append(
                {
                    "metric": "chi",
                    "controller_evidence_strategy": "neutral",
                    "social_group_size": 1,
                    "epistemic_persistence": rho,
                    "intervention_budget": budget,
                    "target_fraction_bin_center": 0.25,
                    "estimate": estimate,
                    "phase_status": status,
                    "units": "target_fraction",
                }
            )
    recipe = {
        "plots": {
            "phase": {
                "source": "state_local_phase_maps",
                "metric": "chi",
                "x": "intervention_budget",
                "y": "target_fraction_bin_center",
                "facet": "epistemic_persistence",
                "kind": "heatmap",
                "color_scale": "independent",
                "status_column": "phase_status",
            }
        }
    }
    paths = _render_plots(
        recipe, {"state_local_phase_maps": pd.DataFrame(rows)}, tmp_path
    )
    assert paths == [str(tmp_path / "phase.png")]
    assert (tmp_path / "phase.png").stat().st_size > 0
