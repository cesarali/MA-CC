from pathlib import Path

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.control import create_control
from mas_cc.games.relational_reasoning.data import load_relational_task
from mas_cc.studies.manifest import discover_study
from mas_cc.studies.preflight import validate_study_preflight_contract
from mas_cc.studies.submission import build_submission_entries


DATASET = Path(
    "src/mas_cc/relational_task_generator/relational_task_generator/datasets/"
    "n12_L2_k3"
)
RHO = [0.7, 0.75, 0.8, 0.85, 0.9, 1.0]


def _source(study: str, filename: str) -> GridSpec:
    loaded = load_run_config_or_grid(
        Path("configs/runs/relational_reasoning") / study / filename
    )
    assert isinstance(loaded, GridSpec)
    return loaded


def test_q1_l2_reference_contracts_are_exact_and_separate():
    for study, semantics in (("population_study_09f", "false only"), ("population_study_09g", "truth only")):
        root = Path("configs/runs/relational_reasoning") / study
        spec = discover_study(root)
        report = validate_study_preflight_contract(spec)
        entries = build_submission_entries(spec, f"/tmp/{study}", git_commit="test")
        assert report["status"] == "permitted"
        assert report["q_values"] == [1]
        assert report["L_values"] == [2]
        assert report["support_redundancy"] == [4]
        assert report["rho_values"] == RHO
        assert report["b_values"] == [3, 6, 9, 12]
        assert report["repetitions"] == [10]
        assert report["target_semantics"] == [semantics]
        assert report["total_cells"] == 24
        assert report["total_episodes"] == 240
        assert entries[0].expected_cell_count == 24
        assert entries[0].expected_episode_count == 240


def test_q1_l2_reference_uses_matching_task_and_semantic_targets():
    task = load_relational_task(DATASET, "task_0002", population_size=12)
    assert task.correct_relation == "NORTHEAST"
    assert len(task.supporting_fact_ids) == 2
    false = _source(
        "population_study_09f", "study09f_task0002_q1_l2_false_persistence.yaml"
    )
    truth = _source(
        "population_study_09g", "study09g_task0002_q1_l2_truth_persistence.yaml"
    )
    assert [(axis.path, list(axis.values)) for axis in false.axes] == [
        ("game.options.epistemic_persistence", RHO),
        ("control.options.intervention_budget", [3, 6, 9, 12]),
    ]
    assert [(axis.path, list(axis.values)) for axis in truth.axes] == [
        (axis.path, list(axis.values)) for axis in false.axes
    ]
    for source, target in ((false, "NORTH"), (truth, "NORTHEAST")):
        for cell in source.cells:
            config = cell.config
            assert config.game.options["social_group_size"] == 1
            assert config.control.options["sensor_sample_size"] == 6
            assert config.execution.repetitions == 10
            assert create_control(config.control).resolved_target_for_task(
                task, config.execution.seed
            ) == target


def test_q1_l2_truth_and_false_configs_are_matched_except_target_identity():
    false = _source(
        "population_study_09f", "study09f_task0002_q1_l2_false_persistence.yaml"
    ).base
    truth = _source(
        "population_study_09g", "study09g_task0002_q1_l2_truth_persistence.yaml"
    ).base
    assert false.llm_provider == truth.llm_provider
    assert false.prompt == truth.prompt
    assert false.game == truth.game
    assert false.execution == truth.execution
    assert false.storage == truth.storage
    assert false.control.options["target"] == "NORTH"
    assert truth.control.options["target"] == "NORTHEAST"
