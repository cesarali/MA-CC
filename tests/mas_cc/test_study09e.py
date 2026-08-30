from pathlib import Path

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.control import create_control
from mas_cc.games.relational_reasoning.data import load_relational_task
from mas_cc.studies.manifest import discover_study
from mas_cc.studies.preflight import validate_study_preflight_contract
from mas_cc.studies.submission import build_submission_entries


ROOT = Path("configs/runs/relational_reasoning/population_study_09e")
DATASET = Path(
    "src/mas_cc/relational_task_generator/relational_task_generator/datasets/"
    "n12_L3_r03_k3"
)


def test_study09e_is_exact_truth_aligned_match_of_study09d():
    spec = discover_study(ROOT)
    report = validate_study_preflight_contract(spec)
    entries = build_submission_entries(spec, "/tmp/test-study09e", git_commit="test")

    assert report["status"] == "permitted"
    assert report["target_semantics"] == ["truth only"]
    assert report["rho_values"] == [0.7, 0.75, 0.8, 0.85, 0.9]
    assert report["b_values"] == [3, 6, 9, 12]
    assert report["repetitions"] == [10]
    assert report["total_cells"] == 20
    assert report["total_episodes"] == 200
    assert entries[0].expected_cell_count == 20
    assert entries[0].expected_episode_count == 200

    source = load_run_config_or_grid(
        ROOT / "study09e_task0002_truth_persistence_refinement.yaml"
    )
    assert isinstance(source, GridSpec)
    task = load_relational_task(DATASET, "task_0002", population_size=12)
    for cell in source.cells:
        config = cell.config
        control = create_control(config.control)
        assert config.execution.seed == 20260830
        assert config.execution.repetitions == 10
        assert control.resolved_target_for_task(task, config.execution.seed) == "NORTH"
        assert config.experiment.metadata["controller_target_is_truth"] is True


def test_study09d_and_09e_differ_only_in_truth_alignment_and_identity():
    false = load_run_config_or_grid(
        "configs/runs/relational_reasoning/population_study_09d/"
        "study09d_task0002_persistence_refinement.yaml"
    )
    truth = load_run_config_or_grid(
        ROOT / "study09e_task0002_truth_persistence_refinement.yaml"
    )
    assert isinstance(false, GridSpec)
    assert isinstance(truth, GridSpec)
    assert [(axis.path, list(axis.values)) for axis in false.axes] == [
        (axis.path, list(axis.values)) for axis in truth.axes
    ]
    false_base = false.base
    truth_base = truth.base
    assert false_base.execution == truth_base.execution
    assert false_base.game == truth_base.game
    assert false_base.prompt == truth_base.prompt
    assert false_base.llm_provider == truth_base.llm_provider
    assert false_base.storage == truth_base.storage
    assert false_base.control.options["target"] == "NORTHWEST"
    assert truth_base.control.options["target"] == "NORTH"
