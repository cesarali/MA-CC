from pathlib import Path

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.core import Seed
from mas_cc.experiments.orchestrator import _grid_cell_seed
from mas_cc.studies.execution import build_cell_execution_entries, plan_cell_execution
from mas_cc.studies.initialization import build_initialization_plan
from mas_cc.studies.manifest import discover_study
from mas_cc.studies.submission import build_submission_entries


ROOT = Path("configs/runs/smoke/blackboard_deepinfra_three_arm")
INITIALIZATIONS = Path(
    "/scratch/df630/MA-CC-results/inputs/"
    "blackboard_deepinfra_three_arm_smoke_v1_initializations"
)


def _grid(name: str) -> GridSpec:
    source = load_run_config_or_grid(ROOT / name)
    assert isinstance(source, GridSpec)
    return source


def test_three_arm_smoke_is_small_provider_safe_and_paired(tmp_path):
    spec = discover_study(ROOT)
    entries = build_submission_entries(spec, tmp_path / "results", git_commit="test")
    shards = build_cell_execution_entries(spec, entries)
    plan = plan_cell_execution(spec, len(shards))
    initializations = build_initialization_plan(spec.configs, INITIALIZATIONS)

    assert [path.name for path in spec.configs] == [
        "no_control.yaml",
        "truth_control.yaml",
        "false_control.yaml",
    ]
    assert len(entries) == len(shards) == 3
    assert all(entry.expected_cell_count == 1 for entry in entries)
    assert all(entry.expected_episode_count == 1 for entry in entries)
    assert len(initializations) == 1
    assert Path(initializations[0].artifact_path).parent == INITIALIZATIONS
    assert plan.array_throttle == 1
    assert plan.cpus_per_task == 1
    assert plan.episode_slots_per_shard == 1
    assert plan.request_concurrency_per_shard == 2
    assert plan.total_request_concurrency == 2
    assert plan.time_limit == "00:10:00"
    for name in ("no_control.yaml", "truth_control.yaml", "false_control.yaml"):
        source = _grid(name)
        common_random_numbers = bool(
            source.base.experiment.metadata.get("common_random_numbers_across_grid")
        )
        runtime_seed = int(
            _grid_cell_seed(
                Seed(source.base.execution.seed),
                source.cells[0].index,
                common_random_numbers=common_random_numbers,
            ).derive("episode:0")
        )
        assert common_random_numbers is True
        assert runtime_seed == initializations[0].episode_seed


def test_controlled_smokes_guarantee_both_dawn_directive_paths():
    no_control = _grid("no_control.yaml").base
    truth = _grid("truth_control.yaml").base
    false = _grid("false_control.yaml").base

    assert no_control.control.mechanism == "none"
    assert truth.control.options["target"] == "correct"
    assert false.control.options["target"] == "ALLOCATION_1"
    for config in (no_control, truth, false):
        initialization = config.game.options["initialization"]
        assert initialization["mode"] == "paired_local_vote"
        assert Path(initialization["artifact_dir"]) == INITIALIZATIONS
        assert initialization["require_artifact"] is True
        assert config.game.options["rounds"] == 1
        assert config.execution.repetitions == 1
        assert config.execution.parallelism == 1
        assert config.logging.comet is False
        assert config.experiment.metadata["common_random_numbers_across_grid"] is True
        assert config.experiment.metadata["paired_initialization_across_grid"] is True
        assert config.experiment.metadata["paired_initialization_across_targets"] is True
    for config in (truth, false):
        assert config.control.options["advocacy_schedule"] == "always"
        assert config.control.options["controller_timing"] == "dawn_only"
        assert config.control.options["controller_actuation_mode"] == "coordination_request"
        assert config.control.options["intervention_budget"] == 3
