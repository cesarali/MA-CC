from pathlib import Path

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.studies.initialization import build_initialization_plan
from mas_cc.studies.manifest import discover_study


ROOT = Path(
    "configs/runs/relational_reasoning/blackboard_game/"
    "iclr_experiments/recomm_only_q12_chatoss"
)


def test_recommendation_only_q12_chatoss_study_is_matched(tmp_path):
    spec = discover_study(ROOT)
    assert spec.name == "recomm_only_q12_chatoss"
    assert [path.name for path in spec.configs] == [
        "no_control.yaml",
        "truth_control.yaml",
        "false_control.yaml",
    ]

    sources = {
        path.name: load_run_config_or_grid(path) for path in spec.configs
    }
    assert all(isinstance(source, GridSpec) for source in sources.values())
    assert {name: len(source.cells) for name, source in sources.items()} == {
        "no_control.yaml": 3,
        "truth_control.yaml": 9,
        "false_control.yaml": 9,
    }
    assert sum(
        cell.config.execution.repetitions
        for source in sources.values()
        for cell in source.cells
    ) == 1260

    reference = sources["truth_control.yaml"].base
    for source in sources.values():
        assert source.base.llm_provider.type == "deepinfra"
        assert source.base.llm_provider.model == "openai/gpt-oss-120b"
        assert source.base.llm_provider.options["reasoning_effort"] == "low"
        assert source.base.game.options["social_group_size"] == 12
        assert source.base.execution.repetitions == 60
        for key in ("game", "execution", "llm_provider", "prompt", "storage"):
            assert source.base.to_dict()[key] == reference.to_dict()[key]

    assert sources["no_control.yaml"].base.control.mechanism == "none"
    for name, target in (
        ("truth_control.yaml", "correct"),
        ("false_control.yaml", "ALLOCATION_2"),
    ):
        source = sources[name]
        assert source.base.control.options["message_mode"] == "recommendation_only"
        assert source.base.control.options["target"] == target
        assert sorted(
            {cell.config.control.options["intervention_budget"] for cell in source.cells}
        ) == [6, 12, 18]

    plan = build_initialization_plan(spec.configs, tmp_path)
    assert len(plan) == 60
    assert len({entry.episode_seed for entry in plan}) == 60
    assert spec.execution["provider_load_control"]["mode"] == "redis_adaptive"
