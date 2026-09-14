from pathlib import Path

import pytest
import yaml

from mas_cc.config import load_run_config_or_grid
from mas_cc.studies.initialization import build_initialization_plan
from mas_cc.studies.manifest import discover_study


ROOT = Path("configs/runs/relational_reasoning/blackboard_game")


@pytest.mark.parametrize("suffix", ["q12_deepinfra_rho3", "30x30_potsdam_rho3"])
def test_no_control_is_matched_and_has_no_budget_sweep(suffix, tmp_path):
    baseline = ROOT / f"astra_task003_no_control_{suffix}"
    reference = ROOT / f"astra_task003_false_control_{suffix}"
    spec = discover_study(baseline)
    source = load_run_config_or_grid(spec.configs[0])
    original = load_run_config_or_grid(reference / "false_control_llm.yaml")
    assert len(source.cells) == 3
    assert sum(cell.config.execution.repetitions for cell in source.cells) == 90
    assert [(axis.path, list(axis.values)) for axis in source.axes] == [
        ("game.options.epistemic_persistence", [0.70, 0.85, 1.00])
    ]
    assert source.base.control.mechanism == "none"
    assert source.base.control.options == {}
    for key in ("game", "execution", "llm_provider", "prompt", "storage"):
        assert source.base.to_dict()[key] == original.base.to_dict()[key]
    # Compatibility and episode seeds must match all three arms, not just labels.
    truth = ROOT / f"astra_task003_truth_control_{suffix}" / "truth_control_llm.yaml"
    plan = build_initialization_plan(
        [spec.configs[0], reference / "false_control_llm.yaml", truth], tmp_path
    )
    assert len(plan) == 30
    recipe = yaml.safe_load((baseline / "analysis.yaml").read_text())
    assert recipe == yaml.safe_load((reference / "analysis.yaml").read_text())
    assert recipe["blackboard_epistemic_phase_outputs"]["enabled"]
    assert recipe["derived_study_aggregates"]["epistemic"]
    assert recipe["resampling"]["bootstrap_resamples"] == 1000
    assert recipe["blackboard_calibration_outputs"]["model_predictions"]["enabled"] is False
