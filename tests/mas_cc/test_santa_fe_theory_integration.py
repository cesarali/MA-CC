"""Real v3 record adapter checks; theory package kernels have their own tests."""
from pathlib import Path
import csv
import json
import yaml

from santa_fe.config import load_config
from santa_fe.runner import run, save_trajectories
from santa_fe.theory_integration import export_cell, _branch_records, _micro_branch, integrate


def test_real_v3_adapter_initial_board_and_action_clock(tmp_path: Path):
    data = yaml.safe_load(Path("configs/santa_fe/v3_pilot.yaml").read_text())
    data["experiment"].update(episodes=2, processes=1)
    data["model"].update(N=8, F=4, F_plus=3, rounds=3)
    data["sweep"]["budget_fraction"] = [0]
    data["output"]["results_dir"] = str(tmp_path / "sim")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    config = load_config(path)
    rounds, micro = run(config)
    save_trajectories(config, rounds, micro)
    out = tmp_path / "adapted"
    resolved, selected = export_cell(config, rounds, 0, out, initial_blocks=2,
                                     theory_replicas=2)
    initials = json.loads((out / "initials.json").read_text())["snapshots"]
    with (out / "simulator.csv").open() as stream:
        adapted = list(csv.DictReader(stream))
    assert len(initials) == 2
    assert len(adapted) == 2 * 4
    assert resolved["F_plus"] == 3 and resolved["F_minus"] == 1
    assert resolved["q_c"] == 4 and resolved["b"] == 0
    for initial in initials:
        assert sum(initial["front_page_counts"]) == 8
        block = [r for r in adapted if r["initial_id"] == initial["snapshot_id"]]
        assert [int(r["round"]) for r in block] == [0, 1, 2, 3]
        assert block[0]["action_next"] == ""
        assert float(block[0]["x"]) == initial["x"]
        for row in block[1:]:
            assert sum(int(row[f"peer_{key}"]) for key in
                       ("plus_fact", "plus_empty", "minus_fact", "minus_empty")) == 8
            assert int(row["posts_next"]) == 0
    snaps, source = _branch_records(selected, resolved, out, snapshots_per_cell=1)
    summary = _micro_branch(source, snaps, resolved, config, pairs=3, seed=31, out=out)
    assert summary[0]["chi_target"] == 0
    with (out / "micro_factual_replay.csv").open() as stream:
        assert next(csv.DictReader(stream))["factual_replay_match"] == "True"


def test_tiny_real_data_theory_overlay(tmp_path: Path):
    data = yaml.safe_load(Path("configs/santa_fe/v3_pilot.yaml").read_text())
    data["experiment"].update(episodes=2, processes=1)
    data["model"].update(N=8, F=4, F_plus=3, rounds=2)
    data["sweep"]["budget_fraction"] = [.25]
    data["output"]["results_dir"] = str(tmp_path / "sim")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    config = load_config(path)
    rounds, micro = run(config)
    save_trajectories(config, rounds, micro)
    output = integrate(path, tmp_path / "integrated", cell_ids=[0], initial_blocks=2,
                       theory_replicas=2, substeps=2, branch_pairs=2,
                       branch_snapshots=1, seed=19)
    assert (output / "regime_pilot.pdf").is_file()
    assert (output / "pilot_report.md").is_file()
    variant = output / "cell_0000" / "sampling_clock_corrected_reduced"
    assert (variant / "comparison" / "overlays.pdf").is_file()
    assert (variant / "comparison" / "late_window_errors.csv").is_file()
    assert (variant / "branch" / "branch_comparison.pdf").is_file()
