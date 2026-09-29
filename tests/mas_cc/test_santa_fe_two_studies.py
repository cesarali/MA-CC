"""Coordinate and estimator-calibration contracts for the two staged v3 studies."""
from pathlib import Path
import pandas as pd
import yaml

from santa_fe.config import load_config, plan
from santa_fe.sample_size_cmi import prepare_reference, run_task, aggregate

ROOT = Path("configs/santa_fe/two_studies")


def _key(cell):
    p = cell.params
    return (cell.beta_regime, p.rho, p.budget, p.q,
            min(p.N, max(1, round(p.sensing_fraction * p.N))))


def test_staged_physical_cells_are_exact_and_nonoverlapping():
    pilot = load_config(ROOT / "study_A_pilot.yaml")
    core = load_config(ROOT / "study_A_core.yaml")
    interaction = load_config(ROOT / "study_A_interaction.yaml")
    full = load_config(ROOT / "study_A_full_manifest_only.yaml")
    calibration = load_config(ROOT / "study_B_trajectory_pilot.yaml")
    keys = {name: {_key(c) for c in cfg.cells} for name, cfg in
            (("pilot", pilot), ("core", core), ("interaction", interaction),
             ("full", full), ("calibration", calibration))}
    assert [len(keys[name]) for name in ("pilot", "core", "interaction", "full", "calibration")] == [24,108,456,2160,10]
    assert not keys["core"] & keys["interaction"]
    assert len(keys["core"] | keys["interaction"]) == 564
    assert keys["pilot"] <= keys["full"] and keys["calibration"] <= keys["full"]
    assert {key[3] for key in keys["full"]} == {1,3,6,12}
    assert {key[4] for key in keys["full"]} == {1,3,6,12,24}
    assert plan(calibration)["sample_size_reference_episodes"] == 1280
    manifest = pd.read_csv(ROOT / "shared_physical_manifest.csv")
    assert len(manifest) == 2160
    assert manifest.A_core.sum() == 108 and manifest.A_interaction.sum() == 456


def test_multistat_sample_size_reuses_one_reference_bank(tmp_path: Path):
    data = yaml.safe_load((ROOT / "study_B_trajectory_pilot.yaml").read_text())
    data["experiment"].update(episodes=5, processes=1)
    data["model"].update(N=8, F=4, F_plus=3, rounds=3)
    data["sweep"] = {"beta_pairs": [{"name": "balanced", "beta_evidence": 1.0,
                                      "beta_social": 1.3}],
                     "rho": [0.7], "budget_fraction": [0.25],
                     "q": [2], "sensing_fraction": [0.5]}
    data["output"]["results_dir"] = str(tmp_path / "sample_size")
    data["sample_size_study"].update(reference_episodes=5, episode_counts=[3,5],
                                     repetitions=2, n_bootstrap=2, n_permutations=3,
                                     statistics=["round_target_sensing_mi",
                                                 "round_target_actuation_cmi"])
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    config = load_config(path)
    prepare_reference(config, 0)
    run_task(config, 0)
    run_task(config, 1)
    aggregate(config)
    reference = config.results_dir / "sample_size_cmi" / "reference_cell_0.parquet"
    assert pd.read_parquet(reference).seed.nunique() == 5
    details = pd.read_parquet(config.results_dir / "sample_size_cmi" /
                              "sample_size_cmi_repetitions.parquet")
    assert len(details) == 2 * 2 * 2
    assert set(details.statistic) == {"round_target_sensing_mi", "round_target_actuation_cmi"}
