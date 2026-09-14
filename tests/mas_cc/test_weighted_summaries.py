import numpy as np
import pandas as pd
import pytest

from mas_cc.studies.weighted_summaries import (
    EPISTEMIC_MEANS, PairedBootstrap, derive_blackboard_summaries,
)


def recipe():
    return {"derived_study_aggregates": {
        "enabled": True, "causal": True, "epistemic": True,
        "groupings": [{"name": "rho", "group_by": ["intervention_budget", "target_semantics"],
                       "marginalize": ["epistemic_persistence"], "weighting": "balanced_cell"}],
        "state_local": {"enabled": True, "x_bins": 8, "weighting": "n_observations",
                        "groupings": [{"name": "rho-x", "group_by": ["intervention_budget", "target_semantics"],
                                       "marginalize": ["epistemic_persistence"]}]},
    }}


def fixture():
    cells = pd.DataFrame([
        {"cell_id": "a", "epistemic_persistence": .7, "intervention_budget": 3, "target_semantics": "false"},
        {"cell_id": "b", "epistemic_persistence": 1., "intervention_budget": 3, "target_semantics": "false"},
    ])
    records = []
    for cell, count, score, x in [("a", 2, .2, .25), ("b", 4, .6, .3)]:
        for ep in range(count):
            records.append({"cell_id": cell, "episode_id": f"{cell}-{ep}", "round_index": 0,
                            "physical_initial_state_hash": f"paired-{ep}",
                            "episode_complete": True, "U_t": ep % 2, "x_t": x,
                            **{f"causal_response_h{h}": score for h in (1, 2, 3)},
                            **{v: score for v in EPISTEMIC_MEANS.values()}})
    frame = pd.DataFrame(records)
    return cells, frame


def run(cells, frame, *, resamples=40):
    plan = PairedBootstrap(frame, resamples=resamples, seed=42)
    tables = {"causal_response_round_inputs": frame, "epistemic_round_timeseries": frame}
    return derive_blackboard_summaries(tables, cells, recipe(),
                                      {"confidence": .95}, "hash", plan)


def test_balanced_cells_ratios_and_local_observation_weighting():
    cells, frame = fixture()
    outputs = run(cells, frame)
    causal = outputs["causal_response_aggregated_metrics"]
    assert causal.query("metric == 'propensity_weighted_causal_response'")["estimate"].tolist() == pytest.approx([.4]*3)
    available = causal.query("metric == 'available_mass_weighted_causal_susceptibility'").iloc[0]
    assert available.estimate == pytest.approx((.2+.6)/(.75+.7))
    assert available.estimate != pytest.approx((.2/.75+.6/.7)/2)
    local = outputs["causal_state_local_aggregated_metrics"].query("target_fraction_bin_index == 2 and metric == 'propensity_weighted_available_susceptibility'").iloc[0]
    assert local.estimate == pytest.approx((2*.2/.75+4*.6/.7)/6)
    missing = outputs["causal_state_local_aggregated_metrics"].query("target_fraction_bin_index == 0").iloc[0]
    assert np.isnan(missing.estimate)
    assert missing.support_status == "unsupported"


def test_epistemic_equal_episode_means_and_missing_is_not_zero():
    cells, frame = fixture()
    # One a episode has nine more rounds; it must retain one episode's weight.
    more = pd.concat([frame.iloc[[0]]]*9, ignore_index=True)
    more["collective_solvable"] = 1.
    frame["collective_solvable"] = 0.
    frame = pd.concat([frame, more], ignore_index=True)
    frame.loc[frame.cell_id == "b", "reference_robustness"] = np.nan
    out = run(cells, frame)["epistemic_aggregated_metrics"].set_index("metric")
    assert out.loc["fraction_rounds_collectively_solvable", "estimate"] == pytest.approx((.9/2+0)/2)
    assert out.loc["mean_reference_robustness", "cell_coverage_fraction"] == .5
    assert out.loc["mean_reference_robustness", "support_status"] == "limited"


def test_shared_block_draws_preserve_pairing_and_cell_counts_with_missing_pairs():
    _, frame = fixture()
    plan = PairedBootstrap(frame, resamples=100, seed=9)
    for cell, group in frame.groupby("cell_id"):
        assert np.all(plan.totals(group, np.ones(len(group))) == len(group))
    a = frame[frame.cell_id == "a"]
    b = frame[(frame.cell_id == "b") & frame.episode_id.isin(["b-0", "b-1"])]
    np.testing.assert_equal(plan.totals(a, np.array([1., -1.])), plan.totals(b, np.array([1., -1.])))
    shuffled = PairedBootstrap(frame.sample(frac=1, random_state=7), resamples=100, seed=9)
    np.testing.assert_equal(plan.weights, shuffled.weights)


def test_paired_anticorrelation_cancels_in_aggregate_interval():
    cells, frame = fixture()
    frame = frame[~frame.episode_id.isin(["b-2", "b-3"])].copy()
    frame["causal_response_h1"] = [.1, .9, .9, .1]
    out = run(cells, frame)["causal_response_aggregated_metrics"]
    row = out.query("metric == 'propensity_weighted_causal_response' and lag == 1").iloc[0]
    assert row.estimate == pytest.approx(.5)
    assert row.ci_low == pytest.approx(.5)
    assert row.ci_high == pytest.approx(.5)


def test_incomplete_episodes_and_saturation_excluded():
    cells, frame = fixture()
    frame.loc[frame.episode_id == "b-3", "episode_complete"] = False
    frame.loc[frame.episode_id == "b-3", "causal_response_h1"] = 100
    frame.loc[frame.cell_id == "a", "x_t"] = 1.
    causal = run(cells, frame)["causal_response_aggregated_metrics"]
    assert causal.query("metric == 'propensity_weighted_causal_response' and lag == 1").estimate.item() == pytest.approx(.4)
    available = causal.query("metric == 'available_mass_weighted_causal_susceptibility'").iloc[0]
    assert available.estimate == pytest.approx(.6/.7)
    assert available.n_contributing_cells == 1
    assert available.support_status == "limited"


def test_unknown_coordinates_and_unrequested_inputs_fail_clearly():
    cells, frame = fixture()
    cells["model"] = ["one", "two"]
    with pytest.raises(ValueError, match="silently mix"):
        run(cells, frame)
    cells = cells.drop(columns="model")
    with pytest.raises(ValueError, match="causal_response_round_inputs"):
        derive_blackboard_summaries({}, cells, recipe(), {"confidence": .95}, "hash",
                                   PairedBootstrap(frame, resamples=2, seed=9))


def test_repetition_numbers_do_not_create_false_pairing():
    _, frame = fixture()
    frame = frame.drop(columns="physical_initial_state_hash")
    frame["repetition_index"] = 0
    plan = PairedBootstrap(frame, resamples=2, seed=9)
    assert len(plan.blocks) == len(frame)


def test_singleton_completion_strata_are_flagged():
    cells, frame = fixture()
    frame = frame[frame.episode_id != "b-3"]
    out = run(cells, frame)["epistemic_aggregated_metrics"]
    assert set(out.singleton_bootstrap_strata) == {1}
    assert set(out.support_status) == {"limited"}


@pytest.mark.parametrize("legacy_label", ["false", False, "False"])
def test_report_alias_uses_new_ratio_and_preserves_empty_bins(legacy_label, tmp_path):
    from mas_cc.studies.weighted_summaries import weighted_rho_aliases

    outputs = {
        "state_local_aggregated_metrics": pd.DataFrame([{
            "intervention_budget": 3, "target_semantics": "false",
            "target_fraction_bin_index": 2, "metric": "eta_ir", "estimate": .2,
            "support_status": "adequate", "marginalized_dimensions": '["epistemic_persistence"]',
            "component_numerator": .1, "component_denominator": .5}]),
        "rho_aggregated_state_local_maps": pd.DataFrame([
            {"intervention_budget": 3, "target_semantics": legacy_label, "target_fraction_bin_index": i,
             "metric": "eta_IR", "estimate": .9, "phase_status": "adequate"} for i in (0, 2)])}
    weighted_rho_aliases(outputs)
    result = outputs["rho_aggregated_state_local_maps"].set_index("target_fraction_bin_index")
    assert result.loc[2, "estimate"] == .2
    assert np.isnan(result.loc[0, "estimate"])
    assert len(result) == 2
    assert set(result.target_semantics) == {"false"}
    from mas_cc.studies.table_io import write_scientific_table
    path = write_scientific_table(tmp_path, "rho_alias", result.reset_index())
    pd.testing.assert_frame_equal(pd.read_parquet(path), result.reset_index())


def test_full_offline_aggregation_exports_weighted_tables_and_zip(tmp_path):
    import json
    import zipfile
    import yaml
    from test_epistemic_phase import test_offline_aggregation_packages_epistemic_outputs
    from mas_cc.studies import aggregate_study
    from mas_cc.studies.table_io import write_scientific_table

    test_offline_aggregation_packages_epistemic_outputs(tmp_path)
    study = tmp_path / "offline-epistemic"
    tables = study / "analysis" / "tables"
    config_path = study / "analysis.yaml"
    config = yaml.safe_load(config_path.read_text())
    config.update(recipe())
    config["derived_study_aggregates"]["bootstrap"] = {"unit": "shared_initialization_block"}
    config["blackboard_phase2_outputs"] = True
    config["rho_aggregated_descriptive"] = True
    config["resampling"]["bootstrap_resamples"] = 4
    from pathlib import Path
    anchor = Path(__file__).resolve().parents[2] / "configs/runs/relational_reasoning/blackboard_game/astra_task003_false_control_30x30_potsdam_rho3/analysis.yaml"
    config["plots"] = {k: v for k, v in yaml.safe_load(anchor.read_text())["plots"].items()
                       if "weighted_by_budget" in k}
    config_path.write_text(yaml.safe_dump(config))
    cells = pd.read_parquet(tables / "cells.parquet")
    cells["target_semantics"] = False
    write_scientific_table(tables, "cells", cells)
    result = aggregate_study(study)
    expected = {"causal_response_aggregated_metrics", "causal_state_local_aggregated_metrics",
                "epistemic_aggregated_metrics", "study_aggregated_metrics", "state_local_aggregated_metrics"}
    with zipfile.ZipFile(result["archive"]) as archive:
        assert {f"tables/{name}.parquet" for name in expected}.issubset(archive.namelist())
        assert any("causal_response_weighted_by_budget" in name for name in archive.namelist())
        assert any("symbolic_solvability_weighted_by_budget" in name for name in archive.namelist())
    ep = pd.read_parquet(tables / "epistemic_aggregated_metrics.parquet")
    assert set(ep.metric) == set(EPISTEMIC_MEANS)
    assert set(ep.bootstrap_unit) == {"shared_initialization_block"}
    validation = json.loads((study / "analysis" / "validation.json").read_text())
    assert validation["causal_response"]["provider_calls"] == 0


def test_four_task003_recipes_enable_weighted_summaries():
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[2] / "configs/runs/relational_reasoning/blackboard_game"
    configs = [yaml.safe_load((root / f"astra_task003_{arm}_control_{provider}" / "analysis.yaml").read_text())
               for arm in ("truth", "false") for provider in ("30x30_potsdam_rho3", "q12_deepinfra_rho3")]
    for config in configs:
        assert config == configs[0]
        derived = config["derived_study_aggregates"]
        assert derived["causal"] and derived["epistemic"]
        assert derived["bootstrap"]["unit"] == "shared_initialization_block"
        assert "eta_ir" in derived["metrics"]
        assert "round_target_information_fraction" in derived["metrics"]
