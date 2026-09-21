from __future__ import annotations

import math
import json

import pandas as pd
import pytest

from mas_cc.analysis.blackboard_calibration import (
    adapt_calibration_inputs, analyze_blackboard_calibration, calibration_settings,
    channel_parameters, mean_map, mixture_parameters, sampling_exposure_probability,
    susceptibility,
)


def fixture_tables(blocks=6):
    micro, rounds, episodes = [], [], []
    for episode in range(blocks * 2):
        ep = f"episode-{episode}"
        episodes.append(dict(cell_id="run/cell", episode_id=ep, status="completed"))
        for action in (0, 1):
            rounds.append(dict(study_id="study", source_run_id="run", cell_id="run/cell",
                episode_id=ep, round_index=action, social_mode="board", board_sampling="uniform",
                analysis_target="Z", N=4, U_k=action, P_U1_given_Y=0.5,
                controller_target_share_before=0.5, controller_target_share=0.75,
                physical_initial_state_hash=f"block-{episode//2}", intervention_budget=2,
                actual_controller_posts=2*action))
            # Directional opportunities include non-target changes and no changes.
            for slot in range(12):
                before = "Z" if slot % 2 else "A"
                exposure = int(action and slot % 3 == 0)
                after = "Z" if slot % 4 == episode % 4 else "B"
                micro.append(dict(cell_id="run/cell", episode_id=ep, round_index=action,
                    micro_slot_index=slot, focal_agent_id=str(slot % 4),
                    focal_opinion_before=before, focal_opinion_after=after,
                    sampled_message_ids=["c"] if exposure else ["p"],
                    sampled_controller_message_ids=["c"] if exposure else [],
                    eligible_peer_message_count=2, eligible_controller_message_count=2*action,
                    board_sample_size=1, focal_selection_rule="uniform_with_replacement"))
    return (pd.DataFrame(micro), pd.DataFrame(rounds), pd.DataFrame(episodes),
            pd.DataFrame([dict(cell_id="run/cell", population_rounds=2, population_size=4)]))


def test_known_counts_and_boundaries():
    p = channel_parameters(20/100, 5/50)
    assert p["gamma"] == pytest.approx(.3)
    assert p["p"] == pytest.approx(2/3)
    assert p["h"] == pytest.approx(math.log(2))
    assert channel_parameters(0, .1)["h"] == -math.inf
    assert channel_parameters(.1, 0)["h"] == math.inf
    zero = channel_parameters(0, 0)
    assert zero["gamma"] == 0 and math.isnan(zero["p"]) and math.isnan(zero["h"])
    assert channel_parameters(.8, .7)["model_status"] == "incompatible_gamma"
    assert channel_parameters(math.nan, .1)["support_status"] == "missing_opportunities"


def test_sampling_and_report_fixture():
    assert sampling_exposure_probability(0, 0, 0) == 0
    assert sampling_exposure_probability(3, 0, 3) == 0
    assert sampling_exposure_probability(2, 2, 3) == 1
    with pytest.raises(ValueError):
        sampling_exposure_probability(1, 1, 3)
    w = sampling_exposure_probability(18, 9, 3)
    c0 = channel_parameters(.35*.7, .35*.3)
    c1 = channel_parameters(.65*.9, .65*.1)
    cb = mixture_parameters(c0, c1, w)
    assert w == pytest.approx(.7210256410)
    assert cb["gamma"] == pytest.approx(.5663076923)
    assert cb["p"] == pytest.approx(.8655166169)
    assert math.isfinite(sampling_exposure_probability(10**9, 10**8, 10))


def test_mean_maps_and_zero_weight_support():
    c0, c1 = channel_parameters(.2, .1), channel_parameters(.1, .2)
    assert susceptibility(c0, c1, 24, 24, .3)["slope"] == 0
    zero = channel_parameters(0, 0)
    assert mean_map(0, 0, 24, 24, .7)["mean"] == .7
    assert mean_map(math.nan, math.nan, 24, 0, .7)["mean"] == .7
    assert math.isnan(susceptibility(c0, c1, 24, 24, 1)["available"])
    assert susceptibility(c0, c0, 24, 24, 1)["chi"] == 0
    absent = channel_parameters(math.nan, math.nan)
    assert mixture_parameters(c0, absent, 0)["p"] == c0["p"]
    assert mixture_parameters(zero, c1, .5)["p"] == c1["p"]
    assert susceptibility(zero, zero, 24, 24, .1)["root_status"] == "all_states"


def test_adapter_missingness_persistent_posts_and_completion():
    micro, rounds, episodes, cells = fixture_tables()
    micro.loc[0, "sampled_controller_message_ids"] = json.dumps(["old"])
    micro.loc[0, "sampled_message_ids"] = json.dumps(["old"])
    micro.loc[0, "eligible_controller_message_count"] = 1
    micro.loc[1, "sampled_controller_message_ids"] = None
    micro["controller_message_directly_exposed"] = False
    micro.loc[2, "controller_message_directly_exposed"] = True
    episodes.loc[1, "status"] = "interrupted"
    frame = adapt_calibration_inputs(micro, rounds, episodes, cells)
    assert len(frame) == len(micro)-24
    assert frame.iloc[0].U == 0 and frame.iloc[0].E == 1
    assert pd.isna(frame.iloc[1].E)
    assert pd.isna(frame.iloc[2].E)
    assert frame.iloc[2].sampling_status == "inapplicable_transient_recommendation"
    assert frame.M.eq(12).all()  # N=4 does not replace actual M.
    legacy = micro.drop(columns=["eligible_peer_message_count", "eligible_controller_message_count", "board_sample_size"])
    old = adapt_calibration_inputs(legacy, rounds, episodes, cells)
    assert old.E.notna().any() and old.sampling_probability.isna().all()
    with pytest.raises(ValueError, match="unique"):
        adapt_calibration_inputs(pd.concat([micro, micro.iloc[:1]]), rounds, episodes, cells)


def test_counts_decomposition_intervals_and_reproducibility():
    tables = fixture_tables()
    out = analyze_blackboard_calibration(*tables, bootstrap_resamples=30, seed=91)
    again = analyze_blackboard_calibration(*tables, bootstrap_resamples=30, seed=91)
    for name in out:
        pd.testing.assert_frame_equal(out[name], again[name])
    counts = out["blackboard_calibration_counts"]
    direct = counts.query("estimator_variant == 'direct_active'").iloc[0]
    conditional = counts.query("estimator_variant == 'mixture_start_vote_weighted'").iloc[0]
    assert direct.D_plus == 72 and direct.D_minus == 72
    assert direct.p_plus == conditional.p_plus
    assert direct.p_minus == conditional.p_minus
    assert direct.n_blocks == 6 and direct.n_episodes == 12
    estimates = out["blackboard_calibration_estimates"]
    assert estimates.bootstrap_valid.max() == 30
    assert estimates.estimate_id.is_unique
    assert not out["blackboard_model_predictions"].shape[0]


def test_starting_vote_exposure_mismatch_is_visible():
    micro, rounds, episodes, cells = fixture_tables()
    active = micro.round_index.eq(1)
    for index in micro[active].index:
        row = micro.loc[index]
        # Target starts exposed more often; both channels retain both starts.
        e = int(row.micro_slot_index in {0, 1, 3, 5, 7, 8})
        micro.at[index, "sampled_controller_message_ids"] = ["c"] if e else []
        micro.at[index, "sampled_message_ids"] = ["c"] if e else ["p"]
        micro.at[index, "focal_opinion_after"] = "Z" if e else "B"
    out = analyze_blackboard_calibration(micro, rounds, episodes, cells, bootstrap_resamples=0)
    diag = out["blackboard_calibration_diagnostics"].query("diagnostic_slice == 'calibration_group'").iloc[0]
    assert diag.starting_vote_exposure_difference != 0
    assert abs(diag.direct_minus_common_entry) > .1
    counts = out["blackboard_calibration_counts"]
    direct = counts.query("estimator_variant == 'direct_active'").iloc[0]
    conditional = counts.query("estimator_variant == 'mixture_start_vote_weighted'").iloc[0]
    assert conditional.p_plus == pytest.approx(direct.p_plus)
    assert conditional.p_minus == pytest.approx(direct.p_minus)


def test_held_out_blocks_and_validation():
    settings = dict(model_predictions=dict(enabled=True, evaluation_fraction=.34,
                    assume_homogeneous_channels=True))
    out = analyze_blackboard_calibration(*fixture_tables(10), settings=settings,
                                        bootstrap_resamples=12, seed=3)
    split = out["blackboard_calibration_splits"]
    assert split.groupby("block_id").split.nunique().max() == 1
    predictions = out["blackboard_model_predictions"]
    assert not predictions.empty
    assert set(predictions.block_id) == set(split.query("split == 'evaluation'").block_id)
    assert predictions.M.eq(12).all()
    assert predictions.analysis_hash.eq("").all()
    validation = out["blackboard_model_validation"]
    assert "held_out_round_weighted_summary" in set(validation.comparison_status)
    assert predictions.model_status.eq("homogeneous_reference_assumed").all()


def test_one_block_no_uncertainty_and_boundary_draws():
    out = analyze_blackboard_calibration(*fixture_tables(1), bootstrap_resamples=10)
    estimates = out["blackboard_calibration_estimates"]
    assert estimates.uncertainty_status.eq("insufficient_independent_units").all()
    from mas_cc.analysis.blackboard_calibration import _interval
    interval = _interval(math.inf, [0., math.inf]*10, 3, .95, 20, True)
    assert interval["ci_high"] == math.inf
    assert interval["bootstrap_valid"] == 20
    assert _interval(0., [0.]*20, 3, .95, 20, True)["uncertainty_status"] == "degenerate_boundary"


@pytest.mark.parametrize("settings", [True, {"made_up": 1}, {"action_stratification": False},
    {"exposure_definition": "controller_message_directly_exposed"},
    {"model_predictions": {"enabled": True}}, {"calibration_scope": "pooled"}])
def test_recipe_rejects_unsupported_designs(settings):
    with pytest.raises(ValueError):
        calibration_settings(settings)


def test_mock_study_retention_packaging_and_offline_reaggregation(tmp_path):
    """Exercise real recorder -> canonical tables -> publication -> retained input."""
    from pathlib import Path
    import shutil
    import zipfile
    import yaml
    from mas_cc.config import load_run_config
    from mas_cc.experiments import run_experiment_sync
    from mas_cc.studies.aggregation import aggregate_study
    from mas_cc.studies.manifest import discover_study
    from mas_cc.studies.submission import build_submission_entries, write_submission_manifest

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    raw = yaml.safe_load(Path("configs/runs/relational_reasoning/misselaneous/relational_blackboard_coordination_smoke.yaml").read_text())
    raw["storage"]["artifact_profile"] = "results_only"
    raw["storage"]["overwrite"] = False
    raw["game"]["options"]["rounds"] = 1
    raw["game"]["horizon"] = 1
    config_path = config_dir / "mock.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    recipe = config_dir / "analysis.yaml"
    recipe.write_text(yaml.safe_dump(dict(theoretical_reference="none",
        blackboard_calibration_outputs={"enabled": True},
        resampling=dict(bootstrap_resamples=5, null_permutations=0, seed=42))))
    spec = discover_study(config_dir)
    study_dir = tmp_path / "study"
    entries = build_submission_entries(spec, study_dir, git_commit="test")
    write_submission_manifest(study_dir / "submission_manifest.csv", entries)
    (study_dir / "study_manifest.json").write_text(json.dumps(dict(
        schema_version=1, study_id=spec.name, analysis_recipe=str(recipe),
        expected_config_count=1, expected_cell_count=1, expected_episode_count=1)))
    config = load_run_config(config_path, environment={})
    run_experiment_sync(config, entries[0].output_dir, resume=False, show_progress=False)
    result = aggregate_study(study_dir)
    assert result["complete"]
    analysis = study_dir / "analysis"
    inputs = pd.read_parquet(analysis / "tables/blackboard_calibration_inputs.parquet")
    assert len(inputs) == 24
    assert inputs.sampling_status.eq("available").all()
    assert inputs.M.eq(24).all()
    assert (inputs.B + inputs.C >= inputs.m).all()
    assert inputs.focal_selection_rule.eq("uniform_with_replacement").all()
    estimates = pd.read_parquet(analysis / "tables/blackboard_calibration_estimates.parquet")
    primary = pd.read_parquet(analysis / "tables/primary_estimates.parquet")
    assert set(estimates.estimate_id) <= set(primary.estimate_id.dropna())
    manifest = json.loads((analysis / "analysis_manifest.json").read_text())
    validation = json.loads((analysis / "validation.json").read_text())
    assert manifest["status"] == "complete"
    assert validation["blackboard_calibration"]["input_rows"] == 24
    assert manifest["blackboard_calibration_hash"]
    with zipfile.ZipFile(result["archive"]) as archive:
        assert "tables/blackboard_calibration_estimates.parquet" in archive.namelist()
    shutil.rmtree(Path(entries[0].output_dir))
    assert aggregate_study(study_dir)["complete"]
    recomputed = pd.read_parquet(analysis / "tables/blackboard_calibration_estimates.parquet")
    pd.testing.assert_frame_equal(estimates.drop(columns="analysis_hash"), recomputed.drop(columns="analysis_hash"))


def test_sampling_comparison_matches_rows_not_average_composition():
    from mas_cc.analysis.blackboard_calibration import _exposure_stats
    micro, rounds, episodes, cells = fixture_tables(1)
    adapted = adapt_calibration_inputs(micro, rounds, episodes, cells)
    rows = adapted.iloc[:3].to_dict("records")
    rows[0].update(B=1, C=1, m=1, E=1, sampling_probability=.5)
    rows[1].update(B=9, C=1, m=1, E=0, sampling_probability=.1)
    rows[2].update(E=1, sampling_probability=math.nan)
    result = _exposure_stats(rows)
    assert result["w"] == pytest.approx(2/3)
    assert result["sampling"] == pytest.approx(.3)
    assert result["sampling_n"] == 2
    assert result["sampling_observed"] == .5
    assert result["residual"] == pytest.approx(.2)
    assert result["sampling"] != sampling_exposure_probability(5, 1, 1)


def test_silent_exposure_suppresses_unexposed_baseline_predictions():
    micro, rounds, episodes, cells = fixture_tables(6)
    for index in micro[micro.round_index.eq(0)].index:
        micro.at[index, "sampled_controller_message_ids"] = ["old"]
        micro.at[index, "sampled_message_ids"] = ["old"]
        micro.at[index, "eligible_controller_message_count"] = 1
    out = analyze_blackboard_calibration(micro, rounds, episodes, cells,
        settings=dict(model_predictions=dict(enabled=True, evaluation_fraction=.3,
                      assume_homogeneous_channels=True)), bootstrap_resamples=5)
    predictions = out["blackboard_model_predictions"]
    assert predictions.model_status.eq("silent_exposure_incompatible").all()
    assert predictions.estimate.isna().all()
    counts = out["blackboard_calibration_counts"]
    assert counts.query("branch == '0' and channel == '1'").n_observations.iloc[0] > 0


def test_variable_round_updates_and_dependencies():
    micro, rounds, episodes, cells = fixture_tables(8)
    micro = micro[~(micro.episode_id.map(lambda s: int(s.split('-')[1]) % 2 == 0)
                     & micro.micro_slot_index.ge(8))]
    out = analyze_blackboard_calibration(micro, rounds, episodes, cells,
        settings=dict(model_predictions=dict(enabled=True, evaluation_fraction=.4,
                      assume_homogeneous_channels=True)), bootstrap_resamples=5)
    prediction = out["blackboard_model_predictions"]
    assert set(prediction.M) == {8, 12}
    estimates = out["blackboard_calibration_estimates"]
    gamma = estimates.query("metric == 'blackboard_effective_compliance' and estimator_variant == 'mixture_common_weight'").estimate.iloc[0]
    relaxation = prediction.query("metric == 'blackboard_round_relaxation' and branch == 'active'")
    for row in relaxation.itertuples():
        assert row.estimate == pytest.approx((1-gamma/row.N)**row.M)
    count_ids = set(out["blackboard_calibration_counts"].count_id)
    for deps in estimates.dependencies_json:
        assert set(json.loads(deps)) <= count_ids
    for deps in prediction.dependencies_json:
        assert set(json.loads(deps)) <= set(estimates.estimate_id)


def test_missing_transition_keeps_starting_vote_exposure_and_detects_missing_slots():
    micro, rounds, episodes, cells = fixture_tables(1)
    micro.loc[0, "focal_opinion_after"] = None
    frame = adapt_calibration_inputs(micro, rounds, episodes, cells)
    assert not frame.iloc[0].valid_update
    assert frame.iloc[0].z_before == 0 and frame.iloc[0].E == 0
    rounds["actual_update_count"] = 12
    with pytest.raises(ValueError, match="actual round update count"):
        adapt_calibration_inputs(micro.iloc[1:], rounds, episodes, cells)


def test_absent_causal_propensity_does_not_discard_calibration():
    micro, rounds, episodes, cells = fixture_tables(8)
    rounds = rounds.drop(columns="P_U1_given_Y")
    out = analyze_blackboard_calibration(micro, rounds, episodes, cells,
        settings=dict(model_predictions=dict(enabled=True, evaluation_fraction=.4,
                      assume_homogeneous_channels=True)), bootstrap_resamples=2)
    assert out["blackboard_calibration_estimates"].estimate.notna().any()
    assert out["blackboard_model_predictions"].estimate.notna().any()
    assert out["blackboard_model_validation"].empirical.isna().all()
    assert "causal_validation" in set(out["blackboard_calibration_diagnostics"].diagnostic_slice)
