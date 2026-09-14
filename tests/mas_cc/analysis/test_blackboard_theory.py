import math
import json

import numpy as np
import pandas as pd
import pytest

from mas_cc.analysis import blackboard_theory_engine as engine
from mas_cc.analysis.blackboard_theory import analyze_blackboard_theory, theory_settings
from mas_cc.analysis.blackboard_calibration import (
    analyze_blackboard_calibration,
    calibration_settings,
    mean_map,
    sampling_exposure_probability,
)
from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import RoundEvent
from test_blackboard_calibration import fixture_tables


def report():
    Q0, Q1 = engine.round_kernels(
        24,
        24,
        (0.35 * 0.7, 0.35 * 0.3),
        (0.65 * 0.9, 0.65 * 0.1),
        sampling_exposure_probability(18, 9, 3),
    )
    S, pi, a = engine.exact_sensor_policy(24, 12, beta=4, theta=0.5)
    return Q0, Q1, a


def test_report_and_mean_identity():
    Q0, Q1, a = report()
    s = engine.state_metrics(Q0, Q1, a)
    expected = [
        (
            0.7253678932,
            0.1347957916,
            0.3284661709,
            0.8480331716,
            0.3873270314,
            0.0317962711,
        ),
        (0.5, 0.1000215251, 0.2772913903, 1.0, 0.2772913903, 0.0260252615),
        (
            0.2746321068,
            0.0652472586,
            0.1339128654,
            0.8480331716,
            0.1579099378,
            0.0182732758,
        ),
    ]
    for n, values in zip((6, 12, 18), expected):
        assert [
            a[n],
            s["chi"][n],
            s["T"][n],
            s["H"][n],
            s["T"][n] / s["H"][n],
            s["B"][n] / s["T"][n],
        ] == pytest.approx(values, abs=1e-8)
    f = np.arange(25) / 24
    for n in range(25):
        assert (Q0 @ f)[n] == pytest.approx(
            mean_map(0.245, 0.105, 24, 24, n / 24)["mean"], abs=1e-10
        )
    assert np.allclose(s["J_total"] - s["J_silent"], s["J_excess"], atol=1e-10)
    assert np.all(s["B"] <= s["T"] + 1e-10)
    assert np.all(s["T"] <= s["H"] + 1e-10)


def test_boundaries_mixture_and_order():
    assert np.array_equal(engine.microscopic_kernel(4, 0, 0), np.eye(5))
    for rates in ((0.8, 0.4), (math.nan, 0.2), (-0.1, 0.2)):
        with pytest.raises(ValueError):
            engine.microscopic_kernel(4, *rates)
    A = engine.microscopic_kernel(4, 0.2, 0.1)
    B = engine.microscopic_kernel(4, 0.6, 0.2)
    assert not np.allclose(
        engine.ordered_product([A, B]), engine.ordered_product([B, A])
    )
    _, Q = engine.round_kernels(4, 4, (0.2, 0.1), (0.6, 0.2), 0.4)
    assert not np.allclose(
        Q, 0.6 * np.linalg.matrix_power(A, 4) + 0.4 * np.linalg.matrix_power(B, 4)
    )
    assert np.allclose(Q.sum(axis=1), 1)
    assert A[0, -1] == A[-1, 0] == 0
    with pytest.raises(ValueError):
        engine.round_kernels(4, True, (0.2, 0.1), (0.3, 0.1))


def test_identical_and_deterministic_branches():
    Q0, Q1, a = report()
    equal = engine.state_metrics(Q0, Q0, a)
    assert np.allclose(equal["chi"], 0)
    assert np.allclose(equal["T"], 0, atol=1e-14)
    for action in (0.0, 1.0):
        s = engine.state_metrics(Q0, Q1, np.full(25, action))
        for key in ("H", "T", "B"):
            assert np.array_equal(s[key], np.zeros(25))
        assert np.any(s["chi"] != 0)
    assert math.isnan(engine.component_ratios(0, 0, 0)["eta_IR"])
    Q0, Q1 = engine.round_kernels(4, 4, (0.2, 0.1), (0.1, 0.2))
    chi = engine.lag_contrast(Q0, Q1, None, 1)
    assert np.ptp(chi) < 1e-12


def test_equal_mean_positive_information():
    Q0 = np.eye(3)
    Q1 = Q0.copy()
    Q1[1] = [0.5, 0, 0.5]
    s = engine.state_metrics(Q0, Q1, np.full(3, 0.5))
    assert s["chi"][1] == 0
    assert s["T"][1] > 0
    assert engine.component_ratios(s["T"][1], s["H"][1], s["B"][1])["eta_IR"] == 0


def test_lags_ipw_and_forward_telescoping():
    Q0, Q1, a = report()
    f = np.arange(25) / 24
    P = engine.feedback_kernel(Q0, Q1, a)
    for lag in (1, 2, 3):
        contrast = engine.lag_contrast(Q0, Q1, a, lag)
        assert np.allclose(contrast, (Q1 - Q0) @ np.linalg.matrix_power(P, lag - 1) @ f)
        for e in (0.2, 0.7):
            enumerated = e / e * (Q1 @ np.linalg.matrix_power(P, lag - 1) @ f - f) - (
                1 - e
            ) / (1 - e) * (Q0 @ np.linalg.matrix_power(P, lag - 1) @ f - f)
            assert np.allclose(contrast, enumerated)
        if lag > 1:
            assert not np.allclose(
                contrast,
                (np.linalg.matrix_power(Q1, lag) - np.linalg.matrix_power(Q0, lag)) @ f,
            )
    initial = np.eye(25)[12]
    trajectory, components, current = engine.forward_ensemble(Q0, Q1, a, initial, 5)
    assert current == pytest.approx(sum(c["J_total"] for c in components))
    assert current == pytest.approx((trajectory[-1] - initial) @ np.arange(25))
    assert not np.isclose(current, sum(c["J_excess"] for c in components))
    missing = a.copy()
    missing[0] = np.nan
    with pytest.raises(ValueError, match="missing_future_policy"):
        engine.lag_contrast(Q0, Q1, missing, 2)


def test_joint_coarsening_entropy_and_component_aggregation():
    Q0, Q1, a = report()
    counts = [6, 12, 18]
    weights = [0.2, 0.3, 0.5]
    exact = engine.joint_law(weights, counts, a[counts], Q0[counts], Q1[counts], counts)
    coarse = engine.joint_law(
        weights, counts, a[counts], Q0[counts], Q1[counts], [0, 0, 0]
    )
    e, c = engine.joint_information(exact), engine.joint_information(coarse)
    assert not np.isclose(e["T"], c["T"])
    assert e["H_action"] > e["H"]
    s = engine.state_metrics(Q0, Q1, a)
    assert e["T"] == pytest.approx(np.asarray(weights) @ s["T"][counts])
    assert not np.isclose(
        e["T"] / e["H"], np.asarray(weights) @ (s["T"][counts] / s["H"][counts])
    )
    mapped = engine.joint_law(
        weights,
        counts,
        a[counts],
        Q0[counts],
        Q1[counts],
        counts,
        [int(n >= 12) for n in range(25)],
    )
    assert engine.joint_information(mapped)["T"] < e["T"]


def test_no_sensor_observation_requires_policy():
    with pytest.raises(ValueError):
        engine.exact_sensor_policy(4, 0)
    S, pi, a = engine.exact_sensor_policy(4, 0, no_observation_probability=0.3)
    assert np.array_equal(S, np.ones((5, 1)))
    assert a == pytest.approx([0.3] * 5)


def prepared(blocks=6, heldout=False):
    tables = fixture_tables(blocks)
    micro, rounds, episodes, cells = tables
    events = []
    for r in rounds.to_dict("records"):
        event = dict(
            occupation_counts_before=[2, 2],
            occupation_counts_after=[1, 3],
            target_count_before=2,
            target_count_after=3,
            controller_action="ADVOCATE_TARGET" if r["U_k"] else "NO_OP",
            delta_p_ctrl=0.25,
            delta_m_ctrl=0.5,
            N=4,
            sensor_target_share=0.5,
            sensor_count_vector=[1, 1],
            possible_answers=["A", "Z"],
            analysis_target="Z",
        )
        events.append(
            RoundEvent(r["cell_id"], r["episode_id"], r["round_index"], event)
        )
    cs = calibration_settings(
        dict(
            shared_unexposed_baseline=True,
            model_predictions=dict(
                enabled=True, evaluation_fraction=0.3, assume_homogeneous_channels=True
            ),
        )
        if heldout
        else dict(shared_unexposed_baseline=True)
    )
    out = analyze_blackboard_calibration(
        *tables, settings=cs, bootstrap_resamples=0, analysis_hash="calibration_hash"
    )
    return out, rounds, cells, events, cs


def test_adapter_uncertainty_splits_statuses_and_reproducibility():
    out, rounds, cells, events, cs = prepared(8, True)
    settings = dict(model="action_branch_calibrated", assume_homogeneous_channels=True)
    kwargs = dict(
        settings=settings,
        calibration_settings=cs,
        events=events,
        bootstrap_resamples=8,
        analysis_hash="theory_hash",
    )
    result = analyze_blackboard_theory(out, rounds, cells, **kwargs)
    again = analyze_blackboard_theory(out, rounds, cells, **kwargs)
    for key in result:
        pd.testing.assert_frame_equal(result[key], again[key])
    manifest = result["theory_model_manifest"].iloc[0]
    assert manifest.split_status == "held_out_prediction"
    assert not set(json.loads(manifest.training_blocks_json)) & set(
        json.loads(manifest.evaluation_blocks_json)
    )
    primary = result["theory_primary_estimates"]
    assert (
        primary.query("metric=='theory_round_target_actuation_cmi'")
        .estimate.notna()
        .all()
    )
    assert (
        primary.query("metric=='theory_propensity_weighted_causal_response' and lag==2")
        .model_status.eq("missing_count_markov_law_for_future_board_evidence")
        .all()
    )
    comparison = result["theory_empirical_comparison"]
    supported = comparison[comparison.residual.notna()]
    assert len(supported) > 0
    assert np.allclose(
        supported.residual,
        supported.empirical_estimate - supported.theoretical_estimate,
    )
    assert set(json.loads(manifest.dependencies_json)) <= set(
        out["blackboard_calibration_estimates"].estimate_id
    )
    assert (
        result["theory_primary_estimates"].calibration_hash.eq("calibration_hash").all()
    )


def test_forward_adapter_and_missing_dependencies():
    out, rounds, cells, events, cs = prepared()
    result = analyze_blackboard_theory(
        out,
        rounds,
        cells,
        events=events,
        calibration_settings=cs,
        settings=dict(
            model="action_branch_calibrated",
            assume_homogeneous_channels=True,
            assume_count_markov=True,
            evaluation_modes=["forward_model_ensemble"],
            forward_policy_source="complete_policy_table",
            complete_policy_table=[0.8, 0.65, 0.5, 0.35, 0.2],
            forward_initial_occupancy=[0, 0, 1, 0, 0],
            forward_horizon=3,
        ),
    )
    derived = result["theory_derived_observables"]
    assert (
        derived.query("metric=='theory_expected_cell_current'").estimate.notna().all()
    )
    assert result["theory_empirical_comparison"].empty
    missing = analyze_blackboard_theory(
        {}, rounds, cells, settings={}, calibration_settings=None
    )
    assert len(missing["theory_validation"]) == 7
    assert missing["theory_primary_estimates"].estimate.isna().all()


@pytest.mark.parametrize(
    "raw",
    [
        True,
        {"bad": 1},
        {"causal_lags": [True]},
        {"matched_policy_source": "fallback"},
        {"uncertainty": "independent_gaussian"},
        {"sensor_policy": {"sampling_law": "with_replacement"}},
    ],
)
def test_strict_settings(raw):
    with pytest.raises(ValueError):
        theory_settings(raw)


def test_offline_package_and_empirical_regression(tmp_path):
    import zipfile
    import yaml
    from mas_cc.studies.aggregation import aggregate_study
    from test_blackboard_calibration import (
        test_mock_study_retention_packaging_and_offline_reaggregation,
    )

    test_mock_study_retention_packaging_and_offline_reaggregation(tmp_path)
    study = tmp_path / "study"
    tables = study / "analysis/tables"
    primary = pd.read_parquet(tables / "primary_estimates.parquet")
    recipe = tmp_path / "configs/analysis.yaml"
    config = yaml.safe_load(recipe.read_text())
    config["blackboard_theory_outputs"] = dict(
        enabled=True, model="action_branch_calibrated", assume_homogeneous_channels=True
    )
    recipe.write_text(yaml.safe_dump(config))
    result = aggregate_study(study)
    pd.testing.assert_frame_equal(
        primary, pd.read_parquet(tables / "primary_estimates.parquet")
    )
    manifest = json.loads((study / "analysis/analysis_manifest.json").read_text())
    assert manifest["blackboard_theory_hash"]
    with zipfile.ZipFile(result["archive"]) as archive:
        for table in (
            "theory_primary_estimates",
            "theory_model_manifest",
            "theory_validation",
        ):
            assert f"tables/{table}.parquet" in archive.namelist()
    before = pd.read_parquet(tables / "theory_primary_estimates.parquet")
    aggregate_study(study)
    pd.testing.assert_frame_equal(
        before, pd.read_parquet(tables / "theory_primary_estimates.parquet")
    )
    config["theoretical_reference"] = "calibrated_blackboard_finite_state_v1"
    recipe.write_text(yaml.safe_dump(config))
    assert aggregate_study(study)["complete"]


def test_available_masks_mean_of_ratios_and_recorded_policy():
    from mas_cc.analysis.blackboard_theory import _evaluate

    rows = []
    for i, n in enumerate([1, 1, 3, 3, 4, 4]):
        action = i % 2
        event = RoundEvent(
            "run/cell",
            f"e{i}",
            0,
            dict(
                occupation_counts_before=[4 - n, n],
                occupation_counts_after=[4 - n, n],
                target_count_before=n,
                target_count_after=n,
                controller_action="ADVOCATE_TARGET" if action else "NO_OP",
                delta_p_ctrl=0.0,
                delta_m_ctrl=0.0,
                N=4,
            ),
        )
        rows.append(
            dict(
                n=n,
                N=4,
                M=4,
                U=action,
                propensity=0.2 if n == 1 else 0.7,
                episode_id=f"e{i}",
                round_index=0,
                _event=event,
                _causal=dict(
                    episode_complete=True, lag_1_available=True, causal_response_h1=0.1
                ),
            )
        )
    rates = [(0.2, 0.1), (0.5, 0.2)]
    settings = theory_settings(
        dict(
            model="action_branch_calibrated",
            matched_policy_source="recorded_propensity_average",
        )
    )
    values, states = _evaluate(rows, rates, settings, 4, "matched_empirical_design")
    Q0, Q1 = engine.round_kernels(4, 4, *rates)
    chi = engine.lag_contrast(Q0, Q1, None, 1)
    summary = next(
        v
        for k, v in values.items()
        if k[0] == "available_causal_susceptibility_summary"
    )
    assert summary["estimate"] == pytest.approx((chi[1] + chi[3]) / (0.75 + 0.25))
    assert summary["n_observations"] == 4
    assert len(json.loads(summary["evaluation_ids_json"])) == 4
    local = [
        v
        for k, v in values.items()
        if k[0] == "available_causal_susceptibility_state_local"
    ]
    assert sorted(v["estimate"] for v in local) == pytest.approx(
        sorted([chi[1] / 0.75, chi[3] / 0.25])
    )
    assert np.mean([v["estimate"] for v in local]) != pytest.approx(summary["estimate"])
    assert {
        s["target_count_before"]: s["policy_probability"] for s in states
    } == pytest.approx({1: 0.2, 3: 0.7, 4: 0.7})
    causal = next(
        v
        for k, v in values.items()
        if k[0] == "propensity_weighted_causal_response" and k[1] == 1
    )
    assert causal["estimate"] == pytest.approx(np.mean(chi[[1, 1, 3, 3, 4, 4]]))


def test_explicit_cross_cell_ratios_and_missing_cells():
    from mas_cc.analysis.blackboard_theory import aggregate_theory_components

    frame = pd.DataFrame(
        [
            dict(
                metric="theory_eta_ir",
                empirical_metric="eta_ir",
                theory_mode="matched_empirical_design",
                policy_source="empirical_action_frequency",
                lag=1,
                grouping_slice_json="{}",
                split_status="in_sample_calibrated",
                cell_id=cid,
                numerator=num,
                denominator=den,
                theory_estimate_id=cid,
            )
            for cid, num, den in [("run1/cell", 0.1, 0.5), ("run2/cell", 0.4, 1.0)]
        ]
    )
    row = aggregate_theory_components(frame, {"run1/cell": 1, "run2/cell": 2})[0]
    assert row["estimate"] == pytest.approx(0.9 / 2.5)
    assert row["estimate"] != pytest.approx((0.2 + 2 * 0.4) / 3)
    assert set(json.loads(row["dependencies_json"])) == set(frame.theory_estimate_id)
    missing = aggregate_theory_components(frame, {"run1/cell": 1, "absent/cell": 1})[0]
    assert math.isnan(missing["estimate"])
    assert missing["model_status"] == "missing_or_incompatible_cell_components"


def test_retained_supported_mixture_and_sensor_provenance(tmp_path):
    out, rounds, cells, events, cs = prepared()
    for event in events:
        event.event.update(
            sensor_sample_size=2,
            controller_policy="soft_target",
            beta=4.0,
            theta=0.5,
            sensor_target_count=1,
        )
    # Exercise the retained calibration boundary: no marginal-CI reconstruction.
    for name, frame in out.items():
        frame.to_parquet(tmp_path / f"{name}.parquet")
    retained = {name: pd.read_parquet(tmp_path / f"{name}.parquet") for name in out}
    config = dict(
        assume_homogeneous_channels=True,
        assume_count_markov=True,
        matched_policy_source="sensor_policy_exact",
        sensor_policy=dict(
            sampling_law="uniform_without_replacement",
            rule="logistic_target_fraction",
            q_c=2,
            beta=4,
            theta=0.5,
        ),
    )
    result = analyze_blackboard_theory(
        retained,
        rounds,
        cells,
        events=events,
        calibration_settings=cs,
        settings=config,
        bootstrap_resamples=4,
    )
    primary = result["theory_primary_estimates"]
    assert (
        primary.query("metric=='theory_propensity_weighted_causal_response' and lag==2")
        .estimate.notna()
        .all()
    )
    for metric in (
        "theory_round_target_sensing_mi",
        "theory_target_sensing_information_nats",
        "theory_round_sensor_mae",
        "theory_round_sensor_mse",
    ):
        selected = primary[primary.metric == metric]
        assert len(selected) == 1 and selected.estimate.notna().all()
    nats = primary.query("metric=='theory_target_sensing_information_nats'").iloc[0]
    assert nats.estimate == pytest.approx(nats.empirical_estimate)
    config["sensor_policy"]["q_c"] = 3
    mismatched = analyze_blackboard_theory(
        retained, rounds, cells, events=events, calibration_settings=cs, settings=config
    )
    assert mismatched["theory_primary_estimates"].estimate.isna().all()
    assert (
        mismatched["theory_primary_estimates"]
        .model_status.str.contains("disagrees_with_recorded")
        .all()
    )
