from pathlib import Path

import pandas as pd
import yaml

from santa_fe.config import load_config, plan
from santa_fe.game import SyntheticGame
from santa_fe.measurements import transitions
from santa_fe.nulls import permutation_test
from santa_fe.runner import run
from santa_fe.state import SimulationParameters


def test_seeded_dynamics_and_zero_budget_action():
    params = SimulationParameters(N=8, F=4, q=2, rounds=5, budget_fraction=0, save_micro=True)
    first = SyntheticGame(params).run_episode(123)
    second = SyntheticGame(params).run_episode(123)
    assert first.rounds == second.rounds or all(
        {k: v for k, v in a.items() if k != "controller_observed_target_share"} ==
        {k: v for k, v in b.items() if k != "controller_observed_target_share"}
        for a, b in zip(first.rounds, second.rounds)
    )
    assert first.micro == second.micro
    assert all(row["controller_effective_U"] == row["budget_used"] == 0 for row in first.rounds)
    assert len(first.micro) == params.N * params.rounds


def test_config_runner_and_nulls(tmp_path: Path):
    source = Path("configs/santa_fe/exploratory.yaml")
    data = yaml.safe_load(source.read_text())
    data["experiment"].update(episodes=4, processes=1)
    data["model"].update(N=6, F=4, rounds=4)
    data["output"].update(results_dir=str(tmp_path / "out"), save_micro_trajectories=True)
    data["sweep"]["budget_fraction"] = [0.0, 0.5]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    config = load_config(path)
    assert plan(config)["total_episodes"] == 8
    assert plan(config)["budgets"] == [{"fraction": 0.0, "integer": 0}, {"fraction": 0.5, "integer": 3}]
    rounds, micro = run(config)
    assert rounds.groupby("cell_id").seed.nunique().to_dict() == {0: 4, 1: 4}
    assert micro.groupby(["cell_id", "seed"]).size().eq(24).all()
    assert rounds.loc[rounds.cell_id == 0, "controller_effective_U"].eq(0).all()
    d = transitions(rounds.loc[rounds.cell_id == 1])
    summary, draws = permutation_test(d, 3, "mean_coverage", 9, 123, progress=False)
    assert set(summary.statistic) == {"I_X_Y_bits", "Tpi_I_U_Xnext_given_X_bits", "Tpi_I_U_Xnext_given_X_K_bits"}
    assert summary.p_value.dropna().between(.1, 1).all()
    assert len(draws) <= 27


def test_conditional_null_requires_stratification(tmp_path: Path):
    data = yaml.safe_load(Path("configs/santa_fe/study.yaml").read_text())
    data["output"]["results_dir"] = str(tmp_path / "out")
    data["analysis"]["permutation_null"]["preserve_conditioning_state"] = False
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    try:
        load_config(path)
    except ValueError as error:
        assert "preserve conditioning strata" in str(error)
    else:
        raise AssertionError("globally shuffled conditional action was accepted")


def test_llm_parallel_reuses_round_engine_at_each_transition(tmp_path: Path):
    from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import round_information_analysis
    from santa_fe.llm_parallel import adapt_trajectories, analyze_existing
    from santa_fe.runner import save_trajectories

    data = yaml.safe_load(Path("configs/santa_fe/exploratory.yaml").read_text())
    data["experiment"].update(episodes=5, processes=1)
    data["model"].update(N=8, F=4, rounds=4)
    data["output"]["results_dir"] = str(tmp_path / "results")
    data["sweep"]["budget_fraction"] = [0.0, 0.5]
    data["analysis"]["llm_parallel"].update(bootstrap_resamples=2, null_permutations=3)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    config = load_config(path)
    rounds, micro = run(config)
    save_trajectories(config, rounds, micro)
    events = adapt_trajectories(rounds, bins=config.bins)
    assert {event.round_index for event in events} == {1, 2, 3, 4}
    assert len(events) == 2 * 5 * 4
    assert all(event.U_k is None for event in events if event.round_index == 4)
    assert all(event.U_k is None for event in events if event.cell_id == "0")
    active = [event for event in events if event.cell_id == "1" and event.round_index == 1]
    direct, direct_nulls = round_information_analysis(
        active, statistics=("round_target_sensing_mi", "round_target_actuation_cmi"),
        bootstrap_resamples=0, null_permutations=0, seed=11)
    assert len(direct) == 2
    assert direct_nulls == []
    output = analyze_existing(config)
    estimates = pd.read_parquet(Path(output["results_dir"]) / "round_information_estimates.parquet")
    assert set(estimates.loc[estimates.scope == "per_round", "round"]) == {1, 2, 3, 4}
    assert not ((estimates["round"] == 4) & estimates.statistic.str.contains("actuation")).any()
    assert not ((estimates.cell_id == 0) & estimates.statistic.str.contains("actuation")).any()
    actual = estimates.loc[(estimates.cell_id == 1) & (estimates.scope == "per_round") &
                           (estimates["round"] == 1) &
                           (estimates.statistic == "round_target_actuation_cmi"), "estimate"].iloc[0]
    assert actual == next(row["estimate"] for row in direct if row["statistic"] == "round_target_actuation_cmi")
    assert (Path(output["results_dir"]) / "analysis_recipe.yaml").read_bytes() == path.read_bytes()
    sequential_raw = yaml.safe_load(path.read_text())
    sequential_raw["analysis"]["llm_parallel"]["processes"] = 1
    sequential_path = tmp_path / "sequential.yaml"
    sequential_path.write_text(yaml.safe_dump(sequential_raw))
    analyze_existing(load_config(sequential_path))
    sequential = pd.read_parquet(Path(output["results_dir"]) / "round_information_estimates.parquet")
    columns = ["cell_id", "scope", "round", "statistic", "estimate", "null_mean", "null_p_value"]
    pd.testing.assert_frame_equal(estimates[columns], sequential[columns])


def test_cluster_cell_array_and_phase_outputs(tmp_path: Path):
    from santa_fe.cluster import (
        aggregate_cells, aggregate_information, prepare, run_cell,
        run_information_cell, write_cygnus_commands,
    )

    data = yaml.safe_load(Path("configs/santa_fe/study_cygnus.yaml").read_text())
    data["experiment"].update(episodes=3, processes=1)
    data["model"].update(N=6, F=4, rounds=3)
    data["output"]["results_dir"] = str(tmp_path / "results")
    data["sweep"] = {"beta_evidence": [1.0, 2.0], "beta_social": [0.5, 1.0],
                     "budget_fraction": [0.0, 0.5]}
    data["analysis"]["llm_parallel"].update(processes=1, bootstrap_resamples=2,
                                               null_permutations=2)
    data["slurm"]["simulation"].update(cpus_per_task=1, throttle=2)
    data["slurm"]["information"].update(cpus_per_task=1, throttle=2)
    config_path = tmp_path / "cluster_config.yaml"
    config_path.write_text(yaml.safe_dump(data))
    config = load_config(config_path)
    manifest = prepare(config)
    assert manifest["parameter_cells"] == 8
    script = write_cygnus_commands(config, python="/shared/home/test/env/bin/python",
                                   repository_root="/shared/home/test/MA-CC-cygnus")
    shell = script.read_text()
    assert "--array=0-7%2" in shell
    assert "--dependency=afterok:$sim_job" in shell
    assert "--dependency=afterok:$info_job" in shell
    assert "--cpus-per-task=1" in shell
    for cell_id in range(8):
        assert run_cell(config, cell_id)["status"] == "complete"
    assert run_cell(config, 0)["status"] == "already_complete"
    combined = aggregate_cells(config)
    assert combined["round_rows"] == 8 * 3 * 4
    assert (config.results_dir / "plots" / "phase_diagrams" /
            "final_target_share_budget_0.5.png").is_file()
    for cell_id in range(8):
        assert run_information_cell(config, cell_id)["status"] == "complete"
    assert run_information_cell(config, 0)["status"] == "already_complete"
    information = aggregate_information(config)
    assert information["cells"] == 8
    assert (config.results_dir / "plots" / "phase_diagrams" /
            "round_target_actuation_cmi_bits_budget_0.5.png").is_file()
    info_table = pd.read_parquet(config.results_dir / "information" / "llm_parallel" /
                                 "round_information_estimates.parquet")
    assert info_table.cell_id.nunique() == 8
    assert not ((info_table.cell_id.isin([0, 2, 4, 6])) &
                info_table.statistic.str.contains("actuation")).any()


def test_targeted_beta_cmi_outputs_and_sample_size(tmp_path: Path):
    from santa_fe.cluster import prepare, run_cell, aggregate_cells, run_information_cell, aggregate_information
    from santa_fe.sample_size_cmi import prepare_reference, run_task, aggregate as aggregate_sample

    data = yaml.safe_load(Path("configs/santa_fe/beta_cmi_sweep.yaml").read_text())
    data["experiment"].update(episodes=5, processes=1)
    data["model"].update(N=8, F=4, rounds=4)
    data["output"]["results_dir"] = str(tmp_path / "sweep")
    data["sweep"]["beta_pairs"] = data["sweep"]["beta_pairs"][:1]
    data["sweep"]["budget_fraction"] = [0.0, .25]
    data["analysis"]["llm_parallel"].update(processes=1, bootstrap_resamples=2, null_permutations=3)
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(yaml.safe_dump(data))
    config = load_config(sweep_path)
    assert len(config.cells) == 4
    assert {cell.beta_regime for cell in config.cells} == {"competition_low"}
    assert {cell.params.rho for cell in config.cells} == {.75, 1.0}
    prepare(config)
    for cell in config.cells:
        run_cell(config, cell.cell_id)
    aggregate_cells(config)
    for cell in config.cells:
        run_information_cell(config, cell.cell_id)
    aggregate_information(config)
    table = pd.read_csv(config.results_dir / "information" / "beta_cmi" / "detectability.csv")
    assert len(table) == 16
    assert table.groupby("cell_id").conditioning.nunique().eq(4).all()
    assert table.loc[table.budget.eq(0), "observed_cmi"].isna().all()
    assert table.loc[table.budget.gt(0), "observed_cmi"].notna().all()
    assert table.loc[table.budget.gt(0), "null_q95"].notna().all()
    assert (config.results_dir / "plots" / "beta_cmi" / "heatmap_observed_cmi_plain_rho_0.75.png").is_file()

    sample = yaml.safe_load(Path("configs/santa_fe/sample_size_cmi.yaml").read_text())
    sample["output"]["results_dir"] = str(tmp_path / "sample")
    sample["experiment"].update(episodes=5, processes=2)
    sample["model"].update(N=8, F=4, rounds=4)
    sample["sweep"]["rho"] = [.75]
    sample["sample_size_study"].update(reference_episodes=5, episode_counts=[3, 5],
                                        repetitions=2, n_bootstrap=2, n_permutations=3)
    sample_path = tmp_path / "sample.yaml"
    sample_path.write_text(yaml.safe_dump(sample))
    sample_config = load_config(sample_path)
    prepare_reference(sample_config, 0)
    run_task(sample_config, 0)
    run_task(sample_config, 1)
    summary = aggregate_sample(sample_config)
    assert summary["summary_rows"] == 2
    assert pd.read_csv(sample_config.results_dir / "sample_size_cmi" / "sample_size_cmi_summary.csv").n_episodes.tolist() == [3, 5]


def test_targeted_scientific_recipe_is_paired():
    config = load_config("configs/santa_fe/beta_cmi_sweep.yaml")
    assert len(config.cells) == 90
    assert {(c.params.beta_evidence, c.params.beta_social) for c in config.cells} == {
        (.75, 1.0), (1.0, 1.3), (1.25, 1.6), (2.0, .5), (.5, 2.0)}
    assert {c.params.rho for c in config.cells} == {.75, 1.0}
    assert {c.params.budget for c in config.cells} == set(range(0, 25, 3))
    sample = load_config("configs/santa_fe/sample_size_cmi.yaml")
    assert len(sample.cells) == 2
    assert {(c.params.beta_evidence, c.params.beta_social, c.params.budget) for c in sample.cells} == {(1.0, 1.3, 6)}
    assert sample.sample_size["episode_counts"] == [10, 20, 30, 50, 75, 100, 200]


def test_v3_semantic_clock_and_event_ledger():
    import json
    from dataclasses import replace
    base = SimulationParameters(
        model_version="santa_fe_epistemic_feedback_v3", persistence_clock="round_boundary",
        peer_posting_mode="vote_aligned_fact", controller_message_mode="target_aligned_fact",
        controller_fact_selection="uniform_with_replacement", N=8, F=4, q=2,
        rounds=4, rho=0.0, sensing_fraction=.5, budget_fraction=.25, save_micro=True)
    result = SyntheticGame(base).run_episode(42)
    assert result.rounds == SyntheticGame(base).run_episode(42).rounds
    assert result.micro == SyntheticGame(base).run_episode(42).micro
    assert len(result.micro) == base.N * base.rounds
    weights = result.fact_weights
    for row in result.rounds[1:]:
        persistence = json.loads(row["persistence_events_json"])
        assert len(persistence) == base.N
        assert all(not event["after_fact_ids"] for event in persistence)
        assert sum(len(event["lost_fact_ids"]) for event in persistence) == sum(
            len(event["before_fact_ids"]) for event in persistence)
        assert row["controller_sensed_messages"] == 4
        assert len(json.loads(row["sensor_sample_ids_json"])) == 4
        peer = json.loads(row["peer_board_json"])
        controller = json.loads(row["controller_board_json"])
        from math import comb
        target_posts = sum(m["vote"] == base.controller_target for m in peer)
        y = row["controller_sensor_target_count"]
        assert row["peer_board_target_count"] == target_posts
        assert abs(row["p_sensor_outcome"] -
                   comb(target_posts, y) * comb(base.N-target_posts, 4-y) / comb(base.N, 4)) < 1e-12
        pool_size = len(json.loads(row["controller_fact_pool_json"]))
        assert abs(row["p_controller_fact_selection"] - pool_size ** (-len(controller))) < 1e-12
        assert len(peer) == base.N
        assert len(controller) == row["budget_used"] == base.budget * row["controller_effective_U"]
        assert all(m["fact_sign"] in (0, m["vote"]) for m in peer)
        assert all(m["vote"] == m["fact_sign"] == base.controller_target for m in controller)
        assert all(weights[m["fact_id"]] == base.controller_target for m in controller)
        assert all(m["fact_id"] is not None for m in controller)
        assert sum(json.loads(row["B_peer_counts_json"]).values()) == base.N
        assert sum(json.loads(row["B_total_counts_json"]).values()) == base.N + len(controller)
    for event in result.micro:
        sampled = json.loads(event["sampled_message_ids_json"])
        assert all(not mid.startswith(f'r{event["round"]}-peer-') for mid in sampled)
        assert event["p_fact_acquisition_given_sample"] == 1.0
        assert set(json.loads(event["controller_only_acquired_fact_ids_json"])).issubset(
            json.loads(event["controller_exposed_fact_ids_json"]))
        if event["sampled_controller_messages"] == 0:
            assert json.loads(event["controller_only_acquired_fact_ids_json"]) == []
        assert event["posted_fact_sign"] in (0, event["vote_after"])
        assert (event["posted_fact_id"] is None) == (not json.loads(event["eligible_post_fact_ids_json"]))
        assert event["p_vote_realized"] > 0
    no_control = SyntheticGame(replace(base, budget_fraction=0)).run_episode(42)
    assert all(row["budget_used"] == 0 for row in no_control.rounds)
    assert all(json.loads(row["controller_board_json"]) == [] for row in no_control.rounds[1:])


def test_v3_config_requires_explicit_semantics(tmp_path: Path):
    data = yaml.safe_load(Path("configs/santa_fe/v3_pilot.yaml").read_text())
    data["output"]["results_dir"] = str(tmp_path / "out")
    data["model"]["peer_posting_mode"] = "random_active_fact"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(data))
    import pytest
    with pytest.raises(ValueError, match="requires peer_posting_mode"):
        load_config(path)


def test_v3_exact_kernel_and_validation_tables(tmp_path: Path):
    import json
    from santa_fe.runner import save_trajectories
    from santa_fe.v3_validation import exact_one_step, validate, finite_size_scaling, sensing_sample_size
    from santa_fe.llm_parallel import analyze_existing
    data = yaml.safe_load(Path("configs/santa_fe/v3_pilot.yaml").read_text())
    data["experiment"].update(episodes=3, processes=1)
    data["model"].update(N=8, F=4, F_plus=3, rounds=3)
    data["sweep"]["budget_fraction"] = [.25]
    data["output"]["results_dir"] = str(tmp_path / "results")
    data["analysis"]["llm_parallel"].update(processes=1, bootstrap_resamples=2, null_permutations=3)
    path = tmp_path / "v3.yaml"
    path.write_text(yaml.safe_dump(data))
    config = load_config(path)
    rounds, micro = run(config)
    save_trajectories(config, rounds, micro)
    row = rounds.loc[rounds["round"] == 1].iloc[0]
    event = micro.loc[micro["round"] == 1].iloc[0]
    classes, emissions, joint = exact_one_step(
        set(json.loads(event.facts_before_json)), int(event.vote_before),
        json.loads(row.front_page_json), json.loads(row.fact_weights_json),
        int(row.q), float(row.beta_evidence), float(row.beta_social))
    assert abs(sum(classes.values()) - 1) < 1e-12
    assert abs(sum(emissions.values()) - 1) < 1e-12
    assert abs(sum(joint.values()) - 1) < 1e-12
    assert not any(category in emissions for category in ("+1_-1", "-1_+1"))
    analyze_existing(config)
    result = validate(config, max_events=15)
    assert result["invariants_passed"]
    assert result["checked_events"] == 15
    assert (config.results_dir / "validation" / "transition_kernel_validation.parquet").is_file()
    assert (config.results_dir / "validation" / "sensing_validation.csv").is_file()
    assert (config.results_dir / "validation" / "response_estimates.parquet").is_file()
    assert pd.read_csv(config.results_dir / "validation" / "path_replay_validation.csv").agents_match.all()
    scaling = finite_size_scaling(config, sizes=(8, 16), repetitions=3, checkpoint_round=1)
    assert len(scaling["slopes"]) >= 2
    sensing = sensing_sample_size(config, counts=(2, 3), repetitions=2, null_permutations=3)
    assert sensing["repetitions"] == 4


def test_v3_sampling_laws_and_local_closure():
    import math
    from santa_fe.v3_validation import (
        fact_encounter_probability, hmf_one_step, hypergeometric_category_law,
        multinomial_category_law, sensor_averaged_propensity,
    )
    counts = {"+1_+1": 2, "+1_0": 0, "-1_-1": 1, "-1_0": 0}
    exact = hypergeometric_category_law(counts, 2)
    approx = multinomial_category_law(counts, 2)
    assert math.isclose(sum(exact.values()), 1)
    assert math.isclose(sum(approx.values()), 1)
    assert exact.get((2, 0, 0, 0)) == 1 / 3
    assert approx.get((2, 0, 0, 0)) == 4 / 9
    assert fact_encounter_probability(3, 2, 2) == 1
    assert math.isclose(fact_encounter_probability(3, 2, 2, replacement=True), 8 / 9)
    assert fact_encounter_probability(0, 0, 2) == 0
    assert hypergeometric_category_law({}, 2) == {(0, 0, 0, 0): 1}
    assert math.isclose(sensor_averaged_propensity(2, 4, 4, 8, .5), .5)
    for sampling in ("hypergeometric", "multinomial"):
        transitions, emissions = hmf_one_step((0, 0, 1), counts, 3, 2, 2, 1, 1,
                                              sampling=sampling)
        assert math.isclose(sum(transitions.values()), 1)
        assert math.isclose(sum(emissions.values()), 1)
        assert not {"+1_-1", "-1_+1"}.intersection(emissions)


def test_v3_explicit_fact_split_and_zero_budget(tmp_path: Path):
    import json
    from santa_fe.v3_validation import _paired_counterfactuals
    data = yaml.safe_load(Path("configs/santa_fe/v3_pilot.yaml").read_text())
    data["experiment"].update(episodes=2, processes=1)
    data["model"].update(N=8, F=5, F_plus=3, rounds=2)
    data["sweep"]["budget_fraction"] = [0]
    data["output"]["results_dir"] = str(tmp_path / "zero_budget")
    path = tmp_path / "zero_budget.yaml"
    path.write_text(yaml.safe_dump(data))
    config = load_config(path)
    rounds, _ = run(config)
    assert all(json.loads(value).count(1) == 3 for value in rounds.fact_weights_json)
    paired, replay, branch_laws = _paired_counterfactuals(rounds, config, 2, 4, 11)
    assert replay.agents_match.all()
    assert (paired.chi_target == 0).all()
    assert (paired.T_pi_nats == 0).all()
    assert (branch_laws.Q0 == branch_laws.Q1).all()


def test_v3_shared_information_engine_and_sensing_source(monkeypatch):
    import json
    from mas_cc.games.hidden_bench.imitation_round_feedback import analysis
    from santa_fe.llm_parallel import adapt_trajectories
    params = SimulationParameters(
        model_version="santa_fe_epistemic_feedback_v3", persistence_clock="round_boundary",
        peer_posting_mode="vote_aligned_fact", controller_message_mode="target_aligned_fact",
        controller_fact_selection="uniform_with_replacement", N=8, F=4, q=2,
        rounds=4, budget_fraction=.25, save_micro=True)
    rows = []
    for seed in range(6):
        result = SyntheticGame(params).run_episode(100 + seed)
        for row in result.rounds:
            rows.append({"cell_id": 0, "seed": seed, "N": params.N, "budget": params.budget,
                         "controller_target": params.controller_target, "model_version": params.model_version,
                         **row})
    events = adapt_trajectories(pd.DataFrame(rows), bins=4)
    first = events[0]
    assert first.sensor_source_target_count == sum(
        m["vote"] == params.controller_target for m in json.loads(rows[1]["peer_board_json"]))
    stats = ("round_target_sensing_mi", "round_target_actuation_cmi",
             "round_kappa_plus_actuation_cmi", "round_kappa_minus_actuation_cmi",
             "round_kappa_ctrl_actuation_cmi", "round_kappa_plus_signed_response")
    monkeypatch.setattr(analysis, "_INFORMATION_ENGINE", "rows")
    reference, reference_nulls = analysis.round_information_analysis(
        events, statistics=stats, bootstrap_resamples=3, null_permutations=4, seed=3)
    monkeypatch.setattr(analysis, "_INFORMATION_ENGINE", "fast")
    fast, fast_nulls = analysis.round_information_analysis(
        events, statistics=stats, bootstrap_resamples=3, null_permutations=4, seed=3)
    assert [row["statistic"] for row in fast] == [row["statistic"] for row in reference]
    for before, after in zip(reference, fast):
        assert abs(before["estimate"] - after["estimate"]) < 1e-10 or (
            pd.isna(before["estimate"]) and pd.isna(after["estimate"]))
        assert before["round_dual_action_event_fraction"] == after["round_dual_action_event_fraction"]
    assert len(reference_nulls) == len(fast_nulls)
