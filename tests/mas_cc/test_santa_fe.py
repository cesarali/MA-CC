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
