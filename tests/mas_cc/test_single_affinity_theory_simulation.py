"""Monte Carlo contracts for the revised single-affinity theory."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mas_cc.cli.main import main
from mas_cc.games.relational_reasoning.imitation_round_feedback.theory_revised import (
    TheoryParameters,
    single_affinity_reference,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.theory_simulation import (
    TheoryInitialization,
    TheorySimulationConfig,
    sample_initial_count,
    simulate_action,
    simulate_controlled_opportunity,
    simulate_cycle,
    simulate_ensemble,
    theory_simulation_config_from_mapping,
)


def _parameters(**overrides) -> TheoryParameters:
    values = dict(N=6, q_c=3, b=4, beta=3.0, theta=0.5, h=1.1, gamma=0.55)
    values.update(overrides)
    return TheoryParameters(**values)


def _config(**overrides) -> TheorySimulationConfig:
    parameters = overrides.pop("parameters", _parameters())
    values = dict(
        parameters=parameters,
        rounds=3,
        episodes=2_000,
        seed=90210,
        initialization=TheoryInitialization.fixed_count(parameters.N, 2),
        validation_samples_per_state=3_000,
    )
    values.update(overrides)
    return TheorySimulationConfig(**values)


def test_reproducibility_is_exact_for_one_seed():
    config = _config(episodes=80, validation_samples_per_state=100)
    first = simulate_ensemble(config)
    second = simulate_ensemble(config)
    assert np.array_equal(first.occupancy_counts, second.occupancy_counts)
    assert np.array_equal(first.transition_counts, second.transition_counts)
    assert np.array_equal(first.terminal_currents, second.terminal_currents)
    assert first.state_validation == second.state_validation


def test_fixed_distribution_and_binomial_initialization():
    rng = np.random.default_rng(4)
    fixed = TheoryInitialization.fixed_count(6, 4)
    assert {sample_initial_count(fixed, rng) for _ in range(100)} == {4}

    explicit = TheoryInitialization.distribution(2, [0.0, 0.0, 1.0])
    assert {sample_initial_count(explicit, rng) for _ in range(100)} == {2}

    binomial = TheoryInitialization.binomial(6, 0.0)
    assert sample_initial_count(binomial, rng) == 0


def test_cycle_uses_realized_sensor_value_for_policy():
    parameters = _parameters(beta=1e6, theta=0.5)
    reference = single_affinity_reference(parameters)
    rng = np.random.default_rng(19)
    observations = [simulate_cycle(3, reference, rng)[0] for _ in range(400)]
    assert {row.y for row in observations} >= {1, 2}
    assert all(row.action == int(row.y / parameters.q_c < 0.5) for row in observations)


def test_noop_is_exact_and_microsteps_stay_in_bounds():
    reference = single_affinity_reference(_parameters())
    rng = np.random.default_rng(8)
    for n in range(reference.parameters.N + 1):
        after, records = simulate_action(n, 0, reference, rng, record_microsteps=True)
        assert after == n
        assert records == ()
        current = n
        for _ in range(100):
            current = simulate_controlled_opportunity(current, reference, rng)
            assert 0 <= current <= reference.parameters.N


def test_one_step_b_step_and_closed_loop_frequencies_match_exact_kernels():
    config = _config(episodes=100, validation_samples_per_state=12_000)
    result = simulate_ensemble(config)
    summary = result.validation_summary
    assert summary["K_max_abs_error"] < 0.025
    assert summary["Q1_max_abs_error"] < 0.025
    assert summary["closed_loop_max_abs_error"] < 0.025
    assert summary["sensor_max_abs_error"] < 0.025
    assert summary["policy_given_y_max_abs_error"] < 0.025
    assert summary["Q0_exact"] is True


def test_local_and_finite_horizon_quantities_converge_to_exact_theory():
    result = simulate_ensemble(_config())
    assert result.validation_summary["chi_max_abs_error"] < 0.035
    assert result.validation_summary["T_pi_max_abs_error_bits"] < 0.04
    assert result.validation_summary["occupancy_max_abs_error"] < 0.04
    assert result.validation_summary["terminal_current_mean_abs_error"] < 0.12
    assert result.validation_summary["terminal_current_variance_abs_error"] < 0.18
    assert result.validation_summary["finite_horizon_J_abs_error"] < 0.12
    assert result.validation_summary["finite_horizon_I_sens_abs_error_nats"] < 0.08
    assert result.validation_summary["finite_horizon_Sigma_abs_error_nats"] < 0.2
    assert (
        abs(result.horizon_summary["thermodynamic_identity_residual_mc_nats"]) < 1e-12
    )
    assert result.horizon_summary["T_pi_units"] == "bits"
    assert result.horizon_summary["thermodynamic_information_units"] == "nats"


def test_zero_actuation_edges_remain_undefined_where_theory_is_undefined():
    parameters = _parameters(b=0, gamma=0.0)
    result = simulate_ensemble(
        _config(parameters=parameters, episodes=100, validation_samples_per_state=100)
    )
    assert np.all(result.terminal_currents == 0)
    assert result.horizon_summary["J_mc"] == 0.0
    assert result.horizon_summary["eta_IR_mc"] != result.horizon_summary["eta_IR_mc"]


def test_strict_config_rejects_q_and_resolves_supported_initializers():
    payload = {
        "simulation": {"type": "single_affinity_theory", "episodes": 2, "rounds": 1},
        "theory": {
            "N": 4,
            "q_c": 2,
            "b": 2,
            "beta": 2.0,
            "theta": 0.5,
            "h": 1.0,
            "gamma": 0.5,
        },
        "initialization": {"type": "binomial", "x0": 0.25},
    }
    config = theory_simulation_config_from_mapping(payload)
    assert config.initialization.as_array().sum() == pytest.approx(1.0)
    payload["theory"]["q"] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        theory_simulation_config_from_mapping(payload)


def test_provider_free_cli_writes_expected_artifacts(tmp_path):
    config = tmp_path / "simulation.yaml"
    config.write_text(
        """
simulation:
  type: single_affinity_theory
  seed: 11
  episodes: 40
  rounds: 2
  validation_samples_per_state: 100

theory:
  N: 4
  q_c: 2
  b: 2
  beta: 2.0
  theta: 0.5
  h: 1.0
  gamma: 0.5

initialization:
  type: fixed_count
  n0: 1

artifacts:
  record_cycles: true
  record_microsteps: false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "result"
    assert (
        main(
            ["theory", "simulate", "--config", str(config), "--output-dir", str(output)]
        )
        == 0
    )
    expected = {
        "resolved_config.json",
        "metadata.json",
        "cycle_trajectories.csv",
        "occupancy_by_round.csv",
        "state_local_validation.csv",
        "horizon_summary.csv",
        "thermodynamics_by_round.csv",
        "validation_summary.json",
    }
    assert expected <= {path.name for path in output.iterdir()}
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["provider_free"] is True
    assert metadata["causal_cycle"] == "n -> Y -> U -> n'"
    assert metadata["theory_semantics_version"] == "single_affinity_v1"
