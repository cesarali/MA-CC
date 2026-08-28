"""Monte Carlo realization of the revised single-affinity reference theory.

The simulated causal cycle is exactly ``n -> Y -> U -> n'``.  This module has
no language-model, agent, prompt, fact, or relational-game dependency.  All
probability laws are sampled from the public objects assembled by
:mod:`theory_revised`; formulas are not duplicated here.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from mas_cc.analysis.estimators import mutual_information_from_counts

from mas_cc.games.relational_reasoning.imitation_round_feedback.theory_revised import (
    THEORY_API_VERSION,
    THEORY_MODULE,
    THEORY_REFERENCE,
    THEORY_SEMANTICS_VERSION,
    SingleAffinityReference,
    TheoryParameters,
    binomial_ensemble,
    finite_horizon_current_moments,
    finite_horizon_thermodynamics,
    information_response_lower_bound,
    single_affinity_reference,
    system_entropy,
    thermodynamic_efficiency,
)


@dataclass(frozen=True, slots=True)
class TheoryInitialization:
    """A probability distribution for the initial count ``n_0``."""

    type: str
    probabilities: tuple[float, ...]

    @classmethod
    def fixed_count(cls, N: int, n0: int) -> "TheoryInitialization":
        if isinstance(n0, bool) or not 0 <= n0 <= N:
            raise ValueError("initialization.n0 must lie in [0,N]")
        values = np.zeros(N + 1, dtype=float)
        values[n0] = 1.0
        return cls("fixed_count", tuple(float(value) for value in values))

    @classmethod
    def binomial(cls, N: int, x0: float) -> "TheoryInitialization":
        values = binomial_ensemble(N, x0)
        return cls("binomial", tuple(float(value) for value in values))

    @classmethod
    def distribution(
        cls, N: int, probabilities: Sequence[float]
    ) -> "TheoryInitialization":
        values = np.asarray(probabilities, dtype=float)
        if values.shape != (N + 1,):
            raise ValueError(f"initialization.probabilities must have length {N + 1}")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                "initialization.probabilities must be finite and non-negative"
            )
        if not np.isclose(values.sum(), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("initialization.probabilities must sum to one")
        return cls("distribution", tuple(float(value) for value in values))

    def as_array(self) -> np.ndarray:
        return np.asarray(self.probabilities, dtype=float)


@dataclass(frozen=True, slots=True)
class TheorySimulationConfig:
    """Complete provider-free simulation configuration."""

    parameters: TheoryParameters
    rounds: int
    episodes: int
    seed: int
    initialization: TheoryInitialization
    validation_samples_per_state: int = 10_000
    record_cycles: bool = False
    record_microsteps: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.rounds, bool) or self.rounds < 1:
            raise ValueError("simulation.rounds must be a positive integer")
        if isinstance(self.episodes, bool) or self.episodes < 1:
            raise ValueError("simulation.episodes must be a positive integer")
        if isinstance(self.seed, bool):
            raise ValueError("simulation.seed must be an integer")
        if (
            isinstance(self.validation_samples_per_state, bool)
            or self.validation_samples_per_state < 1
        ):
            raise ValueError(
                "simulation.validation_samples_per_state must be a positive integer"
            )
        if len(self.initialization.probabilities) != self.parameters.N + 1:
            raise ValueError("initialization does not match theory.N")

    def to_dict(self) -> dict[str, Any]:
        initialization: dict[str, Any]
        if self.initialization.type == "fixed_count":
            initialization = {
                "type": "fixed_count",
                "n0": int(np.argmax(self.initialization.as_array())),
            }
        elif self.initialization.type == "binomial":
            # Preserve the resolved distribution.  Recovering x0 from floating
            # values would add an unnecessary second representation.
            initialization = {
                "type": "distribution",
                "source": "binomial",
                "probabilities": list(self.initialization.probabilities),
            }
        else:
            initialization = {
                "type": "distribution",
                "probabilities": list(self.initialization.probabilities),
            }
        return {
            "simulation": {
                "type": "single_affinity_theory",
                "seed": self.seed,
                "episodes": self.episodes,
                "rounds": self.rounds,
                "validation_samples_per_state": self.validation_samples_per_state,
            },
            "theory": asdict(self.parameters),
            "initialization": initialization,
            "artifacts": {
                "record_cycles": self.record_cycles,
                "record_microsteps": self.record_microsteps,
            },
        }


@dataclass(frozen=True, slots=True)
class ControlledMicrostepRecord:
    episode_id: int
    round_index: int
    controlled_step: int
    n_before: int
    n_after: int
    delta_n: int


@dataclass(frozen=True, slots=True)
class TheoryCycleRecord:
    episode_id: int
    round_index: int
    n_before: int
    y: int
    action: int
    advocacy_probability_given_y: float
    advocacy_probability_given_n: float
    n_after: int
    current: int


@dataclass(frozen=True, slots=True)
class TheoryEpisode:
    episode_id: int
    initial_count: int
    final_count: int
    cycles: tuple[TheoryCycleRecord, ...]
    microsteps: tuple[ControlledMicrostepRecord, ...]

    @property
    def terminal_current(self) -> int:
        return self.final_count - self.initial_count


@dataclass(frozen=True, slots=True)
class TheorySimulationResult:
    config: TheorySimulationConfig
    occupancy_counts: np.ndarray
    sensor_counts_by_round: np.ndarray
    transition_counts: np.ndarray
    action_transition_counts: np.ndarray
    terminal_currents: np.ndarray
    cycle_records: tuple[TheoryCycleRecord, ...]
    microstep_records: tuple[ControlledMicrostepRecord, ...]
    state_validation: tuple[dict[str, Any], ...]
    validation_summary: Mapping[str, Any]
    occupancy_rows: tuple[dict[str, Any], ...]
    horizon_summary: Mapping[str, Any]

    @property
    def occupancy(self) -> np.ndarray:
        return self.occupancy_counts / self.config.episodes


_TOP_LEVEL_KEYS = {"simulation", "theory", "initialization", "artifacts"}
_THEORY_KEYS = {"N", "q_c", "b", "beta", "theta", "h", "gamma"}
_SIMULATION_KEYS = {
    "type",
    "seed",
    "episodes",
    "rounds",
    "validation_samples_per_state",
}
_ARTIFACT_KEYS = {"record_cycles", "record_microsteps"}


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _reject_unknown(values: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"{name} has unknown keys: {sorted(unknown)}")


def theory_simulation_config_from_mapping(
    payload: Mapping[str, Any],
) -> TheorySimulationConfig:
    """Validate and resolve a dedicated theory-simulation mapping."""

    _reject_unknown(payload, _TOP_LEVEL_KEYS, "theory simulation config")
    simulation = _mapping(payload.get("simulation"), "simulation")
    theory = _mapping(payload.get("theory"), "theory")
    initialization = _mapping(payload.get("initialization"), "initialization")
    artifacts = _mapping(payload.get("artifacts"), "artifacts")
    _reject_unknown(simulation, _SIMULATION_KEYS, "simulation")
    _reject_unknown(theory, _THEORY_KEYS, "theory")
    _reject_unknown(artifacts, _ARTIFACT_KEYS, "artifacts")

    simulation_type = simulation.get("type", "single_affinity_theory")
    if simulation_type != "single_affinity_theory":
        raise ValueError("simulation.type must be 'single_affinity_theory'")
    missing = _THEORY_KEYS - set(theory)
    if missing:
        raise ValueError(f"theory is missing required keys: {sorted(missing)}")
    parameters = TheoryParameters(
        N=int(theory["N"]),
        q_c=int(theory["q_c"]),
        b=int(theory["b"]),
        beta=float(theory["beta"]),
        theta=float(theory["theta"]),
        h=float(theory["h"]),
        gamma=float(theory["gamma"]),
    )

    kind = str(initialization.get("type", "fixed_count"))
    if kind == "fixed_count":
        _reject_unknown(initialization, {"type", "n0"}, "initialization")
        if "n0" not in initialization:
            raise ValueError("fixed_count initialization requires n0")
        initial = TheoryInitialization.fixed_count(
            parameters.N, int(initialization["n0"])
        )
    elif kind == "binomial":
        _reject_unknown(initialization, {"type", "x0"}, "initialization")
        if "x0" not in initialization:
            raise ValueError("binomial initialization requires x0")
        initial = TheoryInitialization.binomial(
            parameters.N, float(initialization["x0"])
        )
    elif kind == "distribution":
        _reject_unknown(
            initialization, {"type", "probabilities", "source"}, "initialization"
        )
        if "probabilities" not in initialization:
            raise ValueError("distribution initialization requires probabilities")
        initial = TheoryInitialization.distribution(
            parameters.N, initialization["probabilities"]
        )
    else:
        raise ValueError(
            "initialization.type must be fixed_count, binomial, or distribution"
        )

    return TheorySimulationConfig(
        parameters=parameters,
        rounds=int(simulation.get("rounds", 10)),
        episodes=int(simulation.get("episodes", 100_000)),
        seed=int(simulation.get("seed", 20260828)),
        initialization=initial,
        validation_samples_per_state=int(
            simulation.get("validation_samples_per_state", 10_000)
        ),
        record_cycles=bool(artifacts.get("record_cycles", False)),
        record_microsteps=bool(artifacts.get("record_microsteps", False)),
    )


def load_theory_simulation_config(path: str | Path) -> TheorySimulationConfig:
    """Read one strict, provider-free YAML simulation configuration."""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"theory simulation config {source} is not a mapping")
    return theory_simulation_config_from_mapping(payload)


def sample_initial_count(
    initialization: TheoryInitialization, rng: np.random.Generator
) -> int:
    """Sample ``n_0`` from the configured explicit distribution."""

    probabilities = initialization.as_array()
    return int(rng.choice(len(probabilities), p=probabilities))


def _sample_row(row: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(row.size, p=row))


def simulate_controlled_opportunity(
    n: int, reference: SingleAffinityReference, rng: np.random.Generator
) -> int:
    """Draw one next count from the authoritative microscopic kernel ``K``."""

    if isinstance(n, bool) or not 0 <= n <= reference.parameters.N:
        raise ValueError("n must lie in [0,N]")
    return _sample_row(reference.K[n], rng)


def simulate_action(
    n: int,
    action: int,
    reference: SingleAffinityReference,
    rng: np.random.Generator,
    *,
    episode_id: int = 0,
    round_index: int = 0,
    record_microsteps: bool = False,
) -> tuple[int, tuple[ControlledMicrostepRecord, ...]]:
    """Apply ``Q0=I`` or realize ``Q1=K^b`` as sequential ``K`` draws."""

    if action not in (0, 1):
        raise ValueError("action must be 0 (NoOp) or 1 (advocacy)")
    if not 0 <= n <= reference.parameters.N:
        raise ValueError("n must lie in [0,N]")
    if action == 0:
        return n, ()
    records: list[ControlledMicrostepRecord] = []
    current = int(n)
    for controlled_step in range(reference.parameters.b):
        before = current
        current = simulate_controlled_opportunity(current, reference, rng)
        if record_microsteps:
            records.append(
                ControlledMicrostepRecord(
                    episode_id=episode_id,
                    round_index=round_index,
                    controlled_step=controlled_step,
                    n_before=before,
                    n_after=current,
                    delta_n=current - before,
                )
            )
    return current, tuple(records)


def simulate_cycle(
    n: int,
    reference: SingleAffinityReference,
    rng: np.random.Generator,
    *,
    episode_id: int = 0,
    round_index: int = 0,
    record_microsteps: bool = False,
) -> tuple[TheoryCycleRecord, tuple[ControlledMicrostepRecord, ...]]:
    """Realize one explicit causal cycle ``n -> Y -> U -> n'``."""

    if not 0 <= n <= reference.parameters.N:
        raise ValueError("n must lie in [0,N]")
    y = _sample_row(reference.S[n], rng)
    action = int(rng.random() < reference.pi1[y])
    n_after, microsteps = simulate_action(
        n,
        action,
        reference,
        rng,
        episode_id=episode_id,
        round_index=round_index,
        record_microsteps=record_microsteps,
    )
    return (
        TheoryCycleRecord(
            episode_id=episode_id,
            round_index=round_index,
            n_before=n,
            y=y,
            action=action,
            advocacy_probability_given_y=float(reference.pi1[y]),
            advocacy_probability_given_n=float(reference.advocacy[n]),
            n_after=n_after,
            current=n_after - n,
        ),
        microsteps,
    )


def simulate_episode(
    episode_id: int,
    config: TheorySimulationConfig,
    reference: SingleAffinityReference,
    rng: np.random.Generator,
) -> TheoryEpisode:
    """Simulate one finite trajectory while retaining only this episode."""

    n = sample_initial_count(config.initialization, rng)
    initial = n
    cycles: list[TheoryCycleRecord] = []
    microsteps: list[ControlledMicrostepRecord] = []
    for round_index in range(config.rounds):
        cycle, microscopic = simulate_cycle(
            n,
            reference,
            rng,
            episode_id=episode_id,
            round_index=round_index,
            record_microsteps=config.record_microsteps,
        )
        cycles.append(cycle)
        microsteps.extend(microscopic)
        n = cycle.n_after
    return TheoryEpisode(
        episode_id=episode_id,
        initial_count=initial,
        final_count=n,
        cycles=tuple(cycles),
        microsteps=tuple(microsteps),
    )


def _information_bits(counts: np.ndarray) -> float:
    if counts.sum() <= 0:
        return math.nan
    return float(mutual_information_from_counts(counts).unsmoothed)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else math.nan


def _binomial_standard_error(probability: float, samples: int) -> float:
    if samples <= 0:
        return math.nan
    return math.sqrt(max(0.0, probability * (1.0 - probability)) / samples)


def _state_local_validation(
    reference: SingleAffinityReference,
    rng: np.random.Generator,
    samples: int,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Run fixed-state checks of S, pi/a, K, Q1, Q0, chi, T_pi and Q_pi."""

    N = reference.parameters.N
    rows: list[dict[str, Any]] = []
    sensor_errors: list[float] = []
    policy_y_success = np.zeros(reference.parameters.q_c + 1, dtype=np.int64)
    policy_y_total = np.zeros(reference.parameters.q_c + 1, dtype=np.int64)
    k_errors: list[float] = []
    q1_errors: list[float] = []
    closed_errors: list[float] = []

    for n in range(N + 1):
        sensor = np.zeros(reference.parameters.q_c + 1, dtype=np.int64)
        action_outcome = np.zeros((2, N + 1), dtype=np.int64)
        one_step = np.zeros(N + 1, dtype=np.int64)
        advocacy_outcome = np.zeros(N + 1, dtype=np.int64)
        action_sum = 0
        advocacy_after_sum = 0.0
        advocacy_after_square_sum = 0.0

        for _ in range(samples):
            cycle, _ = simulate_cycle(n, reference, rng)
            sensor[cycle.y] += 1
            policy_y_total[cycle.y] += 1
            policy_y_success[cycle.y] += cycle.action
            action_sum += cycle.action
            action_outcome[cycle.action, cycle.n_after] += 1

            one_step[simulate_controlled_opportunity(n, reference, rng)] += 1
            after, _ = simulate_action(n, 1, reference, rng)
            advocacy_outcome[after] += 1
            advocacy_after_sum += after
            advocacy_after_square_sum += after * after

        sensor_mc = sensor / samples
        a_mc = action_sum / samples
        k_mc = one_step / samples
        q1_mc = advocacy_outcome / samples
        closed_mc = action_outcome.sum(axis=0) / samples
        advocacy_after_mean = advocacy_after_sum / samples
        chi_mc = advocacy_after_mean / N - n / N
        advocacy_after_variance = max(
            0.0,
            advocacy_after_square_sum / samples - advocacy_after_mean**2,
        )
        chi_standard_error = math.sqrt(advocacy_after_variance / samples) / N
        t_mc = _information_bits(action_outcome)
        lower_mc = float(information_response_lower_bound([a_mc], [chi_mc])[0])
        eta_mc = _safe_ratio(lower_mc, t_mc)
        sensor_error = float(np.max(np.abs(sensor_mc - reference.S[n])))
        k_error = float(np.max(np.abs(k_mc - reference.K[n])))
        q1_error = float(np.max(np.abs(q1_mc - reference.Q1[n])))
        closed_error = float(
            np.max(np.abs(closed_mc - reference.closed_loop_kernel[n]))
        )
        sensor_errors.append(sensor_error)
        k_errors.append(k_error)
        q1_errors.append(q1_error)
        closed_errors.append(closed_error)
        rows.append(
            {
                "n": n,
                "x": n / N,
                "samples": samples,
                "a_exact": float(reference.advocacy[n]),
                "a_mc": a_mc,
                "a_difference": a_mc - float(reference.advocacy[n]),
                "a_mc_standard_error": _binomial_standard_error(a_mc, samples),
                "chi_exact": float(reference.chi[n]),
                "chi_mc": chi_mc,
                "chi_difference": chi_mc - float(reference.chi[n]),
                "chi_mc_standard_error": chi_standard_error,
                "T_pi_exact_bits": float(reference.T_pi[n]),
                "T_pi_mc_bits": t_mc,
                "T_pi_difference_bits": t_mc - float(reference.T_pi[n]),
                "T_pi_entropy_ceiling_bits": float(reference.entropy_ceiling()[n]),
                "eta_IR_exact": float(reference.eta_IR[n]),
                "eta_IR_mc": eta_mc,
                "sensor_max_abs_error": sensor_error,
                "sensor_max_standard_error": max(
                    _binomial_standard_error(float(value), samples)
                    for value in sensor_mc
                ),
                "K_max_abs_error": k_error,
                "K_max_standard_error": max(
                    _binomial_standard_error(float(value), samples) for value in k_mc
                ),
                "Q1_max_abs_error": q1_error,
                "Q1_max_standard_error": max(
                    _binomial_standard_error(float(value), samples) for value in q1_mc
                ),
                "closed_loop_max_abs_error": closed_error,
                "closed_loop_max_standard_error": max(
                    _binomial_standard_error(float(value), samples)
                    for value in closed_mc
                ),
                "noop_exact": True,
            }
        )

    policy_mc = np.divide(
        policy_y_success,
        policy_y_total,
        out=np.full(policy_y_success.shape, np.nan, dtype=float),
        where=policy_y_total > 0,
    )
    policy_mask = policy_y_total > 0
    policy_error = float(
        np.max(np.abs(policy_mc[policy_mask] - reference.pi1[policy_mask]))
    )
    summary = {
        "samples_per_state": samples,
        "sensor_max_abs_error": max(sensor_errors),
        "policy_given_y_max_abs_error": policy_error,
        "policy_given_y": [
            {
                "y": y,
                "samples": int(policy_y_total[y]),
                "probability_exact": float(reference.pi1[y]),
                "probability_mc": float(policy_mc[y]),
                "difference": float(policy_mc[y] - reference.pi1[y]),
                "standard_error": _binomial_standard_error(
                    float(policy_mc[y]), int(policy_y_total[y])
                ),
            }
            for y in range(reference.parameters.q_c + 1)
        ],
        "K_max_abs_error": max(k_errors),
        "Q0_exact": True,
        "Q1_max_abs_error": max(q1_errors),
        "closed_loop_max_abs_error": max(closed_errors),
        "chi_max_abs_error": max(abs(float(row["chi_difference"])) for row in rows),
        "T_pi_max_abs_error_bits": max(
            abs(float(row["T_pi_difference_bits"])) for row in rows
        ),
        "T_pi_entropy_ceiling_satisfied_mc": all(
            float(row["T_pi_mc_bits"])
            <= float(row["T_pi_entropy_ceiling_bits"]) + 1e-12
            for row in rows
        ),
    }
    return tuple(rows), summary


def _summaries(
    config: TheorySimulationConfig,
    reference: SingleAffinityReference,
    occupancy_counts: np.ndarray,
    sensor_counts_by_round: np.ndarray,
    transition_counts: np.ndarray,
    terminal_currents: np.ndarray,
    state_rows: tuple[dict[str, Any], ...],
    local_summary: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any], dict[str, Any]]:
    episodes = config.episodes
    p_mc = occupancy_counts / episodes
    exact_horizon = finite_horizon_thermodynamics(
        reference, config.initialization.as_array(), config.rounds
    )
    p_exact = exact_horizon.p_history
    occupancy_rows = tuple(
        {
            "round": round_index,
            "n": n,
            "count": int(occupancy_counts[round_index, n]),
            "probability": float(p_mc[round_index, n]),
            "exact_probability": float(p_exact[round_index, n]),
            "difference": float(p_mc[round_index, n] - p_exact[round_index, n]),
            "standard_error": _binomial_standard_error(
                float(p_mc[round_index, n]), episodes
            ),
        }
        for round_index in range(config.rounds + 1)
        for n in range(config.parameters.N + 1)
    )

    mc_cycle_rows: list[dict[str, Any]] = []
    total_j_mc = 0.0
    total_i_mc = 0.0
    total_sigma_mc = 0.0
    for k in range(config.rounds):
        current_values = (
            np.arange(config.parameters.N + 1)[None, :]
            - np.arange(config.parameters.N + 1)[:, None]
        )
        j_mc = float(np.sum(transition_counts[k] * current_values) / episodes)
        i_mc_bits = _information_bits(sensor_counts_by_round[k])
        i_mc_nats = i_mc_bits * math.log(2.0)
        delta_s_mc = system_entropy(p_mc[k + 1], config.parameters.N) - system_entropy(
            p_mc[k], config.parameters.N
        )
        sigma_mc = delta_s_mc + config.parameters.h * j_mc + i_mc_nats
        eta_mc, c_mc, bounded_mc = thermodynamic_efficiency(
            h=config.parameters.h, J_c=j_mc, I_sens_nats=i_mc_nats
        )
        exact = exact_horizon.cycles[k]
        mc_cycle_rows.append(
            {
                "round": k,
                "J_exact": exact.J_c,
                "J_mc": j_mc,
                "I_sens_exact_nats": exact.I_sens_nats,
                "I_sens_mc_nats": i_mc_nats,
                "delta_S_sys_exact_nats": exact.delta_S_sys_nats,
                "delta_S_sys_mc_nats": delta_s_mc,
                "Sigma_exact_nats": exact.Sigma_nats,
                "Sigma_mc_identity_nats": sigma_mc,
                "C_th_exact_nats": exact.C_th_nats,
                "C_th_mc_nats": c_mc,
                "eta_th_exact": exact.eta_th,
                "eta_th_mc": eta_mc,
                "eta_th_mc_bounded": bounded_mc,
            }
        )
        total_j_mc += j_mc
        total_i_mc += i_mc_nats
        total_sigma_mc += sigma_mc

    moment_exact = finite_horizon_current_moments(
        reference.closed_loop_kernel,
        config.initialization.as_array(),
        config.rounds,
    )
    current_mean_mc = float(np.mean(terminal_currents))
    current_variance_mc = float(np.var(terminal_currents))
    current_std_mc = math.sqrt(current_variance_mc)
    current_mean_standard_error_mc = current_std_mc / math.sqrt(episodes)
    snr2_mc = (
        current_mean_mc**2 / current_variance_mc
        if current_variance_mc > 0.0
        else math.nan
    )
    total_delta_mc = system_entropy(p_mc[-1], config.parameters.N) - system_entropy(
        p_mc[0], config.parameters.N
    )
    eta_th_mc, c_th_mc, eta_th_bounded = thermodynamic_efficiency(
        h=config.parameters.h, J_c=total_j_mc, I_sens_nats=total_i_mc
    )

    local_a = np.asarray([float(row["a_mc"]) for row in state_rows])
    local_chi = np.asarray([float(row["chi_mc"]) for row in state_rows])
    local_t = np.asarray([float(row["T_pi_mc_bits"]) for row in state_rows])
    local_bound = information_response_lower_bound(local_a, local_chi)
    exact_numerator = 0.0
    exact_denominator = 0.0
    mc_occupancy_exact_local_numerator = 0.0
    mc_occupancy_exact_local_denominator = 0.0
    mc_numerator = 0.0
    mc_denominator = 0.0
    for k in range(config.rounds):
        exact_numerator += float(p_exact[k] @ reference.pinsker_bound)
        exact_denominator += float(p_exact[k] @ reference.T_pi)
        mc_occupancy_exact_local_numerator += float(p_mc[k] @ reference.pinsker_bound)
        mc_occupancy_exact_local_denominator += float(p_mc[k] @ reference.T_pi)
        mc_numerator += float(p_mc[k] @ local_bound)
        mc_denominator += float(p_mc[k] @ local_t)

    horizon_summary = {
        "H": config.rounds,
        "episodes": episodes,
        "J_exact": exact_horizon.total_J_c,
        "J_mc": total_j_mc,
        "current_terminal_mean_exact": moment_exact["mean"],
        "current_terminal_mean_mc": current_mean_mc,
        "current_terminal_mean_mc_standard_error": current_mean_standard_error_mc,
        "Var_J_exact": moment_exact["variance"],
        "Var_J_mc": current_variance_mc,
        "Std_J_mc": current_std_mc,
        "SNR_squared_mc": snr2_mc,
        "I_sens_exact_nats": exact_horizon.total_I_sens_nats,
        "I_sens_mc_nats": total_i_mc,
        "delta_S_sys_exact_nats": exact_horizon.total_delta_S_sys_nats,
        "delta_S_sys_mc_nats": total_delta_mc,
        "Sigma_exact_nats": exact_horizon.total_Sigma_nats,
        "Sigma_mc_identity_nats": total_sigma_mc,
        "Sigma_mc_telescope_nats": (
            total_delta_mc + config.parameters.h * total_j_mc + total_i_mc
        ),
        "thermodynamic_identity_residual_mc_nats": total_sigma_mc
        - (total_delta_mc + config.parameters.h * total_j_mc + total_i_mc),
        "C_th_exact_nats": exact_horizon.total_C_th_nats,
        "C_th_mc_nats": c_th_mc,
        "eta_IR_exact": _safe_ratio(exact_numerator, exact_denominator),
        "eta_IR_mc_occupancy_exact_local": _safe_ratio(
            mc_occupancy_exact_local_numerator,
            mc_occupancy_exact_local_denominator,
        ),
        "eta_IR_mc": _safe_ratio(mc_numerator, mc_denominator),
        "eta_IR_exact_numerator_bits": exact_numerator,
        "eta_IR_exact_denominator_bits": exact_denominator,
        "eta_IR_mc_occupancy_exact_local_numerator_bits": (
            mc_occupancy_exact_local_numerator
        ),
        "eta_IR_mc_occupancy_exact_local_denominator_bits": (
            mc_occupancy_exact_local_denominator
        ),
        "eta_IR_mc_numerator_bits": mc_numerator,
        "eta_IR_mc_denominator_bits": mc_denominator,
        "eta_th_exact": exact_horizon.eta_th,
        "eta_th_mc": eta_th_mc,
        "eta_th_mc_bounded": eta_th_bounded,
        "T_pi_units": "bits",
        "thermodynamic_information_units": "nats",
        "cycle_summaries": mc_cycle_rows,
    }
    validation_summary = {
        **local_summary,
        "occupancy_max_abs_error": float(np.max(np.abs(p_mc - p_exact))),
        "terminal_current_mean_abs_error": abs(current_mean_mc - moment_exact["mean"]),
        "terminal_current_variance_abs_error": abs(
            current_variance_mc - moment_exact["variance"]
        ),
        "finite_horizon_J_abs_error": abs(total_j_mc - exact_horizon.total_J_c),
        "finite_horizon_I_sens_abs_error_nats": abs(
            total_i_mc - exact_horizon.total_I_sens_nats
        ),
        "finite_horizon_Sigma_abs_error_nats": abs(
            total_sigma_mc - exact_horizon.total_Sigma_nats
        ),
    }
    return occupancy_rows, horizon_summary, validation_summary


def simulate_ensemble(config: TheorySimulationConfig) -> TheorySimulationResult:
    """Run reproducible trajectories plus fixed-state exact-kernel validation."""

    reference = single_affinity_reference(config.parameters)
    rng = np.random.default_rng(config.seed)
    N = config.parameters.N
    occupancy_counts = np.zeros((config.rounds + 1, N + 1), dtype=np.int64)
    sensor_counts_by_round = np.zeros(
        (config.rounds, N + 1, config.parameters.q_c + 1), dtype=np.int64
    )
    transition_counts = np.zeros((config.rounds, N + 1, N + 1), dtype=np.int64)
    action_transition_counts = np.zeros((N + 1, 2, N + 1), dtype=np.int64)
    terminal_currents = np.zeros(config.episodes, dtype=np.int64)
    cycle_records: list[TheoryCycleRecord] = []
    microstep_records: list[ControlledMicrostepRecord] = []

    for episode_id in range(config.episodes):
        episode = simulate_episode(episode_id, config, reference, rng)
        occupancy_counts[0, episode.initial_count] += 1
        for cycle in episode.cycles:
            occupancy_counts[cycle.round_index + 1, cycle.n_after] += 1
            sensor_counts_by_round[cycle.round_index, cycle.n_before, cycle.y] += 1
            transition_counts[cycle.round_index, cycle.n_before, cycle.n_after] += 1
            action_transition_counts[cycle.n_before, cycle.action, cycle.n_after] += 1
        terminal_currents[episode_id] = episode.terminal_current
        if config.record_cycles:
            cycle_records.extend(episode.cycles)
        if config.record_microsteps:
            microstep_records.extend(episode.microsteps)

    state_rows, local_summary = _state_local_validation(
        reference, rng, config.validation_samples_per_state
    )
    occupancy_rows, horizon_summary, validation_summary = _summaries(
        config,
        reference,
        occupancy_counts,
        sensor_counts_by_round,
        transition_counts,
        terminal_currents,
        state_rows,
        local_summary,
    )
    return TheorySimulationResult(
        config=config,
        occupancy_counts=occupancy_counts,
        sensor_counts_by_round=sensor_counts_by_round,
        transition_counts=transition_counts,
        action_transition_counts=action_transition_counts,
        terminal_currents=terminal_currents,
        cycle_records=tuple(cycle_records),
        microstep_records=tuple(microstep_records),
        state_validation=state_rows,
        validation_summary=validation_summary,
        occupancy_rows=occupancy_rows,
        horizon_summary=horizon_summary,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n")


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_theory_simulation_artifacts(
    result: TheorySimulationResult, output_dir: str | Path
) -> Path:
    """Atomically publish one self-contained simulation result directory."""

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    resolved = result.config.to_dict()
    resolved_path = destination / "resolved_config.json"
    if resolved_path.exists():
        existing = json.loads(resolved_path.read_text(encoding="utf-8"))
        if existing != _json_safe(resolved):
            raise ValueError(
                f"output directory {destination} contains a different resolved config"
            )

    metadata = {
        "theory_reference": THEORY_REFERENCE,
        "theory_semantics_version": THEORY_SEMANTICS_VERSION,
        "theory_api_version": THEORY_API_VERSION,
        "theory_module": THEORY_MODULE,
        "seed": result.config.seed,
        "episodes": result.config.episodes,
        "rounds": result.config.rounds,
        "parameters": asdict(result.config.parameters),
        "causal_cycle": "n -> Y -> U -> n'",
        "Q0": "I",
        "Q1": "K^b sampled sequentially",
        "provider_free": True,
    }
    _atomic_json(resolved_path, resolved)
    _atomic_json(destination / "metadata.json", metadata)
    _atomic_csv(destination / "occupancy_by_round.csv", result.occupancy_rows)
    _atomic_csv(destination / "state_local_validation.csv", result.state_validation)
    horizon_row = {
        key: value
        for key, value in result.horizon_summary.items()
        if key != "cycle_summaries"
    }
    _atomic_csv(destination / "horizon_summary.csv", [horizon_row])
    _atomic_csv(
        destination / "thermodynamics_by_round.csv",
        result.horizon_summary["cycle_summaries"],
    )
    _atomic_json(destination / "validation_summary.json", result.validation_summary)

    if result.config.record_cycles:
        cycle_rows = [
            {
                **asdict(record),
                "seed": result.config.seed,
                "theory_reference": THEORY_REFERENCE,
                "theory_semantics_version": THEORY_SEMANTICS_VERSION,
                "theory_api_version": THEORY_API_VERSION,
            }
            for record in result.cycle_records
        ]
        _atomic_csv(destination / "cycle_trajectories.csv", cycle_rows)
    if result.config.record_microsteps:
        _atomic_csv(
            destination / "controlled_microsteps.csv",
            [asdict(record) for record in result.microstep_records],
        )
    return destination


def run_theory_simulation(
    config_path: str | Path, output_dir: str | Path
) -> TheorySimulationResult:
    """Load, simulate, validate, and publish one provider-free run."""

    config = load_theory_simulation_config(config_path)
    resolved_path = Path(output_dir).resolve() / "resolved_config.json"
    if resolved_path.exists():
        existing = json.loads(resolved_path.read_text(encoding="utf-8"))
        if existing != _json_safe(config.to_dict()):
            raise ValueError(
                f"output directory {resolved_path.parent} contains a different "
                "resolved config"
            )
    result = simulate_ensemble(config)
    write_theory_simulation_artifacts(result, output_dir)
    return result


__all__ = [
    "ControlledMicrostepRecord",
    "TheoryCycleRecord",
    "TheoryEpisode",
    "TheoryInitialization",
    "TheorySimulationConfig",
    "TheorySimulationResult",
    "load_theory_simulation_config",
    "run_theory_simulation",
    "sample_initial_count",
    "simulate_action",
    "simulate_controlled_opportunity",
    "simulate_cycle",
    "simulate_ensemble",
    "simulate_episode",
    "theory_simulation_config_from_mapping",
    "write_theory_simulation_artifacts",
]
