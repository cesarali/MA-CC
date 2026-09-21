"""Exact finite-state calibrated blackboard laws (row-stochastic convention).

These are model probabilities, not a second empirical MI estimator. No sampling,
parameter fitting, provider calls, or marginal-CI reconstruction occurs here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
import numpy as np

from mas_cc.games.relational_reasoning.imitation_round_feedback.theory_revised import (
    sensor_kernel,
    sensing_information_nats as sensing_information_nats,
)


def integer(value, name, minimum=0):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def stochastic(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if (
        matrix.ndim != 2
        or not np.isfinite(matrix).all()
        or (matrix < 0).any()
        or not np.allclose(matrix.sum(axis=1), 1, atol=1e-12, rtol=1e-12)
    ):
        raise ValueError("invalid row-stochastic kernel")
    return matrix


def microscopic_kernel(N: int, entry: float, exit: float) -> np.ndarray:
    N = integer(N, "N", 1)
    if (
        not all(math.isfinite(v) and 0 <= v <= 1 for v in (entry, exit))
        or entry + exit > 1
    ):
        raise ValueError("missing/invalid directional rates or compliance > 1")
    n = np.arange(N + 1)
    up, down = entry * (1 - n / N), exit * n / N
    result = np.diag(1 - up - down)
    result[n[:-1], n[1:]] = up[:-1]
    result[n[1:], n[:-1]] = down[1:]
    return stochastic(result)


def ordered_product(kernels: Sequence[np.ndarray]) -> np.ndarray:
    if not kernels:
        raise ValueError("ordered product requires at least one kernel")
    result = np.eye(len(kernels[0]))
    for kernel in kernels:
        result = result @ stochastic(kernel)
    return stochastic(result)


def round_kernels(N, M, silent_rates, active_rates, exposure_weight=None):
    """Without a weight active_rates is a directly calibrated action branch."""
    M = integer(M, "M")
    K_silent = microscopic_kernel(N, *silent_rates)
    K_active = microscopic_kernel(N, *active_rates)
    if exposure_weight is not None:
        if not math.isfinite(exposure_weight) or not 0 <= exposure_weight <= 1:
            raise ValueError("invalid exposure weight")
        K_active = (1 - exposure_weight) * K_silent + exposure_weight * K_active
    return (
        stochastic(np.linalg.matrix_power(K_silent, M)),
        stochastic(np.linalg.matrix_power(K_active, M)),
    )


def entropy(probabilities, axis=-1):
    probabilities = np.asarray(probabilities, dtype=float)
    logs = np.zeros_like(probabilities)
    np.log2(probabilities, out=logs, where=probabilities > 0)
    return -np.sum(probabilities * logs, axis=axis)


def policy_vector(values, N):
    result = np.asarray(values, dtype=float)
    if (
        result.shape != (N + 1,)
        or not np.isfinite(result).all()
        or ((result < 0) | (result > 1)).any()
    ):
        raise ValueError(
            "missing_future_policy_states: supply a complete probability table"
        )
    return result


def exact_sensor_policy(
    N, q_c, *, beta=None, theta=None, no_observation_probability=None
):
    integer(N, "N", 1)
    integer(q_c, "q_c")
    if q_c > N:
        raise ValueError("q_c exceeds population")
    if q_c == 0:
        if no_observation_probability is None:
            raise ValueError("q_c=0 requires no_observation_probability")
        pi = policy_vector([no_observation_probability] * (N + 1), N)[:1]
        S = np.ones((N + 1, 1))
    else:
        if (
            beta is None
            or theta is None
            or not math.isfinite(beta)
            or not math.isfinite(theta)
        ):
            raise ValueError("logistic policy requires finite beta and theta")
        S = sensor_kernel(N, q_c)
        z = beta * (theta - np.arange(q_c + 1) / q_c)
        pi = np.exp(-np.logaddexp(0, -z))
    return S, pi, S @ pi


def feedback_kernel(Q_silent, Q_active, policy):
    Q_silent, Q_active = stochastic(Q_silent), stochastic(Q_active)
    a = policy_vector(policy, len(Q_silent) - 1)
    return stochastic((1 - a[:, None]) * Q_silent + a[:, None] * Q_active)


def state_metrics(Q_silent, Q_active, policy):
    N = len(Q_silent) - 1
    P = feedback_kernel(Q_silent, Q_active, policy)
    a = np.asarray(policy)
    f = np.arange(N + 1) / N
    d0, d1 = Q_silent @ f - f, Q_active @ f - f
    chi = d1 - d0
    H = entropy(np.stack([1 - a, a], axis=1))
    H_next = entropy(P)
    H_next_action = (1 - a) * entropy(Q_silent) + a * entropy(Q_active)
    T = np.zeros(N + 1)
    for weight, Q in ((1 - a, Q_silent), (a, Q_active)):
        mask = (weight[:, None] > 0) & (Q > 0) & (P > 0)
        logs = np.zeros_like(Q)
        np.log2(np.divide(Q, P, out=np.ones_like(Q), where=mask), out=logs, where=mask)
        T += weight * np.sum(Q * logs, axis=1)
    B = 2 * a * (1 - a) * chi**2 / math.log(2)
    if not np.allclose(T, H_next - H_next_action, atol=1e-10):
        raise ValueError("outcome entropy identity failed")
    if ((B < -1e-10) | (B > T + 1e-10) | (T > H + 1e-10)).any():
        raise ValueError("information bounds failed")
    return dict(
        chi=chi,
        d_silent=d0,
        d_active=d1,
        T=T,
        H=H,
        B=B,
        H_next=H_next,
        H_next_action=H_next_action,
        J_silent=N * d0,
        J_excess=N * a * chi,
        J_total=N * ((1 - a) * d0 + a * d1),
    )


def joint_law(
    weights, counts, policies, silent_rows, active_rows, conditions, outcome_map=None
):
    """Push context mass through branch rows, then map conditioning/outcomes."""
    weights = np.asarray(weights, dtype=float)
    if (
        (weights < 0).any()
        or not np.isfinite(weights).all()
        or not np.isclose(weights.sum(), 1)
    ):
        raise ValueError("context weights must sum to one")
    a = policy_vector(policies, len(weights) - 1)
    q0, q1 = stochastic(silent_rows), stochastic(active_rows)
    if (
        len(counts) != len(weights)
        or len(conditions) != len(weights)
        or q0.shape != q1.shape
        or len(q0) != len(weights)
    ):
        raise ValueError("context shapes disagree")
    mapping = list(range(q0.shape[1])) if outcome_map is None else list(outcome_map)
    if len(mapping) != q0.shape[1]:
        raise ValueError("outcome map shape disagrees")
    cs = list(dict.fromkeys(conditions))
    ys = list(dict.fromkeys(mapping))
    ci, yi = {c: i for i, c in enumerate(cs)}, {y: i for i, y in enumerate(ys)}
    J = np.zeros((len(cs), 2, len(ys)))
    for i, c in enumerate(conditions):
        for m, y in enumerate(mapping):
            J[ci[c], 0, yi[y]] += weights[i] * (1 - a[i]) * q0[i, m]
            J[ci[c], 1, yi[y]] += weights[i] * a[i] * q1[i, m]
    return J


def joint_information(J):
    J = np.asarray(J, dtype=float)
    if (
        J.ndim != 3
        or (J < 0).any()
        or not np.isfinite(J).all()
        or not np.isclose(J.sum(), 1)
    ):
        raise ValueError("invalid conditional joint law")
    c = J.sum(axis=(1, 2))
    cu, cy = J.sum(axis=2), J.sum(axis=1)
    denominator = cu[:, :, None] * cy[:, None, :]
    ratio = np.divide(
        J * c[:, None, None], denominator, out=np.ones_like(J), where=denominator > 0
    )
    logs = np.zeros_like(J)
    np.log2(ratio, out=logs, where=J > 0)
    T = float(np.sum(J * logs))
    H = float(entropy(cu.ravel()) - entropy(c))
    return dict(T=T, H=H, H_action=float(entropy(cu.sum(axis=0))))


def lag_contrast(Q_silent, Q_active, policy, lag):
    integer(lag, "lag", 1)
    f = np.arange(len(Q_silent)) / (len(Q_silent) - 1)
    if lag == 1:
        return (Q_active - Q_silent) @ f
    P = feedback_kernel(Q_silent, Q_active, policy)
    return (Q_active - Q_silent) @ np.linalg.matrix_power(P, lag - 1) @ f


def forward_ensemble(Q_silent, Q_active, policy, initial, horizon):
    integer(horizon, "horizon")
    omega = np.asarray(initial, dtype=float)
    if (
        omega.shape != (len(Q_silent),)
        or not np.isfinite(omega).all()
        or (omega < 0).any()
        or not np.isclose(omega.sum(), 1)
    ):
        raise ValueError("invalid initial occupancy")
    P = feedback_kernel(Q_silent, Q_active, policy)
    state = state_metrics(Q_silent, Q_active, policy)
    trajectory, components = [omega.copy()], []
    for _ in range(horizon):
        components.append({key: float(omega @ value) for key, value in state.items()})
        omega = omega @ P
        trajectory.append(omega.copy())
    current = float((omega - trajectory[0]) @ np.arange(len(omega)))
    if not np.isclose(
        sum(r["J_total"] for r in components), current, atol=1e-10, rtol=1e-10
    ):
        raise ValueError("forward current does not telescope")
    return np.asarray(trajectory), components, current


def component_ratios(T, H, B):
    return dict(
        eta_IF=T / H if H > 0 else math.nan, eta_IR=B / T if T > 0 else math.nan
    )
