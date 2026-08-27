"""Single-affinity finite-compliance feedback theory for the round-feedback clock.

This module is the code counterpart of the revised single-affinity theory report
and the current ICLR paper theory.  It deliberately keeps the same broad shape
as the previous ``theory.py`` (parameters -> sensing/policy -> kernels -> local
information -> assembled reference -> finite-time analysis), but the population
response model is different:

    population count n -> finite sensor Y -> stochastic action U
        -> isolated controlled layer -> next count m

The controlled microscopic channel has one directional affinity ``h`` and one
kinetic compliance ``gamma``.  For a target count ``n`` in a population of size
``N``:

    K(n+1|n) = gamma * (N-n)/N * sigma(h)
    K(n-1|n) = gamma * n/N     * sigma(-h)

The action-conditioned one-cycle kernels are therefore

    Q0 = I                       (NO_OP)
    Q1 = K**b                    (ADVOCATE)

which is the *isolated controlled layer* used by the revised report.  Ordinary
q=1 voter updates may be interspersed in the physical runtime, but they are
mean-neutral and are not part of Q0/Q1 in this reference channel.

The main objects exposed directly by :class:`SingleAffinityReference` are:

``chi``
    Exact controller-induced susceptibility
    ``chi_h,gamma(x) = [sigma(h)-x] * Lambda_b,gamma``.

``T_pi``
    Exact state-local action-to-population information
    ``I(U_k; n_{k+1} | n_k=n)`` in bits.

``eta_IR``
    Bounded information-response efficiency obtained from the Pinsker bound.

For a population ensemble ``p_k``, :meth:`SingleAffinityReference.one_cycle`
returns the finite-time thermodynamic objects:

``J_c``
    Mean controlled target-count current.
``I_sens_nats``
    Sensing information ``I(n_k;Y_k)``.
``delta_S_sys_nats``
    Finite-time coarse system-entropy change.
``Sigma_nats``
    Path irreversibility ``D_KL(P_F || P_R)``.
``C_th_nats``
    Non-storage control expenditure ``Sigma-delta_S = h J_c + I_sens``.
``eta_th``
    Thermodynamic control efficiency ``h J_c / C_th`` when defined.

The exact finite-time identity checked by this module is

    delta_S_sys + h*J_c + I_sens = Sigma >= 0.

All thermodynamic logarithms are natural logarithms (nats).  ``T_pi`` and the
information-response quantities remain in bits, matching the empirical action
channel convention.

This is a deterministic finite-state reference: no Monte Carlo simulation and
no statistical estimation is performed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np


THEORY_REFERENCE = "single_affinity_revised"
THEORY_SEMANTICS_VERSION = "single_affinity_v1"
THEORY_API_VERSION = "1.0"
THEORY_MODULE = (
    "mas_cc.games.relational_reasoning.imitation_round_feedback.theory_revised"
)


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def logistic(z: float) -> float:
    """Numerically stable logistic function ``sigma(z)``."""

    z = float(z)
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def binary_entropy_bits(probability: float) -> float:
    """Binary entropy ``h2(a)`` in bits, the exact ceiling on ``T_pi(n)``."""

    a = float(probability)
    if not 0.0 < a < 1.0:
        return 0.0
    return -a * math.log2(a) - (1.0 - a) * math.log2(1.0 - a)


def _validate_probability_vector(
    values: Sequence[float], *, size: int, name: str
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    if not np.isclose(array.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} must sum to one")
    return array


def _relative_entropy_bits(p: np.ndarray, q: np.ndarray) -> float:
    """``sum p log2(p/q)`` with the convention ``0 log 0 = 0``."""

    total = 0.0
    for value, reference in zip(p, q, strict=True):
        if value <= 0.0:
            continue
        if reference <= 0.0:
            return math.inf
        total += float(value) * math.log2(float(value) / float(reference))
    return total


def _log_binomial(N: int, n: int) -> float:
    """``ln binom(N,n)`` without materializing a potentially huge integer."""

    return math.lgamma(N + 1.0) - math.lgamma(n + 1.0) - math.lgamma(N - n + 1.0)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TheoryParameters:
    """Parameters of the single-affinity finite-compliance reference.

    The revised theory is the exact ``q=1`` controlled layer, so ``q`` is not
    a free parameter.  A read-only :attr:`q` property is retained for output
    compatibility with the previous matched-q-voter theory.
    """

    N: int
    q_c: int
    b: int
    beta: float
    theta: float
    h: float
    gamma: float

    def __post_init__(self) -> None:
        if isinstance(self.N, bool) or self.N < 2:
            raise ValueError("theory requires N >= 2")
        if isinstance(self.q_c, bool) or not 1 <= self.q_c <= self.N:
            raise ValueError("q_c must lie between 1 and N")
        if isinstance(self.b, bool) or not 0 <= self.b <= self.N:
            raise ValueError("b must lie between 0 and N")
        if not math.isfinite(float(self.beta)) or not math.isfinite(float(self.theta)):
            raise ValueError("beta and theta must be finite")
        if not math.isfinite(float(self.h)):
            raise ValueError("finite-time thermodynamics requires finite h")
        if not math.isfinite(float(self.gamma)) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0,1]")

    @property
    def q(self) -> int:
        """The revised closed-form controlled layer is the ``q=1`` theory."""

        return 1

    @property
    def key(self) -> tuple[int, int, int, float, float, float, float]:
        """Stable cache key for one exact reference."""

        return (
            self.N,
            self.q_c,
            self.b,
            float(self.beta),
            float(self.theta),
            float(self.h),
            float(self.gamma),
        )

    @property
    def sensing_fraction(self) -> float:
        return self.q_c / self.N

    @property
    def actuation_fraction(self) -> float:
        return self.b / self.N

    @property
    def p_h(self) -> float:
        """Directional set point ``sigma(h)``."""

        return logistic(self.h)

    @property
    def Lambda_b_gamma(self) -> float:
        """Finite-compliance actuation factor ``1-(1-gamma/N)^b``."""

        return 1.0 - (1.0 - self.gamma / self.N) ** self.b

    def as_fields(self) -> dict[str, Any]:
        """Flat fields suitable for analysis tables."""

        return {
            "theory_mode": THEORY_REFERENCE,
            "theory_N": self.N,
            "theory_q": 1,
            "theory_qc": self.q_c,
            "theory_b": self.b,
            "theory_beta": float(self.beta),
            "theory_theta": float(self.theta),
            "theory_h": float(self.h),
            "theory_gamma": float(self.gamma),
            "theory_p_h": self.p_h,
            "theory_Lambda_b_gamma": self.Lambda_b_gamma,
            "theory_sensing_fraction": self.sensing_fraction,
            "theory_actuation_fraction": self.actuation_fraction,
        }


# ---------------------------------------------------------------------------
# 1. Controller sensing and policy
# ---------------------------------------------------------------------------


def sensor_law(N: int, n: int, q_c: int) -> np.ndarray:
    """Hypergeometric sensor law ``S(y|n)`` for ``y=0,...,q_c``."""

    if N < 1 or not 0 <= n <= N or not 1 <= q_c <= N:
        raise ValueError("require N>=1, 0<=n<=N, and 1<=q_c<=N")
    law = np.zeros(q_c + 1, dtype=float)
    total = math.comb(N, q_c)
    lower = max(0, q_c - (N - n))
    upper = min(q_c, n)
    for y in range(lower, upper + 1):
        law[y] = math.comb(n, y) * math.comb(N - n, q_c - y) / total
    return law


def sensor_kernel(N: int, q_c: int) -> np.ndarray:
    """Matrix ``S[n,y] = P(Y=y|n)`` for all population counts."""

    return np.asarray([sensor_law(N, n, q_c) for n in range(N + 1)], dtype=float)


def policy_advocacy_vector(q_c: int, beta: float, theta: float) -> np.ndarray:
    """``pi(1|y)=sigma(beta*(theta-y/q_c))`` for every sensor outcome."""

    if q_c < 1:
        raise ValueError("q_c must be at least one")
    return np.asarray(
        [logistic(float(beta) * (float(theta) - y / q_c)) for y in range(q_c + 1)],
        dtype=float,
    )


def advocacy_probability_curve(parameters: TheoryParameters) -> np.ndarray:
    """Sensor-averaged advocacy probability ``a_n=P(U=1|n)`` for ``n=0..N``."""

    S = sensor_kernel(parameters.N, parameters.q_c)
    pi1 = policy_advocacy_vector(parameters.q_c, parameters.beta, parameters.theta)
    return S @ pi1


# ---------------------------------------------------------------------------
# 2. Single-affinity microscopic actuation
# ---------------------------------------------------------------------------


def controlled_kernel(parameters: TheoryParameters) -> np.ndarray:
    """One controlled microscopic opportunity ``K_{h,gamma}``.

    The row-stochastic matrix is indexed by the target count ``n=0,...,N``.
    Both directions are present for finite ``h`` and ``gamma>0``, which is the
    microscopic reversibility required by the finite-time path identity.
    """

    N = parameters.N
    gamma = float(parameters.gamma)
    p_h = parameters.p_h
    K = np.zeros((N + 1, N + 1), dtype=float)
    for n in range(N + 1):
        up = gamma * (N - n) / N * p_h
        down = gamma * n / N * (1.0 - p_h)
        if n < N:
            K[n, n + 1] = up
        if n > 0:
            K[n, n - 1] = down
        K[n, n] = 1.0 - up - down
    return K


def noop_kernel(N: int) -> np.ndarray:
    """Isolated controlled-layer NoOp kernel ``Q0=I``."""

    return np.eye(N + 1, dtype=float)


def advocate_kernel(K: np.ndarray, b: int) -> np.ndarray:
    """Advocacy kernel ``Q1=K^b`` for exactly ``b`` controlled opportunities."""

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be square")
    if isinstance(b, bool) or b < 0:
        raise ValueError("b must be a non-negative integer")
    return np.linalg.matrix_power(np.asarray(K, dtype=float), int(b))


# ---------------------------------------------------------------------------
# 3. Exact susceptibility chi
# ---------------------------------------------------------------------------


def susceptibility(x: float, parameters: TheoryParameters) -> float:
    """Exact ``chi_{h,gamma}(x)`` for one initial target fraction ``x``."""

    x = float(x)
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must lie in [0,1]")
    return (parameters.p_h - x) * parameters.Lambda_b_gamma


def susceptibility_curve(parameters: TheoryParameters) -> np.ndarray:
    """``chi(n/N)`` evaluated at all finite population states ``n=0..N``."""

    shares = np.arange(parameters.N + 1, dtype=float) / parameters.N
    return (parameters.p_h - shares) * parameters.Lambda_b_gamma


def kernel_mean_response(Q0: np.ndarray, Q1: np.ndarray, N: int) -> np.ndarray:
    """Kernel check of ``E[x'|ADV,n]-E[x'|NOOP,n]`` for every state."""

    grid = np.arange(N + 1, dtype=float) / N
    return (Q1 @ grid) - (Q0 @ grid)


# ---------------------------------------------------------------------------
# 4. Exact state-local T_pi and bounded eta_IR
# ---------------------------------------------------------------------------


def local_action_information(
    Q0: np.ndarray, Q1: np.ndarray, advocacy: Sequence[float]
) -> np.ndarray:
    """Exact state-local ``T_pi(n)=I(U;n'|n)`` in bits.

    This is the weighted Jensen-Shannon divergence between the NoOp and
    advocacy kernels at the state's own action probability ``a_n``.
    """

    Q0 = np.asarray(Q0, dtype=float)
    Q1 = np.asarray(Q1, dtype=float)
    a = np.asarray(advocacy, dtype=float)
    if Q0.shape != Q1.shape or Q0.ndim != 2 or Q0.shape[0] != Q0.shape[1]:
        raise ValueError("Q0 and Q1 must be equally sized square matrices")
    if a.shape != (Q0.shape[0],):
        raise ValueError("advocacy must be indexed by n=0..N")

    result = np.zeros(Q0.shape[0], dtype=float)
    for n in range(Q0.shape[0]):
        an = float(a[n])
        mixture = (1.0 - an) * Q0[n] + an * Q1[n]
        result[n] = (1.0 - an) * _relative_entropy_bits(Q0[n], mixture) + an * (
            _relative_entropy_bits(Q1[n], mixture)
        )
    return result


def information_response_lower_bound(
    advocacy: Sequence[float], chi: Sequence[float]
) -> np.ndarray:
    """Pinsker lower bound on ``T_pi`` in bits.

    ``T_pi(n) >= 2 a_n(1-a_n) chi(n)^2 / ln(2)``.
    """

    a = np.asarray(advocacy, dtype=float)
    response = np.asarray(chi, dtype=float)
    if a.shape != response.shape:
        raise ValueError("advocacy and chi must have the same shape")
    return 2.0 * a * (1.0 - a) * response**2 / math.log(2.0)


def information_response_efficiency(
    advocacy: Sequence[float], chi: Sequence[float], T_pi: Sequence[float]
) -> np.ndarray:
    """Bounded information-response efficiency ``eta_IR(n)``.

    The ratio is undefined where ``T_pi(n)=0`` and is returned as ``NaN``
    rather than assigning an artificial zero.
    """

    T = np.asarray(T_pi, dtype=float)
    lower = information_response_lower_bound(advocacy, chi)
    if lower.shape != T.shape:
        raise ValueError("T_pi must have the same shape as advocacy and chi")
    eta = np.full(T.shape, np.nan, dtype=float)
    mask = T > 0.0
    eta[mask] = lower[mask] / T[mask]
    # Guard only against floating-point overshoot; a material violation is an
    # implementation error and should remain visible.
    if np.any(eta[mask] > 1.0 + 1e-10):
        raise ArithmeticError("eta_IR exceeded its Pinsker bound")
    eta[(eta > 1.0) & (eta <= 1.0 + 1e-10)] = 1.0
    return eta


# Backwards-friendly name from the previous q-voter module.
local_transfer_entropy = local_action_information


# ---------------------------------------------------------------------------
# 5. Ensemble quantities and finite-time thermodynamics
# ---------------------------------------------------------------------------


def binomial_ensemble(N: int, x0: float) -> np.ndarray:
    """Binomial population-count ensemble used by the theory figures."""

    x0 = float(x0)
    if N < 1 or not 0.0 <= x0 <= 1.0:
        raise ValueError("require N>=1 and x0 in [0,1]")
    p = np.asarray(
        [
            math.comb(N, n) * x0**n * (1.0 - x0) ** (N - n)
            for n in range(N + 1)
        ],
        dtype=float,
    )
    return p / p.sum()


def system_entropy(p: Sequence[float], N: int) -> float:
    """Coarse system entropy ``S_sys[p]`` in nats.

    ``S_sys = -sum p ln p + sum p ln binom(N,n)``.
    """

    distribution = _validate_probability_vector(p, size=N + 1, name="p")
    total = 0.0
    for n, pn in enumerate(distribution):
        if pn > 0.0:
            total += -pn * math.log(float(pn)) + pn * _log_binomial(N, n)
    return float(total)


def sensing_information_nats(
    p_k: Sequence[float], S: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return ``I(n_k;Y_k)`` in nats and the sensor marginal ``p_Y``."""

    S = np.asarray(S, dtype=float)
    if S.ndim != 2:
        raise ValueError("S must be a matrix indexed by (n,y)")
    p = _validate_probability_vector(p_k, size=S.shape[0], name="p_k")
    pY = p @ S
    total = 0.0
    for n, pn in enumerate(p):
        if pn <= 0.0:
            continue
        for y, sy_n in enumerate(S[n]):
            if sy_n > 0.0:
                if pY[y] <= 0.0:
                    return math.inf, pY
                total += pn * sy_n * math.log(float(sy_n) / float(pY[y]))
    return float(total), pY


def mean_controlled_current(
    p_k: Sequence[float], advocacy: Sequence[float], chi: Sequence[float], N: int
) -> float:
    """Mean controlled target-count current ``J_c``.

    Since ``chi`` is a target-*fraction* response, multiplying the
    occupancy/action-weighted response by ``N`` gives the target-count current.
    """

    p = _validate_probability_vector(p_k, size=N + 1, name="p_k")
    a = np.asarray(advocacy, dtype=float)
    response = np.asarray(chi, dtype=float)
    if a.shape != (N + 1,) or response.shape != (N + 1,):
        raise ValueError("advocacy and chi must be indexed by n=0..N")
    return float(N * np.sum(p * a * response))


def thermodynamic_efficiency(
    *, h: float, J_c: float, I_sens_nats: float, tolerance: float = 1e-14
) -> tuple[float, float, bool]:
    """Return ``(eta_th, C_th, bounded_interpretation)``.

    ``C_th = h J_c + I_sens`` and ``eta_th = h J_c / C_th``.  If ``C_th`` is
    zero, the efficiency is undefined and returned as ``NaN``.  A negative
    ``h J_c`` is retained as a signed diagnostic, but ``bounded_interpretation``
    is then false, exactly as stated in the revised report.
    """

    directed = float(h) * float(J_c)
    C_th = directed + float(I_sens_nats)
    if abs(C_th) <= tolerance:
        return math.nan, C_th, False
    eta = directed / C_th
    bounded = directed >= -tolerance and C_th > tolerance
    if bounded:
        if eta < 0.0 and eta >= -1e-12:
            eta = 0.0
        if eta > 1.0 and eta <= 1.0 + 1e-12:
            eta = 1.0
        if not -1e-12 <= eta <= 1.0 + 1e-12:
            raise ArithmeticError("target-directed eta_th left [0,1]")
    return float(eta), float(C_th), bool(bounded)


@dataclass(frozen=True, slots=True)
class CycleThermodynamics:
    """Exact finite-time thermodynamic accounting for one feedback cycle."""

    parameters: TheoryParameters
    p_k: np.ndarray
    p_next: np.ndarray
    p_Y: np.ndarray
    J_c: float
    I_sens_nats: float
    delta_S_sys_nats: float
    Sigma_nats: float
    Sigma_direct_KL_nats: float
    C_th_nats: float
    eta_th: float
    eta_th_has_bounded_interpretation: bool
    identity_residual_nats: float

    @property
    def directed_current_nats(self) -> float:
        """Affinity-weighted directed current ``h J_c``."""

        return float(self.parameters.h * self.J_c)

    @property
    def second_law_lhs_nats(self) -> float:
        """``Delta S_sys + h J_c + I_sens``; equal to ``Sigma``."""

        return float(
            self.delta_S_sys_nats + self.directed_current_nats + self.I_sens_nats
        )

    @property
    def second_law_satisfied(self) -> bool:
        return self.Sigma_nats >= -1e-10

    def as_fields(self) -> dict[str, Any]:
        return {
            **self.parameters.as_fields(),
            "theory_J_c": self.J_c,
            "theory_hJ_c_nats": self.directed_current_nats,
            "theory_I_sens_nats": self.I_sens_nats,
            "theory_delta_S_sys_nats": self.delta_S_sys_nats,
            "theory_Sigma_nats": self.Sigma_nats,
            "theory_Sigma_direct_KL_nats": self.Sigma_direct_KL_nats,
            "theory_C_th_nats": self.C_th_nats,
            "theory_eta_th": self.eta_th,
            "theory_eta_th_bounded": self.eta_th_has_bounded_interpretation,
            "theory_second_law_satisfied": self.second_law_satisfied,
            "theory_identity_residual_nats": self.identity_residual_nats,
        }


def _direct_feedback_path_kl(
    *,
    p_k: np.ndarray,
    p_next: np.ndarray,
    S: np.ndarray,
    p_Y: np.ndarray,
    pi1: np.ndarray,
    Q0: np.ndarray,
    Q1: np.ndarray,
) -> float:
    """Direct ``D_KL(P_F||P_R)`` for the report's reverse reference."""

    total = 0.0
    for n, pn in enumerate(p_k):
        if pn <= 0.0:
            continue
        for y, sy_n in enumerate(S[n]):
            if sy_n <= 0.0:
                continue
            for u in (0, 1):
                pu = (1.0 - pi1[y]) if u == 0 else pi1[y]
                if pu <= 0.0:
                    continue
                Q = Q0 if u == 0 else Q1
                for m in np.flatnonzero(Q[n] > 0.0):
                    forward = pn * sy_n * pu * Q[n, m]
                    if forward <= 0.0:
                        continue
                    reverse = p_next[m] * p_Y[y] * pu * Q[m, n]
                    if reverse <= 0.0:
                        return math.inf
                    total += forward * math.log(float(forward) / float(reverse))
    return float(total)


# ---------------------------------------------------------------------------
# 6. Assembled exact reference
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SingleAffinityReference:
    """Every exact state-local quantity at one parameter tuple, computed once."""

    parameters: TheoryParameters
    S: np.ndarray
    pi1: np.ndarray
    advocacy: np.ndarray
    K: np.ndarray
    Q0: np.ndarray
    Q1: np.ndarray
    chi: np.ndarray
    T_pi: np.ndarray
    pinsker_bound: np.ndarray
    eta_IR: np.ndarray

    @property
    def states(self) -> np.ndarray:
        return np.arange(self.parameters.N + 1, dtype=int)

    @property
    def shares(self) -> np.ndarray:
        return self.states.astype(float) / self.parameters.N

    @property
    def p_h(self) -> float:
        return self.parameters.p_h

    @property
    def set_point(self) -> float:
        """Zero-response point ``x*=sigma(h)``."""

        return self.parameters.p_h

    @property
    def Lambda_b_gamma(self) -> float:
        return self.parameters.Lambda_b_gamma

    @property
    def local_te(self) -> np.ndarray:
        """Compatibility alias: the revised object is ``T_pi``."""

        return self.T_pi

    @property
    def mean_response(self) -> np.ndarray:
        """Compatibility alias: the revised object is ``chi``."""

        return self.chi

    @property
    def closed_loop_kernel(self) -> np.ndarray:
        """Action-marginalized one-cycle kernel ``Q_pi`` row by row."""

        return (
            (1.0 - self.advocacy)[:, None] * self.Q0
            + self.advocacy[:, None] * self.Q1
        )

    def entropy_ceiling(self) -> np.ndarray:
        """Exact action-entropy ceiling ``h2(a_n)`` on ``T_pi(n)`` in bits."""

        return np.asarray([binary_entropy_bits(a) for a in self.advocacy], dtype=float)

    def occupancy_average(
        self, values: Sequence[float], occupancy: Sequence[float]
    ) -> float:
        weights = _validate_probability_vector(
            occupancy, size=self.parameters.N + 1, name="occupancy"
        )
        array = np.asarray(values, dtype=float)
        if array.shape != weights.shape:
            raise ValueError("values must be indexed by n=0..N")
        return float(weights @ array)

    def occupancy_weighted_T_pi(self, occupancy: Sequence[float]) -> float:
        """``sum_n p(n) T_pi(n)`` in bits."""

        return self.occupancy_average(self.T_pi, occupancy)

    def occupancy_weighted_te(self, occupancy: Sequence[float]) -> float:
        """Compatibility alias for :meth:`occupancy_weighted_T_pi`."""

        return self.occupancy_weighted_T_pi(occupancy)

    def propagate(self, p_k: Sequence[float]) -> np.ndarray:
        """Exact one-cycle ensemble propagation under ``Q_pi``."""

        p = _validate_probability_vector(
            p_k, size=self.parameters.N + 1, name="p_k"
        )
        result = p @ self.closed_loop_kernel
        # Clean tiny matrix-arithmetic residue while preserving normalization.
        result[np.abs(result) < 1e-16] = 0.0
        result /= result.sum()
        return result

    def self_occupancy(self, initial: Sequence[float], rounds: int) -> np.ndarray:
        """Transient occupancy history ``p_0,...,p_{rounds-1}``."""

        if isinstance(rounds, bool) or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        p = _validate_probability_vector(
            initial, size=self.parameters.N + 1, name="initial"
        )
        history = [p.copy()]
        for _ in range(int(rounds) - 1):
            p = p @ self.closed_loop_kernel
            history.append(p.copy())
        return np.asarray(history, dtype=float)

    def current(self, p_k: Sequence[float]) -> float:
        """Exact mean controlled current ``J_c`` for occupancy ``p_k``."""

        return mean_controlled_current(
            p_k, self.advocacy, self.chi, self.parameters.N
        )

    def one_cycle(
        self,
        p_k: Sequence[float],
        *,
        check_kl: bool = True,
        atol: float = 2e-10,
        rtol: float = 2e-10,
    ) -> CycleThermodynamics:
        """Compute the complete exact one-cycle finite-time accounting."""

        p = _validate_probability_vector(
            p_k, size=self.parameters.N + 1, name="p_k"
        )
        p_next = self.propagate(p)
        I_sens, p_Y = sensing_information_nats(p, self.S)
        J_c = self.current(p)
        delta_S = system_entropy(p_next, self.parameters.N) - system_entropy(
            p, self.parameters.N
        )
        Sigma_identity = delta_S + self.parameters.h * J_c + I_sens

        Sigma_direct = math.nan
        if check_kl:
            Sigma_direct = _direct_feedback_path_kl(
                p_k=p,
                p_next=p_next,
                S=self.S,
                p_Y=p_Y,
                pi1=self.pi1,
                Q0=self.Q0,
                Q1=self.Q1,
            )
            if not np.isclose(
                Sigma_identity, Sigma_direct, atol=atol, rtol=rtol, equal_nan=False
            ):
                raise ArithmeticError(
                    "finite-time KL identity failed: "
                    f"decomposition={Sigma_identity}, direct={Sigma_direct}"
                )

        eta_th, C_th, bounded = thermodynamic_efficiency(
            h=self.parameters.h, J_c=J_c, I_sens_nats=I_sens
        )
        # The exact identity also requires C_th == Sigma - Delta S.
        cth_from_path = Sigma_identity - delta_S
        if not np.isclose(C_th, cth_from_path, atol=atol, rtol=rtol):
            raise ArithmeticError("non-storage expenditure identity failed")

        residual = (
            Sigma_identity - Sigma_direct if check_kl and math.isfinite(Sigma_direct) else math.nan
        )
        return CycleThermodynamics(
            parameters=self.parameters,
            p_k=p.copy(),
            p_next=p_next,
            p_Y=p_Y,
            J_c=J_c,
            I_sens_nats=I_sens,
            delta_S_sys_nats=delta_S,
            Sigma_nats=Sigma_identity,
            Sigma_direct_KL_nats=Sigma_direct,
            C_th_nats=C_th,
            eta_th=eta_th,
            eta_th_has_bounded_interpretation=bounded,
            identity_residual_nats=residual,
        )


@lru_cache(maxsize=64)
def _reference_for(
    key: tuple[int, int, int, float, float, float, float]
) -> SingleAffinityReference:
    parameters = TheoryParameters(*key)
    S = sensor_kernel(parameters.N, parameters.q_c)
    pi1 = policy_advocacy_vector(parameters.q_c, parameters.beta, parameters.theta)
    advocacy = S @ pi1
    K = controlled_kernel(parameters)
    Q0 = noop_kernel(parameters.N)
    Q1 = advocate_kernel(K, parameters.b)
    chi = susceptibility_curve(parameters)

    # The closed form and the finite matrix kernel must encode the same mean
    # response.  This is an important implementation invariant.
    kernel_chi = kernel_mean_response(Q0, Q1, parameters.N)
    if not np.allclose(chi, kernel_chi, atol=2e-13, rtol=2e-13):
        raise ArithmeticError("closed-form susceptibility disagrees with Q0/Q1")

    T_pi = local_action_information(Q0, Q1, advocacy)
    pinsker = information_response_lower_bound(advocacy, chi)
    eta_IR = information_response_efficiency(advocacy, chi, T_pi)

    if np.any(T_pi > np.asarray([binary_entropy_bits(a) for a in advocacy]) + 1e-10):
        raise ArithmeticError("T_pi exceeded the action-entropy ceiling")
    if np.any(T_pi + 1e-10 < pinsker):
        raise ArithmeticError("T_pi violated the Pinsker response bound")

    return SingleAffinityReference(
        parameters=parameters,
        S=S,
        pi1=pi1,
        advocacy=advocacy,
        K=K,
        Q0=Q0,
        Q1=Q1,
        chi=chi,
        T_pi=T_pi,
        pinsker_bound=pinsker,
        eta_IR=eta_IR,
    )


def single_affinity_reference(parameters: TheoryParameters) -> SingleAffinityReference:
    """Return the cached exact reference for one parameter tuple."""

    return _reference_for(parameters.key)


# ---------------------------------------------------------------------------
# 7. Finite-horizon helpers (same exact one-cycle model)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FiniteHorizonThermodynamics:
    """Sum of exact one-cycle identities over a fixed finite horizon."""

    cycles: tuple[CycleThermodynamics, ...]
    p_history: np.ndarray
    total_J_c: float
    total_I_sens_nats: float
    total_delta_S_sys_nats: float
    total_Sigma_nats: float
    total_C_th_nats: float
    eta_th: float
    eta_th_has_bounded_interpretation: bool

    @property
    def directed_current_nats(self) -> float:
        if not self.cycles:
            return 0.0
        return self.cycles[0].parameters.h * self.total_J_c

    @property
    def identity_residual_nats(self) -> float:
        return self.total_Sigma_nats - (
            self.total_delta_S_sys_nats
            + self.directed_current_nats
            + self.total_I_sens_nats
        )


def finite_horizon_thermodynamics(
    reference: SingleAffinityReference,
    initial: Sequence[float],
    rounds: int,
    *,
    check_kl: bool = True,
) -> FiniteHorizonThermodynamics:
    """Iterate and sum the exact finite-time identity for ``rounds`` cycles."""

    if isinstance(rounds, bool) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    p = _validate_probability_vector(
        initial, size=reference.parameters.N + 1, name="initial"
    )
    p_history = [p.copy()]
    cycles: list[CycleThermodynamics] = []
    for _ in range(int(rounds)):
        cycle = reference.one_cycle(p, check_kl=check_kl)
        cycles.append(cycle)
        p = cycle.p_next
        p_history.append(p.copy())

    total_J = float(sum(c.J_c for c in cycles))
    total_I = float(sum(c.I_sens_nats for c in cycles))
    total_delta = float(sum(c.delta_S_sys_nats for c in cycles))
    total_Sigma = float(sum(c.Sigma_nats for c in cycles))
    eta, C_th, bounded = thermodynamic_efficiency(
        h=reference.parameters.h, J_c=total_J, I_sens_nats=total_I
    )

    return FiniteHorizonThermodynamics(
        cycles=tuple(cycles),
        p_history=np.asarray(p_history, dtype=float),
        total_J_c=total_J,
        total_I_sens_nats=total_I,
        total_delta_S_sys_nats=total_delta,
        total_Sigma_nats=total_Sigma,
        total_C_th_nats=C_th,
        eta_th=eta,
        eta_th_has_bounded_interpretation=bounded,
    )


def finite_horizon_current_moments(
    round_kernel: np.ndarray,
    initial_distribution: Sequence[float],
    rounds: int,
) -> dict[str, float]:
    """Exact moments of terminal net current ``J=n_H-n_0``.

    This helper is retained from the previous module.  In the isolated
    single-affinity controlled layer, every count displacement belongs to the
    controlled channel, so the terminal difference is the integrated current.
    """

    kernel = np.asarray(round_kernel, dtype=float)
    initial = np.asarray(initial_distribution, dtype=float)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("round_kernel must be square")
    initial = _validate_probability_vector(
        initial, size=kernel.shape[0], name="initial_distribution"
    )
    if isinstance(rounds, bool) or rounds < 0:
        raise ValueError("rounds must be a non-negative integer")

    transition = np.linalg.matrix_power(kernel, int(rounds))
    states = np.arange(kernel.shape[0], dtype=float)
    currents = states[None, :] - states[:, None]
    joint = initial[:, None] * transition
    mean = float(np.sum(joint * currents))
    second = float(np.sum(joint * currents * currents))
    variance = second - mean * mean
    if -1e-12 < variance < 0.0:
        variance = 0.0
    if variance < 0.0:
        raise ArithmeticError("finite-horizon current variance became negative")
    return {"mean": mean, "second_moment": second, "variance": variance}


def finite_horizon_current_moments_for_episodes(
    round_kernel: np.ndarray,
    initial_counts: Sequence[int],
    horizons: Sequence[int],
) -> dict[str, float]:
    """Exact terminal-current moments for an empirical mixture of episodes.

    Each episode keeps its own initial count and horizon.  The returned
    variance is therefore the variance of that mixture, including variation
    between episode protocols, rather than an average of conditional
    variances.  This is the deterministic adapter needed by current reports;
    it contains no empirical estimation or I/O.
    """

    kernel = np.asarray(round_kernel, dtype=float)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("round_kernel must be square")
    if len(initial_counts) != len(horizons) or not initial_counts:
        raise ValueError("initial_counts and horizons must be equally sized and non-empty")
    state_count = kernel.shape[0]
    means: list[float] = []
    seconds: list[float] = []
    for initial_count, horizon in zip(initial_counts, horizons, strict=True):
        if isinstance(initial_count, bool) or not 0 <= int(initial_count) < state_count:
            raise ValueError("initial count lies outside the kernel state space")
        initial = np.zeros(state_count, dtype=float)
        initial[int(initial_count)] = 1.0
        moments = finite_horizon_current_moments(kernel, initial, int(horizon))
        means.append(moments["mean"])
        seconds.append(moments["second_moment"])
    mean = float(np.mean(means))
    second = float(np.mean(seconds))
    variance = second - mean * mean
    if -1e-12 < variance < 0.0:
        variance = 0.0
    if variance < 0.0:
        raise ArithmeticError("finite-horizon episode-mixture variance became negative")
    return {"mean": mean, "second_moment": second, "variance": variance}


# ---------------------------------------------------------------------------
# 8. Operational calibration of h and gamma
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AffinityComplianceCalibration:
    """Microscopic calibration from controlled binary vote transitions."""

    p_plus: float
    p_minus: float
    h_eff: float
    gamma_eff: float

    @property
    def forward_reverse_odds(self) -> float:
        return math.exp(self.h_eff)


def calibrate_affinity_compliance(
    p_plus: float, p_minus: float
) -> AffinityComplianceCalibration:
    """Calibrate ``gamma=p_+ + p_-`` and ``h=ln(p_+/p_-)``.

    ``p_plus`` is ``P(non-Z -> Z | controlled)`` and ``p_minus`` is
    ``P(Z -> non-Z | controlled)``.  Finite ``h`` requires both directions to
    be observed with nonzero probability, matching the reversible theory.
    """

    p_plus = float(p_plus)
    p_minus = float(p_minus)
    if not 0.0 <= p_plus <= 1.0 or not 0.0 <= p_minus <= 1.0:
        raise ValueError("p_plus and p_minus must lie in [0,1]")
    gamma = p_plus + p_minus
    if gamma > 1.0 + 1e-12:
        raise ValueError("p_plus + p_minus cannot exceed one")
    if p_plus <= 0.0 or p_minus <= 0.0:
        raise ValueError("finite reversible affinity requires p_plus,p_minus > 0")
    return AffinityComplianceCalibration(
        p_plus=p_plus,
        p_minus=p_minus,
        h_eff=math.log(p_plus / p_minus),
        gamma_eff=min(1.0, gamma),
    )


def calibrate_affinity_compliance_from_counts(
    *,
    plus_transitions: int,
    plus_eligible: int,
    minus_transitions: int,
    minus_eligible: int,
) -> AffinityComplianceCalibration:
    """Count-based wrapper for :func:`calibrate_affinity_compliance`."""

    if plus_eligible <= 0 or minus_eligible <= 0:
        raise ValueError("eligible counts must be positive")
    if not 0 <= plus_transitions <= plus_eligible:
        raise ValueError("plus_transitions must lie in [0, plus_eligible]")
    if not 0 <= minus_transitions <= minus_eligible:
        raise ValueError("minus_transitions must lie in [0, minus_eligible]")
    return calibrate_affinity_compliance(
        plus_transitions / plus_eligible,
        minus_transitions / minus_eligible,
    )


# ---------------------------------------------------------------------------
# 9. Reading protocol parameters from a runtime record
# ---------------------------------------------------------------------------


def theory_parameters_from_record(
    record: Mapping[str, Any],
    *,
    h: float | None = None,
    gamma: float | None = None,
) -> TheoryParameters | None:
    """Build parameters from one round record plus calibrated ``h,gamma``.

    ``h`` and ``gamma`` are properties of the calibrated population response,
    not ordinary controller-resource fields in the current runtime.  They are
    therefore accepted explicitly.  If omitted, this function only looks for
    the unambiguous fields ``theory_h`` and ``theory_gamma`` in ``record``.

    If a record declares ``social_group_size`` and it is not one, ``None`` is
    returned because the revised closed-form theory is the q=1 controlled
    layer.
    """

    if record.get("social_group_size") is not None:
        try:
            if int(record["social_group_size"]) != 1:
                return None
        except (TypeError, ValueError):
            return None

    if h is None:
        h = record.get("theory_h")
    if gamma is None:
        gamma = record.get("theory_gamma")

    required = (
        record.get("N"),
        record.get("sensor_sample_size"),
        record.get("intervention_budget"),
        record.get("controller_beta"),
        record.get("controller_threshold"),
        h,
        gamma,
    )
    if any(value is None for value in required):
        return None

    N, q_c, b, beta, theta, h_value, gamma_value = required
    try:
        return TheoryParameters(
            N=int(N),
            q_c=int(q_c),
            b=int(b),
            beta=float(beta),
            theta=float(theta),
            h=float(h_value),
            gamma=float(gamma_value),
        )
    except (TypeError, ValueError):
        return None


__all__ = [
    "AffinityComplianceCalibration",
    "CycleThermodynamics",
    "FiniteHorizonThermodynamics",
    "SingleAffinityReference",
    "TheoryParameters",
    "THEORY_API_VERSION",
    "THEORY_MODULE",
    "THEORY_REFERENCE",
    "THEORY_SEMANTICS_VERSION",
    "advocacy_probability_curve",
    "advocate_kernel",
    "binary_entropy_bits",
    "binomial_ensemble",
    "calibrate_affinity_compliance",
    "calibrate_affinity_compliance_from_counts",
    "controlled_kernel",
    "finite_horizon_current_moments",
    "finite_horizon_current_moments_for_episodes",
    "finite_horizon_thermodynamics",
    "information_response_efficiency",
    "information_response_lower_bound",
    "kernel_mean_response",
    "local_action_information",
    "local_transfer_entropy",
    "logistic",
    "mean_controlled_current",
    "noop_kernel",
    "policy_advocacy_vector",
    "sensor_kernel",
    "sensor_law",
    "sensing_information_nats",
    "single_affinity_reference",
    "susceptibility",
    "susceptibility_curve",
    "system_entropy",
    "thermodynamic_efficiency",
    "theory_parameters_from_record",
]
