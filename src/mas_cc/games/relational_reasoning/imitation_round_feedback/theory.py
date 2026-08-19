"""Matched finite-`N` controlled q-voter reference for the round-feedback clock.

Nothing in this module is an estimator and nothing in it is a simulation.  Once
`(N, q, q_c, b, beta, theta)` are fixed, every quantity here is a deterministic
number, so a Monte Carlo q-voter would only add sampling noise to something
already exact.

The point of the reference is *not* to claim the LLM population is a q-voter.
It is that the relational experiment and the classical controlled q-voter run
the same control protocol -

    population state -> finite sensing -> stochastic ADVOCATE/NO_OP
        -> budgeted actuation -> next population state

- at the same controller parameters, and differ only in the population-response
kernel: explicit unanimity here, implicit LLM with persistent knowledge there.
Every empirical information number therefore gets a classical number of the
same units measured at the same resources, and a departure becomes readable
rather than merely unexplained.

The microscopic kernel is **not** rewritten here.  `K0`/`K1` are assembled by
calling the round-feedback game's own
`classical.analytical_mesoscopic_transition_probability` on a binary
target/non-target population, which is the repository's single definition of
"one ordinary / one controlled unanimity update".  This module supplies the
count indexing (`n` rather than a vote list), the whole-round composition, and
the information functional on top.

Three things are deliberately exact rather than approximated:

`R1`, the ADVOCATE round kernel
    exactly `b` of the `N` positions are controlled, uniformly preallocated -
    which is what the runtime actually does (`sample_controlled_positions`
    draws `b` distinct positions without replacement).  Bernoulli-`c` control
    of each position has the same mean and the wrong variance, so the dynamic
    recursion over (positions processed, controlled positions used) is used
    instead.  It averages over all schedules without enumerating them.

`a_n`, the state-conditioned advocacy probability
    the controller sees a hypergeometric sample of size `q_c`, not the true
    share, so `a_n` is the sensor law folded through the logistic policy rather
    than the policy evaluated at `n/N`.

`T_qv(n)`, the state-local transfer entropy
    the Jensen-Shannon divergence between the two round kernels at their own
    mixing weight `a_n`.  No counting, no bootstrap, no null.

The binary target/non-target projection is intentional and is the one place
this reference coarse-grains the experiment: the relational task has three
answer options and the classical reference is binary, so the comparable state
is "votes for the controller's target" against everything else.  Callers are
expected to say so in the report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np

# The microscopic unanimity law is imported, never restated. This is the same
# function the HiddenBench round-feedback game uses to describe its own kernel;
# a second copy of `C(n, q) / C(N-1, q)` here would be free to drift away from
# the dynamics it claims to be the reference for.
from ...hidden_bench.imitation_round_feedback.classical import (
    analytical_mesoscopic_transition_probability,
)

TARGET = "Z"
NON_TARGET = "X"
"""The binary alphabet the classical reference runs on.

`Z` is "votes for whatever the controller is targeting", `X` is everything
else, whatever the task's other options happen to be."""


def binary_entropy_bits(probability: float) -> float:
    """`h2(a) = -a log2 a - (1-a) log2 (1-a)`, the ceiling on `T_qv`."""

    a = float(probability)
    if not 0.0 < a < 1.0:
        return 0.0
    return -a * math.log2(a) - (1.0 - a) * math.log2(1.0 - a)


@dataclass(frozen=True, slots=True)
class TheoryParameters:
    """The matched controller protocol, and nothing about the population kernel.

    `q_c` and `b` are independent by construction even though the Study 03
    pilot happens to run them equal: sensing resolution and actuation reach are
    different resources and a later grid is expected to vary them apart.
    """

    N: int
    q: int
    q_c: int
    b: int
    beta: float
    theta: float

    def __post_init__(self) -> None:
        if self.N < 2:
            raise ValueError("theory requires a population of at least two agents")
        if not 1 <= self.q <= self.N - 1:
            raise ValueError("q must lie between 1 and N - 1")
        if not 0 <= self.q_c <= self.N:
            raise ValueError("the sensor sample size must lie between 0 and N")
        if not 0 <= self.b <= self.N:
            raise ValueError("the actuation budget must lie between 0 and N")
        if not math.isfinite(self.beta) or not math.isfinite(self.theta):
            raise ValueError("beta and theta must be finite")

    @property
    def key(self) -> tuple[int, int, int, int, float, float]:
        """The cache key, and the tuple cells must agree on before pooling."""

        return (self.N, self.q, self.q_c, self.b, float(self.beta), float(self.theta))

    @property
    def sensing_fraction(self) -> float:
        """`r_sense = q_c / N`."""

        return self.q_c / self.N

    @property
    def actuation_fraction(self) -> float:
        """`c = r_act = b / N`."""

        return self.b / self.N

    def as_fields(self) -> dict[str, Any]:
        """The `theory_*` resource coordinates carried on every summary row."""

        return {
            "theory_N": self.N,
            "theory_q": self.q,
            "theory_qc": self.q_c,
            "theory_b": self.b,
            "theory_c": self.actuation_fraction,
            "theory_beta": float(self.beta),
            "theory_theta": float(self.theta),
            "theory_sensing_fraction": self.sensing_fraction,
            "theory_actuation_fraction": self.actuation_fraction,
        }


# --------------------------------------------------------------------------
# 1. Controller sensing and policy
# --------------------------------------------------------------------------


def sensor_law(N: int, n: int, q_c: int) -> np.ndarray:
    """`S(y | n)`: `q_c` agents drawn without replacement, `y` of them on target.

    Hypergeometric, because the controller samples distinct agents.  Impossible
    `y` get exactly zero out of `math.comb`, which returns 0 whenever the
    sample exceeds the pool - so no explicit range clamping is needed and no
    invalid combination can leak a nonzero weight.
    """

    law = np.zeros(q_c + 1, dtype=float)
    total = math.comb(N, q_c)
    for y in range(q_c + 1):
        law[y] = math.comb(n, y) * math.comb(N - n, q_c - y) / total
    return law


def advocacy_probability_curve(parameters: TheoryParameters) -> np.ndarray:
    """`a_n = P(ADVOCATE | N_k = n)` for every `n = 0..N`.

    The controller never sees `n`.  It sees `y ~ S(. | n)` and pushes `y / q_c`
    through the logistic policy, so `a_n` is the *sensor-smeared* policy: with a
    small `q_c` it is a much gentler function of `n` than `sigma[beta(theta -
    n/N)]` would be, and reading the empirical action frequencies against the
    unsmeared curve would manufacture a calibration failure that is not there.
    """

    N, q_c = parameters.N, parameters.q_c
    beta, theta = float(parameters.beta), float(parameters.theta)
    if q_c == 0:
        # A controller with no sensor still decides, at the fixed prior share.
        constant = _logistic(beta * theta)
        return np.full(N + 1, constant, dtype=float)
    shares = np.arange(q_c + 1, dtype=float) / q_c
    policy = np.array([_logistic(beta * (theta - share)) for share in shares])
    return np.array(
        [float(sensor_law(N, n, q_c) @ policy) for n in range(N + 1)], dtype=float
    )


def _logistic(z: float) -> float:
    # Split on the sign so the exponential is always of a non-positive number:
    # `beta` is configurable and a large positive `z` would overflow `exp(-z)`'s
    # naive counterpart rather than saturating at one.
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    value = math.exp(z)
    return value / (1.0 + value)


# --------------------------------------------------------------------------
# 2. Microscopic kernels, count-indexed
# --------------------------------------------------------------------------


def _binary_population(N: int, n: int) -> list[str]:
    return [TARGET] * n + [NON_TARGET] * (N - n)


def microscopic_kernels(parameters: TheoryParameters) -> tuple[np.ndarray, np.ndarray]:
    """`(K0, K1)` as `(N+1, N+1)` row-stochastic matrices over the target count.

    `K0` is one ordinary microscopic update: a uniformly chosen focal switches
    only on unanimous social input, in either direction.  `K1` is one
    *controlled* update, where the controller occupies one of the `q` social
    slots and advocates the target - so only `q-1` ordinary target supporters
    are needed to pull a non-target agent across, and a target agent can never
    be pulled away.  `K1(n-1 | n) = 0` is therefore structural, not a modelling
    choice.

    Both rows come out of the shared microscopic law rather than being written
    out here; the only thing added is the projection onto the target count.
    """

    N, q = parameters.N, parameters.q
    K0 = np.zeros((N + 1, N + 1), dtype=float)
    K1 = np.zeros((N + 1, N + 1), dtype=float)
    for n in range(N + 1):
        population = _binary_population(N, n)

        def probability(source: str, destination: str, controlled: bool) -> float:
            return analytical_mesoscopic_transition_probability(
                population_state=population,
                source=source,
                destination=destination,
                social_group_size=q,
                controlled_slot=controlled,
                controller_target=TARGET if controlled else None,
            )

        up = probability(NON_TARGET, TARGET, False)
        down = probability(TARGET, NON_TARGET, False)
        if n + 1 <= N:
            K0[n, n + 1] = up
        if n - 1 >= 0:
            K0[n, n - 1] = down
        K0[n, n] = 1.0 - up - down

        controlled_up = probability(NON_TARGET, TARGET, True)
        if n + 1 <= N:
            K1[n, n + 1] = controlled_up
        # `K1(n-1 | n) = 0`: the shared law already returns zero for a
        # destination that is not the controller target on a controlled slot,
        # so this is asserted by construction rather than assigned.
        K1[n, n] = 1.0 - controlled_up
    return K0, K1


# --------------------------------------------------------------------------
# 3. Whole-round kernels
# --------------------------------------------------------------------------


def no_op_round_kernel(K0: np.ndarray, N: int) -> np.ndarray:
    """`R0 = K0^N` - a round is `N` ordinary microscopic positions."""

    return np.linalg.matrix_power(K0, N)


def advocate_round_kernel(
    K0: np.ndarray, K1: np.ndarray, *, N: int, b: int
) -> np.ndarray:
    """`R1`: one round with exactly `b` of the `N` positions controlled.

    The `b` controlled positions are drawn uniformly without replacement, so
    they are *not* independent across positions - having already spent a
    controlled position makes the next one less likely.  Bernoulli-`c` control
    would reproduce the mean and get the variance wrong, and the variance is
    what the information functional is measuring.

    The exact average over all `C(N, b)` schedules is obtained without
    enumerating any of them, by carrying the joint distribution over (positions
    processed `r`, controlled positions used `j`).  Given `j` used after `r`
    positions, the next position is controlled with probability `(b-j)/(N-r)`
    and ordinary with probability `(N-b-r+j)/(N-r)` - the plain
    without-replacement odds.  `F[N][b]` is then the schedule-averaged kernel.
    """

    size = K0.shape[0]
    current: dict[int, np.ndarray] = {0: np.eye(size, dtype=float)}
    for r in range(N):
        remaining = N - r
        following: dict[int, np.ndarray] = {}
        for j, matrix in current.items():
            controlled_weight = (b - j) / remaining
            ordinary_weight = (N - b - r + j) / remaining
            if controlled_weight > 0.0:
                contribution = controlled_weight * (matrix @ K1)
                following[j + 1] = (
                    contribution
                    if j + 1 not in following
                    else following[j + 1] + contribution
                )
            if ordinary_weight > 0.0:
                contribution = ordinary_weight * (matrix @ K0)
                following[j] = (
                    contribution if j not in following else following[j] + contribution
                )
        current = following
    return current[b]


# --------------------------------------------------------------------------
# 4. State-local transfer entropy
# --------------------------------------------------------------------------


def local_transfer_entropy(
    R0: np.ndarray, R1: np.ndarray, advocacy: np.ndarray
) -> np.ndarray:
    """`T_qv(n)`, the exact state-local controller-to-population TE, in bits.

    This is the Jensen-Shannon divergence between the two whole-round kernels
    at the state's own action mixing weight `a_n`.  Read plainly: how much does
    knowing what the controller *did* this round tell you about where the
    population lands, given where it started.

    It is bounded above by `h2(a_n)` - the controller cannot transmit more than
    its own decision entropy - and vanishes at both ends of that bound, once
    because the action never varies and once because the two kernels agree.
    """

    size = R0.shape[0]
    result = np.zeros(size, dtype=float)
    for n in range(size):
        a = float(advocacy[n])
        mixture = (1.0 - a) * R0[n] + a * R1[n]
        result[n] = (1.0 - a) * _relative_entropy_bits(R0[n], mixture) + a * (
            _relative_entropy_bits(R1[n], mixture)
        )
    return result


def _relative_entropy_bits(p: np.ndarray, q: np.ndarray) -> float:
    """`sum p log2 (p/q)` with `0 log 0 = 0`.

    `q` is always a mixture that contains `p`, so `p > 0` implies `q > 0` and
    the divergence stays finite without any smoothing.
    """

    total = 0.0
    for value, reference in zip(p, q):
        if value > 0.0 and reference > 0.0:
            total += float(value) * math.log2(float(value) / float(reference))
    return total


# --------------------------------------------------------------------------
# 5. Response
# --------------------------------------------------------------------------


def kernel_mean_response(R0: np.ndarray, R1: np.ndarray, N: int) -> np.ndarray:
    """`E[x_{k+1} | n, ADVOCATE] - E[x_{k+1} | n, NO_OP]` from the exact kernels.

    Defined for every `q`, so the response comparison does not have to fall
    back to a closed form that only exists at `q = 1`.  Since `x_k` is fixed
    once `n` is, this difference of next-state means is exactly the
    action-induced mean separation `Delta_mu(n)`.
    """

    grid = np.arange(N + 1, dtype=float) / N
    return (R1 @ grid) - (R0 @ grid)


def q1_mean_response(x: float, *, N: int, b: int) -> float:
    """`Delta_mu_qv(x) = (1-x)[1 - (1-1/N)^b]`, the exact `q = 1` separation.

    At `q = 1` the ordinary kernel is unbiased - a randomly chosen agent copies
    a randomly chosen other, which moves nothing on average - so the whole mean
    separation is the controlled positions' work.  Each of the `b` of them
    converts a uniformly chosen agent, shrinking the non-target pool by a
    factor `(1 - 1/N)`.  Kept as a closed form beside `kernel_mean_response`
    because the two agreeing is a real check on the round-kernel composition.
    """

    return (1.0 - float(x)) * (1.0 - (1.0 - 1.0 / N) ** b)


# --------------------------------------------------------------------------
# 6. Large-N diagnostics
# --------------------------------------------------------------------------


def mean_field_drifts(x: float, q: int) -> dict[str, float]:
    """`f0`, `f1`, `Delta f_q` and `nu0` - interpretable, not authoritative.

    These are the large-`N` limits of the kernels above.  They are reported so
    the shape of a departure can be read off an expression instead of a matrix,
    never in place of the exact finite-`N` numbers.
    """

    x = float(x)
    f0 = (1.0 - x) * x**q - x * (1.0 - x) ** q
    f1 = (1.0 - x) * x ** (q - 1)
    return {
        "f0": f0,
        "f1": f1,
        "delta_f": f1 - f0,
        "nu0": (1.0 - x) * x**q + x * (1.0 - x) ** q - f0 * f0,
    }


def mean_field_transfer_entropy(
    x: float, *, q: int, c: float, N: int, advocacy: float
) -> float:
    """The weak-separation TE approximation, in bits.

    A scaling diagnostic: it says `T ~ N c^2 (Delta f)^2 / nu0`, which is the
    sentence one actually wants when asking how the channel should respond to
    more actuation.  It is unreliable exactly where the local noise `nu0`
    closes - the boundary states, `q = 1, n = 0` being the standard example -
    and returns `nan` there rather than a large number, so a support-opening
    artefact cannot be mistaken for a strong channel.
    """

    drifts = mean_field_drifts(x, q)
    noise = drifts["nu0"]
    if not math.isfinite(noise) or noise <= 1e-12:
        return math.nan
    a = float(advocacy)
    return (
        a * (1.0 - a) / (2.0 * math.log(2.0))
        * N * c * c * drifts["delta_f"] ** 2
        / noise
    )


# --------------------------------------------------------------------------
# 7. The assembled, cached reference
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClassicalReference:
    """Every exact quantity at one parameter tuple, computed once."""

    parameters: TheoryParameters
    advocacy: np.ndarray
    K0: np.ndarray
    K1: np.ndarray
    R0: np.ndarray
    R1: np.ndarray
    local_te: np.ndarray
    mean_response: np.ndarray

    @property
    def states(self) -> np.ndarray:
        return np.arange(self.parameters.N + 1, dtype=int)

    @property
    def shares(self) -> np.ndarray:
        return self.states.astype(float) / self.parameters.N

    def entropy_ceiling(self) -> np.ndarray:
        """`h2(a_n)`, the exact upper bound on `T_qv(n)`."""

        return np.array(
            [binary_entropy_bits(value) for value in self.advocacy], dtype=float
        )

    def occupancy_weighted_te(self, occupancy: Sequence[float]) -> float:
        """`sum_n P(n) T_qv(n)` - the local theory read over a given occupancy."""

        weights = np.asarray(occupancy, dtype=float)
        if weights.shape != self.local_te.shape:
            raise ValueError("occupancy must be indexed by n = 0..N")
        return float(weights @ self.local_te)

    def self_occupancy(self, initial: Sequence[float], rounds: int) -> np.ndarray:
        """The classical closed-loop occupancy `P_k`, propagated under its own kernel.

        `R(m|n) = (1-a_n) R0(m|n) + a_n R1(m|n)`.  Secondary by design: this
        answers "where would the classical process have gone", which is a
        different question from "what would the classical channel have carried
        where the LLM actually went".
        """

        closed = (1.0 - self.advocacy)[:, None] * self.R0 + self.advocacy[:, None] * self.R1
        distribution = np.asarray(initial, dtype=float)
        history = [distribution]
        for _ in range(max(0, rounds - 1)):
            distribution = distribution @ closed
            history.append(distribution)
        return np.asarray(history, dtype=float)


@lru_cache(maxsize=32)
def _reference_for(key: tuple[int, int, int, int, float, float]) -> ClassicalReference:
    parameters = TheoryParameters(*key)
    advocacy = advocacy_probability_curve(parameters)
    K0, K1 = microscopic_kernels(parameters)
    R0 = no_op_round_kernel(K0, parameters.N)
    R1 = advocate_round_kernel(K0, K1, N=parameters.N, b=parameters.b)
    return ClassicalReference(
        parameters=parameters,
        advocacy=advocacy,
        K0=K0,
        K1=K1,
        R0=R0,
        R1=R1,
        local_te=local_transfer_entropy(R0, R1, advocacy),
        mean_response=kernel_mean_response(R0, R1, parameters.N),
    )


def classical_reference(parameters: TheoryParameters) -> ClassicalReference:
    """The exact reference at these parameters, cached by the tuple.

    Cached because a grid repeats the same `(N, q, q_c, b, beta, theta)` across
    every cell that varies something else - the task, the message mode, the
    seed - and `R1` is the one genuinely non-trivial construction here.
    """

    return _reference_for(parameters.key)


# --------------------------------------------------------------------------
# 8. Reading the parameters off a run
# --------------------------------------------------------------------------


def theory_parameters_from_record(record: Mapping[str, Any]) -> TheoryParameters | None:
    """`(N, q, q_c, b, beta, theta)` off one round record, or `None`.

    `None` rather than an exception: a run that never configured a controller
    has no matched classical protocol to be compared against, and that is a
    fact about the run, not an error in reading it.  The caller reports the
    skip.
    """

    required = (
        record.get("N"),
        record.get("social_group_size"),
        record.get("sensor_sample_size"),
        record.get("intervention_budget"),
        record.get("controller_beta"),
        record.get("controller_threshold"),
    )
    if any(value is None for value in required):
        return None
    N, q, q_c, b, beta, theta = required
    try:
        return TheoryParameters(
            N=int(N), q=int(q), q_c=int(q_c), b=int(b),
            beta=float(beta), theta=float(theta),
        )
    except (ValueError, TypeError):
        return None


__all__ = [
    "ClassicalReference",
    "NON_TARGET",
    "TARGET",
    "TheoryParameters",
    "advocacy_probability_curve",
    "advocate_round_kernel",
    "binary_entropy_bits",
    "classical_reference",
    "kernel_mean_response",
    "local_transfer_entropy",
    "mean_field_drifts",
    "mean_field_transfer_entropy",
    "microscopic_kernels",
    "no_op_round_kernel",
    "q1_mean_response",
    "sensor_law",
    "theory_parameters_from_record",
]
