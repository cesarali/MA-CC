"""Exact linear algebra over a finite microstate chain.

Everything here is exact: eigendecomposition and finite sums, no sampling and
no approximation. That is what lets a synthetic game state its own answers
rather than estimating them, and it is affordable only because the microstate
space is small - the whole point of keeping ``N <= 10`` so the action profile
is at most ``2^10 = 1024`` values. A 256x256 transition matrix is a few
milliseconds.

The one structural rule this module exists to enforce is **propagate in the
microstate, lump last**. The macrostate (how many agents play each action) is a
coarse-graining, and the macrostate process is Markov only under strong
lumpability - which holds under full exchangeability and fails the moment the
coupling graph is asymmetric or the noise levels differ. Computing on the
lumped chain when it is not lumpable gives a different and wrong answer. So
every function here takes the full microstate distribution and reduces at the
end; ``lumping_gap`` measures what the shortcut would have cost.
"""

from __future__ import annotations

import math

import numpy as np

_TOLERANCE = 1e-12


def bit_table(n_agents: int, n_states: int | None = None) -> np.ndarray:
    """``bits[x, i]`` - the action agent *i* plays in microstate *x*.

    Microstates are integers whose *i*-th bit is agent *i*'s action index, so
    a population profile and an array index are the same object and every
    projection below is a bit mask rather than a dictionary lookup.
    """

    states = np.arange(n_states if n_states is not None else 1 << n_agents)
    return ((states[:, None] >> np.arange(n_agents)[None, :]) & 1).astype(np.int8)


def entropy(distribution: np.ndarray) -> float:
    """H(p) in bits, with the 0 log 0 limit taken."""

    flat = np.asarray(distribution, dtype=float).ravel()
    positive = flat[flat > _TOLERANCE]
    if positive.size == 0:
        return 0.0
    positive = positive / positive.sum()
    return float(-(positive * np.log2(positive)).sum())


def mutual_information(joint: np.ndarray) -> float:
    """I(X;Y) in bits from an exact joint table."""

    joint = np.asarray(joint, dtype=float)
    total = joint.sum()
    if total <= _TOLERANCE:
        return 0.0
    joint = joint / total
    value = entropy(joint.sum(axis=1)) + entropy(joint.sum(axis=0)) - entropy(joint)
    # Clamp the floating-point residue: a structural zero must read as 0.0, not
    # -3e-17, or every pass/fail check on it becomes a judgement call.
    return max(value, 0.0)


def conditional_mutual_information(joint: np.ndarray) -> float:
    """I(X;Y|Z) in bits from an exact joint with axes ordered (X, Z, Y).

    Axis order matches ``mas_cc.analysis.estimators`` deliberately, so the
    exact value and the estimated one are never accidentally computed from
    differently-oriented tables.
    """

    joint = np.asarray(joint, dtype=float)
    total = joint.sum()
    if total <= _TOLERANCE:
        return 0.0
    joint = joint / total
    value = (
        entropy(joint.sum(axis=2))
        + entropy(joint.sum(axis=0))
        - entropy(joint.sum(axis=(0, 2)))
        - entropy(joint)
    )
    return max(value, 0.0)


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """The chain's stationary law, by eigendecomposition.

    Takes the left eigenvector for eigenvalue 1. Falls back to power iteration
    only if the eigen-solve returns something unusable, which it should not for
    an irreducible chain but can for one made reducible by a zero noise level.
    """

    matrix = np.asarray(transition, dtype=float)
    values, vectors = np.linalg.eig(matrix.T)
    index = int(np.argmin(np.abs(values - 1.0)))
    candidate = np.real(vectors[:, index])
    if candidate.sum() < 0:
        candidate = -candidate
    candidate = np.clip(candidate, 0.0, None)
    if candidate.sum() <= _TOLERANCE:
        candidate = np.full(matrix.shape[0], 1.0 / matrix.shape[0])
        for _ in range(10_000):
            nxt = candidate @ matrix
            if np.abs(nxt - candidate).sum() < 1e-14:
                break
            candidate = nxt
    return candidate / candidate.sum()


def propagate(initial: np.ndarray, transition: np.ndarray, steps: int) -> np.ndarray:
    """``p_0 T^t``, by repeated vector-matrix products rather than matrix powers."""

    distribution = np.asarray(initial, dtype=float).copy()
    for _ in range(steps):
        distribution = distribution @ transition
    return distribution / distribution.sum()


def transition_joint(distribution: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """P(X_t = x, X_{t+1} = x') for a given law over X_t."""

    return np.asarray(distribution, dtype=float)[:, None] * np.asarray(transition, dtype=float)


def project(
    joint: np.ndarray, row_key: np.ndarray, column_key: np.ndarray, rows: int, columns: int
) -> np.ndarray:
    """Sum a state-by-state joint into groups given by per-state keys.

    ``row_key[x]`` is the group of the current state, ``column_key[x']`` the
    group of the next state. Two grouped sums rather than a Python loop over
    cells, so a 1024x1024 joint reduces in milliseconds.
    """

    grouped_rows = np.zeros((rows, joint.shape[1]), dtype=float)
    np.add.at(grouped_rows, row_key, joint)
    grouped = np.zeros((columns, rows), dtype=float)
    np.add.at(grouped, column_key, grouped_rows.T)
    return grouped.T


def entropy_rate(transition: np.ndarray, distribution: np.ndarray) -> float:
    """H(X_{t+1} | X_t) in bits at the given law over X_t."""

    matrix = np.asarray(transition, dtype=float)
    rows = np.array([entropy(matrix[state]) for state in range(matrix.shape[0])])
    return float(np.asarray(distribution, dtype=float) @ rows)


def cross_agent_mutual_information(
    distribution: np.ndarray, bits: np.ndarray, left: int, right: int
) -> float:
    """I(A_i; A_j) between two agents at the same instant."""

    table = np.zeros((2, 2), dtype=float)
    np.add.at(table, (bits[:, left], bits[:, right]), np.asarray(distribution, dtype=float))
    return mutual_information(table)


def directed_conditional_mutual_information(
    joint: np.ndarray, bits: np.ndarray, *, target: int, source: int, condition: int
) -> float:
    """I(A_target^{t+1} ; A_source^t | A_condition^t).

    One function covers both structural zeros the Markov game is built to
    expose, because they are the same quantity with different arguments:

    - **Transfer entropy** ``TE(source -> target)`` is ``condition = target``:
      does the source's present tell you anything about the target's future
      that the target's own present did not already.
    - **The chain's conditional independence** is ``condition`` set to the
      middle agent: on ``1 -> 2 -> 3`` the value with ``target=3, source=1,
      condition=2`` must be exactly zero, because agent 3's next action depends
      on the profile only through agent 2's current one.

    Both are exact zeros rather than small numbers, which is what makes them
    pass/fail rather than close/not-close.
    """

    row_key = (bits[:, source].astype(int) << 1) | bits[:, condition].astype(int)
    column_key = bits[:, target].astype(int)
    table = project(joint, row_key, column_key, 4, 2).reshape(2, 2, 2)
    # Axes land as (source^t, condition^t, target^{t+1}) = (X, Z, Y).
    return conditional_mutual_information(table)


def macrostate_law(distribution: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """P(K = k), the number of agents playing action index 1, for k = 0..N."""

    counts = bits.sum(axis=1)
    law = np.zeros(bits.shape[1] + 1, dtype=float)
    np.add.at(law, counts, np.asarray(distribution, dtype=float))
    return law


def macrostate_pair_law(
    joint: np.ndarray, bits: np.ndarray
) -> np.ndarray:
    """P(K_t = k, K_{t+h} = k'), lumped only after the microstate joint is built."""

    counts = bits.sum(axis=1)
    size = bits.shape[1] + 1
    return project(joint, counts, counts, size, size)


def lumping_gap(transition: np.ndarray, bits: np.ndarray) -> float:
    """How far the macrostate chain is from being Markov, in bits.

    Strong lumpability means every microstate sharing a macrostate has the same
    distribution over next macrostates. This returns the largest total-variation
    departure from that, converted to a simple magnitude: exactly zero means the
    lumped chain is a faithful Markov description and the cheap computation
    would have been correct; anything above zero means it is not, and only the
    microstate propagation is right.
    """

    counts = bits.sum(axis=1)
    size = bits.shape[1] + 1
    rows = np.zeros((transition.shape[0], size), dtype=float)
    np.add.at(rows.T, counts, transition.T)
    worst = 0.0
    for macro in range(size):
        members = rows[counts == macro]
        if len(members) < 2:
            continue
        worst = max(worst, float(np.abs(members - members[0]).sum(axis=1).max()))
    return worst


def kl_divergence_rows(rows: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Row-wise ``D_KL(rows[s] || reference[s])`` in bits, with 0 log 0 = 0.

    Absolute continuity is not checked because the only caller cannot violate
    it: the reference is a mixture that includes the row with positive weight,
    so ``reference > 0`` wherever ``rows > 0``.
    """

    rows = np.asarray(rows, dtype=float)
    reference = np.asarray(reference, dtype=float)
    support = rows > _TOLERANCE
    safe_rows = np.where(support, rows, 1.0)
    safe_reference = np.where(reference > _TOLERANCE, reference, 1.0)
    terms = np.where(support, rows * (np.log2(safe_rows) - np.log2(safe_reference)), 0.0)
    return np.maximum(terms.sum(axis=1), 0.0)


def policy_weights(policy: np.ndarray, n_controls: int, n_states: int) -> np.ndarray:
    """``pi(u|s)`` as a full ``(|U|, |S|)`` table.

    A length-``|U|`` vector is the state-independent policy of the extension
    document's section 2 - the case where ``U_t`` is exactly independent of
    ``S_t``, so there is no mediator and no backdoor. A ``(|U|, |S|)`` table is
    the state-dependent generalisation, which every function here already
    handles so that adding it later is a config change rather than an algebra
    change.
    """

    weights = np.asarray(policy, dtype=float)
    if weights.ndim == 1:
        weights = np.broadcast_to(weights[:, None], (n_controls, n_states))
    if weights.shape != (n_controls, n_states):
        raise ValueError("a policy must be pi(u) of length |U| or pi(u|s) of shape (|U|, |S|)")
    return np.asarray(weights, dtype=float) / weights.sum(axis=0, keepdims=True)


def policy_averaged_kernel(kernels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``T-bar[s,s'] = sum_u pi(u|s) T_u[s,s']``.

    Note this is *not* the kernel of any single control value, and for a
    population it is not a product over agents either: a fixed ``u`` pushes
    every controlled agent toward the same target, so averaging over ``u``
    correlates them. Building it as an explicit mixture rather than by putting
    the mean target into the per-agent formula is what keeps that correlation.
    """

    kernels = np.asarray(kernels, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 1:
        weights = weights[:, None]
    return (weights[:, :, None] * kernels).sum(axis=0)


def control_conditioned_kernels(kernels: np.ndarray, tail: np.ndarray) -> np.ndarray:
    """``[T_u T-bar^{h-1}]`` for every u, given ``tail = T-bar^{h-1}``.

    The tail is passed in rather than recomputed because the caller walks the
    horizon upward and can advance it with one matrix product per step; taking
    a fresh matrix power per horizon would be cubic work repeated H times.
    """

    return np.asarray(kernels, dtype=float) @ np.asarray(tail, dtype=float)


def per_state_empowerment(
    forward: np.ndarray, baseline: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """``E_h(s)``, the policy-weighted KL of the extension document's section 2.

    ``forward[u]`` is ``T_u T-bar^{h-1}``, ``baseline`` is ``T-bar^h``, and the
    result is ``sum_u pi(u|s) D_KL(forward[u][s] || baseline[s])`` - which is
    ``I(U_t; S_{t+h} | S_t = s)`` exactly, because marginalising ``forward``
    against the policy *is* ``baseline``.
    """

    forward = np.asarray(forward, dtype=float)
    total = np.zeros(forward.shape[1], dtype=float)
    for index in range(forward.shape[0]):
        total += weights[index] * kl_divergence_rows(forward[index], baseline)
    return total


def visitation_distribution(
    initial: np.ndarray, transition: np.ndarray, window: tuple[int, int]
) -> np.ndarray:
    """``d(s) = (1/|W|) sum_{t in W} p_t(s)`` - states encountered under the policy.

    **Not the stationary distribution.** The naming game is absorbing, so its
    stationary law sits entirely on consensus states where empowerment is zero;
    averaging over it returns ``E ~ 0`` trivially and for the wrong reason. The
    window is the rounds actually analysed, which is why it is a declared config
    parameter rather than an incidental choice.
    """

    first, last = window
    if last < first:
        raise ValueError("the analysis window must not be empty")
    law = propagate(initial, transition, first)
    total = np.zeros_like(law)
    for round_index in range(first, last + 1):
        total += law
        if round_index < last:
            law = law @ transition
    return total / total.sum()


def discounted_visitation(
    initial: np.ndarray, transition: np.ndarray, gamma: float, horizon: int
) -> np.ndarray:
    """``d_gamma(s) = (1-gamma) sum_t gamma^t p_t(s)``, truncated at ``horizon``.

    The other admissible answer to "states encountered under the policy". On an
    absorbing chain the two disagree, and the config has to say which it meant.
    """

    law = np.asarray(initial, dtype=float)
    law = law / law.sum()
    total = np.zeros_like(law)
    for step in range(horizon + 1):
        total += (1.0 - gamma) * gamma**step * law
        law = law @ transition
    return total / total.sum()


def geometric_truncation(gamma: float, tolerance: float) -> int:
    """The smallest ``H`` with ``gamma^H <= tolerance``.

    ``tau ~ Geom(1-gamma)`` has unbounded support, so the gamma-weighted scalar
    is a truncated sum whose residual mass is ``gamma^H``. Deriving H from a
    declared tolerance, and reporting the residual next to the answer, is what
    keeps the truncation error visible rather than assumed.
    """

    if not 0.0 < gamma < 1.0:
        raise ValueError("the empowerment discount must lie strictly in (0, 1)")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("the horizon tolerance must lie strictly in (0, 1)")
    return max(1, int(math.ceil(math.log(tolerance) / math.log(gamma))))


def blahut_arimoto(
    channel: np.ndarray, *, tolerance: float = 1e-12, max_iterations: int = 10_000
) -> tuple[float, np.ndarray]:
    """Channel capacity in bits, and the input law achieving it.

    ``channel[u, s]`` is ``P(S = s | U = u)``. This is the ceiling on control
    authority - how much the input could tell you about the state under the
    best possible input distribution - as distinct from the *design* mutual
    information under the grid actually swept. Reporting both is what separates
    "the controller is weak" from "the grid was badly chosen".
    """

    matrix = np.asarray(channel, dtype=float)
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    law = np.full(matrix.shape[0], 1.0 / matrix.shape[0])
    capacity = 0.0
    for _ in range(max_iterations):
        output = law @ matrix
        # D_u = sum_s W[u,s] log2(W[u,s] / q[s]), the KL divergence in bits from
        # each input's output law to the current mixture.
        safe_output = np.where(output > 0, output, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.log2(np.where(matrix > 0, matrix, 1.0)) - np.log2(safe_output)[None, :]
        divergence = np.sum(np.where(matrix > 0, matrix * log_ratio, 0.0), axis=1)
        # Blahut-Arimoto brackets the capacity between these two on every
        # iteration, so the gap is a real stopping criterion rather than a
        # guess about when the input law stopped moving.
        shifted = divergence - divergence.max()
        weights = law * np.exp2(shifted)
        total = weights.sum()
        if total <= 0:
            break
        lower = divergence.max() + math.log2(total)
        upper = float(divergence.max())
        capacity = lower
        law = weights / total
        if upper - lower < tolerance:
            break
    return max(capacity, 0.0), law


def design_mutual_information(
    design: np.ndarray, conditionals: np.ndarray
) -> float:
    """I(C;O) for a known input law and known per-condition output laws.

    ``design[c]`` is p(c) - the sweep grid, which we chose and therefore know
    exactly - and ``conditionals[c, o]`` is p(o | c), fixed by dynamics we
    wrote. "Depends on an experimental-design choice" and "is not analytically
    computable" are different claims; conditioning on the design, which we can
    always do because we made it, determines the quantity completely.
    """

    design = np.asarray(design, dtype=float)
    design = design / design.sum()
    conditionals = np.asarray(conditionals, dtype=float)
    joint = design[:, None] * conditionals
    return mutual_information(joint)


def plugin_bias_bits(degrees_of_freedom: int, effective_samples: int) -> float:
    """The Miller-Madow first-order plug-in MI bias, ``dof / (2 N ln 2)`` bits.

    ``effective_samples`` must be the number of **episodes**, not rounds:
    within-episode pairs are correlated and do not buy independent samples,
    which is exactly why the bootstrap resamples over ``episode_id``. Shipping
    this next to the truth is what makes the comparison honest - a lagged CMI
    over an unbinned macrostate can carry more bias than signal.
    """

    if effective_samples <= 0:
        return math.inf
    return degrees_of_freedom / (2.0 * effective_samples * math.log(2.0))


__all__ = [
    "bit_table",
    "blahut_arimoto",
    "conditional_mutual_information",
    "control_conditioned_kernels",
    "cross_agent_mutual_information",
    "design_mutual_information",
    "directed_conditional_mutual_information",
    "discounted_visitation",
    "entropy",
    "entropy_rate",
    "geometric_truncation",
    "kl_divergence_rows",
    "lumping_gap",
    "macrostate_law",
    "macrostate_pair_law",
    "mutual_information",
    "per_state_empowerment",
    "plugin_bias_bits",
    "policy_averaged_kernel",
    "policy_weights",
    "project",
    "propagate",
    "stationary_distribution",
    "transition_joint",
    "visitation_distribution",
]
