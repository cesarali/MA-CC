"""Effective empowerment for Game 3, as Song et al. define it.

Reference: *Estimating the Empowerment of Language Model Agents*, arXiv:2509.22504.
Effective empowerment is ``E(pi) = E[I(a_t; s_* | s_t)]`` with
``tau ~ Geom(1-gamma)`` and the expectation over states encountered under the
policy.

**Why this module exists.** Game 3 draws its control input once per episode and
holds it fixed, which makes the state at round ``t`` causally downstream of the
control. Conditioning on ``S_t`` then blocks most of the effect being measured:
the reported CMI decays with ``t`` for design reasons rather than dynamical
ones, and the answer depends on which rounds were pooled. The paper's ``a_t`` is
drawn *at* round ``t``, so ``s_t`` is pre-action and the conditioning is clean.

Rather than replace one with the other, both are computed on the same chain and
the gap between them is reported as a number:

``3a``  ``u`` drawn once per episode, held fixed - what `mas_cc` control
        experiments have always done, and what the sweep machinery in
        `empowerment.py` treats with ``c := u``. Reported as a curve in ``t``,
        because that dependence is the contamination and quantifying it is the
        point.
``3b``  ``u_t ~ pi(.|s_t)`` resampled every round - the paper's quantity, and a
        policy-weighted KL divergence in closed form.

Everything here is exact linear algebra on the finite microstate chain. The one
approximation is the truncation of the geometric horizon sum, and its residual
``gamma^H`` is reported as a quantity so it is visible rather than assumed.

Cost note: the horizon loop is ``H`` matrix products on ``2^N`` states, so it is
milliseconds at the ``N = 6`` the shipped Game 3 configs use and minutes at the
``N = 10`` cap. Nothing here is quadratic in the analysis window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from . import exact
from .protocols import GroundTruthQuantity

PER_ROUND = "per_round"
EPISODE_FIXED = "episode_fixed"
CONTROL_MODES = (PER_ROUND, EPISODE_FIXED)

VISITATION = "visitation"
DISCOUNTED_VISITATION = "discounted_visitation"
STATIONARY = "stationary"
STATE_DISTRIBUTIONS = (VISITATION, DISCOUNTED_VISITATION, STATIONARY)

MICROSTATE = "microstate"
MACROSTATE = "macrostate"
STATE_REPRESENTATIONS = (MICROSTATE, MACROSTATE)

UNIFORM = "uniform"
STATE_DEPENDENT = "state_dependent"


@dataclass(frozen=True, slots=True)
class EmpowermentSpec:
    """The declared analysis settings, resolved from ``game.options``.

    Two of these fields were incidental choices promoted to declared ones,
    because the measured value moves with them and a reader of the artifact
    would otherwise have no way to know which was used:

    ``analysis_window``     which rounds the state distribution averages over.
    ``state_distribution``  whether that average is flat over the window,
                            geometrically discounted, or the stationary law -
                            which on an absorbing chain is a trap, see below.
    """

    gamma: float = 0.9
    horizon_tolerance: float = 1.0e-3
    state_distribution: str = VISITATION
    analysis_window: tuple[int, int] = (1, 1)
    report_per_state_profile: bool = True
    state_representation: tuple[str, ...] = (MICROSTATE, MACROSTATE)
    control_mode: str = EPISODE_FIXED
    policy: str = UNIFORM
    report_horizons: tuple[int, ...] = (1, 2, 5, 10)

    def __post_init__(self) -> None:
        object.__setattr__(self, "analysis_window", tuple(int(v) for v in self.analysis_window))
        object.__setattr__(
            self, "state_representation", tuple(str(v) for v in self.state_representation)
        )
        object.__setattr__(self, "report_horizons", tuple(int(v) for v in self.report_horizons))
        self.validate()

    def validate(self) -> None:
        if self.control_mode not in CONTROL_MODES:
            raise ValueError(f"game.options.control_mode must be one of {CONTROL_MODES}")
        if self.policy == STATE_DEPENDENT:
            # The algebra in `exact.py` already takes a pi(u|s) table; what does
            # not exist yet is a config syntax for writing one down. The
            # extension document stages it deliberately - verify the
            # state-independent case, where U is exactly independent of S, first.
            raise ValueError(
                "game.options.policy = 'state_dependent' has no config syntax yet. The "
                "closed forms accept a pi(u|s) table, but the state-independent case "
                "must be verified first - it is the one with no mediator and no backdoor."
            )
        if self.policy != UNIFORM:
            raise ValueError(f"game.options.policy must be {UNIFORM!r} or {STATE_DEPENDENT!r}")
        if self.state_distribution not in STATE_DISTRIBUTIONS:
            raise ValueError(
                f"empowerment.state_distribution must be one of {STATE_DISTRIBUTIONS}"
            )
        unknown = set(self.state_representation) - set(STATE_REPRESENTATIONS)
        if unknown or not self.state_representation:
            raise ValueError(
                f"empowerment.state_representation must be a non-empty subset of "
                f"{STATE_REPRESENTATIONS}; got {sorted(unknown) or 'nothing'}"
            )
        first, last = self.analysis_window
        if first < 0 or last < first:
            raise ValueError("empowerment.analysis_window must be a non-empty [first, last]")
        if any(horizon < 1 for horizon in self.report_horizons):
            raise ValueError("empowerment.report_horizons must be positive")
        # Raises for a gamma or tolerance outside (0, 1).
        exact.geometric_truncation(self.gamma, self.horizon_tolerance)

    @classmethod
    def from_config(cls, options: Mapping[str, Any], rounds: int) -> "EmpowermentSpec":
        """Read the two declared blocks: the controller's, and the analysis's.

        ``control_mode`` defaults to ``episode_fixed`` rather than to the
        paper's ``per_round``, so every config written before this module
        existed replays bit-identically. The per-round variant is opted into.
        """

        block = dict(options.get("empowerment", {}) or {})
        window = block.get("analysis_window")
        defaults = cls()
        return cls(
            gamma=float(block.get("gamma", defaults.gamma)),
            horizon_tolerance=float(
                block.get("horizon_tolerance", defaults.horizon_tolerance)
            ),
            state_distribution=str(
                block.get("state_distribution", defaults.state_distribution)
            ),
            analysis_window=(1, max(1, int(rounds)))
            if window is None
            else (int(window[0]), int(window[1])),
            report_per_state_profile=bool(
                block.get("report_per_state_profile", defaults.report_per_state_profile)
            ),
            state_representation=tuple(
                block.get("state_representation", defaults.state_representation)
            ),
            control_mode=str(options.get("control_mode", defaults.control_mode)),
            policy=str(options.get("policy", defaults.policy)),
            report_horizons=tuple(
                block.get("report_horizons", defaults.report_horizons)
            ),
        )

    @property
    def truncation_horizon(self) -> int:
        return exact.geometric_truncation(self.gamma, self.horizon_tolerance)

    @property
    def residual_mass(self) -> float:
        """``gamma^H`` - the geometric mass the truncated sum leaves out."""

        return float(self.gamma**self.truncation_horizon)

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(range(1, self.truncation_horizon + 1))

    @property
    def discount_weights(self) -> np.ndarray:
        """``(1-gamma) gamma^{h-1}`` for ``h = 1..H``, the law of ``tau``."""

        horizons = np.arange(1, self.truncation_horizon + 1)
        return (1.0 - self.gamma) * self.gamma ** (horizons - 1)

    @property
    def reported_horizons(self) -> tuple[int, ...]:
        """The horizons that get their own per-round 3a curve.

        A subset, because the 3a quantity is a curve in ``t`` as well as ``h``
        and emitting the full product would bury the artifact. The whole array
        still goes into the ground truth's parameters.
        """

        limit = self.truncation_horizon
        return tuple(sorted({h for h in self.report_horizons if h <= limit})) or (1,)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gamma": self.gamma,
            "horizon_tolerance": self.horizon_tolerance,
            "state_distribution": self.state_distribution,
            "analysis_window": list(self.analysis_window),
            "report_per_state_profile": self.report_per_state_profile,
            "state_representation": list(self.state_representation),
            "control_mode": self.control_mode,
            "policy": self.policy,
            "report_horizons": list(self.reported_horizons),
            "truncation_horizon": self.truncation_horizon,
        }


@dataclass(frozen=True, slots=True)
class EmpowermentReport:
    """Every exact quantity the extension document asks for, on one chain.

    The per-horizon curve is the primary output and the scalar is derived from
    it - deliberately that way round, because the scalar is recoverable from the
    curve and the curve is not recoverable from the scalar.
    """

    gamma: float
    horizons: tuple[int, ...]
    state_distribution: np.ndarray
    macrostate_occupancy: np.ndarray
    per_state: np.ndarray
    per_macrostate: np.ndarray
    microstate_curve: np.ndarray
    macrostate_curve: np.ndarray
    scalar: float
    macrostate_scalar: float
    stationary_scalar: float
    residual: float
    marginalisation_error: float
    episode_fixed_horizons: tuple[int, ...]
    episode_fixed_rounds: tuple[int, ...]
    episode_fixed: np.ndarray
    episode_fixed_macrostate: np.ndarray

    @property
    def discounted_state_profile(self) -> np.ndarray:
        """``sum_h (1-gamma) gamma^{h-1} E_h(s)`` - the paper's empowerment trace."""

        return self._discount() @ self.per_state

    @property
    def discounted_macrostate_profile(self) -> np.ndarray:
        return self._discount() @ self.per_macrostate

    def _discount(self) -> np.ndarray:
        horizons = np.asarray(self.horizons, dtype=float)
        return (1.0 - self.gamma) * self.gamma ** (horizons - 1.0)

    def episode_fixed_is_monotone(self, horizon: int) -> bool:
        """Whether ``I^3a_h(t)`` decreases in ``t``, as the exact algebra predicts.

        If it does not, the kernel is not actually being applied from round 0 -
        the state at round ``t`` is then not downstream of the control and the
        contamination this whole variant exists to expose is absent.
        """

        index = self.episode_fixed_horizons.index(horizon)
        curve = self.episode_fixed[index]
        return bool(np.all(np.diff(curve) <= 1e-12))

    def gap(self, horizon: int) -> np.ndarray:
        """``Delta_h(t) = E_h^{3b} - I^3a_h(t)``, the number the argument becomes."""

        index = self.episode_fixed_horizons.index(horizon)
        return self.microstate_curve[self.horizons.index(horizon)] - self.episode_fixed[index]


def _state_distribution(
    spec: EmpowermentSpec, initial: np.ndarray, averaged: np.ndarray
) -> np.ndarray:
    """``d(s)``, propagated under the policy-averaged kernel.

    Under the policy, not under any one control value: the paper averages over
    "states encountered under the policy", and a chain driven by a single ``u``
    visits a different set of them.
    """

    if spec.state_distribution == STATIONARY:
        return exact.stationary_distribution(averaged)
    if spec.state_distribution == DISCOUNTED_VISITATION:
        return exact.discounted_visitation(
            initial, averaged, spec.gamma, spec.analysis_window[1]
        )
    return exact.visitation_distribution(initial, averaged, spec.analysis_window)


def _lumped_conditional_mutual_information(
    joint: np.ndarray, counts: np.ndarray, size: int
) -> float:
    """``I(U; M_{t+h} | M_t)`` from a microstate joint with axes (U, S, S').

    Both the conditioning variable and the target are coarse-grained, which is
    why this cannot be bounded against the microstate value in either
    direction: coarse-graining what you condition on can raise a conditional
    mutual information as easily as lower it. A violation of the "obvious"
    inequality is therefore not a bug signal.
    """

    lumped = np.stack(
        [exact.project(plane, counts, counts, size, size) for plane in joint]
    )
    return exact.conditional_mutual_information(lumped)


def analyse(
    kernels: np.ndarray,
    initial: np.ndarray,
    bits: np.ndarray,
    spec: EmpowermentSpec,
    *,
    policy: Sequence[float] | np.ndarray | None = None,
) -> EmpowermentReport:
    """Both variants, both state representations, on one family of kernels.

    ``kernels[u]`` is ``T_u``, the microstate transition matrix under control
    value ``u``. ``policy`` is ``pi(u)``, uniform when omitted; it is required
    to be state-independent because variant 3a - ``u`` drawn once at the start
    of the episode - has no state to depend on.
    """

    kernels = np.asarray(kernels, dtype=float)
    if kernels.ndim != 3 or kernels.shape[1] != kernels.shape[2]:
        raise ValueError("kernels must be a stack of square microstate matrices")
    n_controls, n_states = kernels.shape[0], kernels.shape[1]
    design = (
        np.full(n_controls, 1.0 / n_controls)
        if policy is None
        else np.asarray(policy, dtype=float)
    )
    if design.ndim != 1:
        raise ValueError(
            "variant 3a draws u before any state exists, so the policy must be pi(u)"
        )
    design = design / design.sum()
    weights = exact.policy_weights(design, n_controls, n_states)
    averaged = exact.policy_averaged_kernel(kernels, weights)

    counts = bits.sum(axis=1)
    macro_size = bits.shape[1] + 1
    initial = np.asarray(initial, dtype=float)
    initial = initial / initial.sum()

    distribution = _state_distribution(spec, initial, averaged)
    occupancy = np.zeros(macro_size, dtype=float)
    np.add.at(occupancy, counts, distribution)

    horizons = spec.horizons
    wants_macro = MACROSTATE in spec.state_representation
    per_state = np.zeros((len(horizons), n_states), dtype=float)
    micro_curve = np.zeros(len(horizons), dtype=float)
    macro_curve = np.zeros(len(horizons), dtype=float)
    stationary = exact.stationary_distribution(averaged)
    stationary_curve = np.zeros(len(horizons), dtype=float)
    marginalisation_error = 0.0

    tail = np.eye(n_states)
    for position, _ in enumerate(horizons):
        forward = exact.control_conditioned_kernels(kernels, tail)
        # Marginalising the action must recover the policy-averaged h-step
        # kernel exactly. Computing the baseline as that marginal and comparing
        # it against T-bar @ tail is the identity check, kept as a shipped
        # number because it catches kernel-construction errors immediately.
        baseline = exact.policy_averaged_kernel(forward, weights)
        marginalisation_error = max(
            marginalisation_error, float(np.abs(baseline - averaged @ tail).max())
        )
        values = exact.per_state_empowerment(forward, baseline, weights)
        per_state[position] = values
        micro_curve[position] = float(distribution @ values)
        stationary_curve[position] = float(stationary @ values)
        if wants_macro:
            macro_curve[position] = _lumped_conditional_mutual_information(
                distribution[None, :, None] * weights[:, :, None] * forward, counts, macro_size
            )
        tail = tail @ averaged

    per_macrostate = np.zeros((len(horizons), macro_size), dtype=float)
    for position in range(len(horizons)):
        weighted = np.zeros(macro_size, dtype=float)
        np.add.at(weighted, counts, distribution * per_state[position])
        per_macrostate[position] = np.divide(
            weighted, occupancy, out=np.zeros(macro_size), where=occupancy > 0
        )

    discount = spec.discount_weights
    rounds, micro_fixed, macro_fixed = _episode_fixed_curves(
        kernels, initial, design, counts, macro_size, spec, wants_macro
    )

    return EmpowermentReport(
        gamma=spec.gamma,
        horizons=horizons,
        state_distribution=distribution,
        macrostate_occupancy=occupancy,
        per_state=per_state,
        per_macrostate=per_macrostate,
        microstate_curve=micro_curve,
        macrostate_curve=macro_curve,
        scalar=float(discount @ micro_curve),
        macrostate_scalar=float(discount @ macro_curve),
        stationary_scalar=float(discount @ stationary_curve),
        residual=spec.residual_mass,
        marginalisation_error=marginalisation_error,
        episode_fixed_horizons=spec.reported_horizons,
        episode_fixed_rounds=rounds,
        episode_fixed=micro_fixed,
        episode_fixed_macrostate=macro_fixed,
    )


def _episode_fixed_curves(
    kernels: np.ndarray,
    initial: np.ndarray,
    design: np.ndarray,
    counts: np.ndarray,
    macro_size: int,
    spec: EmpowermentSpec,
    wants_macro: bool,
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    """``I^3a_h(t)`` over the analysis window, micro and macro.

    ``u`` is drawn once from ``p(u)`` and the chain is ``T_u`` throughout, so
    ``p_t(s|u) = [p_0 T_u^t]_s`` and the joint is
    ``p(u) p_t(s|u) [T_u^h]_{ss'}``. The value is a function of ``t`` and that
    dependence is exactly the contamination: ``s_t`` is causally downstream of
    ``u``, so conditioning on it blocks the path being measured.
    """

    horizons = spec.reported_horizons
    first, last = spec.analysis_window
    rounds = tuple(range(first, last + 1))
    powers = np.stack(
        [
            np.stack([np.linalg.matrix_power(kernel, horizon) for kernel in kernels])
            for horizon in horizons
        ]
    )
    laws = np.stack([exact.propagate(initial, kernel, first) for kernel in kernels])

    micro = np.zeros((len(horizons), len(rounds)), dtype=float)
    macro = np.zeros((len(horizons), len(rounds)), dtype=float)
    for offset, round_index in enumerate(rounds):
        for position in range(len(horizons)):
            joint = design[:, None, None] * laws[:, :, None] * powers[position]
            micro[position, offset] = exact.conditional_mutual_information(joint)
            if wants_macro:
                macro[position, offset] = _lumped_conditional_mutual_information(
                    joint, counts, macro_size
                )
        if round_index < last:
            laws = np.einsum("us,ust->ut", laws, kernels)
    return rounds, micro, macro


def _anchor_rounds(rounds: tuple[int, ...], count: int = 8) -> tuple[int, ...]:
    """Evenly spaced rounds from the window, so a long window stays readable."""

    if len(rounds) <= count:
        return rounds
    positions = np.unique(np.linspace(0, len(rounds) - 1, count).round().astype(int))
    return tuple(int(rounds[position]) for position in positions)


def empowerment_quantities(
    report: EmpowermentReport, spec: EmpowermentSpec
) -> tuple[GroundTruthQuantity, ...]:
    """The report as answer-key rows, with the curve kept as the primary output."""

    quantities: list[GroundTruthQuantity] = [
        GroundTruthQuantity(
            name="effective_empowerment",
            value=report.scalar,
            definition=(
                "E(pi) = sum_h (1-gamma) gamma^{h-1} sum_s d(s) E_h(s) over the "
                "microstate, with u resampled every round (variant 3b)."
            ),
        ),
        GroundTruthQuantity(
            name="effective_empowerment_truncation_residual",
            value=report.residual,
            units="probability",
            definition=(
                "gamma^H, the geometric mass the truncated horizon sum omits. Logged "
                "so the truncation error is visible rather than assumed."
            ),
        ),
        GroundTruthQuantity(
            name="effective_empowerment_horizon",
            value=float(spec.truncation_horizon),
            units="rounds",
            definition="H, chosen as ceil(ln(tolerance)/ln(gamma)).",
        ),
        GroundTruthQuantity(
            name="empowerment_marginalisation_error",
            value=report.marginalisation_error,
            units="probability",
            definition=(
                "max |sum_u pi(u|s)[T_u T-bar^{h-1}] - T-bar^h| over the horizon sum. "
                "A structural zero: nonzero means the kernel family is malformed."
            ),
        ),
        GroundTruthQuantity(
            name="stationary_effective_empowerment",
            value=report.stationary_scalar,
            definition=(
                "The same scalar averaged over the stationary law instead of the "
                "visitation one. On an absorbing chain it collapses toward zero for "
                "the wrong reason; shipped so the trap is visible, not so it is used."
            ),
        ),
    ]

    if MACROSTATE in spec.state_representation:
        quantities.append(
            GroundTruthQuantity(
                name="effective_empowerment_macrostate",
                value=report.macrostate_scalar,
                definition=(
                    "The same scalar with both S_t and S_{t+h} coarse-grained to the "
                    "macrostate. No inequality holds against the microstate value in "
                    "either direction - coarse-graining the conditioning variable can "
                    "raise a conditional MI as easily as lower it."
                ),
            )
        )
        quantities.append(
            GroundTruthQuantity(
                name="coarse_graining_gap",
                value=report.scalar - report.macrostate_scalar,
                definition=(
                    "microstate minus macrostate effective empowerment: the cost of "
                    "coarse-graining, measured exactly rather than estimated on "
                    "embeddings. Sign is not constrained."
                ),
            )
        )

    for position, horizon in enumerate(report.horizons):
        quantities.append(
            GroundTruthQuantity(
                name="empowerment_at_horizon",
                value=float(report.microstate_curve[position]),
                subject=(str(horizon),),
                definition="E_h = sum_s d(s) E_h(s), the per-horizon curve (variant 3b).",
            )
        )
        if MACROSTATE in spec.state_representation:
            quantities.append(
                GroundTruthQuantity(
                    name="macrostate_empowerment_at_horizon",
                    value=float(report.macrostate_curve[position]),
                    subject=(str(horizon),),
                    definition="I(U; M_{t+h} | M_t) at this horizon, lumped last.",
                )
            )

    if spec.report_per_state_profile:
        profile = report.discounted_macrostate_profile
        for macro in range(profile.shape[0]):
            quantities.append(
                GroundTruthQuantity(
                    name="empowerment_by_macrostate",
                    value=float(profile[macro]),
                    subject=(str(macro),),
                    definition=(
                        "The gamma-weighted empowerment of states with this many agents "
                        "on the second action - the analogue of the paper's empowerment "
                        "traces. Expected to peak near an even split and vanish at "
                        "consensus. Zero where the macrostate is never visited."
                    ),
                )
            )

    anchors = _anchor_rounds(report.episode_fixed_rounds)
    for position, horizon in enumerate(report.episode_fixed_horizons):
        for round_index in anchors:
            offset = report.episode_fixed_rounds.index(round_index)
            quantities.append(
                GroundTruthQuantity(
                    name="episode_fixed_conditional_mutual_information",
                    value=float(report.episode_fixed[position, offset]),
                    subject=(str(horizon), str(round_index)),
                    definition=(
                        "I(U; S_{t+h} | S_t) with u drawn once per episode (variant 3a). "
                        "A function of t, and that dependence is the contamination."
                    ),
                )
            )
            quantities.append(
                GroundTruthQuantity(
                    name="empowerment_gap",
                    value=float(
                        report.microstate_curve[report.horizons.index(horizon)]
                        - report.episode_fixed[position, offset]
                    ),
                    subject=(str(horizon), str(round_index)),
                    definition=(
                        "Delta_h(t) = E_h^{3b} - I^3a_h(t). The difference between the "
                        "paper's quantity and the episode-fixed design, as a number."
                    ),
                )
            )
        quantities.append(
            GroundTruthQuantity(
                name="episode_fixed_decays_with_round",
                value=1.0 if report.episode_fixed_is_monotone(horizon) else 0.0,
                subject=(str(horizon),),
                units="indicator",
                definition=(
                    "1 when I^3a_h(t) is non-increasing across the window, as the exact "
                    "algebra predicts. 0 means the kernel is not applied from round 0."
                ),
            )
        )
    return tuple(quantities)


def empowerment_parameters(
    report: EmpowermentReport, spec: EmpowermentSpec
) -> dict[str, Any]:
    """The full curves, for the artifact - the quantities carry only anchors."""

    parameters: dict[str, Any] = {
        "empowerment": spec.to_dict(),
        "empowerment_horizon_curve": report.microstate_curve.tolist(),
        "empowerment_episode_fixed_rounds": list(report.episode_fixed_rounds),
        "empowerment_episode_fixed_curve": {
            str(horizon): report.episode_fixed[position].tolist()
            for position, horizon in enumerate(report.episode_fixed_horizons)
        },
        "empowerment_macrostate_occupancy": report.macrostate_occupancy.tolist(),
    }
    if MACROSTATE in spec.state_representation:
        parameters["empowerment_macrostate_horizon_curve"] = report.macrostate_curve.tolist()
        parameters["empowerment_episode_fixed_macrostate_curve"] = {
            str(horizon): report.episode_fixed_macrostate[position].tolist()
            for position, horizon in enumerate(report.episode_fixed_horizons)
        }
    if spec.report_per_state_profile:
        parameters["empowerment_microstate_profile"] = (
            report.discounted_state_profile.tolist()
        )
        parameters["empowerment_macrostate_profile"] = (
            report.discounted_macrostate_profile.tolist()
        )
    return parameters


__all__ = [
    "CONTROL_MODES",
    "DISCOUNTED_VISITATION",
    "EPISODE_FIXED",
    "EmpowermentReport",
    "EmpowermentSpec",
    "MACROSTATE",
    "MICROSTATE",
    "PER_ROUND",
    "STATE_DISTRIBUTIONS",
    "STATIONARY",
    "VISITATION",
    "analyse",
    "empowerment_parameters",
    "empowerment_quantities",
]
