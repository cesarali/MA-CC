"""Empirical single-affinity estimators, in the exact coordinates of the theory.

Everything in this module is estimated from recorded LLM rounds, but every
quantity is *defined* the way ``theory_revised.py`` defines it, so an empirical
number and a theory number of the same name may be compared without a
conversion step.  The unit contract, which the tests enforce:

===========================  ===========================================
quantity                     units
===========================  ===========================================
``chi(n)``                   target FRACTION change per cycle
``T_pi`` / CMI               bits (the direct-counting convention)
Pinsker numerator            bits (it carries the ``/ln 2``)
``h`` (effective affinity)   nats (natural log of a transition-odds ratio)
``gamma`` (compliance)       probability in ``[0,1]``
``J_c``                      target COUNT change per cycle (``chi`` times N)
``h*J_c``, ``I_sens``        nats (per cycle, or summed over the horizon)
``eta_IR``, ``eta_th``       dimensionless
===========================  ===========================================

Three deliberate choices are worth naming, because each one is a place a
plausible-looking shortcut would silently give the wrong number.

1. The response is ``delta_p_ctrl`` (target fraction), never ``delta_m_ctrl``
   (aligned magnetization).  They differ by ``K/(K-1)``, so a magnetization
   response squared inside the Pinsker numerator is inflated by ``(K/(K-1))^2``
   - 2.25x on a three-option task.
2. The current is ``N * sum_n p(n) a(n) chi(n)``, evaluated state by state.
   ``N * mean(a) * mean(chi)`` is a different number whenever the policy and
   the response covary across states, which is exactly the interesting case.
3. Sensing information is the SCALAR channel ``I(n_Z ; Y_Z)``, and the headline
   estimate uses the empirical occupancy with the *exact* hypergeometric sensor
   kernel.  The full K-option ``I(N ; Y)`` is a larger, different quantity.

A "state" here always means the target count before the round, ``n_Z,k``.  A
state is *identified* when both controller actions were observed there: with
only one action there is no difference of conditional means to take, and this
module refuses to pretend the missing arm implies ``chi(n)=0``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np

from mas_cc.analysis.estimators import conditional_mutual_information
from mas_cc.games.relational_reasoning.imitation_round_feedback.theory_revised import (
    THEORY_API_VERSION,
    THEORY_MODULE,
    THEORY_REFERENCE,
    THEORY_SEMANTICS_VERSION,
    calibrate_affinity_compliance_from_counts,
    sensor_kernel,
    sensing_information_nats,
)

NO_OP = "NO_OP"
ADVOCATE_ACTIONS = frozenset({"ADVOCATE_Z", "ADVOCATE_TARGET"})
CONTROLLER_ACTIONS = ADVOCATE_ACTIONS | {NO_OP}
"""Both spellings a game may have written for "the controller advocated".

Named here rather than imported from a game's controller module, the same way
`effective_affinity.py` does it, so this reusable analysis layer does not
depend on any one game package."""

PROVENANCE: Mapping[str, str] = {
    "theory_semantics_version": THEORY_SEMANTICS_VERSION,
    "theory_reference": THEORY_REFERENCE,
    "theory_module": THEORY_MODULE,
    "theory_api_version": THEORY_API_VERSION,
    "response_coordinate": "target_fraction",
    "response_conditioning": "target_count_before",
    "sensing_coordinate": "target_count",
    "sensing_log_base": "e",
    "actuation_information_log_base": "2",
    "affinity_log_base": "e",
    "current_units": "target_count_per_cycle",
    "eta_ir_aggregation": "occupancy_ratio_of_sums",
    "eta_th_aggregation": "finite_horizon_ratio_of_sums",
    "bootstrap_unit": "episode",
}
"""What was done, in machine-readable form, so a stored row is reconstructable.

Persisted next to every ``eta_ir``/``eta_th`` row.  Written as a constant
rather than assembled per call because it describes the estimator's contract,
not one run's data - if a definition here ever changes, this dictionary and
:data:`THEORY_SEMANTICS_VERSION` must change with it."""


# ---------------------------------------------------------------------------
# 0. Row access
# ---------------------------------------------------------------------------


def controlled_rows(rows: Sequence[Any]) -> list[Any]:
    """The rows a controller-response estimate may be built from.

    Exactly the eligibility the target CMI uses, so the occupancy the numerator
    is weighted over is the occupancy the denominator actually saw.
    """

    return [row for row in rows if str(row.U_k) in CONTROLLER_ACTIONS]


def _advocated(row: Any) -> bool:
    return str(row.U_k) in ADVOCATE_ACTIONS


def population_size(rows: Sequence[Any]) -> int | None:
    """The one ``N`` these rows share, or ``None`` if they disagree."""

    sizes = set()
    for row in rows:
        value = row.event.get("N")
        sizes.add(int(value) if value is not None else int(sum(row.N_k)))
    return sizes.pop() if len(sizes) == 1 else None


def sensor_sample_size(rows: Sequence[Any]) -> int | None:
    """The one ``q_c`` these rows share, or ``None`` if they disagree."""

    sizes = {
        int(row.event["sensor_sample_size"])
        for row in rows
        if row.event.get("sensor_sample_size") is not None
    }
    return sizes.pop() if len(sizes) == 1 else None


def episode_key(row: Any) -> tuple[str, str]:
    """``(cell, episode)`` with the episode name the micro-slot rows use.

    Study aggregation prefixes episode ids with the run and cell, but the
    micro-slot table keeps the runtime-local name.  Bootstrapping has to
    resample a round row and its own micro-slot rows together, so both sides
    are keyed on the local name here.
    """

    local = row.event.get("episode_id")
    if local is None:
        local = str(row.episode_id).rsplit("/", 1)[-1]
    return str(row.cell_id), str(local)


# ---------------------------------------------------------------------------
# 1. State-resolved response chi(n) and action weight a(n)
# ---------------------------------------------------------------------------


def state_response_table(rows: Sequence[Any]) -> dict[int, dict[str, Any]]:
    """``chi_hat(n)`` and ``a_hat(n)`` for every visited target count.

    ``chi_hat(n) = E[dx | ADVOCATE, n] - E[dx | NO_OP, n]`` with
    ``dx = delta_p_ctrl``.  Because ``n`` is held fixed, the difference of the
    two conditional means IS the action-induced separation; no before/after
    bookkeeping is needed on top of it.

    Every visited state appears, including single-action ones - they carry
    ``chi = NaN`` and ``identified = False`` so a caller can report the
    occupancy mass it had to leave out instead of absorbing it as a zero.
    """

    buckets: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"advocate": [], "no_op": []}
    )
    for row in controlled_rows(rows):
        delta = row.event.get("delta_p_ctrl")
        if delta is None:
            continue
        arm = "advocate" if _advocated(row) else "no_op"
        buckets[int(row.target_before)][arm].append(float(delta))
    table: dict[int, dict[str, Any]] = {}
    for state, arms in sorted(buckets.items()):
        advocate, no_op = arms["advocate"], arms["no_op"]
        observed = len(advocate) + len(no_op)
        identified = bool(advocate and no_op)
        table[state] = {
            "n_target": state,
            "chi": (
                float(np.mean(advocate)) - float(np.mean(no_op))
                if identified
                else math.nan
            ),
            "a": math.nan if not observed else len(advocate) / observed,
            "advocate_count": len(advocate),
            "no_op_count": len(no_op),
            "observations": observed,
            "identified": identified,
        }
    return table


def pooled_occupancy(rows: Sequence[Any]) -> dict[int, float]:
    """``p_hat(n)`` over the controlled rows, pooled across rounds.

    This is the weighting that matches a pooled CMI denominator, so it is what
    ``eta_ir`` uses.  The thermodynamic quantities use the round-resolved
    occupancy instead - see :func:`round_occupancy`.
    """

    counts: dict[int, int] = defaultdict(int)
    eligible = controlled_rows(rows)
    for row in eligible:
        counts[int(row.target_before)] += 1
    total = len(eligible)
    return (
        {} if not total else {n: count / total for n, count in sorted(counts.items())}
    )


def round_occupancy(rows: Sequence[Any]) -> dict[int, dict[int, float]]:
    """``p_hat_k(n)`` per round index ``k``.

    One distribution per round, never one pooled histogram: episodes end at
    different times, so pooling would silently over-weight the rounds that more
    episodes reached, and the finite-horizon thermodynamic sum is defined
    round by round in the first place.
    """

    by_round: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for row in controlled_rows(rows):
        by_round[int(row.round_index)][int(row.target_before)] += 1
    result: dict[int, dict[int, float]] = {}
    for k, counts in sorted(by_round.items()):
        total = sum(counts.values())
        result[k] = {n: count / total for n, count in sorted(counts.items())}
    return result


def _support_diagnostics(
    table: Mapping[int, Mapping[str, Any]], occupancy: Mapping[int, float]
) -> dict[str, Any]:
    identified = [item for item in table.values() if item["identified"]]
    events = sum(int(item["observations"]) for item in table.values())
    dual_events = sum(int(item["observations"]) for item in identified)
    mass = sum(occupancy.get(int(item["n_target"]), 0.0) for item in identified)
    return {
        "chi_state_count": len(table),
        "chi_identified_state_count": len(identified),
        "chi_dual_action_state_fraction": (
            math.nan if not table else len(identified) / len(table)
        ),
        "chi_dual_action_event_fraction": (
            math.nan if not events else dual_events / events
        ),
        "chi_identified_occupancy_mass": mass,
    }


def susceptibility_summary(rows: Sequence[Any]) -> dict[str, Any]:
    """Occupancy-weighted ``chi``, as a summary of the state-resolved object.

    The state-resolved table is primary; this scalar exists only to put one
    number on a plot axis.  It averages over identified states with the
    occupancy those states actually had, renormalised over the identified mass,
    and reports that mass so a reader can see how much of the run it speaks for.
    """

    table = state_response_table(rows)
    occupancy = pooled_occupancy(rows)
    weighted = 0.0
    mass = 0.0
    for state, item in table.items():
        if not item["identified"]:
            continue
        weight = occupancy.get(state, 0.0)
        weighted += weight * float(item["chi"])
        mass += weight
    return {
        "susceptibility_occupancy_weighted": math.nan if mass <= 0 else weighted / mass,
        **_support_diagnostics(table, occupancy),
    }


# ---------------------------------------------------------------------------
# 2. Information-response efficiency
# ---------------------------------------------------------------------------


def target_actuation_cmi_bits(rows: Sequence[Any]) -> float:
    """``I(U_k ; n_{k+1} | n_k)`` in bits, by direct counting."""

    eligible = controlled_rows(rows)
    if not eligible:
        return math.nan
    return float(
        conditional_mutual_information(
            [str(row.U_k) for row in eligible],
            [int(row.target_after) for row in eligible],
            [int(row.target_before) for row in eligible],
        ).unsmoothed
    )


def eta_ir(rows: Sequence[Any]) -> dict[str, Any]:
    """Occupancy-level information-response efficiency.

    The state-local inequality is ``T_pi(n) >= 2 a_n (1-a_n) chi(n)^2 / ln 2``.
    It survives an occupancy sum term by term, so the headline is a ratio of
    sums::

        eta_IR = sum_n p(n) B_IR(n)  /  I(U ; n' | n)

    and not a mean of state-local ratios, which would give sparse states the
    same say as the states the run actually spent its time in.

    ``a_hat(n)`` comes from the same rows as the denominator, so the numerator's
    action-mixing weight and the denominator's action channel are the same
    object rather than two estimates that happen to be nearby.

    Restricting the numerator to identified states does not bias the ratio
    against itself: a state where only one action was ever taken contributes
    nothing to the CMI in the denominator either, so both sides drop exactly
    the same states.  The reported support mass says how much occupancy that
    was.
    """

    table = state_response_table(rows)
    occupancy = pooled_occupancy(rows)
    numerator = 0.0
    mass = 0.0
    for state, item in table.items():
        if not item["identified"]:
            continue
        weight = occupancy.get(state, 0.0)
        a, chi = float(item["a"]), float(item["chi"])
        numerator += weight * 2.0 * a * (1.0 - a) * chi * chi / math.log(2.0)
        mass += weight
    denominator = target_actuation_cmi_bits(rows)
    valid = bool(math.isfinite(denominator) and denominator > 0.0 and mass > 0.0)
    return {
        "eta_ir": (numerator / denominator) if valid else math.nan,
        "eta_ir_pinsker_numerator_bits": numerator,
        "eta_ir_denominator_T_bits": denominator,
        # Same number under both names the analysis contract asks for.
        "eta_ir_support_mass": mass,
        "eta_ir_identified_occupancy_mass": mass,
        "eta_ir_valid": valid,
        **_support_diagnostics(table, occupancy),
    }


# ---------------------------------------------------------------------------
# 3. Sensing information in the scalar target channel
# ---------------------------------------------------------------------------


def target_sensing_information(rows: Sequence[Any]) -> dict[str, Any]:
    """``I(n_Z,k ; Y_Z,k)`` in nats, per round and summed over the horizon.

    The sensing mechanism is known exactly - the controller draws ``q_c``
    agents without replacement - so the channel ``S(y|n)`` is the
    hypergeometric law, not something to be estimated.  Only the population
    occupancy is empirical.  Combining the two is both unbiased with respect to
    the visited states and far quieter than counting realised ``(n, y)`` pairs.

    The horizon sum is the primary quantity: it is what enters the finite-time
    efficiency.  The per-round mean is reported next to it for reading.
    """

    eligible = controlled_rows(rows)
    N = population_size(eligible)
    q_c = sensor_sample_size(eligible)
    blank = {
        "target_sensing_information_nats": math.nan,
        "target_sensing_information_horizon_nats": math.nan,
        "target_sensing_information_rounds": 0,
        "target_sensing_valid": False,
        "target_sensing_per_round_nats": {},
    }
    if N is None or q_c is None or not 1 <= q_c <= N:
        return blank
    S = sensor_kernel(N, q_c)
    per_round: dict[int, float] = {}
    for k, occupancy in round_occupancy(eligible).items():
        vector = np.zeros(N + 1, dtype=float)
        for n, weight in occupancy.items():
            if 0 <= n <= N:
                vector[n] = weight
        total = float(vector.sum())
        if total <= 0.0:
            continue
        value, _ = sensing_information_nats(vector / total, S)
        if math.isfinite(value):
            per_round[k] = float(value)
    if not per_round:
        return blank
    horizon = float(sum(per_round.values()))
    return {
        "target_sensing_information_nats": horizon / len(per_round),
        "target_sensing_information_horizon_nats": horizon,
        "target_sensing_information_rounds": len(per_round),
        "target_sensing_valid": True,
        "target_sensing_per_round_nats": per_round,
    }


# ---------------------------------------------------------------------------
# 4. Effective affinity and kinetic compliance
# ---------------------------------------------------------------------------


def _micro_target(row: Mapping[str, Any]) -> str | None:
    for key in ("analysis_target", "controller_target", "round_controller_target"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return None


def controlled_transition_counts(
    micro_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """The 2x2 controlled microscopic transition table behind ``h`` and ``gamma``.

    Only slots that were actually intervened on inside an advocating round
    count: those are the opportunities the controlled kernel describes.
    ``plus`` is ``non-target -> target``, ``minus`` is ``target -> non-target``.
    """

    counts = {
        "plus_transitions": 0,
        "plus_eligible": 0,
        "minus_transitions": 0,
        "minus_eligible": 0,
    }
    for row in micro_rows:
        action = str(
            row.get("round_controller_action", row.get("controller_action", ""))
        )
        if action not in ADVOCATE_ACTIONS or not bool(row.get("controlled_slot")):
            continue
        target = _micro_target(row)
        before, after = row.get("focal_opinion_before"), row.get("focal_opinion_after")
        if target is None or before is None or after is None:
            continue
        if str(before) == target:
            counts["minus_eligible"] += 1
            counts["minus_transitions"] += int(str(after) != target)
        else:
            counts["plus_eligible"] += 1
            counts["plus_transitions"] += int(str(after) == target)
    return counts


def affinity_compliance(micro_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Unsmoothed ``h = ln(p_+/p_-)`` in nats and ``gamma = p_+ + p_-``.

    Deliberately unsmoothed.  A direction that was eligible but never fired
    does not have a finite affinity, and adding an epsilon purely so a
    downstream ratio becomes printable would manufacture a number the data does
    not contain.  Such a cell reports ``affinity_valid = False`` and its
    ``eta_th`` is ``NaN``; the raw counts stay in the row so the reason is
    visible.  This routes through the theory module's own calibration so the
    empirical and theoretical calibrations cannot drift apart.
    """

    counts = controlled_transition_counts(micro_rows)
    result: dict[str, Any] = {
        **counts,
        "p_plus": math.nan,
        "p_minus": math.nan,
        "effective_affinity": math.nan,
        "kinetic_compliance": math.nan,
        "affinity_valid": False,
    }
    try:
        calibration = calibrate_affinity_compliance_from_counts(**counts)
    except ValueError:
        return result
    result.update(
        {
            "p_plus": calibration.p_plus,
            "p_minus": calibration.p_minus,
            "effective_affinity": calibration.h_eff,
            "kinetic_compliance": calibration.gamma_eff,
            "affinity_valid": True,
        }
    )
    return result


# ---------------------------------------------------------------------------
# 5. Response-based controlled current
# ---------------------------------------------------------------------------


def controlled_current(rows: Sequence[Any]) -> dict[str, Any]:
    """``J_c,k = N sum_n p_k(n) a_hat(n) chi_hat(n)`` per round, and its sum.

    This is a g-computation current: what the controller's own response moved,
    state by state.  It is NOT the terminal ``n_Z,H - n_Z,0`` of an episode,
    which also contains all the ordinary social dynamics and is a different
    scientific observable.

    The product is taken inside the sum on purpose.  ``N * mean(a) * mean(chi)``
    equals this only when the policy and the response are uncorrelated across
    states, and both are state dependent by construction.
    """

    eligible = controlled_rows(rows)
    N = population_size(eligible)
    table = state_response_table(eligible)
    occupancy = pooled_occupancy(eligible)
    blank = {
        "controlled_current": math.nan,
        "controlled_current_horizon": math.nan,
        "controlled_current_rounds": 0,
        "controlled_current_valid": False,
        "eta_th_identified_occupancy_mass": math.nan,
        "controlled_current_per_round": {},
    }
    if N is None or not table:
        return blank
    identified = {state: item for state, item in table.items() if item["identified"]}
    if not identified:
        return blank
    per_round: dict[int, float] = {}
    masses: list[float] = []
    for k, weights in round_occupancy(eligible).items():
        total = 0.0
        mass = 0.0
        for state, item in identified.items():
            weight = weights.get(state, 0.0)
            total += weight * float(item["a"]) * float(item["chi"])
            mass += weight
        per_round[k] = float(N) * total
        masses.append(mass)
    if not per_round:
        return blank
    horizon = float(sum(per_round.values()))
    return {
        "controlled_current": horizon / len(per_round),
        "controlled_current_horizon": horizon,
        "controlled_current_rounds": len(per_round),
        "controlled_current_valid": True,
        "eta_th_identified_occupancy_mass": float(np.mean(masses)),
        "controlled_current_per_round": per_round,
        **_support_diagnostics(table, occupancy),
    }


# ---------------------------------------------------------------------------
# 6. Thermodynamic efficiency
# ---------------------------------------------------------------------------


def eta_th_from_components(
    *,
    h: float,
    current_horizon: float,
    sensing_horizon: float,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """``h J_c / (h J_c + I_sens)`` over the horizon, as a ratio of sums.

    A mean of per-cycle efficiencies is a different, smaller-sample-noisier
    number; the finite-horizon quantity the theory bounds is total directed
    expenditure over total non-storage expenditure.

    The bounded reading needs ``h J_c >= 0`` and ``C_th > 0``.  When it does
    not hold the signed numerator is still reported - it says the controller
    pushed against its own affinity - but the ratio is not clipped into
    ``[0,1]`` to hide that, and ``eta_th_valid`` is false.
    """

    values = (float(h), float(current_horizon), float(sensing_horizon))
    if not math.isfinite(values[0]):
        reason = "missing_h" if math.isnan(values[0]) else "nonfinite_input"
    elif not math.isfinite(values[1]):
        reason = "missing_current" if math.isnan(values[1]) else "nonfinite_input"
    elif not math.isfinite(values[2]):
        reason = (
            "missing_sensing_information"
            if math.isnan(values[2])
            else "nonfinite_input"
        )
    else:
        reason = None
    directed = values[0] * values[1]
    expenditure = directed + values[2]
    if reason is None and not math.isfinite(expenditure):
        reason = "nonfinite_input"
    if reason is None and abs(expenditure) <= tolerance:
        reason = "zero_control_expenditure"
    numeric_defined = reason is None
    signed = directed / expenditure if numeric_defined else math.nan
    target_directed = bool(math.isfinite(directed) and directed >= 0.0)
    bounded = bool(
        numeric_defined
        and target_directed
        and expenditure > tolerance
        and 0.0 <= signed <= 1.0
    )
    return {
        "affinity_weighted_current_nats": directed,
        "thermodynamic_control_expenditure_nats": expenditure,
        "eta_th": signed if bounded else math.nan,
        "eta_th_signed": signed,
        "eta_th_bounded": signed if bounded else math.nan,
        "eta_th_numeric_defined": numeric_defined,
        "eta_th_has_bounded_interpretation": bounded,
        "eta_th_undefined_reason": reason,
        "eta_th_target_directed": target_directed,
        "eta_th_valid": bounded,
    }


# ---------------------------------------------------------------------------
# 7. The whole estimator, and its whole-episode bootstrap
# ---------------------------------------------------------------------------

_SCALARS = (
    "susceptibility_occupancy_weighted",
    "eta_ir",
    "eta_ir_pinsker_numerator_bits",
    "eta_ir_denominator_T_bits",
    "target_sensing_information_nats",
    "target_sensing_information_horizon_nats",
    "effective_affinity",
    "kinetic_compliance",
    "controlled_current",
    "controlled_current_horizon",
    "affinity_weighted_current_nats",
    "thermodynamic_control_expenditure_nats",
    "eta_th",
    "eta_th_signed",
    "eta_th_bounded",
)
"""The values a confidence interval is formed for.  Every one is recomputed
from scratch inside each bootstrap replicate."""


def point_estimate(
    rows: Sequence[Any], micro_rows: Sequence[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    """The complete single-affinity family for one group of rounds."""

    response = susceptibility_summary(rows)
    information = eta_ir(rows)
    sensing = target_sensing_information(rows)
    affinity = affinity_compliance(micro_rows)
    current = controlled_current(rows)
    thermodynamics = eta_th_from_components(
        h=affinity["effective_affinity"],
        current_horizon=current["controlled_current_horizon"],
        sensing_horizon=sensing["target_sensing_information_horizon_nats"],
    )
    if not affinity["affinity_valid"]:
        thermodynamics["eta_th_undefined_reason"] = (
            "insufficient_support_for_h_calibration"
        )
    elif not current["controlled_current_valid"]:
        thermodynamics["eta_th_undefined_reason"] = "missing_current"
    elif not sensing["target_sensing_valid"]:
        thermodynamics["eta_th_undefined_reason"] = "missing_sensing_information"
    return {
        **response,
        **information,
        **sensing,
        **affinity,
        **current,
        **thermodynamics,
        "n_rounds": len(controlled_rows(rows)),
        "n_episodes": len({episode_key(row) for row in rows}),
        "n_micro_slots": len(micro_rows),
    }


def _by_episode(
    rows: Sequence[Any], micro_rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[Hashable, list[Any]], dict[Hashable, list[Mapping[str, Any]]]]:
    grouped_rounds: dict[Hashable, list[Any]] = defaultdict(list)
    for row in rows:
        grouped_rounds[episode_key(row)].append(row)
    grouped_micro: dict[Hashable, list[Mapping[str, Any]]] = defaultdict(list)
    for row in micro_rows:
        grouped_micro[
            (str(row.get("cell_id", "run")), str(row.get("episode_id", "episode")))
        ].append(row)
    return dict(grouped_rounds), dict(grouped_micro)


def single_affinity_analysis(
    rows: Sequence[Any],
    micro_rows: Sequence[Mapping[str, Any]] = (),
    *,
    bootstrap_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 1,
) -> dict[str, Any]:
    """Point estimates plus percentile intervals from a whole-episode bootstrap.

    Both efficiencies are nonlinear functions of correlated ingredients, so an
    interval built by combining separately bootstrapped intervals for ``h``,
    ``J_c`` and ``I_sens`` would be wrong in a direction nobody can predict.
    Instead each replicate resamples complete episodes - rounds and their own
    micro-slots together - and rebuilds occupancies, response, action weights,
    CMI, transition counts, current and sensing from that resample, so the
    correlations are carried rather than assumed away.
    """

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    estimate = point_estimate(rows, micro_rows)
    grouped_rounds, grouped_micro = _by_episode(rows, micro_rows)
    keys = sorted(grouped_rounds, key=str)
    draws: dict[str, list[float]] = {name: [] for name in _SCALARS}
    if keys and bootstrap_resamples > 0:
        rng = np.random.default_rng(seed)
        for _ in range(int(bootstrap_resamples)):
            selected = [keys[index] for index in rng.integers(0, len(keys), len(keys))]
            replicate = point_estimate(
                [row for key in selected for row in grouped_rounds[key]],
                [row for key in selected for row in grouped_micro.get(key, ())],
            )
            for name in _SCALARS:
                value = float(replicate.get(name, math.nan))
                if math.isfinite(value):
                    draws[name].append(value)
    alpha = (1.0 - confidence) / 2.0
    for name in _SCALARS:
        values = draws[name]
        estimate[f"{name}_ci_low"] = (
            math.nan if not values else float(np.quantile(values, alpha))
        )
        estimate[f"{name}_ci_high"] = (
            math.nan if not values else float(np.quantile(values, 1.0 - alpha))
        )
    estimate["confidence"] = float(confidence)
    estimate["bootstrap_resamples"] = int(bootstrap_resamples)
    estimate.update(PROVENANCE)
    return estimate


# ---------------------------------------------------------------------------
# 8. Side-by-side against the exact theory
# ---------------------------------------------------------------------------


def theory_parameters(rows: Sequence[Any], h: float, gamma: float) -> Any | None:
    """The one protocol tuple these rows share, closed with calibrated h, gamma.

    ``h`` and ``gamma`` are properties of the calibrated population response
    rather than controller settings, so they are supplied rather than read off
    a record.  A group spanning several protocols returns ``None``: averaging
    two different theories into one reference would compare the run against a
    process nobody ran.
    """

    from mas_cc.games.relational_reasoning.imitation_round_feedback.theory_revised import (
        theory_parameters_from_record,
    )

    if any(row.event.get("social_mode", "peer") != "peer" for row in rows):
        return None
    if not math.isfinite(h) or not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        return None
    found = {}
    for row in rows:
        parameters = theory_parameters_from_record(row.event, h=h, gamma=gamma)
        if parameters is not None:
            found[parameters.key] = parameters
    return next(iter(found.values())) if len(found) == 1 else None


def theory_comparison(
    rows: Sequence[Any], micro_rows: Sequence[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    """`chi, T_pi, eta_IR, J_c, I_sens, eta_th`, empirical beside exact.

    The theory side is evaluated on the run's *own* empirical occupancy, so
    the two columns differ only where the LLM population departs from the
    single-affinity kernel - not because they were weighted over different
    states.  Returns ``{"available": False, "reason": ...}`` when the run does
    not pin down one protocol or one calibrated affinity, which is the honest
    answer rather than a table of NaNs.

    This is the *single-affinity* comparison.  The matched q-voter reference
    stays a separate classical null and is never written into these columns.
    """

    from mas_cc.games.relational_reasoning.imitation_round_feedback.theory_revised import (
        single_affinity_reference,
        thermodynamic_efficiency,
    )

    empirical = point_estimate(rows, micro_rows)
    if not empirical["affinity_valid"]:
        return {
            "available": False,
            "reason": "controlled microscopic transitions do not identify a finite h",
            "empirical": empirical,
        }
    parameters = theory_parameters(
        rows, empirical["effective_affinity"], empirical["kinetic_compliance"]
    )
    if parameters is None:
        return {
            "available": False,
            "reason": "rows do not share one (N, q_c, b, beta, theta) protocol",
            "empirical": empirical,
        }

    reference = single_affinity_reference(parameters)
    N = parameters.N
    eligible = controlled_rows(rows)
    occupancies = []
    for weights in round_occupancy(eligible).values():
        vector = np.zeros(N + 1, dtype=float)
        for n, weight in weights.items():
            if 0 <= n <= N:
                vector[n] = weight
        if vector.sum() > 0.0:
            occupancies.append(vector / vector.sum())
    if not occupancies:
        return {
            "available": False,
            "reason": "no controlled rounds to weight the theory over",
            "empirical": empirical,
        }
    pooled = np.mean(occupancies, axis=0)

    theory_current = float(sum(reference.current(p) for p in occupancies))
    theory_sensing = float(
        sum(sensing_information_nats(p, reference.S)[0] for p in occupancies)
    )
    theory_T_pi = reference.occupancy_weighted_T_pi(pooled)
    theory_numerator = float(pooled @ reference.pinsker_bound)
    theory_eta_ir = theory_numerator / theory_T_pi if theory_T_pi > 0.0 else math.nan
    theory_eta_th, _, _ = thermodynamic_efficiency(
        h=parameters.h, J_c=theory_current, I_sens_nats=theory_sensing
    )
    theory_chi = float(pooled @ theory_susceptibility_curve(reference))

    quantities = (
        (
            "chi",
            "target_fraction_per_cycle",
            empirical["susceptibility_occupancy_weighted"],
            theory_chi,
        ),
        (
            "T_pi",
            "bits",
            empirical["eta_ir_denominator_T_bits"],
            theory_T_pi,
        ),
        ("eta_IR", "dimensionless", empirical["eta_ir"], theory_eta_ir),
        (
            "J_c",
            "target_count_per_horizon",
            empirical["controlled_current_horizon"],
            theory_current,
        ),
        (
            "I_sens",
            "nats_per_horizon",
            empirical["target_sensing_information_horizon_nats"],
            theory_sensing,
        ),
        ("eta_th", "dimensionless", empirical["eta_th"], theory_eta_th),
    )
    return {
        "available": True,
        "reason": None,
        "parameters": parameters.as_fields(),
        "empirical": empirical,
        "rows": [
            {
                "quantity": name,
                "units": units,
                "empirical": float(observed),
                "single_affinity_theory": float(exact),
                "residual": float(observed) - float(exact),
                "reference": THEORY_REFERENCE,
            }
            for name, units, observed, exact in quantities
        ],
    }


def finite_horizon_current_comparison(
    rows: Sequence[Any],
    micro_rows: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Revised-theory moments for the terminal target-count coordinate.

    The empirical terminal current includes ordinary social updates, whereas
    this reference evolves only the isolated controlled layer.  Keeping the
    comparison here ensures calibration, protocol validation, and provenance
    are shared with :func:`theory_comparison` and cannot fall back to another
    model.
    """

    from mas_cc.games.relational_reasoning.imitation_round_feedback.theory_revised import (
        finite_horizon_current_moments_for_episodes,
        single_affinity_reference,
    )

    empirical = point_estimate(rows, micro_rows)
    common = {**PROVENANCE, "available": False}
    if not empirical["affinity_valid"]:
        return {
            **common,
            "reason": "controlled microscopic transitions do not identify a finite h",
        }
    parameters = theory_parameters(
        rows, empirical["effective_affinity"], empirical["kinetic_compliance"]
    )
    if parameters is None:
        return {
            **common,
            "reason": "rows do not share one (N, q_c, b, beta, theta) protocol",
        }
    if not episodes:
        return {**common, "reason": "no complete episodes for current comparison"}
    reference = single_affinity_reference(parameters)
    moments = finite_horizon_current_moments_for_episodes(
        reference.closed_loop_kernel,
        [int(item["initial_target_count"]) for item in episodes],
        [int(item["K"]) for item in episodes],
    )
    return {
        **PROVENANCE,
        "available": True,
        "reason": None,
        "parameters": parameters.as_fields(),
        **moments,
    }


def theory_susceptibility_curve(reference: Any) -> np.ndarray:
    """`chi(n)` from a built reference, named so call sites read as the theory."""

    return np.asarray(reference.chi, dtype=float)


__all__ = [
    "ADVOCATE_ACTIONS",
    "CONTROLLER_ACTIONS",
    "PROVENANCE",
    "THEORY_SEMANTICS_VERSION",
    "affinity_compliance",
    "controlled_current",
    "controlled_rows",
    "controlled_transition_counts",
    "episode_key",
    "eta_ir",
    "eta_th_from_components",
    "finite_horizon_current_comparison",
    "point_estimate",
    "pooled_occupancy",
    "population_size",
    "round_occupancy",
    "sensor_sample_size",
    "single_affinity_analysis",
    "state_response_table",
    "susceptibility_summary",
    "target_actuation_cmi_bits",
    "target_sensing_information",
    "theory_comparison",
    "theory_parameters",
]
