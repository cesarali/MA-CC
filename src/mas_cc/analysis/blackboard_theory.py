"""Opt-in theory stage after empirical estimation and blackboard calibration.

Calibration summaries and their retained split blocks are authoritative. Model
predictions remain separate from empirical rows. Unsupported mirrors carry a
reason, never an invented law or a substituted empirical estimator.
"""

from __future__ import annotations

import math
import json
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import pandas as pd

from . import blackboard_calibration as calibration
from . import blackboard_theory_engine as engine
from . import single_affinity
from .causal_response import (
    ESTIMATOR_VERSION as CAUSAL_VERSION,
    AVAILABLE_SUSCEPTIBILITY_VERSION,
)

VERSION = "blackboard_theory_v1"
REFERENCE = "calibrated_blackboard_finite_state_v1"
TABLES = (
    "theory_model_manifest",
    "theory_state_metrics",
    "theory_primary_estimates",
    "theory_derived_observables",
    "theory_empirical_comparison",
    "theory_validation",
)
FAMILIES = {
    "target_response": [
        "round_target_susceptibility",
        "round_target_signed_actuation",
        "round_target_signed_response_share",
    ],
    "causal_response": [
        "propensity_weighted_causal_response",
        "available_causal_susceptibility_state_local",
        "available_causal_susceptibility_summary",
    ],
    "action_entropy": [
        "round_controller_action_entropy",
        "round_controller_action_entropy_given_population",
        "action_entropy_given_target",
        "next_target_entropy_given_target",
        "next_target_entropy_given_target_action",
    ],
    "target_actuation_information": ["round_target_actuation_cmi"],
    "information_efficiencies": ["round_target_information_fraction", "eta_ir"],
    "scalar_sensing": [
        "round_target_sensing_mi",
        "round_sensor_action_mi",
        "round_sensor_mae",
        "round_sensor_mse",
        "target_sensing_information_nats",
        "target_sensing_information_horizon_nats",
    ],
    "currents": [
        "controlled_current",
        "controlled_current_horizon",
        "expected_episode_current",
        "expected_cell_current",
    ],
}
COMPATIBILITY = {
    "round_target_susceptibility": "exact_target_count; dual_action; renormalized_event_mass",
    "round_target_signed_actuation": "susceptibility * K_options/(K_options-1)",
    "round_target_signed_response_share": "action_specific_occupancy; marginal_fraction_increment",
    "round_target_actuation_cmi": "exact_target_count_before -> exact_target_count_after; bits; model_MI_vs_empirical_direct_count",
    "round_target_information_fraction": "T / H(action|target_count); ratio_of_components",
    "eta_ir": "dual_action_Pinsker_numerator / all_controlled_target_CMI; unrenormalized_numerator",
    "round_controller_action_entropy_given_population": "full_population_context; separate_policy_frequency",
    "round_sensor_action_mi": "not_modeled: full_sensor_vector_required",
    "propensity_weighted_causal_response": "existing_causal_adapter; first_action_only; exact_lag_endpoint_mask",
    "available_causal_susceptibility_state_local": "lag1; x<1; eight_bins; mean_of_ratios",
    "available_causal_susceptibility_summary": "lag1; x<1; ratio_of_sums",
}


def theory_settings(raw):
    if raw is None or raw is False:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("blackboard_theory_outputs must be a mapping")
    defaults = dict(
        enabled=True,
        model="frozen_board_exposure_mixture_v1",
        evaluation_modes=["matched_empirical_design"],
        matched_policy_source="empirical_action_frequency",
        forward_policy_source="sensor_policy_exact",
        families=list(FAMILIES),
        causal_lags=[1, 2, 3],
        assume_homogeneous_channels=False,
        assume_count_markov=False,
        sensor_policy=None,
        complete_policy_table=None,
        forward_initial_occupancy=None,
        forward_horizon=None,
        uncertainty="joint_data_and_calibration",
        comparisons=True,
        cross_cell_weights=None,
    )
    if set(raw) - set(defaults):
        raise ValueError(
            f"unknown blackboard theory settings: {sorted(set(raw) - set(defaults))}"
        )
    result = {**defaults, **raw}
    for k in (
        "enabled",
        "assume_homogeneous_channels",
        "assume_count_markov",
        "comparisons",
    ):
        if type(result[k]) is not bool:
            raise ValueError(f"{k} must be boolean")
    if result["model"] not in {
        "frozen_board_exposure_mixture_v1",
        "action_branch_calibrated",
    }:
        raise ValueError("unsupported blackboard theory model")
    for key, allowed in (
        ("evaluation_modes", {"matched_empirical_design", "forward_model_ensemble"}),
        ("families", set(FAMILIES)),
    ):
        value = result[key]
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(v, str) for v in value)
            or len(set(value)) != len(value)
            or not set(value) <= allowed
        ):
            raise ValueError(f"invalid {key}")
    sources = {
        "empirical_action_frequency",
        "recorded_propensity_average",
        "sensor_policy_exact",
        "complete_policy_table",
    }
    if result["matched_policy_source"] not in sources or result[
        "forward_policy_source"
    ] not in {"sensor_policy_exact", "complete_policy_table"}:
        raise ValueError("invalid policy source (no implicit fallback)")
    if result["uncertainty"] not in {
        "none",
        "calibration_only",
        "joint_data_and_calibration",
    }:
        raise ValueError("unsupported uncertainty source")
    lags = result["causal_lags"]
    if not isinstance(lags, list) or not lags or len(set(lags)) != len(lags):
        raise ValueError("causal_lags must be a unique nonempty list")
    for lag in lags:
        engine.integer(lag, "lag", 1)
    if result["forward_horizon"] is not None:
        engine.integer(result["forward_horizon"], "forward_horizon", 1)
    sensor = result["sensor_policy"]
    if sensor is not None:
        if not isinstance(sensor, Mapping) or set(sensor) - {
            "sampling_law",
            "rule",
            "q_c",
            "beta",
            "theta",
            "no_observation_probability",
        }:
            raise ValueError("invalid sensor_policy")
        if (
            sensor.get("sampling_law") != "uniform_without_replacement"
            or sensor.get("rule") != "logistic_target_fraction"
        ):
            raise ValueError("unsupported sensor law or feedback rule")
        engine.integer(sensor.get("q_c"), "q_c")
    weights = result["cross_cell_weights"]
    if weights is not None and (
        not isinstance(weights, Mapping)
        or not weights
        or any(
            not isinstance(k, str)
            or isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or v <= 0
            for k, v in weights.items()
        )
    ):
        raise ValueError(
            "cross_cell_weights must map qualified cell ids to positive finite weights"
        )
    return result if result["enabled"] else None


def _policy(rows, N, source, settings):
    if source == "sensor_policy_exact":
        spec = settings["sensor_policy"]
        if spec is None:
            raise ValueError("missing_exact_sensor_policy_configuration")
        for row in rows:
            event = row.get("_event")
            if event is None:
                raise ValueError("missing_recorded_sensor_policy")
            metadata = event.event
            if metadata.get("controller_policy") != "soft_target":
                raise ValueError("sensor_policy_requires_recorded_soft_target_rule")
            for field, configured in (
                ("sensor_sample_size", "q_c"),
                ("beta", "beta"),
                ("theta", "theta"),
            ):
                recorded = metadata.get(field)
                if (
                    recorded is None
                    or spec.get(configured) is None
                    or not math.isclose(
                        float(recorded), float(spec[configured]), abs_tol=1e-12
                    )
                ):
                    raise ValueError(
                        "sensor_policy_configuration_disagrees_with_recorded_" + field
                    )
        return engine.exact_sensor_policy(
            N, **{k: v for k, v in spec.items() if k not in {"rule", "sampling_law"}}
        )[2]
    if source == "complete_policy_table":
        return engine.policy_vector(settings["complete_policy_table"], N)
    a = np.full(N + 1, np.nan)
    groups = defaultdict(list)
    for row in rows:
        groups[int(row["n"])].append(
            row["U"] if source == "empirical_action_frequency" else row["propensity"]
        )
    for n, values in groups.items():
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError("missing_recorded_propensity")
        a[n] = np.mean(values)
    return a


def _rates(estimates, settings, calibration_settings):
    if not settings["assume_homogeneous_channels"]:
        raise ValueError("homogeneous_branch_law_assumption_required")
    if settings["model"] == "action_branch_calibrated":
        keys = [("0", "all", "direct_silent"), ("1", "all", "direct_active")]
    else:
        if not calibration_settings["shared_unexposed_baseline"]:
            raise ValueError("shared_unexposed_baseline_calibration_required")
        keys = [
            ("pooled", "0", "shared_unexposed"),
            ("1", "mixture", "mixture_common_weight"),
        ]
    values, dependencies = [], []
    for branch, channel, variant in keys:
        rates = []
        for metric in (
            "blackboard_target_entry_probability",
            "blackboard_target_exit_probability",
        ):
            match = estimates[
                (estimates.branch == branch)
                & (estimates.channel == channel)
                & (estimates.estimator_variant == variant)
                & (estimates.metric == metric)
            ]
            if len(match) != 1:
                raise ValueError(
                    f"missing_or_ambiguous_calibration:{branch}:{channel}:{variant}:{metric}"
                )
            row = match.iloc[0]
            if row.model_status != "compatible_rates":
                raise ValueError(f"inapplicable_calibration:{row.model_status}")
            rates.append(float(row.estimate))
            dependencies.append(row.estimate_id)
        values.append(tuple(rates))
    return (
        values,
        dependencies,
        [(int(a) if a in {"0", "1"} else a, c, v) for a, c, v in keys],
    )


def _evaluate(rows, rates, settings, N, mode):
    """Return keyed components and local rows; empirical values reuse dispatch."""
    values, states = {}, []

    def add(
        metric,
        estimate=math.nan,
        *,
        empirical=math.nan,
        numerator=math.nan,
        denominator=math.nan,
        status="available",
        units="dimensionless",
        lag=1,
        group=None,
        support="supported",
        selected=None,
    ):
        group = group or {}
        selected = rows if selected is None else selected
        key = (metric, lag, calibration._json(group))
        values[key] = dict(
            metric="theory_" + metric,
            empirical_metric=metric,
            estimate=float(estimate),
            empirical_estimate=float(empirical),
            numerator=float(numerator),
            denominator=float(denominator),
            model_status=status,
            empirical_support=support,
            units=units,
            lag=lag,
            grouping_slice_json=key[2],
            evaluation_ids_json=calibration._json(
                [[r["episode_id"], r["round_index"]] for r in selected]
            ),
            n_observations=len(selected),
            n_episodes=len({r["episode_id"] for r in selected}),
            horizon=lag if ("horizon" in metric or "expected_" in metric) else None,
        )

    wanted = {m for f in settings["families"] for m in FAMILIES[f]}
    if not rows:
        for m in wanted:
            add(m, status="missing_eligible_evaluation_rows", support="unsupported")
        return values, states
    M_values = {r["M"] for r in rows}
    if len(M_values) != 1:
        for m in wanted:
            add(
                m,
                status="variable_update_count_requires_context_branch_law",
                support="unknown",
            )
        return values, states
    M = int(next(iter(M_values)))
    Q0, Q1 = engine.round_kernels(N, M, *rates)
    source = (
        settings["matched_policy_source"]
        if mode == "matched_empirical_design"
        else settings["forward_policy_source"]
    )
    try:
        policy = _policy(rows, N, source, settings)
    except ValueError as exc:
        for m in wanted:
            add(m, status=str(exc), support="unknown")
        return values, states
    if mode == "forward_model_ensemble":
        if not settings["assume_count_markov"]:
            for m in wanted:
                add(
                    m,
                    status="missing_count_markov_law_for_future_board_evidence",
                    support="not_applicable",
                )
            return values, states
        initial, horizon = (
            settings["forward_initial_occupancy"],
            settings["forward_horizon"],
        )
        if initial is None or horizon is None:
            for m in wanted:
                add(
                    m,
                    status="missing_declared_initial_occupancy_or_horizon",
                    support="not_applicable",
                )
            return values, states
        try:
            trajectory, components, current = engine.forward_ensemble(
                Q0, Q1, policy, initial, horizon
            )
        except ValueError as exc:
            for m in wanted:
                add(m, status=str(exc), support="not_applicable")
            return values, states
        local = engine.state_metrics(Q0, Q1, policy)
        sensor = settings["sensor_policy"]
        sensor_law = None
        if sensor is not None:
            sensor_law = engine.exact_sensor_policy(
                N,
                **{
                    k: v for k, v in sensor.items() if k not in {"rule", "sampling_law"}
                },
            )[0]
        for k, comp in enumerate(components):
            for metric, field, unit in [
                ("round_target_actuation_cmi", "T", "bits"),
                ("action_entropy_given_target", "H", "bits"),
                ("next_target_entropy_given_target", "H_next", "bits"),
                ("next_target_entropy_given_target_action", "H_next_action", "bits"),
                ("controlled_current", "J_excess", "target_count_per_cycle"),
                ("round_target_susceptibility", "chi", "target_fraction_per_cycle"),
            ]:
                add(
                    metric,
                    comp[field],
                    group={"round_index": k},
                    units=unit,
                    support="not_applicable",
                )
            p_action = float(trajectory[k] @ policy)
            add(
                "round_controller_action_entropy",
                engine.entropy([1 - p_action, p_action]),
                units="bits",
                group={"round_index": k},
                support="not_applicable",
            )
            for lag in settings["causal_lags"]:
                if k + lag <= horizon:
                    delta = engine.lag_contrast(Q0, Q1, policy, lag)
                    add(
                        "propensity_weighted_causal_response",
                        trajectory[k] @ delta,
                        lag=lag,
                        units="target_share_change",
                        group={"round_index": k},
                        support="not_applicable",
                    )
            if sensor_law is not None:
                info = engine.sensing_information_nats(trajectory[k], sensor_law)[0]
                add(
                    "round_target_sensing_mi",
                    info / math.log(2),
                    units="bits",
                    group={"round_index": k},
                    support="not_applicable",
                )
                add(
                    "target_sensing_information_nats",
                    info,
                    units="nats",
                    group={"round_index": k},
                    support="not_applicable",
                )
                q = sensor["q_c"]
                if q:
                    error = (
                        np.arange(q + 1)[None, :] / q - np.arange(N + 1)[:, None] / N
                    )
                    for metric, loss in (
                        ("round_sensor_mae", abs(error)),
                        ("round_sensor_mse", error**2),
                    ):
                        add(
                            metric,
                            np.sum(trajectory[k, :, None] * sensor_law * loss),
                            units="target_fraction"
                            if metric.endswith("mae")
                            else "target_fraction_squared",
                            group={"round_index": k},
                            support="not_applicable",
                        )
            for n in range(N + 1):
                states.append(
                    dict(
                        round_index=k,
                        target_count_before=n,
                        occupancy=trajectory[k, n],
                        policy_probability=policy[n],
                        **{key: float(value[n]) for key, value in local.items()},
                        **engine.component_ratios(
                            local["T"][n], local["H"][n], local["B"][n]
                        ),
                    )
                )
        if sensor_law is not None:
            add(
                "target_sensing_information_horizon_nats",
                sum(
                    engine.sensing_information_nats(v, sensor_law)[0]
                    for v in trajectory[:-1]
                ),
                units="nats",
                lag=horizon,
                support="not_applicable",
            )
        T, H, B = (sum(c[f] for c in components) for f in ("T", "H", "B"))
        add(
            "round_target_information_fraction",
            engine.component_ratios(T, H, B)["eta_IF"],
            numerator=T,
            denominator=H,
            lag=horizon,
            support="not_applicable",
        )
        add(
            "eta_ir",
            engine.component_ratios(T, H, B)["eta_IR"],
            numerator=B,
            denominator=T,
            lag=horizon,
            support="not_applicable",
        )
        add(
            "expected_cell_current",
            current,
            units="target_count",
            lag=horizon,
            support="not_applicable",
        )
        add(
            "controlled_current_horizon",
            sum(c["J_excess"] for c in components),
            units="target_count",
            lag=horizon,
            support="not_applicable",
        )
    else:
        # Missing policies outside observed states are not filled for rollouts.
        # Local arithmetic uses arbitrary zero only on exactly zero occupancy.
        local_policy = np.where(np.isfinite(policy), policy, 0.0)
        state = engine.state_metrics(Q0, Q1, local_policy)
        occupancy = np.bincount([r["n"] for r in rows], minlength=N + 1) / len(rows)
        actions = {
            n: {r["U"] for r in rows if r["n"] == n} for n in np.flatnonzero(occupancy)
        }
        dual = np.array([actions.get(n) == {0, 1} for n in range(N + 1)])
        mass = float(occupancy @ dual)
        events = [r["_event"] for r in rows if r.get("_event") is not None]
        compatible_events = len(events) == len(rows)
        from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import (
            _diagnostic_for,
            _estimate_for,
            MAIN_ESTIMATOR_VARIANT,
        )

        def empirical(metric):
            if not compatible_events:
                return math.nan
            if metric == "eta_ir":
                return single_affinity.eta_ir(events)["eta_ir"]
            if metric == "round_target_actuation_cmi":
                return getattr(_estimate_for(metric, events), MAIN_ESTIMATOR_VARIANT)
            if metric in {
                "round_target_susceptibility",
                "round_target_signed_actuation",
                "round_target_signed_response_share",
                "round_controller_action_entropy",
                "round_controller_action_entropy_given_population",
                "round_target_information_fraction",
            }:
                return _diagnostic_for(metric, events)
            return math.nan

        support = "supported" if mass > 0 else "unsupported"
        T, H = (float(occupancy @ state[f]) for f in ("T", "H"))
        B = float(occupancy @ (state["B"] * dual))
        chi = float(occupancy @ (state["chi"] * dual)) / mass if mass else math.nan
        for n in np.flatnonzero(occupancy):
            states.append(
                dict(
                    target_count_before=int(n),
                    occupancy=occupancy[n],
                    policy_probability=policy[n],
                    empirical_support="dual_action" if dual[n] else "single_action",
                    model_support="available",
                    **{k: float(v[n]) for k, v in state.items()},
                    **engine.component_ratios(
                        state["T"][n], state["H"][n], state["B"][n]
                    ),
                )
            )
        add(
            "round_target_susceptibility",
            chi,
            empirical=empirical("round_target_susceptibility"),
            numerator=float(occupancy @ (state["chi"] * dual)),
            denominator=mass,
            units="target_fraction_per_cycle",
            support=support,
            selected=[r for r in rows if dual[r["n"]]],
        )
        options = {len(e.N_k) for e in events}
        K = next(iter(options)) if len(options) == 1 else None
        add(
            "round_target_signed_actuation",
            chi * K / (K - 1) if K and K > 1 else math.nan,
            empirical=empirical("round_target_signed_actuation"),
            status="available" if K and K > 1 else "missing_K_options",
            units="aligned_magnetization_per_cycle",
            support=support,
            selected=[r for r in rows if dual[r["n"]]],
        )
        p = float(occupancy @ local_policy)
        marginal = (
            float(occupancy @ (local_policy * state["d_active"])) / p
            - float(occupancy @ ((1 - local_policy) * state["d_silent"])) / (1 - p)
            if 0 < p < 1
            else math.nan
        )
        add(
            "round_target_signed_response_share",
            marginal,
            empirical=empirical("round_target_signed_response_share"),
            units="target_fraction_per_cycle",
            support=support,
        )
        for metric, value in [
            ("round_target_actuation_cmi", T),
            ("action_entropy_given_target", H),
            ("round_controller_action_entropy", float(engine.entropy([1 - p, p]))),
            ("next_target_entropy_given_target", float(occupancy @ state["H_next"])),
            (
                "next_target_entropy_given_target_action",
                float(occupancy @ state["H_next_action"]),
            ),
        ]:
            add(
                metric,
                value,
                empirical=empirical(metric),
                units="bits",
                support=support,
            )
        # Full context entropy uses its own exact empirical conditioning policy.
        if compatible_events and source == "empirical_action_frequency":
            full_H = empirical("round_controller_action_entropy_given_population")
            add(
                "round_controller_action_entropy_given_population",
                full_H,
                empirical=full_H,
                units="bits",
            )
        else:
            add(
                "round_controller_action_entropy_given_population",
                status="full_context_policy_adapter_required",
                units="bits",
            )
        for metric, num, den in [
            ("round_target_information_fraction", T, H),
            ("eta_ir", B, T),
        ]:
            value = (
                num / den if den > 0 and (metric != "eta_ir" or mass > 0) else math.nan
            )
            add(
                metric,
                value,
                empirical=empirical(metric),
                numerator=num,
                denominator=den,
                support=support,
            )
        # Currents use equal round-index weighting, matching existing horizon semantics.
        currents = []
        for k in sorted({r["round_index"] for r in rows}):
            selected = [r for r in rows if r["round_index"] == k]
            currents.append(
                float(
                    np.mean(
                        [
                            N
                            * local_policy[r["n"]]
                            * state["chi"][r["n"]]
                            * dual[r["n"]]
                            for r in selected
                        ]
                    )
                )
            )
        observed_currents = (
            single_affinity.controlled_current(events) if compatible_events else {}
        )
        add(
            "controlled_current",
            float(np.mean(currents)) if mass else math.nan,
            empirical=observed_currents.get("controlled_current", math.nan),
            units="target_count_per_cycle",
            support=support,
        )
        add(
            "controlled_current_horizon",
            sum(currents) if mass else math.nan,
            empirical=observed_currents.get("controlled_current_horizon", math.nan),
            units="target_count",
            lag=len(currents),
            support=support,
        )
        # Canonical causal adapter supplies both eligibility and the empirical IPW contribution.
        from .causal_response import _support_status

        complete_causal = [
            r for r in rows if r.get("_causal", {}).get("episode_complete", False)
        ]
        causal_support = _support_status(
            sum(r["U"] == 1 for r in complete_causal),
            sum(r["U"] == 0 for r in complete_causal),
        )
        for lag in settings["causal_lags"]:
            selected = [
                r
                for r in rows
                if r.get("_causal", {}).get("episode_complete", False)
                and r["_causal"].get(f"lag_{lag}_available", False)
            ]
            reason = None
            if not selected:
                reason = "missing_eligible_causal_lag_endpoints_or_propensities"
            elif lag > 1 and not settings["assume_count_markov"]:
                reason = "missing_count_markov_law_for_future_board_evidence"
            elif lag > 1 and not np.isfinite(policy).all():
                reason = "missing_future_policy_states"
            if reason:
                add(
                    "propensity_weighted_causal_response",
                    status=reason,
                    lag=lag,
                    units="target_share_change",
                    support=causal_support,
                    selected=selected,
                )
                continue
            contrast = engine.lag_contrast(Q0, Q1, policy, lag)
            predicted = [contrast[r["n"]] for r in selected]
            observed = [r["_causal"][f"causal_response_h{lag}"] for r in selected]
            add(
                "propensity_weighted_causal_response",
                np.mean(predicted),
                empirical=np.mean(observed),
                lag=lag,
                units="target_share_change",
                support=causal_support,
                selected=selected,
            )
            if lag == 1:
                available = [r for r in selected if r["n"] < N]
                if available:
                    total_mass = sum(1 - r["n"] / N for r in available)
                    add(
                        "available_causal_susceptibility_summary",
                        sum(contrast[r["n"]] for r in available) / total_mass,
                        empirical=sum(
                            r["_causal"]["causal_response_h1"] for r in available
                        )
                        / total_mass,
                        numerator=sum(contrast[r["n"]] for r in available),
                        denominator=total_mass,
                        selected=available,
                        units="target_share_response_per_available_target_mass",
                        support=_support_status(
                            sum(r["U"] == 1 for r in available),
                            sum(r["U"] == 0 for r in available),
                        ),
                    )
                    for b in sorted({min(int(8 * r["n"] / N), 7) for r in available}):
                        br = [r for r in available if min(int(8 * r["n"] / N), 7) == b]
                        add(
                            "available_causal_susceptibility_state_local",
                            np.mean([contrast[r["n"]] / (1 - r["n"] / N) for r in br]),
                            empirical=np.mean(
                                [
                                    r["_causal"]["causal_response_h1"]
                                    / (1 - r["n"] / N)
                                    for r in br
                                ]
                            ),
                            group={"target_fraction_bin_index": b},
                            selected=br,
                            units="target_share_response_per_available_target_mass",
                            support=_support_status(
                                sum(r["U"] == 1 for r in br),
                                sum(r["U"] == 0 for r in br),
                            ),
                        )
        sensor = settings["sensor_policy"]
        sensor_reason = "missing_exact_sensor_law_or_eligible_sensor_rows"
        if sensor is not None and (
            not compatible_events
            or any(e.event.get("sensor_sample_size") != sensor["q_c"] for e in events)
        ):
            sensor = None
            sensor_reason = "missing_or_incompatible_recorded_sensor_sample_size"
        if sensor is not None:
            S, _, _ = engine.exact_sensor_policy(
                N,
                **{
                    k: v for k, v in sensor.items() if k not in {"rule", "sampling_law"}
                },
            )
            sensor_rows = [
                r
                for r in rows
                if r.get("_event") is not None
                and r["_event"].sensor_target_count is not None
            ]
            if sensor_rows:
                sensor_occupancy = np.bincount(
                    [r["n"] for r in sensor_rows], minlength=N + 1
                ) / len(sensor_rows)
                information, _ = engine.sensing_information_nats(sensor_occupancy, S)
                empirical_sensor = getattr(
                    _estimate_for(
                        "round_target_sensing_mi", [r["_event"] for r in sensor_rows]
                    ),
                    MAIN_ESTIMATOR_VARIANT,
                )
                add(
                    "round_target_sensing_mi",
                    information / math.log(2),
                    empirical=empirical_sensor,
                    units="bits",
                    selected=sensor_rows,
                )
            q = sensor["q_c"]
            if q:
                errors = np.arange(q + 1)[None, :] / q - np.arange(N + 1)[:, None] / N
                for metric, error in [
                    ("round_sensor_mae", abs(errors)),
                    ("round_sensor_mse", errors**2),
                ]:
                    eligible = [
                        r
                        for r in rows
                        if r.get("_event") is not None
                        and r["_event"].event.get("sensor_target_share") is not None
                    ]
                    if eligible:
                        v = np.bincount(
                            [r["n"] for r in eligible], minlength=N + 1
                        ) / len(eligible)
                        add(
                            metric,
                            float(np.sum(v[:, None] * S * error)),
                            empirical=_diagnostic_for(
                                metric, [r["_event"] for r in eligible]
                            ),
                            units="target_fraction"
                            if metric.endswith("mae")
                            else "target_fraction_squared",
                            selected=eligible,
                        )
            per_round = []
            for k in sorted({r["round_index"] for r in rows}):
                selected = [r for r in rows if r["round_index"] == k]
                v = np.bincount([r["n"] for r in selected], minlength=N + 1) / len(
                    selected
                )
                per_round.append(engine.sensing_information_nats(v, S)[0])
            observed_sensing = (
                single_affinity.target_sensing_information(events)
                if compatible_events
                else {}
            )
            add(
                "target_sensing_information_nats",
                np.mean(per_round),
                empirical=observed_sensing.get(
                    "target_sensing_information_nats", math.nan
                ),
                units="nats",
            )
            add(
                "target_sensing_information_horizon_nats",
                sum(per_round),
                empirical=observed_sensing.get(
                    "target_sensing_information_horizon_nats", math.nan
                ),
                units="nats",
                lag=len(per_round),
            )
        # Matched episode predictions require a complete autonomous future law.
        if settings["assume_count_markov"] and np.isfinite(policy).all():
            predictions = []
            observed_predictions = []
            for ep in sorted({r["episode_id"] for r in rows}):
                er = sorted(
                    [r for r in rows if r["episode_id"] == ep],
                    key=lambda r: r["round_index"],
                )
                if [r["round_index"] for r in er] != list(range(len(er))):
                    continue
                initial = np.eye(N + 1)[er[0]["n"]]
                _, _, current = engine.forward_ensemble(
                    Q0, Q1, policy, initial, len(er)
                )
                predictions.append(current)
                observed = er[-1]["_event"].target_after - er[0]["_event"].target_before
                observed_predictions.append(observed)
                add(
                    "expected_episode_current",
                    current,
                    empirical=observed,
                    selected=er,
                    lag=len(er),
                    group={"episode_id": ep},
                    units="target_count",
                )
            if predictions:
                add(
                    "expected_cell_current",
                    np.mean(predictions),
                    empirical=np.mean(observed_predictions),
                    units="target_count",
                )
    if mode == "matched_empirical_design":
        for value in values.values():
            value["identified_occupancy_mass"] = mass
            value["numerator_denominator_support_mismatch"] = bool(
                mass < 1 and source != "empirical_action_frequency"
            )
            value["missing_policy_states_json"] = calibration._json(
                np.flatnonzero(~np.isfinite(policy)).tolist()
            )
    if mode == "matched_empirical_design" and sensor is None:
        for metric in FAMILIES["scalar_sensing"]:
            if not any(key[0] == metric for key in values):
                add(
                    metric,
                    status="full_sensor_vector_required"
                    if metric == "round_sensor_action_mi"
                    else sensor_reason,
                    support="unknown",
                )
    present = {key[0] for key in values}
    for metric in wanted - present:
        reason = (
            "full_sensor_vector_required"
            if metric == "round_sensor_action_mi"
            else "missing_exact_sensor_law_or_eligible_sensor_rows"
            if metric in FAMILIES["scalar_sensing"]
            else "missing_complete_policy_or_trajectory_law"
            if "current" in metric
            else "missing_matching_state_or_eligible_mask"
        )
        add(metric, status=reason, support="unknown")
    return {k: v for k, v in values.items() if k[0] in wanted}, states


def analyze_blackboard_theory(
    calibration_outputs,
    rounds,
    cells,
    *,
    settings,
    calibration_settings,
    events=(),
    bootstrap_resamples=0,
    confidence=0.95,
    seed=1,
    analysis_hash="",
    provisional=False,
    progress=None,
):
    """Consume current calibration outputs, refitting coupled blocks for intervals."""
    settings = theory_settings(settings)
    output = {name: [] for name in TABLES}
    if settings is None:
        return {name: pd.DataFrame() for name in TABLES}
    inputs = calibration_outputs.get("blackboard_calibration_inputs", pd.DataFrame())
    estimates = calibration_outputs.get(
        "blackboard_calibration_estimates", pd.DataFrame()
    )
    splits = calibration_outputs.get("blackboard_calibration_splits", pd.DataFrame())
    event_lookup = {
        (str(e.cell_id), str(e.episode_id), e.round_index): e
        for e in single_affinity.controlled_rows(events)
    }
    causal_lookup = {}
    from .causal_response import build_causal_response_inputs

    if not inputs.empty:
        completed = set(zip(inputs.cell_id.astype(str), inputs.episode_id.astype(str)))
        selected_rounds = rounds[
            [
                (str(r.cell_id), str(r.episode_id)) in completed
                for r in rounds.itertuples()
            ]
        ]
        for cid, frame in selected_rounds.groupby("cell_id"):
            try:
                causal = build_causal_response_inputs(
                    frame, cells, lags=settings["causal_lags"]
                )
                for r in causal.to_dict("records"):
                    causal_lookup[
                        (str(r["cell_id"]), str(r["episode_id"]), r["round_index"])
                    ] = r
            except ValueError as exc:
                output["theory_validation"].append(
                    dict(
                        cell_id=cid,
                        family="causal_response",
                        model_status="missing_dependency",
                        reason=str(exc),
                    )
                )
    if inputs.empty or estimates.empty or splits.empty:
        for family in settings["families"]:
            output["theory_validation"].append(
                dict(
                    family=family,
                    model_status="missing_dependency",
                    reason="missing_blackboard_calibration_outputs",
                )
            )
            for metric in FAMILIES[family]:
                output["theory_primary_estimates"].append(
                    dict(
                        metric="theory_" + metric,
                        empirical_metric=metric,
                        estimate=math.nan,
                        model_status="missing_blackboard_calibration_outputs",
                    )
                )
    else:
        for training_id, group_estimates in estimates.groupby(
            "training_group_id", sort=True
        ):
            identity = group_estimates.iloc[0]
            group_splits = splits[splits.training_group_id == training_id]
            blocks = set(group_splits.block_id)
            frame = inputs[
                (inputs.cell_id == identity.cell_id) & inputs.block_id.isin(blocks)
            ]
            budget = identity.b_budget
            frame = (
                frame[frame.b_budget.isna()]
                if pd.isna(budget)
                else frame[frame.b_budget == budget]
            )
            eval_blocks = set(
                group_splits.loc[group_splits.split == "evaluation", "block_id"]
            )
            train_blocks = blocks - eval_blocks
            train = frame[frame.block_id.isin(train_blocks)].to_dict("records")
            evaluation = frame[frame.block_id.isin(eval_blocks or train_blocks)]
            retained = evaluation.drop_duplicates(
                ["cell_id", "episode_id", "round_index"]
            ).to_dict("records")
            evaluation_rows = []
            for r in retained:
                key = (str(r["cell_id"]), str(r["episode_id"]), r["round_index"])
                event = event_lookup.get(key)
                if event is None:
                    continue
                if not math.isfinite(float(r["N"])) or not float(r["N"]).is_integer():
                    continue
                r["n"] = int(event.target_before)
                r["_event"] = event
                r["_causal"] = causal_lookup.get(key, {})
                if not 0 <= r["n"] <= r["N"] or not math.isclose(
                    r["n"] / r["N"], r["x"], abs_tol=1e-12
                ):
                    raise ValueError(
                        "theory/calibration target orientation disagreement"
                    )
                evaluation_rows.append(r)
            dependency_ids = []
            failure = None
            try:
                rates, dependency_ids, keys = _rates(
                    group_estimates, settings, calibration_settings
                )
                sizes = frame.N.dropna().unique()
                if len(sizes) != 1 or not float(sizes[0]).is_integer():
                    raise ValueError("missing_unique_population_size")
                N = int(sizes[0])
                for rate in rates:
                    engine.microscopic_kernel(N, *rate)
                if settings["model"] == "frozen_board_exposure_mixture_v1":
                    summary = calibration._summaries(train, calibration_settings)
                    diagnostic = calibration._diagnostics(train, summary)
                    status = calibration._prediction_status(train, summary, diagnostic)
                    if status != "homogeneous_reference_assumed":
                        raise ValueError(status)
                    eval_summary = calibration._summaries(
                        evaluation.to_dict("records"), calibration_settings
                    )
                    eval_diagnostic = calibration._diagnostics(
                        evaluation.to_dict("records"), eval_summary
                    )
                    if (
                        eval_diagnostic["rounds_with_varying_board"]
                        or eval_diagnostic["sampler_inapplicable"]
                        or eval_diagnostic["eligible_composition_missing"]
                    ):
                        raise ValueError("evaluation_frozen_board_law_unverified")
                    if ((evaluation.U == 0) & (evaluation.E == 1)).any():
                        raise ValueError("silent_exposure_incompatible")
                if frame.focal_selection_rule.ne("uniform_with_replacement").any():
                    raise ValueError("unverified_uniform_focal_selection")
            except ValueError as exc:
                failure = str(exc)
            for mode in settings["evaluation_modes"]:
                source = (
                    settings["matched_policy_source"]
                    if mode == "matched_empirical_design"
                    else settings["forward_policy_source"]
                )
                base = dict(
                    study_id=identity.study_id,
                    source_run_id=identity.source_run_id,
                    cell_id=identity.cell_id,
                    qualified_cell_id=identity.cell_id,
                    grouping_json=identity.grouping_json,
                    conditioning_json=calibration._json(
                        {
                            "state": "target_count_before",
                            "outcome": "target_count_after",
                        }
                    ),
                    theory_mode=mode,
                    policy_source=source,
                    occupancy_source="canonical_controlled_evaluation_rounds"
                    if mode == "matched_empirical_design"
                    else "declared_initial_forward_propagation",
                    calibration_variant=settings["model"],
                    calibration_version=calibration.VERSION,
                    calibration_hash=identity.analysis_hash,
                    training_group_id=training_id,
                    split_status="held_out_prediction"
                    if eval_blocks
                    else "in_sample_calibrated",
                    dependencies_json=calibration._json(dependency_ids),
                    evaluation_ids_json=calibration._json(
                        [[r["episode_id"], r["round_index"]] for r in evaluation_rows]
                    ),
                    mask_id=calibration._id(
                        [[r["episode_id"], r["round_index"]] for r in evaluation_rows]
                    ),
                    binning_id="exact_target_count",
                    weight_unit="controlled_round",
                    extrapolated_mass=0.0
                    if mode == "matched_empirical_design"
                    else math.nan,
                    uncertainty_source=settings["uncertainty"],
                    theoretical_reference=REFERENCE,
                    model_interpretation="homogeneous_calibrated_coarse_reference",
                    causal_identification="assumed_branch_law; completed_endpoint_selection",
                    calibration_evaluation_disjoint=bool(eval_blocks),
                )
                output["theory_model_manifest"].append(
                    {
                        **base,
                        "settings_json": calibration._json(settings),
                        "compatibility_map_json": calibration._json(COMPATIBILITY),
                        "provider_calls": 0,
                        "model_status": failure or "homogeneous_reference_assumed",
                        "training_blocks_json": calibration._json(sorted(train_blocks)),
                        "evaluation_blocks_json": calibration._json(
                            sorted(eval_blocks or train_blocks)
                        ),
                        "uncertainty_method": "paired_block_refit"
                        if settings["uncertainty"] != "none"
                        else "point_prediction",
                    }
                )
                if failure:
                    for family in settings["families"]:
                        output["theory_validation"].append(
                            {
                                **base,
                                "family": family,
                                "model_status": "missing_dependency",
                                "reason": failure,
                            }
                        )
                        for metric in FAMILIES[family]:
                            output["theory_primary_estimates"].append(
                                {
                                    **base,
                                    "metric": "theory_" + metric,
                                    "empirical_metric": metric,
                                    "estimate": math.nan,
                                    "model_status": failure,
                                    "uncertainty_status": "unavailable",
                                }
                            )
                    continue
                values, states = _evaluate(evaluation_rows, rates, settings, N, mode)
                for state in states:
                    output["theory_state_metrics"].append(
                        {**base, **state, "N": N, "M": int(frame.M.iloc[0])}
                    )
                draws = defaultdict(list)
                residual_draws = defaultdict(list)
                rng = np.random.default_rng(
                    seed + int(calibration._id([training_id, mode])[:8], 16)
                )
                requested = (
                    bootstrap_resamples if settings["uncertainty"] != "none" else 0
                )
                for replicate in range(requested):
                    if (
                        settings["uncertainty"] == "joint_data_and_calibration"
                        and not eval_blocks
                    ):
                        combined = [{**r, "_kind": "training"} for r in train] + [
                            {**r, "_kind": "evaluation"} for r in evaluation_rows
                        ]
                        sampled = calibration._resample(combined, rng)
                        sample_train = [r for r in sampled if r["_kind"] == "training"]
                        sample_eval = [r for r in sampled if r["_kind"] == "evaluation"]
                    else:
                        sample_train = calibration._resample(train, rng)
                        sample_eval = (
                            calibration._resample(evaluation_rows, rng)
                            if settings["uncertainty"] == "joint_data_and_calibration"
                            else evaluation_rows
                        )
                    summary = calibration._summaries(sample_train, calibration_settings)
                    sample_rates = [
                        (summary[k]["p_plus"], summary[k]["p_minus"]) for k in keys
                    ]
                    try:
                        sample_values, _ = _evaluate(
                            sample_eval, sample_rates, settings, N, mode
                        )
                    except ValueError:
                        sample_values = {}
                    for key in values:
                        v = sample_values.get(key, {})
                        predicted = v.get("estimate", math.nan)
                        if math.isfinite(predicted):
                            draws[key].append(predicted)
                        observed = v.get("empirical_estimate", math.nan)
                        if math.isfinite(predicted) and math.isfinite(observed):
                            residual_draws[key].append(observed - predicted)
                    if progress and replicate % 25 == 0:
                        progress(
                            dict(
                                stage="blackboard_theory_bootstrap",
                                cell_id=identity.cell_id,
                                replicate=replicate,
                                requested=requested,
                            )
                        )
                for key, value in values.items():
                    nblocks = min(len(train_blocks), len(eval_blocks or train_blocks))
                    interval = calibration._interval(
                        value["estimate"], draws[key], nblocks, confidence, requested
                    )
                    row = {
                        **base,
                        **value,
                        **interval,
                        "confidence": confidence,
                        "theory_estimate_id": calibration._id(
                            [analysis_hash, training_id, mode, key]
                        ),
                        "estimand_contract": COMPATIBILITY.get(
                            key[0], "see_blackboard_theory_reference"
                        ),
                        "model_support": "available"
                        if math.isfinite(value["estimate"])
                        else "unavailable",
                    }
                    row["grouping_json"] = calibration._json(
                        {**json.loads(identity.grouping_json), **json.loads(key[2])}
                    )
                    row["mask_id"] = calibration._id(
                        [row["evaluation_ids_json"], key[0], key[1], key[2]]
                    )
                    if key[0] in {
                        "available_causal_susceptibility_state_local",
                        "available_causal_susceptibility_summary",
                    }:
                        row["empirical_metric"] = (
                            "propensity_weighted_available_susceptibility"
                            if key[0].endswith("state_local")
                            else "available_mass_weighted_causal_susceptibility"
                        )
                    if key[0] == "available_causal_susceptibility_state_local":
                        row["binning_id"] = (
                            "eight_equal_width_fraction_bins; exact_state_law_inside_bin"
                        )
                    if key[0] == "propensity_weighted_causal_response":
                        row["conditioning_json"] = "{}"
                        if value["model_status"] == "missing_future_policy_states":
                            row["extrapolated_mass"] = math.nan
                    if key[0] == "round_controller_action_entropy_given_population":
                        row["conditioning_json"] = calibration._json(
                            {"state": "full_population"}
                        )
                    derived = (
                        key[0]
                        in FAMILIES["information_efficiencies"] + FAMILIES["currents"]
                    )
                    output[
                        "theory_derived_observables"
                        if derived
                        else "theory_primary_estimates"
                    ].append(row)
                    if settings["comparisons"] and mode == "matched_empirical_design":
                        empirical = value["empirical_estimate"]
                        residual = empirical - value["estimate"]
                        paired = calibration._interval(
                            residual,
                            residual_draws[key],
                            nblocks,
                            confidence,
                            requested,
                        )
                        output["theory_empirical_comparison"].append(
                            {
                                **row,
                                "theoretical_estimate": value["estimate"],
                                "empirical_estimate": empirical,
                                "residual": residual,
                                "residual_ci_low": paired["ci_low"],
                                "residual_ci_high": paired["ci_high"],
                                "residual_uncertainty_status": paired[
                                    "uncertainty_status"
                                ],
                                "comparison_status": "matched_recomputed_empirical_adapter"
                                if math.isfinite(residual)
                                else "missing_compatible_empirical_or_model_estimate",
                                "empirical_estimator_version": CAUSAL_VERSION
                                if key[0] == "propensity_weighted_causal_response"
                                else AVAILABLE_SUSCEPTIBILITY_VERSION
                                if key[0].startswith("available_causal_susceptibility")
                                else single_affinity.PROVENANCE[
                                    "theory_semantics_version"
                                ]
                                if key[0]
                                in {
                                    "eta_ir",
                                    "controlled_current",
                                    "controlled_current_horizon",
                                    "target_sensing_information_nats",
                                    "target_sensing_information_horizon_nats",
                                }
                                else "round-feedback-v1",
                                "empirical_estimator_variant": "unsmoothed"
                                if key[0]
                                in {
                                    "round_target_actuation_cmi",
                                    "round_target_sensing_mi",
                                    "round_target_information_fraction",
                                    "eta_ir",
                                }
                                else "existing_empirical_definition",
                                "theory_estimator_variant": "exact_finite_state_summation",
                                "empirical_null_adjustment": "none; exact_model_information_has_no_null_offset",
                            }
                        )
                    if value["model_status"] != "available" or not math.isfinite(
                        value["estimate"]
                    ):
                        output["theory_validation"].append(
                            {
                                **base,
                                "metric": value["metric"],
                                "lag": value["lag"],
                                "model_status": value["model_status"],
                                "reason": value["model_status"]
                                if value["model_status"] != "available"
                                else "undefined_denominator_or_empirical_support",
                            }
                        )
                output["theory_validation"].append(
                    {
                        **base,
                        "model_status": "validated" if states else "unavailable",
                        "reason": "row_stochastic; information_bounds; entropy_identity; current_identity"
                        if states
                        else "no_evaluable_model_states",
                        "numerical_tolerance": 1e-10,
                        "N": N,
                    }
                )
    if settings["cross_cell_weights"] is not None:
        output["theory_derived_observables"].extend(
            aggregate_theory_components(
                pd.DataFrame(output["theory_derived_observables"]),
                settings["cross_cell_weights"],
            )
        )
    result = {name: pd.DataFrame(records) for name, records in output.items()}
    for frame in result.values():
        frame["model_version"] = VERSION
        frame["analysis_hash"] = analysis_hash
        frame["provisional"] = provisional
    return result


def aggregate_theory_components(frame, weights):
    """Explicit cross-cell ratios of bits components; never average kernels/ratios.

    Cross-cell intervals require a study-level coupled block plan and therefore
    are deliberately unavailable from these compact per-cell summaries.
    """
    if frame.empty or not {
        "metric",
        "numerator",
        "denominator",
        "grouping_slice_json",
    } <= set(frame):
        return []
    selected = frame[
        frame.metric.isin(["theory_eta_ir", "theory_round_target_information_fraction"])
        & frame.cell_id.isin(weights)
    ]
    result = []
    for keys, group in selected.groupby(
        [
            "metric",
            "theory_mode",
            "policy_source",
            "lag",
            "grouping_slice_json",
            "split_status",
        ],
        dropna=False,
    ):
        ids = set(group.cell_id)
        valid = (
            ids == set(weights)
            and not group.cell_id.duplicated().any()
            and np.isfinite(group[["numerator", "denominator"]].to_numpy()).all()
        )
        numerator = (
            sum(weights[r.cell_id] * r.numerator for r in group.itertuples())
            if valid
            else math.nan
        )
        denominator = (
            sum(weights[r.cell_id] * r.denominator for r in group.itertuples())
            if valid
            else math.nan
        )
        result.append(
            dict(
                metric=keys[0],
                empirical_metric=group.iloc[0].empirical_metric,
                theory_mode=keys[1],
                policy_source=keys[2],
                lag=keys[3],
                grouping_slice_json=keys[4],
                split_status=keys[5],
                grouping_json=calibration._json({"qualified_cell_weights": weights}),
                cell_id=None,
                qualified_cell_id=None,
                units="dimensionless",
                estimate=numerator / denominator if denominator > 0 else math.nan,
                numerator=numerator,
                denominator=denominator,
                model_status="available"
                if valid
                else "missing_or_incompatible_cell_components",
                estimand_variant="explicit_cross_cell_component_ratio",
                uncertainty_status="unavailable_requires_joint_cross_cell_block_plan",
                ci_low=math.nan,
                ci_high=math.nan,
                dependencies_json=calibration._json(group.theory_estimate_id.tolist()),
                theory_estimate_id=calibration._id(
                    [keys, weights, group.theory_estimate_id.tolist()]
                ),
                missing_cell_ids_json=calibration._json(sorted(set(weights) - ids)),
            )
        )
    return result
