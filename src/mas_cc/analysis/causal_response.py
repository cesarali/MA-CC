"""Offline causal-response and communication-efficiency analysis.

The estimand implemented here is the Horvitz--Thompson round contribution

    [U/e - (1-U)/(1-e)] * (x[t+h] - x[t])

for the randomized binary controller gate.  Communication mode and realized
delivery are deliberately kept out of the causal grouping: both occur after
the gate and therefore have no silent-arm counterpart.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ESTIMATOR_VERSION = "propensity_weighted_causal_response_v1"
AVAILABLE_SUSCEPTIBILITY_VERSION = "available_causal_susceptibility_v1"
DEFAULT_LAGS = (1, 2, 3)
FORBIDDEN_CAUSAL_STRATA = frozenset(
    {
        "chosen_message_mode",
        "communication_mode",
        "actual_controller_posts",
        "controller_posts",
        "controller_message_exposures",
        "controller_unique_readers",
    }
)
COST_FIELDS = {
    "actual_posts": "actual_controller_posts",
    "exposures": "controller_message_exposures",
    "unique_readers_per_round": "controller_unique_readers",
    "new_controller_facts": "new_controller_facts",
    "reactivated_controller_facts": "reactivated_controller_fact_count",
}


def _first_present(frame: pd.DataFrame, names: Sequence[str]) -> pd.Series:
    for name in names:
        if name in frame:
            return frame[name]
    return pd.Series(math.nan, index=frame.index, dtype=float)


def _numeric(frame: pd.DataFrame, names: Sequence[str]) -> pd.Series:
    return pd.to_numeric(_first_present(frame, names), errors="coerce")


def _sequence(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        return list(decoded) if isinstance(decoded, (list, tuple)) else []
    return list(value) if isinstance(value, (list, tuple, set, np.ndarray)) else []


def _target_share(frame: pd.DataFrame, *, before: bool) -> pd.Series:
    suffix = "before" if before else "after"
    direct = _numeric(
        frame,
        (
            f"target_fraction_{suffix}",
            "controller_target_share_before" if before else "controller_target_share",
        ),
    )
    counts_name = "occupation_counts_before" if before else "occupation_counts_after"
    if not {"analysis_target", "possible_answers", counts_name}.issubset(frame.columns):
        return direct
    derived = pd.Series(math.nan, index=frame.index, dtype=float)
    for index in frame.index:
        options = [str(item) for item in _sequence(frame.at[index, "possible_answers"])]
        counts = _sequence(frame.at[index, counts_name])
        target = str(frame.at[index, "analysis_target"])
        if target not in options or len(options) != len(counts) or not counts:
            continue
        total = sum(int(value) for value in counts)
        if total > 0:
            derived.at[index] = int(counts[options.index(target)]) / total
    comparable = direct.notna() & derived.notna()
    if comparable.any() and not np.allclose(
        direct.loc[comparable], derived.loc[comparable], rtol=0, atol=1e-12
    ):
        raise ValueError(
            "recorded target share disagrees with analysis_target orientation"
        )
    direct.loc[direct.isna()] = derived.loc[direct.isna()]
    return direct


def _expected_rounds(
    group: pd.DataFrame, cell_row: Mapping[str, Any] | None
) -> tuple[int, ...]:
    if cell_row is not None:
        for field in ("population_rounds", "horizon"):
            raw = cell_row.get(field)
            try:
                count = int(raw)
            except (TypeError, ValueError):
                continue
            if count > 0:
                return tuple(range(count))
    patterns = Counter(
        tuple(sorted(int(value) for value in episode["round_index"].tolist()))
        for _, episode in group.groupby("episode_id", sort=False)
    )
    if not patterns:
        return ()
    return max(patterns, key=lambda value: (patterns[value], len(value), value))


def build_causal_response_inputs(
    rounds: pd.DataFrame,
    cells: pd.DataFrame | None = None,
    *,
    lags: Sequence[int] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """Validate randomized rounds and build one wide estimator-input row each."""

    required = {"cell_id", "episode_id", "round_index"}
    if rounds.empty or not required.issubset(rounds.columns):
        return pd.DataFrame()
    requested_lags = tuple(sorted({int(value) for value in lags}))
    if not requested_lags or requested_lags[0] < 1:
        raise ValueError("causal-response lags must be positive integers")

    source = rounds.copy()
    source["round_index"] = pd.to_numeric(source["round_index"], errors="raise").astype(
        int
    )
    if (source["round_index"] < 0).any():
        raise ValueError("causal-response round indices must be non-negative")
    source["U_t"] = _numeric(source, ("U_k", "controller_sampled_U"))
    source["e_t"] = _numeric(
        source,
        (
            "P_U1_given_Y",
            "controller_probability_U1_given_Y",
            "controller_action_probability",
        ),
    )
    partial = source["U_t"].isna() ^ source["e_t"].isna()
    if partial.any():
        raise ValueError("causal-response rows must retain both U_t and e_t")
    source = source[source["U_t"].notna()].copy()
    if source.empty:
        return pd.DataFrame()
    if not source["U_t"].isin([0, 1]).all():
        raise ValueError("causal-response U_t must be exactly zero or one")
    if (~source["e_t"].between(0.0, 1.0, inclusive="neither")).any():
        raise ValueError("causal-response e_t must be strictly between zero and one")
    duplicated = source.duplicated(["cell_id", "episode_id", "round_index"], keep=False)
    if duplicated.any():
        raise ValueError("causal-response round identities must be unique")

    source["x_t"] = _target_share(source, before=True)
    source["x_after"] = _target_share(source, before=False)
    if source[["x_t", "x_after"]].isna().any().any():
        raise ValueError("causal-response target-oriented shares are missing")
    if (
        not source["x_t"].between(0, 1).all()
        or not source["x_after"].between(0, 1).all()
    ):
        raise ValueError("causal-response target shares must lie in [0, 1]")
    source["ipw_contrast_weight"] = np.where(
        source["U_t"] == 1,
        1.0 / source["e_t"],
        -1.0 / (1.0 - source["e_t"]),
    )

    cells_by_id: dict[str, Mapping[str, Any]] = {}
    if cells is not None and not cells.empty and "cell_id" in cells:
        cells_by_id = {
            str(row["cell_id"]): row for row in cells.to_dict(orient="records")
        }
    completeness: dict[tuple[str, str], bool] = {}
    for cell_id, cell_group in source.groupby("cell_id", sort=True):
        expected = _expected_rounds(cell_group, cells_by_id.get(str(cell_id)))
        for episode_id, episode in cell_group.groupby("episode_id", sort=True):
            actual = tuple(sorted(int(value) for value in episode["round_index"]))
            completeness[(str(cell_id), str(episode_id))] = actual == expected

    source = source.sort_values(["cell_id", "episode_id", "round_index"]).reset_index(
        drop=True
    )
    source["episode_complete"] = [
        completeness[(str(row.cell_id), str(row.episode_id))]
        for row in source.itertuples()
    ]
    source["initialization_block_id"] = _first_present(
        source, ("physical_initial_state_hash", "initialization_artifact_hash")
    ).astype("string")
    fallback = _first_present(source, ("initialization_repetition", "repetition_index"))
    source["initialization_block_id"] = source["initialization_block_id"].fillna(
        fallback.map(lambda value: None if pd.isna(value) else f"repetition:{value}")
    )
    source["initialization_block_id"] = source["initialization_block_id"].fillna(
        source["cell_id"].astype(str) + "/" + source["episode_id"].astype(str)
    )

    lookup = {
        (str(row.cell_id), str(row.episode_id), int(row.round_index)): float(
            row.x_after
        )
        for row in source.itertuples()
    }
    for lag in requested_lags:
        outcomes = [
            lookup.get(
                (str(row.cell_id), str(row.episode_id), int(row.round_index) + lag - 1)
            )
            for row in source.itertuples()
        ]
        source[f"x_t_plus_{lag}"] = outcomes
        source[f"delta_x_h{lag}"] = source[f"x_t_plus_{lag}"] - source["x_t"]
        source[f"causal_response_h{lag}"] = (
            source["ipw_contrast_weight"] * source[f"delta_x_h{lag}"]
        )
        source[f"lag_{lag}_available"] = source[f"x_t_plus_{lag}"].notna()
    return source


def _support_status(action: int, silence: int) -> str:
    if action == 0 or silence == 0:
        return "unsupported"
    if min(action, silence) < 2:
        return "limited"
    return "adequate"


def _bootstrap_draws(
    complete: pd.DataFrame, *, resamples: int, seed: int
) -> list[pd.DataFrame]:
    if resamples <= 0 or complete.empty:
        return []
    blocks = sorted(complete["initialization_block_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    by_block = {
        block: complete[complete["initialization_block_id"].astype(str) == block]
        for block in blocks
    }
    return [
        pd.concat(
            [
                by_block[block]
                for block in rng.choice(blocks, size=len(blocks), replace=True)
            ],
            ignore_index=True,
        )
        for _ in range(resamples)
    ]


def _interval(values: Sequence[float], confidence: float) -> tuple[float, float]:
    finite = np.asarray(
        [value for value in values if math.isfinite(value)], dtype=float
    )
    if not len(finite):
        return math.nan, math.nan
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(finite, alpha)), float(np.quantile(finite, 1.0 - alpha))


def estimate_causal_response(
    inputs: pd.DataFrame,
    *,
    lags: Sequence[int] = DEFAULT_LAGS,
    bootstrap_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    """Return cell effects, support diagnostics, and transient block draws."""

    if inputs.empty:
        return pd.DataFrame(), pd.DataFrame(), []
    complete = inputs[inputs["episode_complete"]].copy()
    draws = _bootstrap_draws(complete, resamples=bootstrap_resamples, seed=seed)
    effect_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    incomplete_by_cell = (
        inputs.loc[~inputs["episode_complete"]]
        .groupby("cell_id")["episode_id"]
        .nunique()
        .to_dict()
    )
    for cell_id, cell in complete.groupby("cell_id", sort=True):
        action = int((cell["U_t"] == 1).sum())
        silence = int((cell["U_t"] == 0).sum())
        for lag in sorted({int(value) for value in lags}):
            field = f"causal_response_h{lag}"
            eligible = cell[field].dropna()
            estimates = []
            for draw in draws:
                values = draw.loc[draw["cell_id"] == cell_id, field].dropna()
                estimates.append(float(values.mean()) if len(values) else math.nan)
            ci_low, ci_high = _interval(estimates, confidence)
            status = _support_status(action, silence)
            common = {
                "cell_id": cell_id,
                "lag": lag,
                "n_observations": int(len(eligible)),
                "n_episodes": int(
                    cell.loc[cell[field].notna(), "episode_id"].nunique()
                ),
                "n_initialization_blocks": int(
                    cell.loc[cell[field].notna(), "initialization_block_id"].nunique()
                ),
                "action_count": action,
                "silence_count": silence,
                "support_status": status,
                "missing_lag_count": int(cell[field].isna().sum()),
                "incomplete_episode_count": int(incomplete_by_cell.get(cell_id, 0)),
            }
            effect_rows.append(
                {
                    **common,
                    "metric": "propensity_weighted_causal_response",
                    "estimator_version": ESTIMATOR_VERSION,
                    "estimate": float(eligible.mean()) if len(eligible) else math.nan,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "confidence": confidence,
                    "bootstrap_resamples": bootstrap_resamples,
                    "bootstrap_unit": "shared_initialization_block",
                    "units": "target_share_change",
                }
            )
            support_rows.append(
                {
                    **common,
                    "e_min": float(cell["e_t"].min()),
                    "e_q05": float(cell["e_t"].quantile(0.05)),
                    "e_median": float(cell["e_t"].median()),
                    "e_q95": float(cell["e_t"].quantile(0.95)),
                    "e_max": float(cell["e_t"].max()),
                    "effective_action_weight_sum": float(
                        (
                            cell.loc[cell["U_t"] == 1, "U_t"]
                            / cell.loc[cell["U_t"] == 1, "e_t"]
                        ).sum()
                    ),
                    "effective_silence_weight_sum": float(
                        (1.0 / (1.0 - cell.loc[cell["U_t"] == 0, "e_t"])).sum()
                    ),
                }
            )
    return pd.DataFrame(effect_rows), pd.DataFrame(support_rows), draws


def _available_support(group: pd.DataFrame) -> dict[str, Any]:
    """Return audit fields for rows contributing to an available-mass estimate."""

    action = int((group["U_t"] == 1).sum())
    silence = int((group["U_t"] == 0).sum())
    return {
        "n_rounds": int(len(group)),
        "n_episodes": int(group["episode_id"].nunique()),
        "n_initialization_blocks": int(group["initialization_block_id"].nunique()),
        "n_action": action,
        "n_silence": silence,
        "support_status": _support_status(action, silence),
        "propensity_min": float(group["e_t"].min()) if len(group) else math.nan,
        "propensity_q05": float(group["e_t"].quantile(0.05))
        if len(group)
        else math.nan,
        "propensity_median": float(group["e_t"].median()) if len(group) else math.nan,
        "propensity_q95": float(group["e_t"].quantile(0.95))
        if len(group)
        else math.nan,
        "propensity_max": float(group["e_t"].max()) if len(group) else math.nan,
        "available_mass": float(group["available_mass"].sum()) if len(group) else 0.0,
    }


def _with_available_susceptibility(frame: pd.DataFrame) -> pd.DataFrame:
    """Add transient availability fields without changing retained round inputs."""

    result = frame.copy()
    result["available_mass"] = 1.0 - result["x_t"]
    result["saturated"] = result["x_t"] == 1.0
    result["available_susceptibility_defined"] = (
        result["lag_1_available"] & ~result["saturated"]
    )
    result["available_causal_susceptibility_h1"] = np.where(
        result["available_susceptibility_defined"],
        result["causal_response_h1"] / result["available_mass"],
        math.nan,
    )
    result["target_fraction_bin_index"] = np.minimum(
        np.floor(8.0 * result["x_t"]).astype(int), 7
    )
    return result


def estimate_available_causal_susceptibility(
    inputs: pd.DataFrame,
    draws: Sequence[pd.DataFrame],
    *,
    bins: int = 8,
    bootstrap_resamples: int = 1000,
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate immediate causal response per pre-treatment available mass.

    The state-local table averages each round's normalized contribution.  The
    cell summary instead divides the sum of causal contributions by the sum of
    available mass, including that ratio calculation inside every bootstrap
    replicate. Saturated rows are reported but never assigned a finite value.
    """

    if inputs.empty:
        return pd.DataFrame(), pd.DataFrame()
    if not {"causal_response_h1", "lag_1_available", "x_t"}.issubset(inputs.columns):
        raise ValueError(
            "available susceptibility requires the immediate causal response"
        )
    if bins != 8:
        raise ValueError("available susceptibility uses the fixed eight-bin convention")

    complete = _with_available_susceptibility(inputs[inputs["episode_complete"]])
    available_draws = [_with_available_susceptibility(draw) for draw in draws]
    state_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    common_estimator = {
        "estimator_version": AVAILABLE_SUSCEPTIBILITY_VERSION,
        "confidence": confidence,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_unit": "shared_initialization_block",
        "units": "target_share_response_per_available_target_mass",
    }
    for cell_id, cell in complete.groupby("cell_id", sort=True):
        saturated_cell = cell[cell["saturated"] & cell["lag_1_available"]]
        eligible_cell = cell[cell["available_susceptibility_defined"]]
        for bin_index in range(bins):
            in_bin = cell["target_fraction_bin_index"] == bin_index
            eligible = cell[in_bin & cell["available_susceptibility_defined"]]
            saturated = cell[in_bin & cell["saturated"] & cell["lag_1_available"]]
            draw_estimates: list[float] = []
            for draw in available_draws:
                subset = draw[
                    (draw["cell_id"] == cell_id)
                    & (draw["target_fraction_bin_index"] == bin_index)
                    & draw["available_susceptibility_defined"]
                ]
                values = subset["available_causal_susceptibility_h1"].dropna()
                draw_estimates.append(float(values.mean()) if len(values) else math.nan)
            ci_low, ci_high = _interval(draw_estimates, confidence)
            values = eligible["available_causal_susceptibility_h1"].dropna()
            state_rows.append(
                {
                    "cell_id": cell_id,
                    "x_bin": bin_index,
                    "target_fraction_bin_index": bin_index,
                    "target_fraction_bin_lower": bin_index / bins,
                    "target_fraction_bin_upper": (bin_index + 1) / bins,
                    "target_fraction_bin_center": (bin_index + 0.5) / bins,
                    "target_fraction_bin_count": bins,
                    "metric": "propensity_weighted_available_susceptibility",
                    "estimator_name": "propensity_weighted_available_susceptibility",
                    "estimate": float(values.mean()) if len(values) else math.nan,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_saturated_excluded": int(len(saturated)),
                    **_available_support(eligible),
                    **common_estimator,
                }
            )

        draw_ratios: list[float] = []
        for draw in available_draws:
            subset = draw[
                (draw["cell_id"] == cell_id) & draw["available_susceptibility_defined"]
            ]
            denominator = float(subset["available_mass"].sum())
            draw_ratios.append(
                float(subset["causal_response_h1"].sum()) / denominator
                if denominator > 0
                else math.nan
            )
        ci_low, ci_high = _interval(draw_ratios, confidence)
        denominator = float(eligible_cell["available_mass"].sum())
        numerator = float(eligible_cell["causal_response_h1"].sum())
        summary_rows.append(
            {
                "cell_id": cell_id,
                "metric": "available_mass_weighted_causal_susceptibility",
                "estimator_name": "available_mass_weighted_causal_susceptibility",
                "estimate": numerator / denominator if denominator > 0 else math.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "causal_response_sum": numerator,
                "n_saturated_excluded": int(len(saturated_cell)),
                **_available_support(eligible_cell),
                **common_estimator,
            }
        )
    return pd.DataFrame(state_rows), pd.DataFrame(summary_rows)


def _micro_audit(
    funnel: pd.DataFrame, micro_slots: pd.DataFrame | None
) -> pd.DataFrame:
    if micro_slots is None or micro_slots.empty:
        funnel["micro_slot_audit_available"] = False
        return funnel
    required = {"cell_id", "episode_id", "round_index"}
    if not required.issubset(micro_slots.columns):
        funnel["micro_slot_audit_available"] = False
        return funnel
    micro = micro_slots.copy()
    for field in (
        "sampled_controller_message_ids",
        "new_controller_fact_ids",
        "reactivated_controller_fact_ids",
    ):
        if field not in micro:
            funnel["micro_slot_audit_available"] = False
            return funnel
        micro[field] = micro[field].map(_sequence)
    micro["_reader"] = np.where(
        micro["sampled_controller_message_ids"].map(bool),
        _first_present(micro, ("focal_agent_id",)).astype("string"),
        pd.NA,
    )
    grouped = micro.groupby(["cell_id", "episode_id", "round_index"], dropna=False)
    audit = grouped.agg(
        micro_controller_exposures=(
            "sampled_controller_message_ids",
            lambda values: sum(len(value) for value in values),
        ),
        micro_unique_readers=("_reader", "nunique"),
        micro_new_controller_facts=(
            "new_controller_fact_ids",
            lambda values: sum(len(value) for value in values),
        ),
        micro_reactivated_controller_facts=(
            "reactivated_controller_fact_ids",
            lambda values: sum(len(value) for value in values),
        ),
    ).reset_index()
    result = funnel.merge(
        audit, on=["cell_id", "episode_id", "round_index"], how="left"
    )
    checks = [
        ("exposures", "micro_controller_exposures"),
        ("new_controller_facts", "micro_new_controller_facts"),
        ("reactivated_controller_facts", "micro_reactivated_controller_facts"),
    ]
    if "focal_agent_id" in micro_slots:
        checks.append(("unique_readers_per_round", "micro_unique_readers"))
    result["micro_slot_audit_available"] = result[checks[0][1]].notna()
    for round_field, micro_field in checks:
        mismatch = result["micro_slot_audit_available"] & (
            pd.to_numeric(result[round_field], errors="coerce")
            != pd.to_numeric(result[micro_field], errors="coerce")
        )
        if mismatch.any():
            raise ValueError(
                f"round communication count disagrees with micro slots: {round_field}"
            )
    return result


def build_communication_funnel(
    inputs: pd.DataFrame, micro_slots: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Link gate assignment through delivery and target-oriented response."""

    if inputs.empty:
        return pd.DataFrame()
    funnel = inputs[
        [
            column
            for column in inputs.columns
            if column
            in {
                "study_id",
                "source_run_id",
                "cell_id",
                "episode_id",
                "episode_key",
                "repetition_index",
                "initialization_repetition",
                "physical_initial_state_hash",
                "initialization_block_id",
                "round_index",
                "U_t",
                "e_t",
                "episode_complete",
                "x_t",
                "chosen_message_mode",
                "controller_actuation_mode",
                "controller_communication_policy",
                "communication_policy",
                "intervention_budget",
            }
        ]
    ].copy()
    funnel["controller_action"] = inputs["U_t"].astype(int)
    for label, source in COST_FIELDS.items():
        alternatives = (
            (source, "controller_posts") if label == "actual_posts" else (source,)
        )
        funnel[label] = _numeric(inputs, alternatives).fillna(0.0)
    silent_cost = funnel["U_t"] == 0
    if (funnel.loc[silent_cost, list(COST_FIELDS)].fillna(0) != 0).any().any():
        raise ValueError(
            "silent randomized rounds cannot have controller communication cost"
        )
    funnel["unique_reader_scope"] = "per_round_not_episode_wide"
    for lag in DEFAULT_LAGS:
        for prefix in ("delta_x", "causal_response"):
            field = f"{prefix}_h{lag}"
            if field in inputs:
                funnel[field] = inputs[field]
    return _micro_audit(funnel, micro_slots)


def communication_efficiency(
    inputs: pd.DataFrame,
    effects: pd.DataFrame,
    draws: Sequence[pd.DataFrame],
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Estimate activation cost separately, then form aggregate response/cost."""

    if inputs.empty or effects.empty:
        return pd.DataFrame()
    complete = inputs[inputs["episode_complete"]].copy()
    rows: list[dict[str, Any]] = []
    for effect in effects.to_dict(orient="records"):
        cell_id, lag = effect["cell_id"], int(effect["lag"])
        cell = complete[complete["cell_id"] == cell_id]
        for metric, source in COST_FIELDS.items():
            alternatives = (
                (source, "controller_posts") if metric == "actual_posts" else (source,)
            )
            cost = _numeric(cell, alternatives).fillna(0.0)
            contribution = cell["U_t"] * cost / cell["e_t"]
            expected_cost = (
                float(contribution.mean()) if len(contribution) else math.nan
            )
            draw_costs: list[float] = []
            draw_ratios: list[float] = []
            response_field = f"causal_response_h{lag}"
            for draw in draws:
                subset = draw[draw["cell_id"] == cell_id]
                if subset.empty:
                    continue
                draw_source = _numeric(subset, alternatives).fillna(0.0)
                draw_cost = float((subset["U_t"] * draw_source / subset["e_t"]).mean())
                draw_response = float(subset[response_field].dropna().mean())
                draw_costs.append(draw_cost)
                draw_ratios.append(
                    draw_response / draw_cost if draw_cost > 0 else math.nan
                )
            cost_low, cost_high = _interval(draw_costs, confidence)
            ratio_low, ratio_high = _interval(draw_ratios, confidence)
            rows.append(
                {
                    **effect,
                    "cost_metric": metric,
                    "expected_activation_cost": expected_cost,
                    "cost_ci_low": cost_low,
                    "cost_ci_high": cost_high,
                    "response_per_expected_cost": (
                        float(effect["estimate"]) / expected_cost
                        if expected_cost > 0 and pd.notna(effect["estimate"])
                        else math.nan
                    ),
                    "ratio_ci_low": ratio_low,
                    "ratio_ci_high": ratio_high,
                    "zero_cost_round_count": int((cost == 0).sum()),
                    "zero_denominator": bool(expected_cost == 0),
                    "cost_estimand": "IPW expected cost under policy activation",
                    "causal_conditioning_excludes_realized_mode_and_cost": True,
                }
            )
    return pd.DataFrame(rows)


def response_cost_frontier(efficiency: pd.DataFrame) -> pd.DataFrame:
    """Mark the best supported observed response reached by each cost level."""

    if efficiency.empty:
        return pd.DataFrame()
    grouping = [
        column
        for column in ("target_semantics", "cost_metric", "lag")
        if column in efficiency
    ]
    if "cost_metric" not in grouping:
        grouping.append("cost_metric")
    if "lag" not in grouping:
        grouping.append("lag")
    rows: list[dict[str, Any]] = []
    eligible = efficiency[
        efficiency["support_status"].isin(["adequate", "limited"])
        & pd.to_numeric(efficiency["estimate"], errors="coerce").notna()
        & pd.to_numeric(efficiency["expected_activation_cost"], errors="coerce").notna()
    ]
    for _, group in eligible.groupby(grouping, dropna=False, sort=True):
        best = -math.inf
        ordered = group.sort_values(
            ["expected_activation_cost", "estimate"], ascending=[True, False]
        )
        for record in ordered.to_dict(orient="records"):
            value = float(record["estimate"])
            on_frontier = value > best
            best = max(best, value)
            rows.append(
                {
                    **record,
                    "frontier_response": best,
                    "on_response_cost_frontier": on_frontier,
                    "frontier_kind": "best_observed_causal_response_at_or_below_cost",
                    "thermodynamic_efficiency": False,
                }
            )
    return pd.DataFrame(rows)


def descriptive_mode_response(inputs: pd.DataFrame) -> pd.DataFrame:
    """Post-treatment mode summaries, explicitly non-causal."""

    if inputs.empty or "chosen_message_mode" not in inputs:
        return pd.DataFrame()
    acted = inputs[(inputs["U_t"] == 1) & inputs["chosen_message_mode"].notna()]
    rows: list[dict[str, Any]] = []
    for (cell_id, mode), group in acted.groupby(
        ["cell_id", "chosen_message_mode"], sort=True
    ):
        for lag in DEFAULT_LAGS:
            values = group[f"delta_x_h{lag}"].dropna()
            rows.append(
                {
                    "cell_id": cell_id,
                    "communication_mode": mode,
                    "lag": lag,
                    "mean_observed_response": float(values.mean())
                    if len(values)
                    else math.nan,
                    "n_observations": int(len(values)),
                    "causal": False,
                    "interpretation": "descriptive post-treatment mode breakdown",
                }
            )
    return pd.DataFrame(rows)


def analyze_causal_communication(
    rounds: pd.DataFrame,
    cells: pd.DataFrame,
    micro_slots: pd.DataFrame | None = None,
    *,
    bootstrap_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 1,
) -> dict[str, pd.DataFrame]:
    """Build every Section 2/3 table from canonical observations only."""

    coordinates = cells.drop_duplicates("cell_id") if not cells.empty else cells
    attach = lambda frame: (
        frame
        if frame.empty or coordinates.empty or "cell_id" not in frame
        else frame.merge(
            coordinates, on="cell_id", how="left", suffixes=("", "_coordinate")
        )
    )
    inputs = attach(build_causal_response_inputs(rounds, cells))
    effects, support, draws = estimate_causal_response(
        inputs,
        bootstrap_resamples=bootstrap_resamples,
        confidence=confidence,
        seed=seed,
    )
    available_state_local, available_summary = estimate_available_causal_susceptibility(
        inputs,
        draws,
        bootstrap_resamples=bootstrap_resamples,
        confidence=confidence,
    )
    funnel = build_communication_funnel(inputs, micro_slots)
    efficiency = communication_efficiency(inputs, effects, draws, confidence=confidence)
    effects, support, available_state_local, available_summary, funnel, efficiency = (
        map(
            attach,
            (
                effects,
                support,
                available_state_local,
                available_summary,
                funnel,
                efficiency,
            ),
        )
    )
    frontier = response_cost_frontier(efficiency)
    modes = attach(descriptive_mode_response(inputs))
    return {
        "causal_response_round_inputs": inputs,
        "causal_response_effects": effects,
        "causal_response_support": support,
        "available_causal_susceptibility_state_local": available_state_local,
        "available_causal_susceptibility_summary": available_summary,
        "communication_funnel": funnel,
        "communication_efficiency": efficiency,
        "response_cost_frontier": frontier,
        "communication_mode_descriptive_response": modes,
    }


__all__ = [
    "AVAILABLE_SUSCEPTIBILITY_VERSION",
    "DEFAULT_LAGS",
    "ESTIMATOR_VERSION",
    "FORBIDDEN_CAUSAL_STRATA",
    "analyze_causal_communication",
    "build_causal_response_inputs",
    "build_communication_funnel",
    "communication_efficiency",
    "descriptive_mode_response",
    "estimate_causal_response",
    "estimate_available_causal_susceptibility",
    "response_cost_frontier",
]
