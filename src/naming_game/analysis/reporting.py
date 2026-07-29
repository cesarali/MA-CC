"""Human-facing reports built exclusively from stored experiment histories."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .metrics import GROUP_COLUMNS

NON_ESTIMABLE_MESSAGE = "empowerment not estimable from this run"
EXPECTED_POLICIES = {
    "neutral": {"always_A", "always_B", "no_committee"},
    "consensus_attack": {
        "support_incumbent",
        "promote_alternative",
        "no_committee",
    },
    "pulse": {"alternative_pulse", "no_pulse"},
}


def _missing_series(frame: pd.DataFrame, value: Any = None) -> pd.Series:
    return pd.Series([value] * len(frame), index=frame.index, dtype=object)


def _derive_direction(row: pd.Series) -> str | None:
    incumbent = row.get("incumbent_name")
    promoted = row.get("promoted_name")
    if pd.isna(incumbent) or pd.isna(promoted):
        return None
    strong = row.get("strong_name")
    weak = row.get("weak_name")
    if pd.notna(strong) and pd.notna(weak):
        if incumbent == strong and promoted == weak:
            return "strong_to_weak"
        if incumbent == weak and promoted == strong:
            return "weak_to_strong"
    return f"{incumbent}_to_{promoted}"


def normalize_histories(
    interactions: pd.DataFrame, episodes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Normalize schema-v1 and schema-v2 histories without external metadata."""

    trajectory = interactions.copy()
    summaries = episodes.copy()
    warnings: list[str] = []
    for frame in (trajectory, summaries):
        if "incumbent_name" not in frame:
            frame["incumbent_name"] = frame.get("incumbent", _missing_series(frame))
        if "promoted_name" not in frame:
            frame["promoted_name"] = frame.get("alternative", _missing_series(frame))
        for column in ("strong_name", "weak_name", "convention_role_source"):
            if column not in frame:
                frame[column] = _missing_series(frame)
        if "attack_direction" not in frame:
            frame["attack_direction"] = frame.apply(_derive_direction, axis=1)
        else:
            missing = frame["attack_direction"].isna()
            if missing.any():
                frame.loc[missing, "attack_direction"] = frame.loc[missing].apply(
                    _derive_direction, axis=1
                )

    promoted_present = summaries["promoted_name"].notna()
    if "ever_crossed" not in summaries:
        legacy = summaries.get("takeover", False)
        summaries["ever_crossed"] = pd.Series(legacy, index=summaries.index).fillna(False)
    summaries["ever_crossed"] = (
        summaries["ever_crossed"].fillna(False).astype(bool) & promoted_present
    )
    if "terminal_takeover" not in summaries:
        summaries["terminal_takeover"] = (
            promoted_present
            & (summaries["final_convention"] == summaries["promoted_name"])
        )
    if "incumbent_survives" not in summaries:
        summaries["incumbent_survives"] = (
            summaries["incumbent_name"].notna()
            & (summaries["final_convention"] == summaries["incumbent_name"])
        )
    summaries["terminal_takeover"] = summaries["terminal_takeover"].fillna(False).astype(bool)
    summaries["incumbent_survives"] = summaries["incumbent_survives"].fillna(False).astype(bool)
    summaries["takeover"] = summaries["ever_crossed"]

    if "recovery_time_population_rounds" not in summaries:
        numerator = pd.to_numeric(
            summaries.get("recovery_time_interactions", _missing_series(summaries)),
            errors="coerce",
        )
        denominator = pd.to_numeric(summaries["N"], errors="coerce")
        summaries["recovery_time_population_rounds"] = numerator / denominator

    metadata_columns = [
        "strong_name",
        "weak_name",
        "convention_role_source",
        "incumbent_name",
        "promoted_name",
        "attack_direction",
    ]
    episode_metadata = summaries.set_index("episode_id")[metadata_columns]
    for column in metadata_columns:
        mapped = trajectory["episode_id"].map(episode_metadata[column])
        if column not in trajectory:
            trajectory[column] = mapped
        else:
            trajectory[column] = trajectory[column].where(
                trajectory[column].notna(), mapped
            )

    attack_rows = summaries["regime"].isin(("consensus_attack", "pulse"))
    if attack_rows.any() and summaries.loc[attack_rows, "strong_name"].isna().all():
        warnings.append(
            "Convention-role calibration is absent; attack directions use neutral name-to-name labels."
        )
    return trajectory, summaries, warnings


def attach_null_summary(estimates: pd.DataFrame, nulls: pd.DataFrame) -> pd.DataFrame:
    result = estimates.copy()
    numeric_columns = (
        "shuffle_null_median",
        "shuffle_null_ci_low",
        "shuffle_null_ci_high",
        "empowerment_above_shuffle_median",
    )
    for column in numeric_columns:
        result[column] = math.nan
    result["above_shuffle_97_5pct"] = pd.Series(
        pd.NA, index=result.index, dtype="boolean"
    )
    if result.empty or nulls.empty:
        return result
    terminal_null = nulls[
        (nulls["statistic"] == "terminal_all")
        & (nulls["null_type"] == "episode_label_shuffle_null")
    ]
    if terminal_null.empty:
        return result
    key_columns = [column for column in result.columns if column in GROUP_COLUMNS and column != "committee_policy"]
    key_columns += ["statistic"]
    quantiles = (
        terminal_null.groupby(key_columns, dropna=False)["jeffreys"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .rename(
            columns={
                0.025: "shuffle_null_ci_low_join",
                0.5: "shuffle_null_median_join",
                0.975: "shuffle_null_ci_high_join",
            }
        )
        .reset_index()
    )
    merged = result.merge(quantiles, on=key_columns, how="left")
    terminal_mask = merged["statistic"] == "terminal_all"
    for target, source in (
        ("shuffle_null_ci_low", "shuffle_null_ci_low_join"),
        ("shuffle_null_median", "shuffle_null_median_join"),
        ("shuffle_null_ci_high", "shuffle_null_ci_high_join"),
    ):
        merged.loc[terminal_mask, target] = merged.loc[terminal_mask, source]
    merged.loc[terminal_mask, "empowerment_above_shuffle_median"] = (
        merged.loc[terminal_mask, "jeffreys"]
        - merged.loc[terminal_mask, "shuffle_null_median"]
    )
    valid_comparison = (
        terminal_mask
        & np.isfinite(merged["jeffreys"])
        & np.isfinite(merged["shuffle_null_ci_high"])
    )
    merged.loc[valid_comparison, "above_shuffle_97_5pct"] = (
        merged.loc[valid_comparison, "jeffreys"]
        > merged.loc[valid_comparison, "shuffle_null_ci_high"]
    )
    return merged.drop(columns=[column for column in merged if column.endswith("_join")])


def collect_warnings(
    episodes: pd.DataFrame, estimates: pd.DataFrame, nulls: pd.DataFrame
) -> list[str]:
    warnings: list[str] = []
    if not (episodes["committee_size"] == 0).any():
        warnings.append("No zero-committee baseline was found.")

    cell_columns = [
        "regime",
        "N",
        "committee_size",
        "initial_condition",
        "provider",
        "model",
        "prompt_version",
        "pulse_rounds",
    ]
    missing_cells: list[str] = []
    low_cells: list[str] = []
    for keys, group in episodes.groupby(cell_columns, dropna=False, sort=True):
        regime = str(group["regime"].iloc[0])
        expected = EXPECTED_POLICIES.get(regime, set())
        actual = set(group["committee_policy"].dropna().astype(str))
        missing = sorted(expected - actual)
        label = ", ".join(f"{column}={value}" for column, value in zip(cell_columns, keys, strict=True))
        if missing:
            missing_cells.append(f"{label}: {', '.join(missing)}")
        counts = group.groupby("committee_policy")["episode_id"].nunique()
        if not counts.empty and int(counts.min()) < 10:
            low_cells.append(f"{label}: minimum {int(counts.min())} episode(s) per observed policy")
    attack = episodes[episodes["regime"].isin(("consensus_attack", "pulse"))]
    if not attack.empty:
        if attack["strong_name"].notna().any():
            expected_directions = {"strong_to_weak", "weak_to_strong"}
        else:
            observed_pairs = {
                (str(left), str(right))
                for left, right in attack[["incumbent_name", "promoted_name"]].dropna().itertuples(index=False, name=None)
            }
            expected_directions = {
                f"{left}_to_{right}" for left, right in observed_pairs | {(right, left) for left, right in observed_pairs}
            }
        direction_cell_columns = [
            "regime", "N", "committee_size", "provider", "model", "prompt_version", "pulse_rounds"
        ]
        for keys, group in attack.groupby(direction_cell_columns, dropna=False, sort=True):
            actual_directions = set(group["attack_direction"].dropna().astype(str))
            missing_directions = sorted(expected_directions - actual_directions)
            if missing_directions:
                label = ", ".join(
                    f"{column}={value}"
                    for column, value in zip(direction_cell_columns, keys, strict=True)
                )
                missing_cells.append(
                    f"{label}: attack direction(s) {', '.join(missing_directions)}"
                )
    if missing_cells:
        preview = "; ".join(missing_cells[:8])
        suffix = f"; and {len(missing_cells) - 8} more" if len(missing_cells) > 8 else ""
        warnings.append(f"Missing policies / empty cells — {preview}{suffix}.")
    if low_cells:
        preview = "; ".join(low_cells[:8])
        suffix = f"; and {len(low_cells) - 8} more" if len(low_cells) > 8 else ""
        warnings.append(f"Too few episodes for stable reporting — {preview}{suffix}.")
    if not estimates.empty and (estimates["estimate_status"] == "non_estimable").any():
        warnings.append(NON_ESTIMABLE_MESSAGE + ".")
    if nulls.empty:
        warnings.append("No shuffle-null or circular-shift results are available.")
    return warnings


def _rolling_success(frame: pd.DataFrame) -> pd.Series:
    values = frame["success"].astype(float).to_numpy()
    counts = frame["rolling_window_count"].astype(int).to_numpy()
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    positions = np.arange(1, len(values) + 1)
    starts = positions - counts
    totals = cumulative[positions] - cumulative[starts]
    return pd.Series(totals / counts, index=frame.index)


def round_end_trajectories(interactions: pd.DataFrame) -> pd.DataFrame:
    frame = interactions.sort_values(["episode_id", "interaction_index"]).copy()
    pieces: list[pd.DataFrame] = []
    for _, episode in frame.groupby("episode_id", sort=False):
        episode = episode.copy()
        episode["rolling_success_probability"] = _rolling_success(episode)
        pieces.append(episode)
    frame = pd.concat(pieces, ignore_index=False).sort_values(
        ["episode_id", "interaction_index"]
    )
    endpoints = frame.groupby(["episode_id", "population_round"], sort=False).tail(1).copy()
    promoted_a = endpoints["promoted_name"] == "A"
    promoted_b = endpoints["promoted_name"] == "B"
    endpoints["alternative_share"] = np.where(
        promoted_a,
        endpoints["rolling_share_A"],
        np.where(promoted_b, 1.0 - endpoints["rolling_share_A"], endpoints["rolling_share_A"]),
    )
    endpoints["committee_fraction"] = endpoints["committee_size"] / endpoints["N"]
    endpoints["direction_label"] = endpoints["attack_direction"].fillna("neutral")
    return endpoints


def _trajectory_summary(
    frame: pd.DataFrame,
    value_column: str,
    *,
    line_columns: list[str],
    resamples: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for keys, group in frame.groupby(line_columns, dropna=False, sort=True):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        pivot = group.pivot_table(
            index="episode_id",
            columns="population_round",
            values=value_column,
            aggfunc="last",
        ).sort_index(axis=1)
        if pivot.empty:
            continue
        values = pivot.to_numpy(dtype=float)
        means = np.nanmean(values, axis=0)
        low = np.full(len(means), np.nan)
        high = np.full(len(means), np.nan)
        if resamples > 0 and len(pivot) > 1:
            draws = rng.integers(0, len(pivot), size=(resamples, len(pivot)))
            boot = np.nanmean(values[draws], axis=1)
            alpha = (1 - confidence) / 2
            low = np.nanquantile(boot, alpha, axis=0)
            high = np.nanquantile(boot, 1 - alpha, axis=0)
        base = dict(zip(line_columns, key_tuple, strict=True))
        for population_round, mean, lower, upper in zip(
            pivot.columns, means, low, high, strict=True
        ):
            rows.append(
                {
                    **base,
                    "population_round": float(population_round),
                    "mean": float(mean),
                    "ci_low": float(lower),
                    "ci_high": float(upper),
                    "episodes": len(pivot),
                }
            )
    return pd.DataFrame(rows)


def _atomic_save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    try:
        fig.savefig(temporary, dpi=180, bbox_inches="tight")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _direction_values(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["not available"]
    return sorted(frame["direction_label"].fillna("neutral").astype(str).unique())


def _format_direction(value: str) -> str:
    return value.replace("_to_", " → ").replace("_", " ")


def _plot_trajectory_axis(
    ax: Any,
    summary: pd.DataFrame,
    *,
    ylabel: str,
    fractions: Iterable[float],
) -> None:
    import matplotlib.pyplot as plt

    fraction_values = sorted(set(float(value) for value in fractions))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, max(1, len(fraction_values))))
    color_map = dict(zip(fraction_values, colors, strict=True))
    for fraction in fraction_values:
        line = summary[np.isclose(summary["committee_fraction"], fraction)].sort_values(
            "population_round"
        )
        if line.empty:
            continue
        label = "no committee" if math.isclose(fraction, 0.0) else f"{fraction:.3g}"
        ax.plot(line["population_round"], line["mean"], color=color_map[fraction], label=label)
        finite = np.isfinite(line["ci_low"]) & np.isfinite(line["ci_high"])
        if finite.any():
            ax.fill_between(
                line.loc[finite, "population_round"],
                line.loc[finite, "ci_low"],
                line.loc[finite, "ci_high"],
                color=color_map[fraction],
                alpha=0.14,
                linewidth=0,
            )
    ax.set(xlabel="population round", ylabel=ylabel, ylim=(-0.03, 1.03))
    ax.grid(alpha=0.2)


def _metric_for_regime(metrics: pd.DataFrame, regime: str) -> pd.DataFrame:
    frame = metrics[metrics["regime"] == regime].copy()
    policy = {
        "consensus_attack": "promote_alternative",
        "pulse": "alternative_pulse",
    }.get(regime)
    if policy is not None and (frame["committee_policy"] == policy).any():
        frame = frame[frame["committee_policy"] == policy]
    frame["committee_fraction"] = frame["committee_size"] / frame["N"]
    frame["direction_label"] = frame["attack_direction"].fillna("neutral")
    return frame


def make_experiment_summary(
    interactions: pd.DataFrame,
    metrics: pd.DataFrame,
    estimates: pd.DataFrame,
    output_path: Path,
    *,
    bootstrap_resamples: int,
    confidence: float,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    rounds = round_end_trajectories(interactions)
    available = set(rounds["regime"].unique())
    regime = next(
        (candidate for candidate in ("consensus_attack", "pulse", "neutral") if candidate in available),
        None,
    )
    if regime is None:
        raise ValueError("No supported experiment regime is present in the histories.")
    policy = {
        "consensus_attack": "promote_alternative",
        "pulse": "alternative_pulse",
    }.get(regime)
    selected = rounds[rounds["regime"] == regime].copy()
    if policy is not None and (selected["committee_policy"] == policy).any():
        selected = selected[selected["committee_policy"] == policy]
    if regime == "pulse" and selected["pulse_rounds"].nunique(dropna=True) > 1:
        selected = selected[selected["pulse_rounds"] == selected["pulse_rounds"].dropna().min()]
    directions = _direction_values(selected)
    fractions = sorted(selected["committee_fraction"].dropna().unique())
    convention = _trajectory_summary(
        selected,
        "alternative_share",
        line_columns=["direction_label", "committee_fraction"],
        resamples=bootstrap_resamples,
        confidence=confidence,
        seed=seed + 700,
    )
    coordination = _trajectory_summary(
        selected,
        "rolling_success_probability",
        line_columns=["direction_label", "committee_fraction"],
        resamples=bootstrap_resamples,
        confidence=confidence,
        seed=seed + 701,
    )
    outcome_metrics = _metric_for_regime(metrics, regime)
    if regime == "pulse" and outcome_metrics["pulse_rounds"].nunique(dropna=True) > 1:
        outcome_metrics = outcome_metrics[
            outcome_metrics["pulse_rounds"] == outcome_metrics["pulse_rounds"].dropna().min()
        ]
    terminal = estimates[
        (estimates["regime"] == regime) & (estimates["statistic"] == "terminal_all")
    ].copy()
    if regime == "pulse" and terminal["pulse_rounds"].nunique(dropna=True) > 1:
        terminal = terminal[terminal["pulse_rounds"] == terminal["pulse_rounds"].dropna().min()]
    terminal["committee_fraction"] = terminal["committee_size"] / terminal["N"]
    terminal["direction_label"] = terminal["attack_direction"].fillna("neutral")

    figure = plt.figure(figsize=(8 * max(2, len(directions)), 14))
    outer = figure.add_gridspec(2, 2, hspace=0.28, wspace=0.22)
    axes_a = outer[0, 0].subgridspec(1, len(directions), wspace=0.25).subplots(squeeze=False)[0]
    axes_b = outer[0, 1].subgridspec(1, len(directions), wspace=0.25).subplots(squeeze=False)[0]
    axes_c = outer[1, 0].subgridspec(1, len(directions), wspace=0.25).subplots(squeeze=False)[0]
    axes_d = outer[1, 1].subgridspec(2, len(directions), hspace=0.08, wspace=0.25).subplots(squeeze=False)

    for index, direction in enumerate(directions):
        title = _format_direction(direction)
        a_data = convention[convention["direction_label"] == direction]
        b_data = coordination[coordination["direction_label"] == direction]
        _plot_trajectory_axis(
            axes_a[index], a_data, ylabel="mean promoted-convention share", fractions=fractions
        )
        _plot_trajectory_axis(
            axes_b[index], b_data, ylabel="rolling success probability", fractions=fractions
        )
        axes_a[index].set_title(("A  Convention dynamics\n" if index == 0 else "") + title)
        axes_b[index].set_title(("B  Coordination dynamics\n" if index == 0 else "") + title)
        if index == 0 and not a_data.empty:
            axes_a[index].legend(title="committee fraction", fontsize=7)

        cell = outcome_metrics[outcome_metrics["direction_label"] == direction].sort_values(
            "committee_fraction"
        )
        ax_c = axes_c[index]
        if regime == "neutral":
            series = (
                ("terminal_A_probability", "A terminal", "o", "terminal_A_ci_low", "terminal_A_ci_high"),
                ("terminal_B_probability", "B terminal", "s", "terminal_B_ci_low", "terminal_B_ci_high"),
                ("unresolved_probability", "unresolved", "^", "unresolved_ci_low", "unresolved_ci_high"),
            )
        else:
            series = (
                ("terminal_takeover_probability", "terminal takeover", "o", "terminal_takeover_ci_low", "terminal_takeover_ci_high"),
                ("ever_crossed_probability", "ever crossed", "x", "ever_crossed_ci_low", "ever_crossed_ci_high"),
                ("incumbent_survival_probability", "incumbent survives", "s", "incumbent_survival_ci_low", "incumbent_survival_ci_high"),
                ("unresolved_probability", "unresolved", "^", "unresolved_ci_low", "unresolved_ci_high"),
            )
        for column, label, marker, low_column, high_column in series:
            if column not in cell:
                continue
            values = cell[column].astype(float)
            finite = values.notna() & cell[low_column].notna() & cell[high_column].notna()
            if finite.any():
                ax_c.errorbar(
                    cell.loc[finite, "committee_fraction"],
                    values.loc[finite],
                    yerr=np.vstack(
                        (
                            values.loc[finite] - cell.loc[finite, low_column],
                            cell.loc[finite, high_column] - values.loc[finite],
                        )
                    ),
                    marker=marker,
                    capsize=2,
                    label=label,
                )
        if regime != "neutral" and not cell.empty:
            reached = cell[cell["terminal_takeover_probability"] >= 0.95]
            if not reached.empty:
                first = reached.sort_values("committee_fraction").iloc[0]
                x_value = float(first["committee_fraction"])
                ax_c.scatter([x_value], [first["terminal_takeover_probability"]], marker="*", s=140, color="black", zorder=5)
                ax_c.axvline(x_value, color="black", linestyle=":", alpha=0.45)
                ax_c.annotate(f"smallest tested ≥0.95: {x_value:.3g}", (x_value, 0.95), fontsize=7, rotation=90, va="top")
        ax_c.set(xlabel="committee fraction", ylabel="episode probability", ylim=(-0.03, 1.03))
        ax_c.set_title(("C  Final population outcomes\n" if index == 0 else "") + title)
        ax_c.grid(alpha=0.2)
        if index == 0:
            ax_c.legend(fontsize=7)

        ax_empowerment = axes_d[0, index]
        estimate_cell = terminal[terminal["direction_label"] == direction].sort_values(
            "committee_fraction"
        )
        finite = estimate_cell[np.isfinite(estimate_cell["jeffreys"])]
        if finite.empty:
            ax_empowerment.text(0.5, 0.5, NON_ESTIMABLE_MESSAGE, ha="center", va="center", transform=ax_empowerment.transAxes, wrap=True)
        else:
            ax_empowerment.plot(
                finite["committee_fraction"],
                finite["jeffreys"],
                marker="o",
                color="tab:blue",
                label="observed empowerment",
            )
            interval = finite[
                np.isfinite(finite["ci_low"]) & np.isfinite(finite["ci_high"])
            ]
            if not interval.empty:
                ax_empowerment.errorbar(
                    interval["committee_fraction"],
                    interval["jeffreys"],
                    yerr=np.vstack(
                        (
                            interval["jeffreys"] - interval["ci_low"],
                            interval["ci_high"] - interval["jeffreys"],
                        )
                    ),
                    fmt="none",
                    color="tab:blue",
                    capsize=3,
                )
            null_finite = finite[
                np.isfinite(finite["shuffle_null_ci_low"])
                & np.isfinite(finite["shuffle_null_ci_high"])
            ]
            if not null_finite.empty:
                ax_empowerment.fill_between(
                    null_finite["committee_fraction"],
                    null_finite["shuffle_null_ci_low"],
                    null_finite["shuffle_null_ci_high"],
                    color="tab:gray",
                    alpha=0.25,
                    label="shuffle-null 95% band",
                )
                ax_empowerment.plot(
                    null_finite["committee_fraction"],
                    null_finite["shuffle_null_median"],
                    color="tab:gray",
                    linestyle="--",
                    linewidth=1,
                )
        ax_empowerment.set(ylabel="terminal empowerment (bits)")
        ax_empowerment.set_title(("D  Committee empowerment\n" if index == 0 else "") + title)
        ax_empowerment.grid(alpha=0.2)
        ax_empowerment.tick_params(labelbottom=False)
        if index == 0 and not finite.empty:
            ax_empowerment.legend(fontsize=7)

        ax_takeover = axes_d[1, index]
        if not cell.empty and regime != "neutral":
            ax_takeover.plot(cell["committee_fraction"], cell["terminal_takeover_probability"], marker="o", label="terminal takeover")
            ax_takeover.plot(cell["committee_fraction"], cell["ever_crossed_probability"], marker="x", linestyle="--", label="ever crossed")
            if index == 0:
                ax_takeover.legend(fontsize=7)
        else:
            ax_takeover.text(0.5, 0.5, "takeover not defined for neutral starts", ha="center", va="center", transform=ax_takeover.transAxes, wrap=True)
        ax_takeover.set(xlabel="committee fraction", ylabel="takeover probability", ylim=(-0.03, 1.03))
        ax_takeover.grid(alpha=0.2)

    figure.suptitle(f"Committee-empowerment experiment summary — {regime.replace('_', ' ')}", fontsize=16)
    _atomic_save(figure, output_path)
    plt.close(figure)


def make_pulse_summary(
    interactions: pd.DataFrame,
    metrics: pd.DataFrame,
    output_path: Path,
    *,
    bootstrap_resamples: int,
    confidence: float,
    seed: int,
) -> bool:
    import matplotlib.pyplot as plt

    rounds = round_end_trajectories(interactions)
    pulse = rounds[rounds["regime"] == "pulse"].copy()
    if pulse.empty:
        return False
    durations = sorted(int(value) for value in pulse["pulse_rounds"].dropna().unique())
    directions = _direction_values(pulse)
    if not durations:
        return False
    figure = plt.figure(
        figsize=(5.2 * max(len(durations), len(directions)), 4.2 * len(directions) + 10)
    )
    outer = figure.add_gridspec(2, 1, height_ratios=[max(1, len(directions)), 1.6], hspace=0.24)
    top = outer[0].subgridspec(len(directions), len(durations), hspace=0.3, wspace=0.2).subplots(squeeze=False)
    bottom = outer[1].subgridspec(3, len(directions), hspace=0.42, wspace=0.24).subplots(squeeze=False)

    for row_index, direction in enumerate(directions):
        for column_index, duration in enumerate(durations):
            ax = top[row_index, column_index]
            cell = pulse[
                (pulse["direction_label"] == direction)
                & (pulse["pulse_rounds"] == duration)
            ]
            active = cell[cell["committee_policy"] == "alternative_pulse"]
            summary = _trajectory_summary(
                active,
                "alternative_share",
                line_columns=["committee_fraction"],
                resamples=bootstrap_resamples,
                confidence=confidence,
                seed=seed + 900 + row_index * 100 + column_index,
            )
            fractions = sorted(active["committee_fraction"].dropna().unique())
            _plot_trajectory_axis(ax, summary, ylabel="promoted share", fractions=fractions)
            for line in ax.lines:
                x_data = np.asarray(line.get_xdata(), dtype=float)
                y_data = np.asarray(line.get_ydata(), dtype=float)
                line.set_data(np.concatenate(([0.0], x_data)), np.concatenate(([0.0], y_data)))
            control = cell[cell["committee_policy"] == "no_pulse"]
            if not control.empty:
                control_summary = _trajectory_summary(
                    control,
                    "alternative_share",
                    line_columns=["direction_label"],
                    resamples=bootstrap_resamples,
                    confidence=confidence,
                    seed=seed + 950 + row_index * 100 + column_index,
                )
                ax.plot(
                    np.concatenate(([0.0], control_summary["population_round"].to_numpy())),
                    np.concatenate(([0.0], control_summary["mean"].to_numpy())),
                    color="black",
                    linestyle="--",
                    linewidth=1.3,
                    label="matched no-pulse control",
                )
            ax.axvline(0, color="black", linestyle=":", linewidth=1)
            ax.axvline(duration, color="tab:red", linestyle=":", linewidth=1)
            ax.set_title(f"{_format_direction(direction)} — pulse {duration} round(s)")
            if row_index == 0 and column_index == 0:
                ax.legend(fontsize=6, ncol=2)

    pulse_metrics = metrics[
        (metrics["regime"] == "pulse")
        & (metrics["committee_policy"] == "alternative_pulse")
    ].copy()
    pulse_metrics["committee_fraction"] = pulse_metrics["committee_size"] / pulse_metrics["N"]
    pulse_metrics["direction_label"] = pulse_metrics["attack_direction"].fillna("neutral")
    outcome_specs = (
        ("recovery_probability", "recovery probability", "recovery_ci_low", "recovery_ci_high"),
        ("permanent_flip_probability", "permanent-flip probability", "permanent_flip_ci_low", "permanent_flip_ci_high"),
        (
            "median_recovery_time_population_rounds",
            "median recovery time (population rounds)",
            "median_recovery_time_population_rounds_ci_low",
            "median_recovery_time_population_rounds_ci_high",
        ),
    )
    markers = ("o", "s", "^", "D", "v", "P")
    for column_index, direction in enumerate(directions):
        direction_metrics = pulse_metrics[pulse_metrics["direction_label"] == direction]
        for row_index, (column, ylabel, low_column, high_column) in enumerate(outcome_specs):
            ax = bottom[row_index, column_index]
            for duration_index, duration in enumerate(durations):
                line = direction_metrics[direction_metrics["pulse_rounds"] == duration].sort_values(
                    "committee_fraction"
                )
                if line.empty:
                    continue
                values = line[column].astype(float)
                ax.plot(
                    line["committee_fraction"],
                    values,
                    marker=markers[duration_index % len(markers)],
                    label=f"{duration} round(s)",
                )
                finite = values.notna() & line[low_column].notna() & line[high_column].notna()
                if finite.any():
                    ax.errorbar(
                        line.loc[finite, "committee_fraction"],
                        values.loc[finite],
                        yerr=np.vstack(
                            (
                                values.loc[finite] - line.loc[finite, low_column],
                                line.loc[finite, high_column] - values.loc[finite],
                            )
                        ),
                        fmt="none",
                        capsize=2,
                        alpha=0.65,
                    )
            ax.set(xlabel="committee fraction", ylabel=ylabel)
            if "probability" in ylabel:
                ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.2)
            if row_index == 0:
                ax.set_title(_format_direction(direction))
            if row_index == 0 and column_index == 0:
                ax.legend(title="pulse duration", fontsize=7)
    figure.suptitle("Pulse intervention summary", fontsize=16)
    _atomic_save(figure, output_path)
    plt.close(figure)
    return True


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    available = [column for column in columns if column in frame]
    if frame.empty or not available:
        return ["No estimable rows were found."]
    rows = ["| " + " | ".join(column.replace("_", " ") for column in available) + " |"]
    rows.append("| " + " | ".join("---" for _ in available) + " |")
    for record in frame[available].head(200).to_dict("records"):
        rows.append("| " + " | ".join(_format_value(record[column]) for column in available) + " |")
    if len(frame) > 200:
        rows.append(f"\n_Table truncated to 200 of {len(frame)} rows._")
    return rows


def write_summary_markdown(
    episodes: pd.DataFrame,
    metrics: pd.DataFrame,
    estimates: pd.DataFrame,
    warnings: list[str],
    output_path: Path,
) -> None:
    sizes = sorted(int(value) for value in episodes["committee_size"].unique())
    populations = sorted(int(value) for value in episodes["N"].unique())
    policies = {
        str(regime): sorted(group["committee_policy"].astype(str).unique())
        for regime, group in episodes.groupby("regime", sort=True)
    }
    policy_text = "; ".join(
        f"{regime}: {', '.join(values)}" for regime, values in policies.items()
    )
    directions = sorted(episodes["attack_direction"].dropna().astype(str).unique())
    sources = sorted(episodes["convention_role_source"].dropna().astype(str).unique())
    lines = [
        "# Committee-empowerment experiment summary",
        "",
        "## Run inventory",
        "",
        f"- Completed episodes: {episodes['episode_id'].nunique()}",
        f"- Population sizes: {', '.join(map(str, populations))}",
        f"- Committee sizes: {', '.join(map(str, sizes))}",
        f"- Committee fractions: {', '.join(f'{size / populations[0]:.3g}' for size in sizes) if len(populations) == 1 else 'vary by population size'}",
        f"- Policies found: {policy_text}",
        f"- Attack directions found: {', '.join(directions) if directions else 'none'}",
        f"- Convention-role source: {', '.join(sources) if sources else 'not supplied'}",
        "",
        "## Takeover and final outcomes",
        "",
    ]
    outcomes = metrics[
        metrics["committee_policy"].isin(("promote_alternative", "alternative_pulse"))
    ].copy()
    outcomes["committee_fraction"] = outcomes["committee_size"] / outcomes["N"]
    lines.extend(
        _markdown_table(
            outcomes.sort_values(["regime", "attack_direction", "pulse_rounds", "committee_fraction"]),
            [
                "regime",
                "attack_direction",
                "pulse_rounds",
                "committee_size",
                "committee_fraction",
                "episodes",
                "terminal_takeover_probability",
                "terminal_takeover_ci_low",
                "terminal_takeover_ci_high",
                "ever_crossed_probability",
                "incumbent_survival_probability",
                "unresolved_probability",
            ],
        )
    )
    lines.extend(["", "### Smallest tested fraction reaching 0.95 terminal takeover", ""])
    threshold_rows: list[str] = []
    for keys, group in outcomes.groupby(["regime", "attack_direction", "pulse_rounds"], dropna=False):
        reached = group[group["terminal_takeover_probability"] >= 0.95].sort_values("committee_fraction")
        label = " / ".join(_format_value(value) for value in keys)
        threshold_rows.append(
            f"- {label}: {float(reached.iloc[0]['committee_fraction']):.3g} (smallest tested; not an interpolated critical mass)"
            if not reached.empty
            else f"- {label}: not reached on the sampled grid"
        )
    lines.extend(threshold_rows or ["- No attack cells were found."])

    lines.extend(["", "## Terminal empowerment and shuffle null", ""])
    terminal = estimates[estimates["statistic"] == "terminal_all"].copy()
    terminal["committee_fraction"] = terminal["committee_size"] / terminal["N"]
    lines.extend(
        _markdown_table(
            terminal.sort_values(["regime", "attack_direction", "pulse_rounds", "committee_fraction"]),
            [
                "regime",
                "attack_direction",
                "pulse_rounds",
                "committee_size",
                "committee_fraction",
                "estimate_status",
                "jeffreys",
                "ci_low",
                "ci_high",
                "shuffle_null_median",
                "shuffle_null_ci_low",
                "shuffle_null_ci_high",
                "empowerment_above_shuffle_median",
                "above_shuffle_97_5pct",
                "status_reason",
            ],
        )
    )
    if (terminal["estimate_status"] == "non_estimable").any():
        lines.extend(["", f"**{NON_ESTIMABLE_MESSAGE}.**"])

    lines.extend(["", "## Consensus and recovery", ""])
    recovery = outcomes.copy()
    lines.extend(
        _markdown_table(
            recovery.sort_values(["regime", "attack_direction", "pulse_rounds", "committee_fraction"]),
            [
                "regime",
                "attack_direction",
                "pulse_rounds",
                "committee_fraction",
                "consensus_probability",
                "mean_consensus_time_population_rounds",
                "recovery_probability",
                "permanent_flip_probability",
                "median_recovery_time_population_rounds",
                "median_recovery_time_population_rounds_ci_low",
                "median_recovery_time_population_rounds_ci_high",
            ],
        )
    )
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in warnings] or ["- None."])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    try:
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "NON_ESTIMABLE_MESSAGE",
    "attach_null_summary",
    "collect_warnings",
    "make_experiment_summary",
    "make_pulse_summary",
    "normalize_histories",
    "round_end_trajectories",
    "write_summary_markdown",
]
