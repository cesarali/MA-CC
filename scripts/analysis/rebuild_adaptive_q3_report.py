#!/usr/bin/env python3
"""Rebuild the adaptive-communication q=3 analysis from retained Parquet tables.

This is a derived, read-only post-processing pipeline. It never changes source
study outputs and never launches episodes. State-local estimates come directly
from primary_estimates.parquet and derived_observables.parquet; the known-bad
state_local_phase_maps.parquet status column is used only for diagnosis.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import textwrap
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import yaml

RHO = [0.70, 0.775, 0.85, 0.925, 1.00]
BUDGETS = [3, 6, 9, 12, 15, 18, 21]
SEMANTICS = ["truth", "false"]
STATE_METRICS = {
    "T_pi": ("round_target_actuation_cmi", r"$T_\pi$", "bits"),
    "chi": ("round_target_susceptibility", r"$\chi$", "target-share change"),
    "eta_IF": ("round_target_information_fraction", r"$\eta_{\rm IF}$", "fraction"),
    "eta_IR": ("eta_ir_state_local", r"$\eta_{\rm IR}$", "fraction"),
}
WHOLE_METRICS = {
    "T_pi": ("primary", "round_target_actuation_cmi", r"$T_\pi$", "bits"),
    "T_pi_excess": (
        "primary",
        "round_target_actuation_cmi",
        r"$T_\pi-T_{\rm null}$",
        "bits",
    ),
    "chi": (
        "derived",
        "susceptibility_occupancy_weighted",
        r"$\chi$",
        "target-share change",
    ),
    "eta_IF": (
        "primary",
        "round_target_information_fraction",
        r"$\eta_{\rm IF}$",
        "fraction",
    ),
    "eta_IR": ("derived", "eta_ir", r"$\eta_{\rm IR}$", "fraction"),
}
SEED = 20260907


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--prior-zip", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args()


def extract_archive(path: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(destination)


def read_table(root: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(root / "tables" / f"{name}.parquet")


def normalize_semantics(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"false", "wrong", "deceptive"}:
        return "false"
    if text in {"truth", "true", "correct"}:
        return "truth"
    return "none"


def attach_coordinates(frame: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "target_semantics" in result:
        result["target_semantics"] = result["target_semantics"].map(normalize_semantics)
    needed = [
        "cell_id",
        "target_semantics",
        "epistemic_persistence",
        "intervention_budget",
    ]
    coords = cells[needed].copy()
    coords["target_semantics"] = coords["target_semantics"].map(normalize_semantics)
    for column in needed[1:]:
        if column in result:
            result = result.drop(columns=column)
    result = result.merge(coords, on="cell_id", how="left")
    result["epistemic_persistence"] = pd.to_numeric(
        result["epistemic_persistence"], errors="coerce"
    ).round(3)
    result["intervention_budget"] = (
        pd.to_numeric(result["intervention_budget"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    return result


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    valid = values.notna() & weights.notna() & (weights > 0)
    return (
        float(np.average(values[valid], weights=weights[valid]))
        if valid.any()
        else math.nan
    )


def support_rollup(values: Iterable[object]) -> str:
    statuses = {str(value) for value in values}
    if "adequate" in statuses and "limited" not in statuses:
        return "adequate"
    if statuses & {"adequate", "limited"}:
        return "limited"
    return "unsupported"


def build_state_local(primary: pd.DataFrame, derived: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, (source, _, units) in STATE_METRICS.items():
        frame = derived if source == "eta_ir_state_local" else primary
        selected = frame[
            (frame["metric"] == source) & frame["target_fraction_bin_index"].notna()
        ].copy()
        for column in [
            "estimate",
            "ci_low",
            "ci_high",
            "null_mean",
            "p_value",
            "n_observations",
        ]:
            if column not in selected:
                selected[column] = np.nan
        selected["metric"] = label
        selected["source_metric"] = source
        selected["null_estimate"] = selected["null_mean"]
        selected["units"] = units
        selected["x_bin"] = selected["target_fraction_bin_index"].astype(int)
        selected["rho"] = selected["epistemic_persistence"].round(3)
        selected["b"] = selected["intervention_budget"].astype(int)
        rows.append(selected)
    state = pd.concat(rows, ignore_index=True, sort=False)
    state["target_semantics"] = state["target_semantics"].map(normalize_semantics)
    keep = [
        "cell_id",
        "target_semantics",
        "rho",
        "b",
        "x_bin",
        "target_fraction_bin_lower",
        "target_fraction_bin_upper",
        "target_fraction_bin_center",
        "metric",
        "source_metric",
        "estimate",
        "ci_low",
        "ci_high",
        "null_estimate",
        "p_value",
        "n_observations",
        "n_episodes",
        "support_status",
        "units",
    ]
    state = state[keep].sort_values(["target_semantics", "rho", "b", "x_bin", "metric"])
    return state


def aggregate_state(
    state: pd.DataFrame, group_columns: list[str], scope: str
) -> pd.DataFrame:
    rows = []
    bin_cols = [
        "x_bin",
        "target_fraction_bin_lower",
        "target_fraction_bin_upper",
        "target_fraction_bin_center",
    ]
    for key, group in state.groupby(
        group_columns + bin_cols + ["metric"], dropna=False, sort=True
    ):
        valid = group[
            group["support_status"].isin(["adequate", "limited"])
            & group["estimate"].notna()
        ]
        record = dict(zip(group_columns + bin_cols + ["metric"], key, strict=True))
        record.update(
            {
                "estimate": weighted_mean(valid["estimate"], valid["n_observations"]),
                "ci_low": math.nan,
                "ci_high": math.nan,
                "null_estimate": weighted_mean(
                    valid["null_estimate"], valid["n_observations"]
                ),
                "p_value": math.nan,
                "n_observations": int(
                    pd.to_numeric(valid["n_observations"], errors="coerce").sum()
                ),
                "support_status": support_rollup(valid["support_status"])
                if not valid.empty
                else "unsupported",
                "aggregation_scope": scope,
                "aggregation_weight": "n_observations",
                "descriptive_only": True,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def bootstrap_mean(
    values: np.ndarray, rng: np.random.Generator, n: int
) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan, math.nan
    draws = np.mean(rng.choice(values, size=(n, len(values)), replace=True), axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def episode_endpoints(rounds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cell_id, episode_id), group in rounds.groupby(
        ["cell_id", "episode_id"], sort=False
    ):
        group = group.sort_values("round_index")
        first, last = group.iloc[0], group.iloc[-1]
        options = (
            json.loads(first["possible_answers"])
            if isinstance(first["possible_answers"], str)
            else list(first["possible_answers"])
        )
        before = (
            json.loads(first["occupation_counts_before"])
            if isinstance(first["occupation_counts_before"], str)
            else list(first["occupation_counts_before"])
        )
        after = (
            json.loads(last["occupation_counts_after"])
            if isinstance(last["occupation_counts_after"], str)
            else list(last["occupation_counts_after"])
        )
        initial = dict(zip(options, map(int, before), strict=True))
        final = dict(zip(options, map(int, after), strict=True))
        truth = str(first["correct_answer"])
        target = str(
            first.get("controller_target")
            if pd.notna(first.get("controller_target"))
            else first["analysis_target"]
        )
        population = sum(initial.values())
        max_count = max(final.values())
        winners = [name for name, count in final.items() if count == max_count]
        winner = winners[0] if len(winners) == 1 else None
        rows.append(
            {
                "cell_id": cell_id,
                "episode_id": episode_id,
                "target_semantics": first["target_semantics"],
                "rho": first["rho"],
                "b": int(first["b"]),
                "initial_p_truth": initial[truth] / population,
                "final_p_truth": final[truth] / population,
                "initial_p_target": initial[target] / population,
                "final_p_target": final[target] / population,
                "truth_plurality": winner == truth,
                "target_plurality": winner == target,
                "false_target_plurality": bool(
                    first["target_semantics"] == "false" and winner == target
                ),
                "final_tie": winner is None,
            }
        )
    return pd.DataFrame(rows)


def outcome_summary(endpoints: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    keys = ["target_semantics", "rho", "b"]
    metrics = [
        "initial_p_truth",
        "final_p_truth",
        "initial_p_target",
        "final_p_target",
        "truth_plurality",
        "target_plurality",
        "false_target_plurality",
        "final_tie",
    ]
    for key, group in endpoints.groupby(keys, sort=True):
        record = dict(zip(keys, key, strict=True))
        record["n_completed_episodes"] = len(group)
        for metric in metrics:
            values = group[metric].astype(float).to_numpy()
            low, high = bootstrap_mean(values, rng, n_boot)
            record[metric] = float(np.mean(values))
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        rows.append(record)
    result = pd.DataFrame(rows)
    baseline = result[result.target_semantics == "none"][
        ["rho", "final_p_truth", "truth_plurality"]
    ].rename(
        columns={
            "final_p_truth": "baseline_final_p_truth",
            "truth_plurality": "baseline_truth_plurality",
        }
    )
    result = result.merge(baseline, on="rho", how="left")
    result["effect_final_truth_vs_no_control"] = (
        result["final_p_truth"] - result["baseline_final_p_truth"]
    )
    result["effect_truth_plurality_vs_no_control"] = (
        result["truth_plurality"] - result["baseline_truth_plurality"]
    )
    return result


def whole_metrics(
    primary: pd.DataFrame,
    derived: pd.DataFrame,
    outcomes: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for label, (source_name, source_metric, _, units) in WHOLE_METRICS.items():
        frame = primary if source_name == "primary" else derived
        selected = frame[
            (frame.metric == source_metric)
            & frame.target_fraction_bin_index.isna()
            & frame.target_semantics.map(normalize_semantics).isin(SEMANTICS)
            & frame.intervention_budget.notna()
        ].copy()
        for row in selected.to_dict("records"):
            estimate = float(row.get("estimate", math.nan))
            null = (
                float(row.get("null_mean", math.nan))
                if pd.notna(row.get("null_mean"))
                else math.nan
            )
            if label == "T_pi_excess":
                estimate = estimate - null
            rows.append(
                {
                    "cell_id": row["cell_id"],
                    "target_semantics": normalize_semantics(row["target_semantics"]),
                    "rho": round(float(row["epistemic_persistence"]), 3),
                    "b": int(row["intervention_budget"]),
                    "metric": label,
                    "estimate": estimate,
                    "ci_low": row.get("ci_low"),
                    "ci_high": row.get("ci_high"),
                    "null_estimate": null,
                    "p_value": row.get("p_value"),
                    "n_observations": row.get("n_observations"),
                    "n_episodes": row.get("n_episodes"),
                    "support_status": row.get("support_status"),
                    "units": units,
                }
            )
    outcome_metrics = [
        "final_p_truth",
        "final_p_target",
        "truth_plurality",
        "false_target_plurality",
    ]
    for row in outcomes.to_dict("records"):
        for metric in outcome_metrics:
            rows.append(
                {
                    "cell_id": None,
                    "target_semantics": row["target_semantics"],
                    "rho": row["rho"],
                    "b": row["b"],
                    "metric": metric,
                    "estimate": row[metric],
                    "ci_low": row.get(metric + "_ci_low"),
                    "ci_high": row.get(metric + "_ci_high"),
                    "null_estimate": math.nan,
                    "p_value": math.nan,
                    "n_observations": row["n_completed_episodes"],
                    "n_episodes": row["n_completed_episodes"],
                    "support_status": "adequate",
                    "units": "fraction",
                }
            )
    diag_specs = {
        "P_U1": lambda d: (
            d.act_report_rounds + d.act_request_rounds + d.act_directive_rounds
        )
        / d.rounds,
        "controller_posts_per_active_round": lambda d: d.controller_posts
        / (d.act_report_rounds + d.act_request_rounds + d.act_directive_rounds).replace(
            0, np.nan
        ),
        "controller_exposure": lambda d: d.directives_read / d.rounds,
        "active_evidence_coverage": lambda d: d.active_latent_coverage_mean,
        "historical_evidence_coverage": lambda d: d.historical_latent_coverage_mean,
        "peer_exposure": lambda d: (
            d.peer_report_exposures_with_controller_actuation
            + d.peer_report_exposures_without_controller_actuation
        )
        / d.rounds,
        "REQUEST_rate": lambda d: d.request_count / d.rounds,
        "REPORT_rate": lambda d: d.report_count / d.rounds,
        "DIRECTIVE_rate": lambda d: d.directives_posted / d.rounds,
    }
    for metric, function in diag_specs.items():
        values = function(diagnostics)
        for (_, row), value in zip(diagnostics.iterrows(), values, strict=True):
            rows.append(
                {
                    "cell_id": row.cell_id,
                    "target_semantics": row.target_semantics,
                    "rho": row.epistemic_persistence,
                    "b": row.intervention_budget,
                    "metric": metric,
                    "estimate": value,
                    "ci_low": math.nan,
                    "ci_high": math.nan,
                    "null_estimate": math.nan,
                    "p_value": math.nan,
                    "n_observations": row.rounds,
                    "n_episodes": row.episodes,
                    "support_status": "adequate",
                    "units": "per round",
                }
            )
    return pd.DataFrame(rows)


def communication_summary(rounds: pd.DataFrame) -> pd.DataFrame:
    active = rounds[
        (rounds.U_k == 1)
        & rounds.chosen_message_mode.isin(["REQUEST", "REPORT", "DIRECTIVE"])
    ].copy()
    active["x_bin"] = np.minimum(
        (active.controller_target_share_before * 8).astype(int), 7
    )
    active["delta_target_share"] = (
        active.controller_target_share - active.controller_target_share_before
    )
    rows = []
    groupings = [
        (["target_semantics", "rho", "b", "chosen_message_mode"], "rho_b"),
        (["target_semantics", "chosen_message_mode"], "arm"),
        (["target_semantics", "rho", "b", "x_bin", "chosen_message_mode"], "state"),
    ]
    for keys, resolution in groupings:
        denominators = active.groupby(keys[:-1], dropna=False).size()
        for key, group in active.groupby(keys, dropna=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            record = dict(zip(keys, key_tuple, strict=True))
            denom_key = key_tuple[:-1]
            if len(denom_key) == 1:
                denom_key = denom_key[0]
            record.update(
                {
                    "resolution": resolution,
                    "active_rounds": int(denominators.loc[denom_key]),
                    "mode_rounds": len(group),
                    "mode_probability_given_U1": len(group)
                    / int(denominators.loc[denom_key]),
                    "mean_delta_target_share": group.delta_target_share.mean(),
                    "delta_ci_low": group.delta_target_share.quantile(0.025),
                    "delta_ci_high": group.delta_target_share.quantile(0.975),
                    "mean_posts": group.actual_controller_posts.mean(),
                    "mean_exposures": group.controller_fact_exposures.mean(),
                    "mean_unique_readers": group.controller_unique_readers.mean(),
                }
            )
            rows.append(record)
    return pd.DataFrame(rows).rename(columns={"chosen_message_mode": "mode"})


def budget_summary(rounds: pd.DataFrame) -> pd.DataFrame:
    active = rounds[rounds.U_k == 1].copy()
    active["x_bin"] = np.minimum(
        (active.controller_target_share_before * 8).astype(int), 7
    )
    rows = []
    for keys, resolution in [
        (["target_semantics", "rho", "b"], "rho_b"),
        (["target_semantics", "b"], "rho_aggregated"),
    ]:
        for key, group in active.groupby(keys, dropna=False):
            record = dict(
                zip(keys, key if isinstance(key, tuple) else (key,), strict=True)
            )
            posts = group.actual_controller_posts.astype(float)
            exposures = group.controller_fact_exposures.astype(float)
            adoptions = group.controlled_adoption_count.astype(float)
            record.update(
                {
                    "resolution": resolution,
                    "active_rounds": len(group),
                    "nominal_b": float(group.b.mean()),
                    "controller_posts_per_active_round": posts.mean(),
                    "controller_posts_ci_low": posts.quantile(0.025),
                    "controller_posts_ci_high": posts.quantile(0.975),
                    "controller_exposures_per_active_round": exposures.mean(),
                    "unique_readers_per_active_round": group.controller_unique_readers.mean(),
                    "target_adoptions_per_active_round": adoptions.mean(),
                    "target_adoption_per_exposure": adoptions.sum() / exposures.sum()
                    if exposures.sum()
                    else math.nan,
                    "exposure_per_controller_post": exposures.sum() / posts.sum()
                    if posts.sum()
                    else math.nan,
                }
            )
            rows.append(record)
    return pd.DataFrame(rows)


def cooperation_summary(rounds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in rounds.groupby(["target_semantics", "rho", "b"], dropna=False):
        requests = group.request_count.sum()
        rows.append(
            {
                "target_semantics": key[0],
                "rho": key[1],
                "b": key[2],
                "rounds": len(group),
                "participant_requests_per_round": requests / len(group),
                "participant_reports_per_round": group.report_count.sum() / len(group),
                "no_post_rate": 1
                - (group.report_count.sum() + requests) / (24 * len(group)),
                "reply_rate_to_request": group.controller_direct_replies.sum()
                / requests
                if requests
                else math.nan,
                "evidence_acquisition_per_request": group.new_evidence_acquisitions.sum()
                / requests
                if requests
                else math.nan,
                "peer_exposure_per_round": group.peer_fact_exposures.sum() / len(group),
                "controller_exposure_per_round": group.controller_fact_exposures.sum()
                / len(group),
                "active_evidence_coverage": group.active_mean_supporting_fact_coverage_after.mean(),
                "historical_evidence_coverage": group.historical_mean_supporting_fact_coverage_after.mean(),
            }
        )
    return pd.DataFrame(rows)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 190,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.grid": False,
        }
    )


def heat(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    title: str,
    diverging: bool = False,
    annotate: bool = False,
):
    xs = sorted(frame[x].dropna().unique())
    ys = sorted(frame[y].dropna().unique())
    grid = frame.pivot_table(index=y, columns=x, values=value, aggfunc="mean").reindex(
        index=ys, columns=xs
    )
    data = np.ma.masked_invalid(grid.to_numpy(float))
    if diverging and np.isfinite(data).any():
        maximum = max(abs(float(data.min())), abs(float(data.max())), 1e-12)
        image = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(0, -maximum, maximum),
        )
    else:
        image = ax.imshow(data, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(xs)), [f"{v:g}" for v in xs], rotation=0)
    ax.set_yticks(range(len(ys)), [f"{v:.3g}" for v in ys])
    ax.set_title(title)
    for row_index, row in enumerate(ys):
        for column_index, column in enumerate(xs):
            if np.ma.is_masked(data[row_index, column_index]):
                ax.add_patch(
                    Rectangle(
                        (column_index - 0.5, row_index - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="///",
                        edgecolor="0.55",
                        linewidth=0,
                    )
                )
            elif annotate:
                ax.text(
                    column_index,
                    row_index,
                    f"{data[row_index, column_index]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white"
                    if abs(float(data[row_index, column_index]))
                    > 0.55 * max(abs(float(data.min())), abs(float(data.max())), 1e-12)
                    else "black",
                )
    return image


def save_state_figures(
    state: pd.DataFrame,
    agg: pd.DataFrame,
    pooled_rho: pd.DataFrame,
    pooled_all: pd.DataFrame,
    figures: Path,
) -> list[Path]:
    paths = []
    for metric, (_, label, units) in STATE_METRICS.items():
        selected = state[state.metric == metric]
        finite = selected[selected.support_status.isin(["adequate", "limited"])].copy()
        finite.loc[~finite.support_status.isin(["adequate", "limited"]), "estimate"] = (
            np.nan
        )
        fig, axes = plt.subplots(
            2, 5, figsize=(15.2, 6.2), sharex=True, sharey=True, constrained_layout=True
        )
        image = None
        for i, sem in enumerate(SEMANTICS):
            for j, rho in enumerate(RHO):
                panel = finite[
                    (finite.target_semantics == sem) & np.isclose(finite.rho, rho)
                ].copy()
                panel.loc[
                    ~panel.support_status.isin(["adequate", "limited"]), "estimate"
                ] = np.nan
                image = heat(
                    axes[i, j],
                    panel,
                    "b",
                    "target_fraction_bin_center",
                    "estimate",
                    f"{sem}; $\\rho={rho:g}$",
                    metric in {"chi"},
                )
                if i == 1:
                    axes[i, j].set_xlabel("nominal budget $b$")
                if j == 0:
                    axes[i, j].set_ylabel("target fraction $x$")
        fig.colorbar(image, ax=axes, shrink=0.78, label=f"{label} [{units}]")
        fig.suptitle(
            f"Persistence-resolved state maps: {label}\nHatched cells are unsupported; estimates come from primary estimator records",
            fontsize=13,
        )
        path = figures / f"state_{metric}_rho_resolved.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

        fig, axes = plt.subplots(
            1, 3, figsize=(12.5, 3.9), sharey=True, constrained_layout=True
        )
        panels = [
            (agg[agg.metric == metric], "truth", "truth; $\\rho$ aggregated"),
            (agg[agg.metric == metric], "false", "false; $\\rho$ aggregated"),
            (
                pooled_all[pooled_all.metric == metric],
                "control pooled",
                "truth + false; $\\rho$ aggregated",
            ),
        ]
        for ax, (frame, sem, title) in zip(axes, panels, strict=True):
            panel = (
                frame
                if sem == "control pooled"
                else frame[frame.target_semantics == sem]
            )
            image = heat(
                ax,
                panel,
                "b",
                "target_fraction_bin_center",
                "estimate",
                title,
                metric == "chi",
            )
            ax.set_xlabel("nominal budget $b$")
        axes[0].set_ylabel("target fraction $x$")
        fig.colorbar(image, ax=axes, shrink=0.82, label=f"{label} [{units}]")
        fig.suptitle(
            f"Observation-weighted descriptive state maps: {label}", fontsize=13
        )
        path = figures / f"state_{metric}_aggregated.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

        fig, axes = plt.subplots(
            1, 5, figsize=(15.2, 3.4), sharex=True, sharey=True, constrained_layout=True
        )
        for ax, rho in zip(axes, RHO, strict=True):
            panel = pooled_rho[
                (pooled_rho.metric == metric) & np.isclose(pooled_rho.rho, rho)
            ]
            image = heat(
                ax,
                panel,
                "b",
                "target_fraction_bin_center",
                "estimate",
                f"$\\rho={rho:g}$",
                metric == "chi",
            )
            ax.set_xlabel("$b$")
        axes[0].set_ylabel("target fraction $x$")
        fig.colorbar(image, ax=axes, shrink=0.8, label=f"{label} [{units}]")
        fig.suptitle(
            f"Truth + false control pooled over target semantics: {label}", fontsize=13
        )
        path = figures / f"state_{metric}_truth_false_pooled_by_rho.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_occupancy(state: pd.DataFrame, figures: Path) -> Path:
    occupancy = state[state.metric == "T_pi"]
    fig, axes = plt.subplots(
        2, 5, figsize=(15.2, 6.2), sharex=True, sharey=True, constrained_layout=True
    )
    for i, sem in enumerate(SEMANTICS):
        for j, rho in enumerate(RHO):
            panel = occupancy[
                (occupancy.target_semantics == sem) & np.isclose(occupancy.rho, rho)
            ]
            image = heat(
                axes[i, j],
                panel,
                "b",
                "target_fraction_bin_center",
                "n_observations",
                f"{sem}; $\\rho={rho:g}$",
            )
            if i == 1:
                axes[i, j].set_xlabel("nominal budget $b$")
            if j == 0:
                axes[i, j].set_ylabel("target fraction $x$")
    fig.colorbar(image, ax=axes, shrink=0.78, label="round observations")
    fig.suptitle(
        "State occupancy supporting the reconstructed estimator maps", fontsize=13
    )
    path = figures / "state_occupancy_rho_resolved.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_whole_phase(metrics: pd.DataFrame, figures: Path) -> list[Path]:
    paths = []
    groups = [
        [
            "final_p_truth",
            "final_p_target",
            "truth_plurality",
            "false_target_plurality",
        ],
        ["T_pi", "T_pi_excess", "chi", "eta_IF", "eta_IR"],
        [
            "P_U1",
            "controller_posts_per_active_round",
            "controller_exposure",
            "active_evidence_coverage",
            "historical_evidence_coverage",
        ],
    ]
    for index, names in enumerate(groups, 1):
        fig, axes = plt.subplots(
            len(SEMANTICS),
            len(names),
            figsize=(3.1 * len(names), 5.8),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        if len(names) == 1:
            axes = np.array(axes).reshape(2, 1)
        for i, sem in enumerate(SEMANTICS):
            for j, name in enumerate(names):
                panel = metrics[
                    (metrics.target_semantics == sem) & (metrics.metric == name)
                ]
                image = heat(
                    axes[i, j],
                    panel,
                    "b",
                    "rho",
                    "estimate",
                    f"{sem}: {name}",
                    name in {"T_pi_excess", "chi"},
                )
                if i == 1:
                    axes[i, j].set_xlabel("nominal budget $b$")
                if j == 0:
                    axes[i, j].set_ylabel("persistence $\\rho$")
                fig.colorbar(image, ax=axes[i, j], shrink=0.70)
        fig.suptitle("Whole-cell $\\rho\\times b$ phase diagrams", fontsize=13)
        path = figures / f"whole_cell_phase_group_{index}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_lines(metrics: pd.DataFrame, figures: Path) -> list[Path]:
    requested = [
        "final_p_truth",
        "final_p_target",
        "T_pi",
        "T_pi_excess",
        "chi",
        "eta_IF",
        "eta_IR",
        "P_U1",
        "controller_posts_per_active_round",
        "controller_exposure",
        "active_evidence_coverage",
        "historical_evidence_coverage",
        "peer_exposure",
        "REQUEST_rate",
        "REPORT_rate",
        "DIRECTIVE_rate",
    ]
    paths = []
    for chunk_index in range(0, len(requested), 4):
        chunk = requested[chunk_index : chunk_index + 4]
        fig, axes = plt.subplots(
            2,
            len(chunk),
            figsize=(3.5 * len(chunk), 6.0),
            sharex=True,
            constrained_layout=True,
        )
        if len(chunk) == 1:
            axes = np.array(axes).reshape(2, 1)
        for i, sem in enumerate(SEMANTICS):
            for j, name in enumerate(chunk):
                ax = axes[i, j]
                panel = metrics[
                    (metrics.target_semantics == sem) & (metrics.metric == name)
                ]
                for rho, g in panel.groupby("rho"):
                    g = g.sort_values("b")
                    ax.plot(
                        g.b,
                        g.estimate,
                        marker="o",
                        label=f"$\\rho={rho:g}$",
                        linewidth=1.2,
                    )
                    if g.ci_low.notna().any():
                        ax.fill_between(g.b, g.ci_low, g.ci_high, alpha=0.12)
                aggregate = panel.groupby("b", as_index=False).apply(
                    lambda g: pd.Series(
                        {"estimate": weighted_mean(g.estimate, g.n_observations)}
                    ),
                    include_groups=False,
                )
                ax.plot(
                    aggregate.b,
                    aggregate.estimate,
                    color="black",
                    marker="s",
                    linewidth=2,
                    label="all $\\rho$",
                )
                ax.set_title(f"{sem}: {name}")
                ax.set_xlabel("nominal budget $b$")
                ax.grid(alpha=0.2)
                if i == 0 and j == 0:
                    ax.legend(ncol=2, fontsize=6)
        path = figures / f"lines_vs_b_group_{chunk_index // 4 + 1}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_communication(
    comm: pd.DataFrame, budget: pd.DataFrame, cooperation: pd.DataFrame, figures: Path
) -> list[Path]:
    paths = []
    arm = comm[comm.resolution == "arm"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for sem, color in zip(SEMANTICS, ["#2673b8", "#c4423b"], strict=True):
        g = (
            arm[arm.target_semantics == sem]
            .set_index("mode")
            .reindex(["REQUEST", "REPORT", "DIRECTIVE"])
        )
        axes[0].plot(
            g.index, g.mode_probability_given_U1, marker="o", label=sem, color=color
        )
        axes[1].plot(
            g.index, g.mean_delta_target_share, marker="o", label=sem, color=color
        )
    axes[0].set_ylabel(r"$P(\mathrm{mode}\mid U=1)$")
    axes[1].set_ylabel("mean immediate change in target share")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Adaptive communication mode selection and immediate response")
    path = figures / "communication_modes.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    rb = budget[budget.resolution == "rho_b"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for i, sem in enumerate(SEMANTICS):
        for rho, g in rb[rb.target_semantics == sem].groupby("rho"):
            g = g.sort_values("b")
            axes[i].plot(
                g.b,
                g.controller_posts_per_active_round,
                marker="o",
                label=f"$\\rho={rho:g}$",
            )
        axes[i].plot(BUDGETS, BUDGETS, "k--", alpha=0.5, label="nominal = realized")
        axes[i].set(
            title=sem, xlabel="nominal budget $b$", ylabel="posts per active round"
        )
        axes[i].grid(alpha=0.2)
        axes[i].legend(fontsize=6, ncol=2)
    fig.suptitle("Realized communication dose and saturation")
    path = figures / "budget_realization.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for i, metric in enumerate(
        ["participant_requests_per_round", "evidence_acquisition_per_request"]
    ):
        for sem in SEMANTICS:
            g = (
                cooperation[cooperation.target_semantics == sem]
                .groupby("b", as_index=False)[metric]
                .mean()
            )
            axes[i].plot(g.b, g[metric], marker="o", label=sem)
        axes[i].set(xlabel="nominal budget $b$", ylabel=metric.replace("_", " "))
        axes[i].grid(alpha=0.2)
        axes[i].legend()
    fig.suptitle("Participant information seeking and evidence acquisition")
    path = figures / "participant_cooperation.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def prior_comparison(adaptive_root: Path, prior_root: Path) -> pd.DataFrame:
    rows = []
    for study, root in [("adaptive", adaptive_root), ("report_only", prior_root)]:
        cells = read_table(root, "cells")
        rounds = attach_coordinates(read_table(root, "rounds"), cells)
        rounds["rho"] = rounds["epistemic_persistence"]
        rounds["b"] = rounds["intervention_budget"]
        primary = read_table(root, "primary_estimates")
        primary["target_semantics"] = primary.target_semantics.map(normalize_semantics)
        derived = read_table(root, "derived_observables")
        derived["target_semantics"] = derived.target_semantics.map(normalize_semantics)
        endpoints = outcome_summary(episode_endpoints(rounds), 500)
        for row in endpoints[endpoints.target_semantics.isin(SEMANTICS)].to_dict(
            "records"
        ):
            for metric in ["final_p_target", "false_target_plurality"]:
                rows.append(
                    {
                        "study": study,
                        "target_semantics": row["target_semantics"],
                        "rho": row["rho"],
                        "b": row["b"],
                        "metric": metric,
                        "estimate": row[metric],
                    }
                )
        for frame, source, label in [
            (primary, "round_target_actuation_cmi", "T_pi_excess"),
            (derived, "susceptibility_occupancy_weighted", "chi"),
        ]:
            selected = frame[
                (frame.metric == source)
                & frame.target_fraction_bin_index.isna()
                & frame.target_semantics.isin(SEMANTICS)
            ]
            for row in selected.to_dict("records"):
                value = row["estimate"] - (
                    row.get("null_mean")
                    if label == "T_pi_excess" and pd.notna(row.get("null_mean"))
                    else 0
                )
                rows.append(
                    {
                        "study": study,
                        "target_semantics": row["target_semantics"],
                        "rho": round(float(row["epistemic_persistence"]), 3),
                        "b": int(row["intervention_budget"]),
                        "metric": label,
                        "estimate": value,
                    }
                )
        for sem, g in rounds[rounds.target_semantics.isin(SEMANTICS)].groupby(
            ["target_semantics", "rho", "b"]
        ):
            active = g[g.U_k == 1]
            posts_column = (
                "actual_controller_posts"
                if "actual_controller_posts" in active
                else "controller_posts"
            )
            rows.extend(
                [
                    {
                        "study": study,
                        "target_semantics": sem[0],
                        "rho": sem[1],
                        "b": sem[2],
                        "metric": "P_U1",
                        "estimate": len(active) / len(g),
                    },
                    {
                        "study": study,
                        "target_semantics": sem[0],
                        "rho": sem[1],
                        "b": sem[2],
                        "metric": "posts_per_active_round",
                        "estimate": active[posts_column].mean(),
                    },
                ]
            )
    return pd.DataFrame(rows)


def plot_prior(comparison: pd.DataFrame, figures: Path) -> Path:
    metrics = [
        "final_p_target",
        "false_target_plurality",
        "chi",
        "T_pi_excess",
        "P_U1",
        "posts_per_active_round",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, metric in zip(axes.flat, metrics, strict=True):
        panel = comparison[comparison.metric == metric]
        for (study, sem), g in panel.groupby(["study", "target_semantics"]):
            g = g.groupby("b", as_index=False).estimate.mean()
            ax.plot(g.b, g.estimate, marker="o", label=f"{study}; {sem}")
        ax.set(title=metric, xlabel="nominal budget $b$")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6)
    fig.suptitle(
        "Descriptive comparison with truthful-report-only study\nDifferent prompt version and initialization archive: not a controlled ablation"
    )
    path = figures / "prior_report_only_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def diagnosis(source: Path, primary: pd.DataFrame) -> dict[str, object]:
    broken = read_table(source, "state_local_phase_maps")
    state = primary[primary.target_fraction_bin_index.notna()]
    return {
        "broken_export_rows": len(broken),
        "broken_status_counts": broken.phase_status.value_counts().to_dict(),
        "primary_state_rows": len(state),
        "primary_state_finite_estimates": int(state.estimate.notna().sum()),
        "cause": "export expected config-level cell labels while primary estimates use resolved hashed cell IDs; structural presence therefore failed even though estimator rows exist",
    }


def latex_escape(text: object) -> str:
    value = str(text)
    for old, new in [
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ]:
        value = value.replace(old, new)
    return value


def write_report(
    output: Path,
    figures: list[Path],
    validation: dict,
    findings: dict,
    comparison_available: bool,
) -> Path:
    figure_blocks = {}
    for path in figures:
        caption = path.stem.replace("_", " ").capitalize()
        figure_blocks[path.stem] = (
            f"\\begin{{figure}}[H]\n\\centering\n\\includegraphics[width=0.98\\textwidth]{{figures/{latex_escape(path.name)}}}\n\\caption{{{latex_escape(caption)}. Blank hatched cells are unsupported, not zero.}}\n\\end{{figure}}"
        )
    state_resolved = "\n".join(
        figure_blocks[f"state_{metric}_rho_resolved"] for metric in STATE_METRICS
    )
    state_aggregated = "\n".join(
        figure_blocks[f"state_{metric}_aggregated"] for metric in STATE_METRICS
    )
    state_pooled = "\n".join(
        figure_blocks[f"state_{metric}_truth_false_pooled_by_rho"]
        for metric in STATE_METRICS
    )
    whole_phase = "\n".join(
        figure_blocks[f"whole_cell_phase_group_{index}"] for index in range(1, 4)
    )
    line_plots = "\n".join(
        figure_blocks[f"lines_vs_b_group_{index}"] for index in range(1, 5)
    )
    comparison_text = (
        "A descriptive comparison is included. It is not a controlled ablation because the prior run used prompt version 3 and a different paired-initialization archive; the adaptive study used prompt version 4."
        if comparison_available
        else "The prior study archive was not supplied, so no cross-study comparison was made."
    )
    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.72in]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,graphicx,float,caption,microtype,xcolor,hyperref,longtable,array}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{0.45em}}
\title{{Adaptive Communication at $q=3$: Rebuilt Phase Diagrams and Full Analysis}}
\author{{Derived analysis of \texttt{{musr\_blackboard\_adaptive\_communication\_q3\_deepinfra}}}}
\date{{7 September 2026}}
\begin{{document}}\maketitle
\begin{{abstract}}
This report rebuilds state-resolved phase maps directly from retained Parquet estimator records. The bundled map export was broken: all {validation["broken_export_rows"]} rows were labelled structural-cell-not-run, although \texttt{{primary\_estimates.parquet}} contains {validation["primary_state_finite_estimates"]} finite state-local estimator values. The study is \textbf{{incomplete and provisional}}: 723 of 750 planned episodes completed, with 27 missing and no failed or aborted episodes. No game episode was launched and no production estimator was replaced.
\end{{abstract}}
\tableofcontents\newpage
\section{{Executive summary}}
\textbf{{Strongest behavioral result.}} {latex_escape(findings["behavior"])}

\textbf{{Strongest susceptibility result.}} {latex_escape(findings["chi"])}

\textbf{{Transfer information relative to null.}} {latex_escape(findings["tpi"])}

\textbf{{Adaptive versus report-only.}} {latex_escape(findings["comparison"])}

\textbf{{REQUEST and false control.}} {latex_escape(findings["request"])}

\textbf{{Budget saturation.}} {latex_escape(findings["saturation"])}

\textbf{{Persistence structure.}} {latex_escape(findings["rho"])}

\section{{Validation and provisional status}}
Expected episodes: 750. Completed episodes: 723. Missing episode keys: 27. Failed: 0. Aborted: 0. Retained rows: 7,230 population rounds and 173,520 microscopic update slots. The source validator marks the package invalid because it is incomplete. Every estimate in this report is therefore provisional.

The broken gray plots are an export failure, not absence of local estimates. The old exporter compared incompatible cell identifiers and marked every expected map row \texttt{{structural\_cell\_not\_run}}. This rebuild joins and plots the actual resolved estimator rows. It preserves estimator support: unsupported rows are blank and hatched.

\section{{Experimental parameters}}
The experiment uses MuSR Team Allocation task 001, $N=24$ agents, 10 rounds, social sample $q=3$, controller sensor size $q_c=12$, $\beta=4$, and threshold $\theta=0.5$. Persistence is $\rho\in\{{0.70,0.775,0.85,0.925,1.00\}}$ and nominal intervention budget is $b\in\{{3,6,9,12,15,18,21\}}$. The board retains messages for one round. Participant REQUEST is enabled. Adaptive controller REQUEST, REPORT, and DIRECTIVE modes are enabled under \texttt{{contextual\_weighted\_v1}} at dawn. The model is DeepSeek-V4-Flash through DeepInfra, with prompt \texttt{{relational\_blackboard\_ballot@4}}.

There are no-control, truth-control, and false-control arms. Truth control targets the correct answer. False control targets \texttt{{ALLOCATION\_1}}, while the correct answer is \texttt{{ALLOCATION\_0}}.

\section{{Methods and weighting}}
The state coordinate is target fraction $x$. The eight archived bins have width 0.125 and centers 0.0625 through 0.9375. Persistence-resolved maps use the archived cell-local point estimates. $\rho$-aggregated maps and truth-plus-false maps are descriptive observation-weighted means; they are not replacement pooled estimators. A cell contributes only when its estimator support is adequate or limited and its estimate is finite.

The principal quantities are $T_\pi=I(U_k;N_{{k+1}}\mid N_k)$, susceptibility $\chi$, information fraction $\eta_{{\rm IF}}$, and bounded information--response efficiency $\eta_{{\rm IR}}$. The binary control variable remains $U\in\{{0,1\}}$; communication mode is secondary. Whole-cell confidence intervals are the retained whole-episode bootstrap intervals. State-local rows were generated without bootstrap or permutation resampling, so this report does not invent local confidence intervals or local null values.

Raw $T_\pi$ is always interpreted beside its policy-conditional randomization null at whole-cell resolution. State-local $T_\pi-T_{{\rm null}}$ maps cannot be produced from this archive because local permutations were not retained or run. This is an explicit unsupported deliverable, not a zero-valued result.

\section{{Behavioral outcomes and whole-cell phase diagrams}}
{whole_phase}

\section{{$\rho$-resolved state maps}}
{state_resolved}

\section{{$\rho$-aggregated descriptive state maps}}
{state_aggregated}

\section{{Truth plus false control pooled state maps}}
{state_pooled}

\section{{Occupancy and support}}
{figure_blocks["state_occupancy_rho_resolved"]}

\section{{Aggregated line plots versus budget}}
{line_plots}

\section{{Adaptive communication modes}}
{figure_blocks["communication_modes"]}
The three plotted mode probabilities are conditional on $U=1$. Immediate response is the same-round change in controller-target vote share. It is descriptive and does not replace the binary-action transfer-information estimator.

\section{{Realized budget and saturation}}
{figure_blocks["budget_realization"]}
Nominal $b$ is a capacity coordinate. REPORT can use up to the budget, while REQUEST and DIRECTIVE often post one message. Realized posts, exposures, readers, and adoptions are exported in \texttt{{budget\_realization\_summary.csv}}.

\section{{Participant cooperation and information seeking}}
{figure_blocks["participant_cooperation"]}
Participant REQUEST and REPORT counts are ordinary-agent public-board actions. Reply and evidence-acquisition ratios are descriptive because a round can contain several requests and several acquisitions.

\section{{Comparison with truthful-report-only control}}
{comparison_text}
{figure_blocks.get("prior_report_only_comparison", "")}

\section{{Thermodynamic efficiency}}
$\eta_{{\rm th}}$ is unavailable. The archive reports insufficient support for effective-affinity $h$ calibration under this adaptive blackboard actuator. The report does not reuse the old direct-actuation calibration and does not fabricate a thermodynamic efficiency. Supported sensing information and controlled-current quantities remain in the source archive.

\section{{Interpretation and limitations}}
The analysis tests rather than assumes the claim that adaptive communication trades steering for information seeking. Conclusions are based on reconstructed maps and exported tables. Important limits are the missing 27 episodes, sparse high-$x$ state bins, absent state-local null distributions, observational communication-mode comparisons, and the non-controlled cross-study comparison.

\section{{Recommended next experiments}}
First fill the 27 missing episode keys and rerun strict aggregation. Then run state-local whole-episode bootstrap and policy-conditional randomization so $T_\pi-T_{{\rm null}}$ has local uncertainty. For mechanism, pre-register a within-prompt, paired-initialization ablation between adaptive communication and truthful-report-only control. A participant-REQUEST ON/OFF axis would directly test whether ordinary-agent information seeking protects against false control.

\appendix\section{{Exported tables}}
The report directory contains \texttt{{cell\_metrics.csv}}, \texttt{{rho\_b\_outcomes.csv}}, \texttt{{state\_local\_metrics.csv}}, \texttt{{state\_local\_rho\_aggregated.csv}}, \texttt{{state\_local\_truth\_false\_pooled.csv}}, \texttt{{communication\_mode\_summary.csv}}, \texttt{{budget\_realization\_summary.csv}}, \texttt{{uncertainty\_summary.csv}}, and additional cooperation/comparison tables.
\end{{document}}
"""
    path = output / "adaptive_communication_q3_report.tex"
    path.write_text(tex, encoding="utf-8")
    return path


def numeric_findings(
    outcomes: pd.DataFrame,
    metrics: pd.DataFrame,
    comm: pd.DataFrame,
    budget: pd.DataFrame,
    comparison: pd.DataFrame | None,
) -> dict[str, str]:
    controlled = outcomes[outcomes.target_semantics.isin(SEMANTICS)]
    best = controlled.loc[controlled.final_p_target.idxmax()]
    chi = metrics[(metrics.metric == "chi") & metrics.target_semantics.isin(SEMANTICS)]
    chi_row = chi.loc[chi.estimate.abs().idxmax()]
    t = metrics[
        (metrics.metric == "T_pi_excess") & metrics.target_semantics.isin(SEMANTICS)
    ]
    significant = metrics[(metrics.metric == "T_pi") & (metrics.p_value < 0.05)]
    arm = comm[comm.resolution == "arm"]
    request = arm[arm["mode"] == "REQUEST"].set_index("target_semantics")
    realized = budget[budget.resolution == "rho_aggregated"]
    ratios = realized.assign(
        ratio=realized.controller_posts_per_active_round / realized.nominal_b
    )
    rho_range = controlled.groupby("rho").final_p_target.mean()
    result = {
        "behavior": f"The largest mean final target share is {best.final_p_target:.3f} for {best.target_semantics} control at rho={best.rho:g}, b={int(best.b)}; all values remain provisional.",
        "chi": f"The largest absolute whole-cell susceptibility is {chi_row.estimate:.4f} for {chi_row.target_semantics} control at rho={chi_row.rho:g}, b={int(chi_row.b)}.",
        "tpi": f"Whole-cell null-adjusted T_pi ranges from {t.estimate.min():.3f} to {t.estimate.max():.3f} bits; {len(significant)} of {len(t)} controlled cells have retained permutation p<0.05. State-local null estimates are unavailable.",
        "request": f"REQUEST accounts for {request.loc['false', 'mode_probability_given_U1']:.1%} of false-control active rounds and its mean immediate false-target change is {request.loc['false', 'mean_delta_target_share']:+.3f}. This observational contrast cannot establish protection causally.",
        "saturation": f"Across rho, realized posts at b=21 average {realized[realized.b == 21].controller_posts_per_active_round.mean():.2f} per active round, or {ratios[ratios.b == 21].ratio.mean():.1%} of nominal capacity, showing strong dose saturation.",
        "rho": f"Mean controlled final target share ranges from {rho_range.min():.3f} to {rho_range.max():.3f} across rho. The sampled grid supports smooth persistence-dependent structure, not a demonstrated sharp phase transition.",
    }
    if comparison is None:
        result["comparison"] = "No prior archive was supplied."
    else:
        means = comparison.groupby(
            ["study", "target_semantics", "metric"]
        ).estimate.mean()
        a = means.get(("adaptive", "false", "final_p_target"), math.nan)
        p = means.get(("report_only", "false", "final_p_target"), math.nan)
        result["comparison"] = (
            f"Adaptive false control has mean final target share {a:.3f}, versus {p:.3f} in the report-only archive. This is descriptive, not a controlled ablation, because prompt version and initialization differ."
        )
    return result


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    source = (args.work_dir or output.parent / (output.name + "_source")).resolve()
    prior_source = output.parent / (output.name + "_prior_source")
    extract_archive(args.source_zip.resolve(), source)
    if output.exists():
        shutil.rmtree(output)
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True)
    tables.mkdir(parents=True)
    cells = read_table(source, "cells")
    cells["target_semantics"] = cells.target_semantics.map(normalize_semantics)
    primary = read_table(source, "primary_estimates")
    primary["target_semantics"] = primary.target_semantics.map(normalize_semantics)
    derived = read_table(source, "derived_observables")
    derived["target_semantics"] = derived.target_semantics.map(normalize_semantics)
    rounds = attach_coordinates(read_table(source, "rounds"), cells)
    rounds["rho"] = rounds["epistemic_persistence"]
    rounds["b"] = rounds["intervention_budget"]
    diagnostics = read_table(source, "blackboard_diagnostics")
    diagnostics["target_semantics"] = diagnostics.target_semantics.map(
        normalize_semantics
    )
    diagnostics["epistemic_persistence"] = diagnostics.epistemic_persistence.round(3)
    validation = json.loads((source / "validation.json").read_text())
    diag = diagnosis(source, primary)
    state = build_state_local(primary, derived)
    state_rho = aggregate_state(
        state, ["target_semantics", "b"], "rho-aggregated descriptive maps"
    )
    pooled_rho = aggregate_state(
        state, ["rho", "b"], "truth+false pooled descriptive maps"
    )
    pooled_rho["target_semantics"] = "control pooled"
    pooled_all = aggregate_state(
        state, ["b"], "truth+false and rho pooled descriptive maps"
    )
    pooled_all["target_semantics"] = "control pooled"
    endpoints = episode_endpoints(rounds)
    outcomes = outcome_summary(endpoints, args.bootstrap_resamples)
    metrics = whole_metrics(primary, derived, outcomes, diagnostics)
    comm = communication_summary(rounds)
    budget = budget_summary(rounds)
    cooperation = cooperation_summary(rounds)
    uncertainty = metrics[
        metrics.metric.isin(["chi", "eta_IR", "T_pi", "T_pi_excess"])
    ].copy()
    state.to_csv(tables / "state_local_metrics.csv", index=False)
    state_rho.to_csv(tables / "state_local_rho_aggregated.csv", index=False)
    pd.concat([pooled_rho, pooled_all], ignore_index=True).to_csv(
        tables / "state_local_truth_false_pooled.csv", index=False
    )
    outcomes.to_csv(tables / "rho_b_outcomes.csv", index=False)
    metrics.to_csv(tables / "cell_metrics.csv", index=False)
    comm.to_csv(tables / "communication_mode_summary.csv", index=False)
    budget.to_csv(tables / "budget_realization_summary.csv", index=False)
    uncertainty.to_csv(tables / "uncertainty_summary.csv", index=False)
    cooperation.to_csv(tables / "cooperation_summary.csv", index=False)
    endpoints.to_csv(tables / "episode_outcomes.csv", index=False)
    configure_style()
    figure_paths = []
    figure_paths += save_state_figures(
        state, state_rho, pooled_rho, pooled_all, figures
    )
    figure_paths.append(plot_occupancy(state, figures))
    figure_paths += plot_whole_phase(metrics, figures)
    figure_paths += plot_lines(metrics, figures)
    figure_paths += plot_communication(comm, budget, cooperation, figures)
    comparison = None
    if args.prior_zip:
        extract_archive(args.prior_zip.resolve(), prior_source)
        comparison = prior_comparison(source, prior_source)
        comparison.to_csv(tables / "truthful_report_only_comparison.csv", index=False)
        figure_paths.append(plot_prior(comparison, figures))
    findings = numeric_findings(outcomes, metrics, comm, budget, comparison)
    report_validation = {**validation, **diag}
    (output / "rebuild_validation.json").write_text(
        json.dumps(report_validation, indent=2, default=str) + "\n"
    )
    (output / "analysis_summary.json").write_text(
        json.dumps(
            {
                "status": "incomplete_provisional",
                "source_archive": str(args.source_zip.resolve()),
                "findings": findings,
                "weighting": "observation weighted descriptive aggregation",
                "state_local_null": "unavailable",
                "eta_th": "unsupported",
                "figures": [str(p.relative_to(output)) for p in figure_paths],
                "tables": [
                    str(p.relative_to(output)) for p in sorted(tables.glob("*.csv"))
                ],
            },
            indent=2,
        )
        + "\n"
    )
    tex = write_report(output, figure_paths, diag, findings, comparison is not None)
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex.name],
        cwd=output,
        check=True,
    )
    generated = output / (tex.stem + ".pdf")
    target = output / "Adaptive_Communication_q3_Analysis_Report.pdf"
    generated.replace(target)
    for suffix in ["aux", "fdb_latexmk", "fls", "log", "out", "toc"]:
        (output / f"{tex.stem}.{suffix}").unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "output": str(output),
                "pdf": str(target),
                "tex": str(tex),
                "figures": len(figure_paths),
                "tables": len(list(tables.glob("*.csv"))),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
