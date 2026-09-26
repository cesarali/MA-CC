"""Plots from saved tables."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


EVOLUTION = ["truth_share", "target_share", "kappa_mean_coverage", "kappa_population_coverage",
             "vote_entropy_bits", "controller_effective_U"]


def save_plots(rounds: pd.DataFrame, final: pd.DataFrame, root: Path,
               null_summary: pd.DataFrame | None = None, null_draws: pd.DataFrame | None = None) -> None:
    evolution = root / "evolution"
    phase = root / "phase_diagrams"
    evolution.mkdir(parents=True, exist_ok=True)
    phase.mkdir(parents=True, exist_ok=True)
    for cell, d in rounds.groupby("cell_id"):
        fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True)
        for ax, key in zip(axes.flat, EVOLUTION):
            stats = d.groupby("round")[key].agg(["mean", "std", "count"])
            error = stats["std"].fillna(0) / stats["count"].pow(.5)
            ax.plot(stats.index, stats["mean"])
            ax.fill_between(stats.index, stats["mean"] - 1.96 * error, stats["mean"] + 1.96 * error, alpha=.2)
            ax.set_ylabel(key)
        axes[-1, 0].set_xlabel("round")
        axes[-1, 1].set_xlabel("round")
        fig.suptitle(f"Cell {cell}; bands = 1.96 SE across episodes")
        fig.tight_layout()
        fig.savefig(evolution / f"cell_{cell}.png", dpi=150)
        plt.close(fig)
    for (be, bs), g in final.groupby(["beta_evidence", "beta_social"]):
        g = g.sort_values("budget_fraction")
        fig, ax = plt.subplots(figsize=(7, 5))
        for key in ("final_target_share", "final_truth_share", "final_mean_coverage"):
            ax.plot(g.budget_fraction, g[key], marker="o", label=key)
        ax.set(xlabel="budget fraction b/N", ylabel="final share", title=f"βe={be:g}, βs={bs:g}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(phase / f"budget_be_{be:g}_bs_{bs:g}.png", dpi=150)
        plt.close(fig)
    if final.beta_evidence.nunique() > 1 and final.beta_social.nunique() > 1:
        for budget, g in final.groupby("budget_fraction"):
            pivot = g.pivot(index="beta_evidence", columns="beta_social", values="final_target_share")
            fig, ax = plt.subplots(figsize=(6, 5))
            image = ax.imshow(pivot, origin="lower", aspect="auto")
            ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
            ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
            ax.set(xlabel="beta_social", ylabel="beta_evidence", title=f"Final target share; b/N={budget:g}")
            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            fig.savefig(phase / f"heatmap_budget_{budget:g}.png", dpi=150)
            plt.close(fig)
    if null_summary is not None and null_draws is not None and not null_draws.empty:
        out = root / "null_histograms"
        out.mkdir(exist_ok=True)
        for (cell, statistic), g in null_draws.groupby(["cell_id", "statistic"]):
            observed = null_summary.loc[(null_summary.cell_id == cell) &
                                        (null_summary.statistic == statistic), "observed"].iloc[0]
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(g.value, bins=30)
            ax.axvline(observed, color="red", label="observed")
            ax.set(xlabel="information (bits)", title=f"Cell {cell}: {statistic}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / f"cell_{cell}_{statistic}.png", dpi=150)
            plt.close(fig)


def save_analysis_plots(final: pd.DataFrame, information: pd.DataFrame | None,
                        susceptibility: pd.DataFrame | None, root: Path) -> None:
    for table, folder, columns in (
        (information, "information", ["I_X_Y_bits", "Tpi_I_U_Xnext_given_X_bits", "Tpi_I_U_Xnext_given_X_K_bits"]),
        (susceptibility, "susceptibility", ["chi_empirical"]),
    ):
        if table is None or table.empty:
            continue
        output = root / folder
        output.mkdir(parents=True, exist_ok=True)
        joined = final[["cell_id", "beta_evidence", "beta_social", "budget_fraction"]].merge(table, on="cell_id")
        for (be, bs), g in joined.groupby(["beta_evidence", "beta_social"]):
            g = g.sort_values("budget_fraction")
            fig, ax = plt.subplots(figsize=(7, 5))
            for column in columns:
                ax.plot(g.budget_fraction, g[column], marker="o", label=column)
            ax.set(xlabel="budget fraction b/N", ylabel="estimate", title=f"βe={be:g}, βs={bs:g}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(output / f"budget_be_{be:g}_bs_{bs:g}.png", dpi=150)
            plt.close(fig)


def save_sample_size_plots(summary: pd.DataFrame, root: Path) -> None:
    output = root / "sample_size"
    output.mkdir(parents=True, exist_ok=True)
    for (cell, metric), g in summary.groupby(["cell_id", "metric"]):
        g = g.sort_values("n_episodes")
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, column in zip(axes.flat, ["bias", "variance", "ci_coverage", "detection_probability"]):
            ax.plot(g.n_episodes, g[column], marker="o")
            ax.set(xlabel="episodes", ylabel=column)
            if column == "detection_probability" and g.false_positive_rate.notna().any():
                ax.plot(g.n_episodes, g.false_positive_rate, marker="x", label="false-positive rate")
                ax.legend()
        fig.suptitle(f"Cell {cell}: {metric}")
        fig.tight_layout()
        fig.savefig(output / f"cell_{cell}_{metric}.png", dpi=150)
        plt.close(fig)


def save_phase_diagrams(final: pd.DataFrame, pooled_information: pd.DataFrame | None, root: Path) -> None:
    """Render observed beta_evidence × beta_social maps at every budget.

    Each point is one physical scientific cell. Information maps use the
    complete cell estimate; no per-round or scheduler-shard estimates are averaged.
    """
    root.mkdir(parents=True, exist_ok=True)
    table = final.copy()
    metrics = ["final_target_share", "final_truth_share", "final_mean_coverage",
               "final_population_coverage"]
    if pooled_information is not None and not pooled_information.empty:
        requested = ("round_target_sensing_mi", "round_target_actuation_cmi",
                     "round_kappa_target_actuation_cmi")
        subset = pooled_information.loc[pooled_information.statistic.isin(requested),
                                        ["cell_id", "statistic", "estimate", "estimate_minus_null"]]
        values = subset.pivot(index="cell_id", columns="statistic", values="estimate")
        values.columns = [f"{name}_bits" for name in values.columns]
        excess = subset.pivot(index="cell_id", columns="statistic", values="estimate_minus_null")
        excess.columns = [f"{name}_minus_null_bits" for name in excess.columns]
        table = table.merge(values, left_on="cell_id", right_index=True, how="left")
        table = table.merge(excess, left_on="cell_id", right_index=True, how="left")
        metrics += list(values.columns) + list(excess.columns)
    table.to_parquet(root / "phase_coordinates.parquet", index=False)
    if table.beta_evidence.nunique() < 2 or table.beta_social.nunique() < 2 or table.rho.nunique() > 1:
        return
    for metric in metrics:
        for budget, rows in table.groupby("budget_fraction"):
            pivot = rows.pivot(index="beta_evidence", columns="beta_social", values=metric)
            if pivot.isna().all().all():
                continue
            fig, ax = plt.subplots(figsize=(6, 5))
            image = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto")
            ax.set_xticks(range(len(pivot.columns)), labels=[f"{x:g}" for x in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), labels=[f"{x:g}" for x in pivot.index])
            ax.set(xlabel="beta_social", ylabel="beta_evidence",
                   title=f"{metric}; b/N={budget:g}")
            fig.colorbar(image, ax=ax, label=metric)
            fig.tight_layout()
            fig.savefig(root / f"{metric}_budget_{budget:g}.png", dpi=150)
            plt.close(fig)
