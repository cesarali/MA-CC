"""Targeted beta-regime CMI tables, figures and data-grounded study report."""
from __future__ import annotations

from pathlib import Path
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .measurements import transitions

CMI = {
    "plain": "round_target_actuation_cmi",
    "mean": "round_kappa_target_actuation_cmi",
    "population": "round_phi_target_actuation_cmi",
    "both": "round_epistemic_target_actuation_cmi",
}


def detectability_table(rounds: pd.DataFrame, final: pd.DataFrame,
                        susceptibility: pd.DataFrame, estimates: pd.DataFrame) -> pd.DataFrame:
    """One row per cell and conditioning variant; zero-budget CMI stays undefined."""
    d = transitions(rounds)
    action = d.groupby("cell_id").agg(
        n_U0=("controller_effective_U", lambda x: int((x == 0).sum())),
        n_U1=("controller_effective_U", lambda x: int((x == 1).sum())),
        controller_action_rate=("controller_effective_U", "mean"))
    action["action_support_fraction"] = 2 * action[["n_U0", "n_U1"]].min(axis=1) / (action.n_U0 + action.n_U1)
    base = final.merge(susceptibility[["cell_id", "chi_empirical", "n_supported"]].rename(
        columns={"n_supported": "susceptibility_supported_count"}), on="cell_id", how="left")
    base = base.merge(action, left_on="cell_id", right_index=True, how="left")
    pooled = estimates.loc[(estimates.scope == "pooled") & estimates.statistic.isin(CMI.values())].copy()
    pooled["conditioning"] = pooled.statistic.map({value: key for key, value in CMI.items()})
    columns = ["cell_id", "conditioning", "statistic", "estimate", "null_mean", "null_std", "null_q95",
               "null_p_value", "estimate_minus_null", "null_z_score", "n_rounds", "n_episodes",
               "round_dual_action_event_fraction", "bootstrap_ci_low", "bootstrap_ci_high"]
    observed = pooled[[c for c in columns if c in pooled]].rename(columns={"estimate": "observed_cmi"})
    table = pd.concat([base.assign(conditioning=key).merge(
        observed.loc[observed.conditioning == key].drop(columns="conditioning"), on="cell_id", how="left")
        for key in CMI], ignore_index=True)
    table["supported_sample_count"] = (table.n_rounds * table.round_dual_action_event_fraction).round().astype("Int64")
    table["detectable_p05"] = (table.null_p_value < .05).astype("boolean")
    table.loc[table.null_p_value.isna(), "detectable_p05"] = pd.NA
    return table.sort_values(["rho", "beta_regime", "budget_fraction", "conditioning"]).reset_index(drop=True)


def _lines(table: pd.DataFrame, root: Path) -> None:
    metrics = {
        "observed_cmi": "Observed CMI (bits)", "estimate_minus_null": "CMI − null mean (bits)",
        "null_z_score": "CMI null z-score", "chi_empirical": "Empirical susceptibility",
        "final_target_share": "Final controller-target share", "final_truth_share": "Final truth share",
        "action_support_fraction": "Action-support fraction",
    }
    for (rho, regime), rows in table.groupby(["rho", "beta_regime"]):
        for column, label in metrics.items():
            fig, ax = plt.subplots(figsize=(7, 4.5))
            keys = CMI if column in {"observed_cmi", "estimate_minus_null", "null_z_score"} else {"plain": CMI["plain"]}
            for key in keys:
                g = rows.loc[rows.conditioning == key].sort_values("budget_fraction")
                ax.plot(g.budget_fraction, g[column], marker="o", label=key)
            ax.set(xlabel="budget fraction b/N", ylabel=label, title=f"{regime}; rho={rho:g}")
            if len(keys) > 1:
                ax.legend(title="conditioning")
            fig.tight_layout()
            fig.savefig(root / f"{column}_{regime}_rho_{rho:g}.png", dpi=150)
            plt.close(fig)


def _heatmaps(table: pd.DataFrame, root: Path) -> None:
    for (rho, conditioning), rows in table.groupby(["rho", "conditioning"]):
        for column in ("observed_cmi", "estimate_minus_null", "chi_empirical"):
            if column == "chi_empirical" and conditioning != "plain":
                continue
            pivot = rows.pivot(index="beta_regime", columns="budget_fraction", values=column)
            if pivot.isna().all().all():
                continue
            fig, ax = plt.subplots(figsize=(9, 4.5))
            image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
            ax.set_xticks(range(len(pivot.columns)), [f"{x:g}" for x in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), pivot.index)
            ax.set(xlabel="budget fraction b/N", ylabel="beta regime", title=f"{column}; {conditioning}; rho={rho:g}")
            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            fig.savefig(root / f"heatmap_{column}_{conditioning}_rho_{rho:g}.png", dpi=150)
            plt.close(fig)


def _report(table: pd.DataFrame, root: Path, sample_summary: Path | None = None) -> None:
    active = table.loc[(table.conditioning == "plain") & (table.budget > 0)].copy()
    lines = ["# Targeted Santa Fe beta/CMI study", "",
             "Predictive CMI is not a causal effect. CMI uses transitions 1 through R−1; at b=0 no physical action exists and CMI is undefined.",
             "Population-only conditioning uses the shared engine's `phi` slot for synthetic population coverage. It does not carry HiddenBench phi semantics.",
             "Action-support fraction = 2 min(n_U0,n_U1)/(n_U0+n_U1). Detection means the shared engine's policy-resampling null p<0.05.", "",
             "## Questions and observed evidence", ""]
    if active.empty:
        lines.append("No active-budget results are available.")
    else:
        def fmt(row) -> str:
            return (f"{row.beta_regime}, rho={row.rho:g}, b={int(row.budget)}: "
                    f"chi={row.chi_empirical:.3f}, corrected CMI={row.estimate_minus_null:.3f} bits, "
                    f"p={row.null_p_value:.3f}, action support={row.action_support_fraction:.2f}")
        baseline = active.loc[(active.beta_regime == "competition_baseline") & (active.budget == 6)]
        lines += ["1. **Baseline competition:** at b=6, compare the following baseline response and channel support with the controls:"]
        lines += [f"   - {fmt(row)}" for row in baseline.itertuples()] or ["   - Baseline b=6 was not included."]
        lines += ["2. **Off-regime controls:** at b=6:"]
        controls = active.loc[active.beta_regime.isin(["evidence_dominated", "social_dominated"]) & active.budget.eq(6)]
        lines += [f"   - {fmt(row)}" for row in controls.itertuples()] or ["   - No control b=6 rows."]
        for rho, group in active.groupby("rho"):
            supported = group.loc[group.n_U0.gt(0) & group.n_U1.gt(0)]
            if supported.empty:
                lines.append(f"rho={rho:g}: no cells with both actions observed; CMI comparisons are not supported.")
                continue
            chi_cut = supported.chi_empirical.abs().quantile(.75)
            cmi_cut = supported.estimate_minus_null.quantile(.75)
            high_chi = supported.loc[supported.chi_empirical.abs().ge(chi_cut)]
            detectable = supported.loc[supported.null_p_value.lt(.05)]
            overlap = high_chi.loc[high_chi.cell_id.isin(detectable.cell_id)]
            response_only = high_chi.loc[~high_chi.cell_id.isin(detectable.cell_id)]
            info_only = supported.loc[supported.estimate_minus_null.ge(cmi_cut) &
                                      ~supported.cell_id.isin(high_chi.cell_id)]
            lines += [f"3. **Strong susceptibility, rho={rho:g}:** upper-quartile |chi| cells: " +
                      ("; ".join(fmt(row) for row in high_chi.itertuples()) or "none"),
                      f"4. **Detectable CMI, rho={rho:g}:** {len(detectable)}/{len(group)} active cells with both actions and p<0.05.",
                      f"5. **Response and CMI together, rho={rho:g}:** {len(overlap)} upper-quartile response cells have detectable CMI.",
                      f"6. **Response without detection, rho={rho:g}:** {len(response_only)} cells; high corrected CMI with weaker net response: {len(info_only)} cells."]
            largest = group.sort_values("budget").groupby("beta_regime").tail(1)
            lines.append(f"7. **High-budget saturation, rho={rho:g}:** " + "; ".join(
                f"{row.beta_regime} target={row.final_target_share:.2f}, action support={row.action_support_fraction:.2f}"
                for row in largest.itertuples()))
        if active.rho.nunique() == 2:
            paired = active.pivot_table(index=["beta_regime", "budget"], columns="rho",
                                        values=["chi_empirical", "estimate_minus_null"])
            rho_low, rho_high = sorted(active.rho.unique())
            diff_chi = (paired["chi_empirical"][rho_high] - paired["chi_empirical"][rho_low]).dropna()
            diff_cmi = (paired["estimate_minus_null"][rho_high] - paired["estimate_minus_null"][rho_low]).dropna()
            lines.append(f"8. **Persistence:** mean matched-cell change from rho={rho_low:g} to {rho_high:g}: "
                         f"chi={diff_chi.mean():+.3f}, corrected CMI={diff_cmi.mean():+.3f} bits. "
                         "Inspect paired curves for heterogeneous effects.")
        size = sample_summary or root.parent.parent / "sample_size_cmi" / "sample_size_cmi_summary.csv"
        if size.is_file():
            calibration = pd.read_csv(size)
            lines.append("9. **50 episodes:** observed detection probability by rho: " + "; ".join(
                f"rho={row.rho:g}: {row.detection_probability:.2f}"
                for row in calibration.loc[calibration.n_episodes.eq(50)].itertuples()))
        else:
            lines.append("9. **50 episodes:** pending the separate `sample_size_cmi.yaml` calibration; the sweep cannot settle this question.")
        lines += ["", "Select scientifically useful cells from response, overlap, unsaturated share and null-corrected CMI together; do not rank by raw CMI alone."]
    lines += ["", "Complete per-cell data: `detectability.csv`; required curves and heatmaps: `plots/beta_cmi/`.", ""]
    (root / "beta_cmi_report.md").write_text("\n".join(lines), encoding="utf-8")


def produce_beta_outputs(config, rounds: pd.DataFrame, final: pd.DataFrame,
                         susceptibility: pd.DataFrame, estimates: pd.DataFrame) -> pd.DataFrame:
    root = config.results_dir / "information" / "beta_cmi"
    root.mkdir(parents=True, exist_ok=True)
    table = detectability_table(rounds, final, susceptibility, estimates)
    table.to_parquet(root / "detectability.parquet", index=False)
    table.to_csv(root / "detectability.csv", index=False)
    plots = config.results_dir / "plots" / "beta_cmi"
    plots.mkdir(parents=True, exist_ok=True)
    _lines(table, plots)
    _heatmaps(table, plots)
    _report(table, root, config.results_dir.parent / "santa_fe_sample_size_cmi" / "sample_size_cmi" / "sample_size_cmi_summary.csv")
    (root / "analysis_recipe.yaml").write_bytes(config.path.read_bytes())
    return table


def main(argv=None):
    import argparse
    from .config import load_config
    parser = argparse.ArgumentParser(description="Refresh beta/CMI report after sample-size calibration")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-size-summary")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    root = config.results_dir / "information" / "beta_cmi"
    table = pd.read_csv(root / "detectability.csv")
    sample = Path(args.sample_size_summary).expanduser() if args.sample_size_summary else (
        config.results_dir.parent / "santa_fe_sample_size_cmi" / "sample_size_cmi" / "sample_size_cmi_summary.csv")
    _report(table, root, sample)
    print(root / "beta_cmi_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
