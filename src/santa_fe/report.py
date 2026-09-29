"""Descriptive exploratory report from saved trajectories."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def _table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame[columns].iterrows():
        lines.append("| " + " | ".join(str(int(row[key])) if key in {"cell_id", "round", "budget", "n_episodes", "controller_U", "controller_effective_U", "budget_used"} and pd.notna(row[key]) else f"{row[key]:.3f}" if isinstance(row[key], (float, np.floating)) else str(row[key]) for key in columns) + " |")
    return lines


def write_report(rounds: pd.DataFrame, final: pd.DataFrame, path: Path) -> None:
    lines = ["# Santa Fe synthetic control: exploratory report", "",
             "This report describes simulated outcomes; it does not make causal or theoretical claims.", "",
             "## Final outcomes by parameter cell", ""]
    lines += _table(final, ["cell_id", "beta_evidence", "beta_social", "budget_fraction", "budget",
                             "n_episodes", "final_truth_share", "final_target_share", "final_mean_coverage"])
    lines += ["", "## Observed patterns", ""]
    for (be, bs), g in final.groupby(["beta_evidence", "beta_social"]):
        g = g.sort_values("budget_fraction")
        if len(g) > 1:
            delta = g.final_target_share.iloc[-1] - g.final_target_share.iloc[0]
            monotonic = bool(np.all(np.diff(g.final_target_share) >= -1e-9))
            lines.append(f"- βe={be:g}, βs={bs:g}: final target share changed by {delta:+.3f} across the budget range; "
                         f"the cell means were {'nondecreasing' if monotonic else 'not monotonic'}.")
    if final.beta_evidence.nunique() > 1 or final.beta_social.nunique() > 1:
        means = final.groupby(["beta_evidence", "beta_social"])["final_target_share"].mean()
        lines.append(f"- Across beta pairs, mean final target share ranged from {means.min():.3f} to {means.max():.3f}.")
    for cell, d in rounds.groupby("cell_id"):
        final_rows = d.loc[d.groupby("seed")["round"].idxmax()].copy()
        median = final_rows.target_share.median()
        selected = {
            "median outcome": int(final_rows.iloc[(final_rows.target_share - median).abs().argmin()].seed),
            "strongest controller success": int(final_rows.loc[final_rows.target_share.idxmax(), "seed"]),
            "strongest controller failure": int(final_rows.loc[final_rows.target_share.idxmin(), "seed"]),
        }
        volatility = d.groupby("seed").target_share.apply(lambda x: float(x.diff().abs().sum()))
        selected["unusual trajectory (largest total change)"] = int(volatility.idxmax())
        by_round = d.groupby("round").agg(target=("target_share", "mean"),
                                           truth=("truth_share", "mean"))
        early = by_round.target.iloc[min(10, len(by_round) - 1)]
        late = by_round.target.iloc[-1]
        variability = final_rows.target_share.std()
        if len(by_round) >= 11:
            recent = by_round.target.iloc[-5:]
            prior = by_round.target.iloc[-10:-5]
            lines.append(f"- Cell {cell}: mean target share in the last five recorded rounds differed from the preceding five by {recent.mean() - prior.mean():+.3f}.")
        lines += ["", f"## Cell {cell}", "",
                  f"Final target-share SD across episodes: {variability:.3f}. "
                  f"Mean target share changed by {late - early:+.3f} after round {min(10, len(by_round)-1)}.", "",
                  "Representative episode stories:", ""]
        for label, seed in selected.items():
            episode = d.loc[d.seed == seed].sort_values("round")
            lines += [f"### {label}: seed {seed}", ""]
            cols = ["round", "truth_share", "target_share", "kappa_mean_coverage",
                    "kappa_population_coverage", "controller_observed_target_share",
                    "controller_U", "controller_effective_U", "budget_used"]
            lines += _table(episode, cols)
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
