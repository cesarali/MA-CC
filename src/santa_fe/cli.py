"""Run a YAML-configured synthetic Santa Fe study."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from .bootstrap import analyze as bootstrap_analyze
from .config import load_config, plan
from .measurements import final_summary, information_summary, susceptibility_summary, susceptibility_strata, transitions
from .nulls import analyze as null_analyze
from .plotting import save_plots, save_analysis_plots, save_sample_size_plots
from .report import write_report
from .runner import run, save_trajectories
from .sample_size import calibrate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML experiment config")
    parser.add_argument("--dry-run", action="store_true", help="validate and print plan without simulation")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    print(json.dumps(plan(config), indent=2))
    if args.dry_run:
        return 0
    root = config.results_dir
    if root.exists() and any(root.iterdir()):
        parser.error(f"results directory is not empty: {root}")
    rounds, micro = run(config)
    save_trajectories(config, rounds, micro)
    # All downstream calculations read the persisted canonical trajectories.
    if config.save_round:
        rounds = pd.read_parquet(root / "trajectories" / "round_trajectories.parquet")
    final = final_summary(rounds)
    summaries = root / "summaries"
    summaries.mkdir(exist_ok=True)
    final.to_csv(summaries / "final_summary.csv", index=False)
    information = chi = None
    if config.information:
        information = information_summary(rounds, config.bins, config.coordinate)
        chi = susceptibility_summary(rounds, config.bins, config.coordinate)
        information.to_csv(summaries / "information_summary.csv", index=False)
        chi.to_csv(summaries / "susceptibility.csv", index=False)
        local = []
        for cell, d in transitions(rounds).groupby("cell_id"):
            rows = susceptibility_strata(d, config.bins, config.coordinate)
            if not rows.empty:
                rows.insert(0, "cell_id", cell)
                local.append(rows)
        local_table = pd.concat(local, ignore_index=True) if local else pd.DataFrame(
            columns=["cell_id", "chi_empirical", "n_U0", "n_U1", "n_supported"])
        local_table.to_csv(summaries / "susceptibility_strata.csv", index=False)
    null_summary = null_draws = None
    if config.permutation.get("enabled", False):
        null_summary, null_draws = null_analyze(rounds, config.bins, config.coordinate, config.permutation)
        out = root / "nulls"
        out.mkdir(exist_ok=True)
        null_summary.to_csv(out / "null_summary.csv", index=False)
        null_draws.to_parquet(out / "null_samples.parquet", index=False)
    if config.bootstrap.get("enabled", False):
        out = root / "bootstrap"
        out.mkdir(exist_ok=True)
        bootstrap_analyze(rounds, config.bins, config.coordinate, config.bootstrap).to_csv(
            out / "bootstrap_summary.csv", index=False)
    calibration = None
    if config.sample_size.get("enabled", False):
        calibration = calibrate(config)
    if config.save_plots:
        save_plots(rounds, final, root / "plots",
                   null_summary if config.histograms.get("save_null_histograms", True) else None,
                   null_draws if config.histograms.get("enabled", True) else None)
        save_analysis_plots(final, information, chi, root / "plots")
        if calibration is not None:
            save_sample_size_plots(calibration, root / "plots")
    if config.save_reports:
        write_report(rounds, final, root / "report.md")
    print(f"Results: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
