"""Summarise a Simulation 1 run: one row per cell, compared with its silent baseline.

The runner calls this at the end of every run. To redo it by hand, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_1/simulator/summarize_simulation_1.py <run folder>
"""
from __future__ import annotations
import json, pathlib, sys

import pandas as pd

ABSTAINED = 4


def summarize(run: str | pathlib.Path) -> pd.DataFrame:
    run = pathlib.Path(run)
    rows = []
    for cell in sorted(p for p in (run / "cells").iterdir() if p.is_dir()):
        prm = json.loads((cell / "params.json").read_text())
        days = pd.read_parquet(cell / "days.parquet")
        agents = pd.read_parquet(cell / "agents.parquet", columns=["vote", "post_reason"])
        last = days[days.day == prm["M"] - 1]
        row = {k: prm[k] for k in ("setup", "arm", "q", "qc", "b", "rho")}
        row.update({
            "episodes": days.episode.nunique(),
            "final_share_A0": last.share_A0.mean(), "final_share_A1": last.share_A1.mean(),
            "final_share_A2": last.share_A2.mean(),
            "final_mean_p_A0": last.mean_p_A0.mean(), "final_mean_p_A2": last.mean_p_A2.mean(),
            "final_proof_rate_A0": last.proof_rate_A0.mean(),
            "final_proof_rate_A0_own_evidence": last.proof_rate_A0_own_evidence.mean(),
            "episodes_majority_A0": (last.share_A0 > 0.5).mean(),
            "episodes_majority_A2": (last.share_A2 > 0.5).mean(),
            "abstention_rate": (agents.post_reason == ABSTAINED).mean(),
        })
        for v in range(3):
            voted = agents[agents.vote == v]
            row[f"abstention_rate_when_voting_A{v}"] = (voted.post_reason == ABSTAINED).mean() if len(voted) else None
        nights_file = cell / "nights.parquet"
        if nights_file.exists():
            n = pd.read_parquet(nights_file, columns=["episode", "n_posts_read", "v_hat", "true_target_share",
                                                      "target_share_among_posters", "decision", "n_posted"])
            row["nights_acting"] = n.decision.isin(["posted", "posted_proved", "fallback"]).mean()
            row["facts_posted_per_episode"] = n.n_posted.sum() / n.episode.nunique()
            for d in ("proved", "gate", "posted", "posted_proved", "fallback"):
                row[f"nights_{d}"] = (n.decision == d).mean()
            sensed = n.dropna(subset=["v_hat"])
            row["mean_posts_read_by_controller"] = n.n_posts_read.mean()
            row["sensing_bias_vs_all_agents"] = (sensed.v_hat - sensed.true_target_share).mean()
            row["posters_vs_all_agents"] = (n.target_share_among_posters - n.true_target_share).mean()
        rows.append(row)
    s = pd.DataFrame(rows)
    silent = s[s.arm == "silent"].set_index(["setup", "q", "rho"])
    if len(silent):
        for col in ("final_share_A0", "final_share_A1", "final_share_A2"):
            s[f"{col}_minus_silent"] = s.apply(
                lambda r: r[col] - silent.loc[(r.setup, r.q, r.rho), col]
                if (r.setup, r.q, r.rho) in silent.index else None, axis=1)
    s = s.sort_values(["setup", "arm", "rho", "q", "qc", "b"], na_position="first")
    s.to_csv(run / "summary.csv", index=False, float_format="%.4f")
    return s


if __name__ == "__main__":
    out = summarize(sys.argv[1])
    print(f"{len(out)} cells summarised")
