"""Summarise a Simulation 1 run: one row per cell, compared with its silent baseline.

Writes summary.csv next to the cell folders. From the repository root:
    .venv/bin/python analysis/task003_llm_free/simulator/summarize_simulation_1.py
"""
from __future__ import annotations
import argparse, json, pathlib

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(HERE.parent / "results" / "simulation_1"))
    run = pathlib.Path(ap.parse_args().run)
    rows = []
    for cell in sorted(p for p in run.iterdir() if p.is_dir()):
        prm = json.loads((cell / "params.json").read_text())
        days = pd.read_parquet(cell / "days.parquet")
        last = days[days.day == prm["M"] - 1]
        row = {k: prm[k] for k in ("setup", "arm", "q", "qc", "b", "rho")}
        row.update({
            "episodes": days.episode.nunique(),
            "final_share_A0": last.share_A0.mean(), "final_share_A2": last.share_A2.mean(),
            "final_mean_p_A0": last.mean_p_A0.mean(), "final_mean_p_A2": last.mean_p_A2.mean(),
            "final_proof_rate_A0": last.proof_rate_A0.mean(),
            "episodes_majority_A0": (last.share_A0 > 0.5).mean(),
            "episodes_majority_A2": (last.share_A2 > 0.5).mean(),
        })
        nights_file = cell / "nights.parquet"
        if nights_file.exists():
            n = pd.read_parquet(nights_file)
            row["nights_posted"] = n.decision.isin(["posted", "fallback"]).mean()
            row["facts_posted_per_episode"] = n.n_posted.sum() / n.episode.nunique()
            for d in ("proved", "gate", "posted", "fallback"):
                row[f"nights_{d}"] = (n.decision == d).mean()
        rows.append(row)
    s = pd.DataFrame(rows)
    silent = s[s.arm == "silent"].set_index(["setup", "q", "rho"])
    for col in ("final_share_A0", "final_share_A2"):
        s[f"{col}_minus_silent"] = s.apply(
            lambda r: r[col] - silent.loc[(r.setup, r.q, r.rho), col], axis=1)
    s = s.sort_values(["setup", "arm", "rho", "q", "qc", "b"], na_position="first")
    s.to_csv(run / "summary.csv", index=False, float_format="%.4f")
    print(f"wrote {run / 'summary.csv'}: {len(s)} cells")


if __name__ == "__main__":
    main()
