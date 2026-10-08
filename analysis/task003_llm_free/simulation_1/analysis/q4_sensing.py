"""Q4 — how well does the controller sense the population? (ANALYSIS_PLAN.md §4)

Every night the controller estimates v_hat, the share of the agent posts it read whose
author voted for its target, and stays silent when v_hat >= theta_vote (0.75). The
simulator also records the TRUE share of all 24 agents voting for the target that day,
and the share among agents who posted (they differ only through abstention).

Per controlled cell (main runs), over all nights:
  bias                mean(v_hat - true share)
  rmse                sqrt(mean((v_hat - true share)^2))
  posters_bias        mean(share among posters - true share): the part due to abstention
  wrongly_silent      share of nights the gate kept it silent although the true share < theta
  wrongly_acting      share of nights it posted although the true share >= theta
  posts_read          mean posts read (can fall below qc when agents abstain)
Abstention by vote comes from the summary tables (agent-days with post_reason abstained,
split by the vote).

Outputs: tables/q4_sensing.csv, figures/q4_*.png.
From the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_1/analysis/q4_sensing.py
"""
from __future__ import annotations
import multiprocessing, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import SIM, VARIANT_COLOR, VARIANT_LABEL, VARIANTS, cells_table, out_dir, write_index  # noqa: E402

QCS = (3, 6, 12, 24)


def cell(row: dict) -> dict:
    n = pd.read_parquet(pathlib.Path(row["path"]) / "nights.parquet",
                        columns=["day", "n_posts_read", "v_hat", "true_target_share", "target_share_among_posters",
                                 "decision"])
    theta = row["theta_vote"]
    sensed = n.dropna(subset=["v_hat"])
    err = sensed.v_hat - sensed.true_target_share
    acting = n.decision.isin(["posted", "posted_proved", "fallback"])
    return {**{k: row[k] for k in ("variant", "setup", "arm", "q", "qc", "b", "rho")},
            "nights": len(n), "posts_read": n.n_posts_read.mean(),
            "bias": err.mean(), "rmse": float(np.sqrt((err ** 2).mean())),
            "posters_bias": (n.target_share_among_posters - n.true_target_share).mean(),
            "wrongly_silent": ((n.decision == "gate") & (n.true_target_share < theta)).mean(),
            "wrongly_acting": (acting & (n.true_target_share >= theta)).mean(),
            "nights_true_share_below_theta": (n.true_target_share < theta).mean()}


def main():
    d = out_dir()
    c = cells_table()
    rows = c[(c.purpose == "main") & (c.arm != "silent")].to_dict("records")
    with multiprocessing.Pool(12) as pool:
        t = pd.DataFrame(pool.map(cell, rows))
    t = t.sort_values(["variant", "setup", "arm", "rho", "q", "qc", "b"]).reset_index(drop=True)
    t.to_csv(d / "tables" / "q4_sensing.csv", index=False, float_format="%.5f")

    entries = [{"question": "Q4", "file": "tables/q4_sensing.csv",
                "what": "per cell: sensing bias, rmse, abstention bias, gate errors, posts read", "cells": "main controlled"}]
    for metric, label in (("rmse", "error of v̂ (root mean square)"),
                          ("wrongly_silent", "share of nights silent although true share < 0.75"),
                          ("wrongly_acting", "share of nights posting although true share ≥ 0.75")):
        fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
        for i, setup in enumerate(("task003_nosolution", "task003_symmetric")):
            for j, (arm, rho) in enumerate([(a, r) for a in ("truth", "false") for r in (0.75, 1.0)]):
                ax = axes[i, j]
                for v in VARIANTS:
                    s = t[(t.variant == v) & (t.setup == setup) & (t.arm == arm) & (t.rho == rho)]
                    m = s.groupby("qc")[metric].mean()
                    ax.plot(range(len(QCS)), [m.get(qc, np.nan) for qc in QCS], "o-", color=VARIANT_COLOR[v],
                            label=VARIANT_LABEL[v])
                ax.set_xticks(range(len(QCS)), [str(q) for q in QCS])
                ax.set_title(f"{setup.split('_')[1]} · {arm} · ρ = {rho}", fontsize=9)
                if i == 1:
                    ax.set_xlabel("qc (posts read per night)")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"Q4 · {label} (mean over q and b)", fontsize=11)
        fig.tight_layout()
        name = f"figures/q4_{metric}_vs_qc.png"
        fig.savefig(d / name, dpi=115)
        plt.close(fig)
        entries.append({"question": "Q4", "file": name, "what": f"{metric} against qc, by variant", "cells": "main controlled"})
    write_index(d, entries)

    pd.set_option("display.width", 220)
    print("SENSING, mean over setups, arms, q, b, rho")
    print(t.groupby(["variant", "qc"])[["posts_read", "bias", "rmse", "posters_bias", "wrongly_silent",
                                        "wrongly_acting"]].mean().round(4).to_string())
    print("\nBY SETUP AND ARM (qc = 12)")
    print(t[t.qc == 12].groupby(["setup", "arm", "variant"])[["bias", "rmse", "wrongly_silent", "wrongly_acting",
                                                              "nights_true_share_below_theta"]].mean().round(4).to_string())
    # abstention by vote: from the committed run summaries of the main runs
    ab = []
    for v in VARIANTS:
        f = SIM / "run_summaries" / f"2026-10-07_{v}_summary.csv"
        x = pd.read_csv(f)
        x["variant"] = v
        ab.append(x)
    ab = pd.concat(ab)
    cols = [f"abstention_rate_when_voting_A{k}" for k in range(3)]
    print("\nABSTENTION RATE BY VOTE (mean over cells)")
    print(ab.groupby(["variant", "setup", "arm"])[["abstention_rate"] + cols].mean().round(4).to_string())

if __name__ == "__main__":
    main()
