"""Q2 — does control work, and how? Computation (ANALYSIS_PLAN.md §3, §4). Figures: q2_figures.py.

For each group of cells that differ only in the arm -- (variant, setup, q, qc, b, rho)
with its truth cell, false cell and silent twin (same variant, setup, q, rho) -- and
each day t = 1..30:

  paired gain        G_t^(a)[arm] = mean_e( x_a,t(arm, e) - x_a,t(silent, e) )
  shared part        M_t^(a) = (G^(a)[truth] + G^(a)[false]) / 2
  directional part   D_t^(a) = (G^(a)[truth] - G^(a)[false]) / 2   (full switch = 2D)
  delivered dose     C_t[arm] = facts posted on nights 1..t, mean per episode

x_a,t is the share of agents voting a at the end of day t. Episode e uses the same
random draws in all three arms, so every difference is taken within an episode, and
the bootstrap (1,000 resamples of episodes) resamples the three arms together.

Final-day quantities per group and arm: own-target gain (truth: a = 0, false: a = 2),
dose, budget efficiency 24 * G_30 / C_29 (extra supporting agents per fact posted),
time to 75% (first day the target's share >= 0.75; episodes that never reach it are
counted as not reached, never dropped), and the unanimity split on day 30.

Main runs only (stopping rule on); Q5 handles the no-proof-stop runs.
Outputs: results/simulation_1/analysis/<date>/tables/q2_time.csv, q2_final.csv.

From the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_1/analysis/q2_compute.py
"""
from __future__ import annotations
import hashlib, multiprocessing, pathlib, sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import cells_table, out_dir, write_index  # noqa: E402

N_BOOT = 1000
TARGET = {"truth": 0, "false": 2}
GROUP_KEYS = ["variant", "setup", "q", "qc", "b", "rho"]


def shares(path: str) -> np.ndarray:
    """(3, episodes, days) vote shares."""
    d = pd.read_parquet(pathlib.Path(path) / "days.parquet",
                        columns=["episode", "day", "share_A0", "share_A1", "share_A2"]).sort_values(["episode", "day"])
    E, T = d.episode.nunique(), d.day.nunique()
    return np.stack([d[f"share_A{k}"].to_numpy().reshape(E, T) for k in range(3)])


def dose(path: str, E: int, T: int) -> np.ndarray:
    """(episodes, days): facts posted on nights 1..t (a night's posts are read the next day)."""
    n = pd.read_parquet(pathlib.Path(path) / "nights.parquet", columns=["episode", "day", "n_posted"])
    per_night = np.zeros((E, T))
    per_night[n.episode.to_numpy(), n.day.to_numpy()] = n.n_posted.to_numpy()
    return per_night.cumsum(axis=1)


def boot(x: np.ndarray, idx: np.ndarray):
    """x: (E, T). Mean and 95% percentile interval over episode resamples idx (B, E)."""
    b = x[idx].mean(axis=1)
    return x.mean(axis=0), np.percentile(b, 2.5, axis=0), np.percentile(b, 97.5, axis=0)


def group(job):
    key, paths = job
    rec = dict(zip(GROUP_KEYS, key))
    S = shares(paths["silent"])
    X = {arm: shares(paths[arm]) for arm in ("truth", "false")}
    _, E, T = S.shape
    C = {arm: dose(paths[arm], E, T) for arm in ("truth", "false")}
    # fixed seed per group: Python's hash() of strings changes between processes
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:8], "big"))
    idx = rng.integers(0, E, size=(N_BOOT, E))
    days = np.arange(1, T + 1)
    time_rows, final_rows = [], []

    def add(quantity, a, arm, arr):
        m, lo, hi = boot(arr, idx)
        time_rows.extend({**rec, "arm": arm, "quantity": quantity, "a": a, "day": int(t), "mean": m[i],
                          "lo": lo[i], "hi": hi[i]} for i, t in enumerate(days))
        return m[-1], lo[-1], hi[-1]

    finals = {}
    for arm in ("truth", "false"):
        for a in range(3):
            finals[("G", a, arm)] = add("G", a, arm, X[arm][a] - S[a])
        finals[("dose", None, arm)] = add("dose", None, arm, C[arm])
    for a in range(3):
        finals[("M", a, "both")] = add("M", a, "both", (X["truth"][a] + X["false"][a]) / 2 - S[a])
        finals[("D", a, "both")] = add("D", a, "both", (X["truth"][a] - X["false"][a]) / 2)
    for k in range(3):                                            # silent twin's own shares, for reference
        add("share_silent", k, "silent", S[k])

    for arm, tgt in TARGET.items():
        x_t = X[arm][tgt]
        g, glo, ghi = finals[("G", tgt, arm)]
        c29 = C[arm][:, -2].mean()                                 # posts on nights 1..29 (none after day 30)
        reached = (x_t >= 0.75)
        first = np.where(reached.any(axis=1), reached.argmax(axis=1) + 1, np.inf)
        frac_by_day = np.array([(first <= t).mean() for t in days])
        med = int(days[np.argmax(frac_by_day >= 0.5)]) if (frac_by_day >= 0.5).any() else None
        sil_first = np.where((S[tgt] >= 0.75).any(axis=1), (S[tgt] >= 0.75).argmax(axis=1) + 1, np.inf)
        final_rows.append({
            **rec, "arm": arm, "target": tgt,
            "own_gain_30": g, "own_gain_30_lo": glo, "own_gain_30_hi": ghi,
            "gain_A1_30": finals[("G", 1, arm)][0],
            "target_share_30": x_t[:, -1].mean(), "silent_target_share_30": S[tgt][:, -1].mean(),
            "A1_share_30": X[arm][1][:, -1].mean(),
            "dose_29": c29, "nights_acting": (np.diff(np.c_[np.zeros(E), C[arm]], axis=1)[:, :-1] > 0).mean(),
            "efficiency_agents_per_fact": (24 * g / c29) if c29 > 0 else np.nan,
            "reached_75_by_30": frac_by_day[-1], "silent_reached_75_by_30": np.isfinite(sil_first).mean(),
            "median_day_to_75": med,
            "unanimous_target_30": (x_t[:, -1] == 1).mean(),
            "unanimous_other_30": ((X[arm][2 - tgt][:, -1] == 1) | (X[arm][1][:, -1] == 1)).mean(),
            "silent_unanimous_target_30": (S[tgt][:, -1] == 1).mean(),
        })
    for a in range(3):
        final_rows.append({**rec, "arm": "both", "target": a,
                           "M_30": finals[("M", a, "both")][0], "M_30_lo": finals[("M", a, "both")][1],
                           "M_30_hi": finals[("M", a, "both")][2],
                           "D_30": finals[("D", a, "both")][0], "D_30_lo": finals[("D", a, "both")][1],
                           "D_30_hi": finals[("D", a, "both")][2]})
    return time_rows, final_rows


def main():
    d = out_dir()
    cells = cells_table()
    main_runs = cells[cells.purpose == "main"]
    silent = main_runs[main_runs.arm == "silent"].set_index(["variant", "setup", "q", "rho"]).path
    jobs = []
    for key, g in main_runs[main_runs.arm != "silent"].groupby(GROUP_KEYS):
        arms = g.set_index("arm").path
        v, s, q, qc, b, rho = key
        jobs.append((key, {"truth": arms["truth"], "false": arms["false"], "silent": silent[(v, s, q, rho)]}))
    print(f"{len(jobs)} groups (truth + false + silent twin)", flush=True)
    time_rows, final_rows = [], []
    with multiprocessing.Pool(12) as pool:
        for i, (t, f) in enumerate(pool.imap_unordered(group, jobs, chunksize=4), 1):
            time_rows += t
            final_rows += f
            if i % 120 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    tt, ff = pd.DataFrame(time_rows), pd.DataFrame(final_rows)
    # workers finish in any order: sort so the files are identical from run to run
    tt = tt.sort_values(GROUP_KEYS + ["arm", "quantity", "a", "day"], na_position="first").reset_index(drop=True)
    ff = ff.sort_values(GROUP_KEYS + ["arm", "target"]).reset_index(drop=True)
    tt.to_parquet(d / "tables" / "q2_time.parquet", index=False)
    ff.to_csv(d / "tables" / "q2_final.csv", index=False, float_format="%.5f")
    write_index(d, [
        {"question": "Q2", "file": "tables/q2_time.parquet",
         "what": "per day: paired gain G (a=0,1,2) per arm, M and D, dose, silent shares; 95% intervals", "cells": "main runs"},
        {"question": "Q2", "file": "tables/q2_final.csv",
         "what": "day-30 own gain, dose, efficiency, time to 75%, unanimity; M and D per allocation", "cells": "main runs"},
    ])
    print(f"wrote {len(tt):,} time rows and {len(ff):,} final rows")


if __name__ == "__main__":
    main()
