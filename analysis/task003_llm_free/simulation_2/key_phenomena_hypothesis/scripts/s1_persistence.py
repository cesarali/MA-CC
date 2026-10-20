"""Step 1.4: how long the false controller's effect persists after its last post, from existing light records.

For each setting (setup, case, rho, rate; q = 6, qc = 12, gate on) and episode e:
  t*_e          time of the controller's last post (controller.parquet; no post -> episode left out)
  delta_e(t)    x_A2(e, false arm, t) - x_A2(e, silent arm, t), on the 0.25 grid up to t = 40
Fixed clock: delta(30), delta(40) with paired standard errors.
Aligned to the last post: delta(s) = mean_e delta_e(t*_e + s) over the episodes still observed at t*_e + s <= 40
(right-censoring; n(s) is reported), s = 0, 0.25, ...; the silent arm gets the same clock shift.
  half_time     first s with delta(s) <= delta(0) / 2, among s with n(s) >= 100 (else censored: '> s_max')
  ratio_s10/s20/s30  delta(s) / delta(0) at s = 10, 20, 30 (where n(s) >= 100)
Reference times: an unread fact's half-life ln2 / (-ln rho); a post stays readable about W / 24 time units.
Writes results/key_phenomena/persistence.parquet (one row per setting) and persistence_curves.parquet (delta(s)).

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/scripts/s1_persistence.py
"""
from __future__ import annotations
import math, multiprocessing, pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = ROOT / "results"
OUT = RES / "key_phenomena"
STEP, T_END, N_MIN = 0.25, 40.0, 100
RUNS = ["2026-10-15_sim2_symmetric_fine", "2026-10-16_sim2_symmetric_fine_extra_rho", "2026-10-16_sim2_nosolution_fine",
        "2026-10-18_sim2_symmetric_budget100", "2026-10-18_sim2_symmetric_window24",
        "2026-10-18_sim2_symmetric_window240", "2026-10-18_sim2_nosolution_budget100",
        "2026-10-18_sim2_nosolution_window24", "2026-10-18_sim2_nosolution_window240"]


def case_of(r):
    return "budget100" if r.budget != 30 else (f"window{int(r.W)}" if r.W != 48 else "standard")


def x2(folder):
    s = pd.read_parquet(folder / "snapshots.parquet", columns=["episode", "grid_index", "share_A2", "share_voted"])
    s["x2"] = np.where(s.share_voted > 0, s.share_A2 / s.share_voted.clip(lower=1e-12), np.nan)
    return s.pivot(index="episode", columns="grid_index", values="x2").sort_index()


def run(job):
    key, folder, silent, W = job
    folder, silent = pathlib.Path(folder), pathlib.Path(silent)
    c = pd.read_parquet(folder / "controller.parquet", columns=["episode", "t", "post_id"])
    tstar = c[c.post_id >= 0].groupby("episode").t.max()
    d = (x2(folder) - x2(silent)).loc[tstar.index].to_numpy()          # episodes x grid (0, 0.25, ..., 40)
    n_grid = d.shape[1]
    k0 = np.ceil(tstar.to_numpy() / STEP).astype(int)                     # first grid point at or after t*
    rows_curve, smax = [], int(T_END / STEP)
    for j in range(smax + 1):
        idx = k0 + j
        ok = idx < n_grid
        if ok.sum() < 1:
            break
        v = d[np.nonzero(ok)[0], idx[ok]]
        v = v[~np.isnan(v)]
        rows_curve.append((*key, j * STEP, v.mean(), v.std() / math.sqrt(len(v)), len(v)))
    cur = pd.DataFrame(rows_curve, columns=["setup", "case", "rho", "lambda_c", "s", "delta", "se", "n"])
    good = cur[cur.n >= N_MIN]
    d0 = good.delta.iloc[0]
    below = good[good.delta <= d0 / 2]
    half = below.s.iloc[0] if len(below) else np.nan
    smax_obs = good.s.iloc[-1]
    at = lambda s: good.loc[good.s == s, "delta"].iloc[0] / d0 if (good.s == s).any() and d0 > 0 else np.nan  # noqa: E731
    fixed = {tt: d[:, int(tt / STEP)] for tt in (30.0, 40.0)}
    rho = key[2]
    row = dict(zip(["setup", "case", "rho", "lambda_c"], key), episodes_with_post=len(tstar),
               tstar_mean=tstar.mean(), tstar_median=tstar.median(), delta_s0=d0, half_time=half,
               half_censored=bool(np.isnan(half)), s_max_observed=smax_obs, ratio_s10=at(10.0), ratio_s20=at(20.0),
               ratio_s30=at(30.0),
               delta30=np.nanmean(fixed[30.0]), delta30_se=np.nanstd(fixed[30.0]) / math.sqrt(len(tstar)),
               delta40=np.nanmean(fixed[40.0]), delta40_se=np.nanstd(fixed[40.0]) / math.sqrt(len(tstar)),
               d40_minus_d30=np.nanmean(fixed[40.0] - fixed[30.0]),
               d40_minus_d30_se=np.nanstd(fixed[40.0] - fixed[30.0]) / math.sqrt(len(tstar)),
               fact_half_life=(math.log(2) / -math.log(rho)) if rho < 1 else np.inf, board_life=W / 24)
    return row, cur


def main():
    inv = pd.read_csv(OUT / "inventory_cells.csv")
    inv = inv[inv.run.isin(RUNS)].copy()
    inv["case"] = inv.apply(case_of, axis=1)
    inv["folder"] = [str(RES / "simulation_2" / r / "cells" / c) for r, c in zip(inv.run, inv.cell)]
    sil = {(r.run, r.q, r.rho): r.folder for r in inv[inv.arm == "silent"].itertuples()}
    fa = inv[(inv.arm == "false") & (inv.q == 6) & (inv.qc == 12) & (inv.gate == "on")]
    jobs = [((r.setup, r.case, r.rho, r.lambda_c), r.folder, sil[(r.run, 6, r.rho)], r.W) for r in fa.itertuples()]
    with multiprocessing.Pool(12) as pool:
        parts = pool.map(run, jobs, chunksize=2)
    pd.DataFrame([p[0] for p in parts]).to_parquet(OUT / "persistence.parquet", index=False)
    pd.concat([p[1] for p in parts], ignore_index=True).to_parquet(OUT / "persistence_curves.parquet", index=False)
    print(len(parts), "settings ->", OUT / "persistence.parquet")


if __name__ == "__main__":
    main()
