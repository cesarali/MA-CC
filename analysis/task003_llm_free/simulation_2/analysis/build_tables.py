"""Build the analysis tables of plan §2-§3 for one Simulation 2 grid run.

Outputs (results/simulation_2/analysis/<date>/tables/):
  cells.csv            one row per cell, reused cells included
  episode_values.parquet  one row per cell folder and episode
  gains.csv            per controlled cell: G(30), G(40), persistence, average gain, vote gains,
                       messages, efficiency, budget use; 95% intervals and verdicts (plan §6)
  parts.csv            shared and directional parts M, D (truth and false of the same settings)
  gate_effect.csv      gate off - gate on, same episode
  stop_effect.csv      proof stop off - on, symmetric truth
  rate_steps.csv       G(30) at each rate minus the next lower rate, same episode (plan Q3)

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/analysis/build_tables.py [--run <run folder>] [--tag <name>]
"""
from __future__ import annotations
import argparse, pathlib

import numpy as np
import pandas as pd

from common import (SIZE, TARGET, boot_index, cells_table, episode_values, mean_ci, out_dir, ratio_ci,
                    run_dir, verdict, write_index)

KEYS = ["setup", "arm", "q", "qc", "rho", "lambda_c", "gate", "stop"]


def arrays(ev: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {f: g.sort_values("episode").reset_index(drop=True) for f, g in ev.groupby("folder")}


def build(run: pathlib.Path, tag: str, d: pathlib.Path):
    cells = cells_table(run)
    cells.to_csv(d / "tables" / f"{tag}cells.csv", index=False)
    ev = episode_values(run, cells.folder)
    ev.to_parquet(d / "tables" / f"{tag}episode_values.parquet", index=False)
    V = arrays(ev)
    n = len(next(iter(V.values())))
    assert all(len(v) == n for v in V.values()), "every cell must have the same episodes"
    idx = boot_index(n, run.name)
    folder_of = dict(zip(cells.cell, cells.folder))
    ctrl = cells[cells.arm != "silent"].copy()

    rows = []
    for c in ctrl.itertuples():
        k = TARGET[c.arm]
        x, s = V[c.folder], V[folder_of[c.silent_cell]]
        r = {key: getattr(c, key) for key in KEYS}
        r.update(cell=c.cell, folder=c.folder, silent_cell=c.silent_cell)
        d30 = (x[f"mean_p_A{k}_t30"] - s[f"mean_p_A{k}_t30"]).to_numpy()
        d40 = (x[f"mean_p_A{k}_t40"] - s[f"mean_p_A{k}_t40"]).to_numpy()
        for name, v in (("G30", d30), ("G40", d40), ("persistence", d40 - d30),
                        ("avg_gain", (x[f"avg_p_A{k}_0_30"] - s[f"avg_p_A{k}_0_30"]).to_numpy()),
                        ("vote_gain30", (x[f"share_A{k}_t30"] - s[f"share_A{k}_t30"]).to_numpy()),
                        ("vote_gain40", (x[f"share_A{k}_t40"] - s[f"share_A{k}_t40"]).to_numpy()),
                        ("messages", x.controller_messages.to_numpy(float)),
                        ("reads", x.controller_reads.to_numpy(float))):
            r[name], r[f"{name}_lo"], r[f"{name}_hi"] = mean_ci(v, idx)
        r["efficiency"], r["efficiency_lo"], r["efficiency_hi"] = ratio_ci(24 * d30, x.controller_messages.to_numpy(float), idx)
        r["m_target30_controlled"] = x[f"mean_p_A{k}_t30"].mean()
        r["m_target30_silent"] = s[f"mean_p_A{k}_t30"].mean()
        ex = x.budget_exhausted_at
        r["share_budget_spent"] = ex.notna().mean()
        r["median_time_budget_spent"] = ex.median() if ex.notna().any() else np.nan
        r["G30_verdict"] = verdict(r["G30"], r["G30_lo"], r["G30_hi"])
        r["persistence_verdict"] = verdict(r["persistence"], r["persistence_lo"], r["persistence_hi"])
        rows.append(r)
    gains = pd.DataFrame(rows)
    gains.to_csv(d / "tables" / f"{tag}gains.csv", index=False)

    # shared and directional parts: truth and false at the same settings (stop on)
    parts = []
    base = ctrl[ctrl.stop == "on"]
    for key, g in base.groupby(["setup", "q", "qc", "rho", "lambda_c", "gate"]):
        if set(g.arm) != {"truth", "false"}:
            continue
        t, f = (V[g[g.arm == a].folder.iloc[0]] for a in ("truth", "false"))
        s = V[folder_of[g.silent_cell.iloc[0]]]
        for a in (0, 2):
            gt = (t[f"mean_p_A{a}_t30"] - s[f"mean_p_A{a}_t30"]).to_numpy()
            gf = (f[f"mean_p_A{a}_t30"] - s[f"mean_p_A{a}_t30"]).to_numpy()
            r = dict(zip(["setup", "q", "qc", "rho", "lambda_c", "gate"], key), belief=f"A{a}")
            r["M"], r["M_lo"], r["M_hi"] = mean_ci((gt + gf) / 2, idx)
            r["D"], r["D_lo"], r["D_hi"] = mean_ci((gt - gf) / 2, idx)
            parts.append(r)
    pd.DataFrame(parts).to_csv(d / "tables" / f"{tag}parts.csv", index=False)

    def contrast(by: str, a_val, b_val, extra_filter=None):
        """(cell with by=a_val) - (cell with by=b_val), everything else equal: G(30) and messages."""
        out = []
        sel = ctrl if extra_filter is None else ctrl[extra_filter(ctrl)]
        others = [k for k in KEYS if k != by]
        for key, g in sel.groupby(others):
            if set(g[by]) < {a_val, b_val}:
                continue
            ca, cb = (g[g[by] == v].iloc[0] for v in (a_val, b_val))
            k = TARGET[ca.arm]
            xa, xb = V[ca.folder], V[cb.folder]
            r = dict(zip(others, key))
            r["effect_G30"], r["effect_G30_lo"], r["effect_G30_hi"] = mean_ci(
                (xa[f"mean_p_A{k}_t30"] - xb[f"mean_p_A{k}_t30"]).to_numpy(), idx)
            r["effect_messages"], r["effect_messages_lo"], r["effect_messages_hi"] = mean_ci(
                (xa.controller_messages - xb.controller_messages).to_numpy(float), idx)
            r["verdict"] = verdict(r["effect_G30"], r["effect_G30_lo"], r["effect_G30_hi"])
            out.append(r)
        return pd.DataFrame(out)

    contrast("gate", "off", "on").to_csv(d / "tables" / f"{tag}gate_effect.csv", index=False)
    contrast("stop", "off", "on",
             lambda c: (c.setup == "task003_symmetric") & (c.arm == "truth")).to_csv(
        d / "tables" / f"{tag}stop_effect.csv", index=False)

    # rate steps: G(30) at each rate minus the next lower rate
    steps = []
    others = [k for k in KEYS if k != "lambda_c"]
    for key, g in ctrl.groupby(others):
        g = g.sort_values("lambda_c")
        k = TARGET[g.arm.iloc[0]]
        for lo_c, hi_c in zip(g.itertuples(), list(g.itertuples())[1:]):
            diff = (V[hi_c.folder][f"mean_p_A{k}_t30"] - V[lo_c.folder][f"mean_p_A{k}_t30"]).to_numpy()
            r = dict(zip(others, key), rate_from=lo_c.lambda_c, rate_to=hi_c.lambda_c)
            r["step_G30"], r["step_G30_lo"], r["step_G30_hi"] = mean_ci(diff, idx)
            r["reversal"] = r["step_G30_hi"] < 0                       # higher rate significantly worse
            r["reversal_size_ok"] = r["reversal"] and r["step_G30"] <= -SIZE
            steps.append(r)
    pd.DataFrame(steps).to_csv(d / "tables" / f"{tag}rate_steps.csv", index=False)

    write_index(d, [
        {"file": f"tables/{tag}{f}", "question": q, "what": w} for f, q, w in [
            ("cells.csv", "all", "one row per cell"),
            ("episode_values.parquet", "all", "per cell and episode values"),
            ("gains.csv", "Q2-Q4", "G(30), G(40), persistence, dose, efficiency with intervals"),
            ("parts.csv", "Q2", "shared and directional parts"),
            ("gate_effect.csv", "Q5", "gate off - on"),
            ("stop_effect.csv", "Q6", "stop off - on, symmetric truth"),
            ("rate_steps.csv", "Q3", "G(30) between neighbouring rates")]])
    print(f"{len(cells)} cells, {len(gains)} controlled; tables in {d / 'tables'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run folder (default: the latest official sim2_grid run)")
    ap.add_argument("--tag", default="", help="prefix for the output tables, e.g. confirm_")
    ap.add_argument("--date", help="analysis folder date (default today)")
    a = ap.parse_args()
    run = pathlib.Path(a.run).resolve() if a.run else run_dir("sim2_grid")
    build(run, a.tag, out_dir(a.date))


if __name__ == "__main__":
    main()
