"""Q3, criterion 4 — does the large-b fall replicate with new random draws (seed 2027)?

For every line in the confirmation runs (task003-nosolution; variant, arm, q, qc, rho;
b = 3, 6, 9), compute the own-target paired gain at day 30 for seed 2026 (main runs)
and seed 2027 (confirmation runs), and the step where the fall was found:
  truth: b 3 -> 6   (the pool has 4 facts on A0's side)
  false: b 6 -> 9   (the pool has 6 facts on A2's side)
A line replicates when the seed-2027 step is significantly negative (95% interval
below 0) and at least one agent in size (<= -1/24), like seed 2026.

Output: tables/q3_confirmation.csv.
From the repository root:
    .venv/bin/python analysis/task003_llm_free/sim1_analysis/q3_confirm.py
"""
from __future__ import annotations
import hashlib, pathlib, sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import cells_table, out_dir, write_index  # noqa: E402

TARGET = {"truth": 0, "false": 2}
STEP = {"truth": (3, 6), "false": (6, 9)}
N_BOOT = 1000


def final_shares(path):
    d = pd.read_parquet(pathlib.Path(path) / "days.parquet", columns=["episode", "day", "share_A0", "share_A1", "share_A2"])
    return d[d.day == d.day.max()].sort_values("episode")[["share_A0", "share_A1", "share_A2"]].to_numpy()


def main():
    d = out_dir()
    c = cells_table()
    c = c[c.setup == "task003_nosolution"]
    c = c[c.purpose.isin(["main", "confirmation"])]
    c["seed_run"] = np.where(c.purpose == "confirmation", 2027, 2026)
    conf = c[c.seed_run == 2027]
    rows = []
    for (variant, arm, q, qc, rho), g in conf[conf.arm != "silent"].groupby(["variant", "arm", "q", "qc", "rho"]):
        a = TARGET[arm]
        b0, b1 = STEP[arm]
        rec = {"variant": variant, "arm": arm, "q": q, "qc": qc, "rho": rho, "step": f"b {b0}->{b1}"}
        for seed in (2026, 2027):
            sub = c[(c.seed_run == seed) & (c.variant == variant) & (c.q == q) & (c.rho == rho)]
            sil = final_shares(sub[sub.arm == "silent"].path.iloc[0])
            x0 = final_shares(sub[(sub.arm == arm) & (sub.qc == qc) & (sub.b == b0)].path.iloc[0])
            x1 = final_shares(sub[(sub.arm == arm) & (sub.qc == qc) & (sub.b == b1)].path.iloc[0])
            diff = x1[:, a] - x0[:, a]
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(repr((variant, arm, q, qc, rho, seed)).encode()).digest()[:8], "big"))
            boots = diff[rng.integers(0, len(diff), size=(N_BOOT, len(diff)))].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rec.update({f"gain_b{b0}_seed{seed}": (x0[:, a] - sil[:, a]).mean(),
                        f"gain_b{b1}_seed{seed}": (x1[:, a] - sil[:, a]).mean(),
                        f"step_seed{seed}": diff.mean(), f"step_lo_seed{seed}": lo, f"step_hi_seed{seed}": hi})
        rec["fall_in_2026"] = rec["step_hi_seed2026"] < 0 and rec["step_seed2026"] <= -1 / 24
        rec["replicates_in_2027"] = rec["step_hi_seed2027"] < 0 and rec["step_seed2027"] <= -1 / 24
        rec["seeds_agree_within_intervals"] = not (rec["step_hi_seed2026"] < rec["step_lo_seed2027"]
                                                   or rec["step_hi_seed2027"] < rec["step_lo_seed2026"])
        rows.append(rec)
    t = pd.DataFrame(rows).sort_values(["variant", "arm", "rho", "q", "qc"])
    t.to_csv(d / "tables" / "q3_confirmation.csv", index=False, float_format="%.4f")
    write_index(d, [{"question": "Q3", "file": "tables/q3_confirmation.csv",
                     "what": "large-b fall: seed 2026 vs seed 2027, per line", "cells": "nosolution confirmation runs"}])
    f = t[t.fall_in_2026]
    print(f"lines with a material fall in seed 2026: {len(f)} of {len(t)}")
    print(f"  of those, replicated in seed 2027: {int(f.replicates_in_2027.sum())}")
    print(f"  seeds agree within their intervals: {int(t.seeds_agree_within_intervals.sum())} of {len(t)} lines (all lines)")
    print(f"lines with a material fall in seed 2027 but not 2026: {int((t.replicates_in_2027 & ~t.fall_in_2026).sum())}")
    print(t.groupby(["variant", "arm", "rho"])[["step_seed2026", "step_seed2027", "fall_in_2026", "replicates_in_2027"]]
          .agg({"step_seed2026": "mean", "step_seed2027": "mean", "fall_in_2026": "sum", "replicates_in_2027": "sum"})
          .round(3).to_string())


if __name__ == "__main__":
    main()
