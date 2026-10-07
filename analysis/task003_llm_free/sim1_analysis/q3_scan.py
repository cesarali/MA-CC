"""Q3 — non-monotonicity in b and qc (ANALYSIS_PLAN.md §4, criteria §6).

Scans every line of cells that differ in one budget only:
  along b  (1, 2, 3, 6, 9)  at fixed variant, setup, arm, q, qc, rho  -> 384 lines
  along qc (3, 6, 12, 24)   at fixed variant, setup, arm, q, b, rho   -> 480 lines
for two day-30 outcomes: the paired gain in the controller's own target, and the
paired gain in A1 (the answer nobody is pushed toward: "confusion").

Steps between neighbouring cells are paired within episodes (cells share their
episode draws), so a step is mean_e[x(cell_k+1, e) - x(cell_k, e)], with a 95%
bootstrap interval (1,000 episode resamples). A step is "up" or "down" when its
interval excludes 0. A line is NON-MONOTONE when it has a significant up and a
significant down (criterion 1).

Each non-monotone line is then annotated with:
  neighbour support (criterion 2): the same reversal, at the same position, at an
      adjacent q or the other rho, all else fixed;
  dose (criterion 3, along b): whether the delivered dose rises along the whole
      line (if not, the "reversal" may be a dose artefact);
  gate (criterion 3, along qc): whether the reversal involves qc = 3 or 6, where
      the effective gate is stricter.
  size (§6, "large enough to matter"): the significant rise and the significant fall
      must each move the gain by at least one agent (1/24 = 0.042).
Criterion 4 (seed-2027 confirmation) is a separate run.

Outputs: tables/q3_lines.csv (every line), tables/q3_nonmonotone.csv.
From the repository root:
    .venv/bin/python analysis/task003_llm_free/sim1_analysis/q3_scan.py
"""
from __future__ import annotations
import hashlib, multiprocessing, pathlib, sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import cells_table, out_dir, write_index  # noqa: E402

N_BOOT = 1000
TARGET = {"truth": 0, "false": 2}
BS, QCS = (1, 2, 3, 6, 9), (3, 6, 12, 24)


def final_shares(path: str) -> np.ndarray:
    d = pd.read_parquet(pathlib.Path(path) / "days.parquet", columns=["episode", "day", "share_A0", "share_A1", "share_A2"])
    last = d[d.day == d.day.max()].sort_values("episode")
    return last[["share_A0", "share_A1", "share_A2"]].to_numpy()          # (E, 3)


def final_dose(path: str) -> float:
    n = pd.read_parquet(pathlib.Path(path) / "nights.parquet", columns=["episode", "n_posted"])
    return n.n_posted.sum() / n.episode.nunique()


def scan_line(job):
    key, axis, values, paths, silent_path = job
    variant, setup, arm, q, fixed, rho = key
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(repr((key, axis)).encode()).digest()[:8], "big"))
    S = final_shares(silent_path)
    X = [final_shares(p) for p in paths]
    E = S.shape[0]
    idx = rng.integers(0, E, size=(N_BOOT, E))
    doses = [final_dose(p) for p in paths]
    rows = []
    for outcome, a in (("own_target", TARGET[arm]), ("A1", 1)):
        G = [x[:, a] - S[:, a] for x in X]
        gains = [g.mean() for g in G]
        steps, sizes = [], []
        for k in range(len(values) - 1):
            diff = X[k + 1][:, a] - X[k][:, a]
            b = diff[idx].mean(axis=1)
            lo, hi = np.percentile(b, [2.5, 97.5])
            steps.append("up" if lo > 0 else "down" if hi < 0 else "flat")
            sizes.append(diff.mean())
        ups = [k for k, s in enumerate(steps) if s == "up"]
        downs = [k for k, s in enumerate(steps) if s == "down"]
        if ups and downs:
            pattern = "peak" if min(ups) < max(downs) and max(ups) < max(downs) else \
                      "valley" if min(downs) < max(ups) and max(downs) < max(ups) else "zigzag"
        elif ups:
            pattern = "rising"
        elif downs:
            pattern = "falling"
        else:
            pattern = "flat"
        rows.append({"variant": variant, "setup": setup, "arm": arm, "q": q, "rho": rho, "axis": axis,
                     "fixed": ("qc" if axis == "b" else "b") + f"={fixed}", "outcome": outcome,
                     "values": "/".join(map(str, values)), "gains": "/".join(f"{g:.3f}" for g in gains),
                     "steps": "/".join(steps), "pattern": pattern,
                     "doses": "/".join(f"{x:.1f}" for x in doses),
                     "dose_rises_throughout": bool(all(np.diff(doses) > 0)),
                     "largest_significant_rise": max([sz for sz, st in zip(sizes, steps) if st == "up"], default=0.0),
                     "largest_significant_fall": min([sz for sz, st in zip(sizes, steps) if st == "down"], default=0.0),
                     "reversal_at": "/".join(f"{values[k]}->{values[k+1]}:{steps[k]}" for k in range(len(steps))
                                             if steps[k] != "flat")})
    return rows


def main():
    d = out_dir()
    cells = cells_table()
    main_runs = cells[(cells.stopping_rule == "stop_when_proved")]
    silent = main_runs[main_runs.arm == "silent"].set_index(["variant", "setup", "q", "rho"]).path
    ctl = main_runs[main_runs.arm != "silent"].set_index(["variant", "setup", "arm", "q", "qc", "b", "rho"]).path
    jobs = []
    for (variant, setup, arm, q, rho) in ctl.index.droplevel(["qc", "b"]).unique():
        sp = silent[(variant, setup, q, rho)]
        for qc in QCS:
            jobs.append(((variant, setup, arm, q, qc, rho), "b", BS,
                         [ctl[(variant, setup, arm, q, qc, b, rho)] for b in BS], sp))
        for b in BS:
            jobs.append(((variant, setup, arm, q, b, rho), "qc", QCS,
                         [ctl[(variant, setup, arm, q, qc, b, rho)] for qc in QCS], sp))
    print(f"{len(jobs)} lines", flush=True)
    with multiprocessing.Pool(12) as pool:
        rows = [r for res in pool.map(scan_line, jobs) for r in res]
    L = pd.DataFrame(rows).sort_values(["outcome", "axis", "variant", "setup", "arm", "fixed", "q", "rho"]).reset_index(drop=True)

    # criterion 2: the same pattern AND the same step signature at an adjacent q or the other rho
    sig = L.set_index(["outcome", "axis", "variant", "setup", "arm", "fixed", "q", "rho"])
    def neighbours(r):
        out = []
        for q2 in (3, 6, 12):
            if abs((3, 6, 12).index(q2) - (3, 6, 12).index(r.q)) == 1:
                out.append((r.outcome, r.axis, r.variant, r.setup, r.arm, r.fixed, q2, r.rho))
        out.append((r.outcome, r.axis, r.variant, r.setup, r.arm, r.fixed, r.q, 1.0 if r.rho == 0.75 else 0.75))
        return [n for n in out if n in sig.index and sig.loc[n].pattern == r.pattern
                and sig.loc[n].reversal_at == r.reversal_at]
    nm = L[L.pattern.isin(["peak", "valley", "zigzag"])].copy()
    nm["neighbour_support"] = [len(neighbours(r)) for r in nm.itertuples()]
    nm["gate_involved"] = (nm.axis == "qc") & nm.reversal_at.str.contains(r"(?:^|/)(?:3->6|6->12)")
    nm["material"] = (nm.largest_significant_rise >= 1 / 24) & (nm.largest_significant_fall <= -1 / 24)
    nm["passes_criteria_1_to_3"] = (nm.neighbour_support > 0) & ~nm.gate_involved & \
                                   ((nm.axis == "qc") | nm.dose_rises_throughout) & nm.material
    L.to_csv(d / "tables" / "q3_lines.csv", index=False)
    nm.to_csv(d / "tables" / "q3_nonmonotone.csv", index=False)
    write_index(d, [
        {"question": "Q3", "file": "tables/q3_lines.csv", "what": "every b-line and qc-line: gains, paired steps, pattern, doses", "cells": "main runs"},
        {"question": "Q3", "file": "tables/q3_nonmonotone.csv", "what": "non-monotone lines with criteria 2-3 annotations", "cells": "main runs"},
    ])
    print(L.groupby(["outcome", "axis", "pattern"]).size().unstack(fill_value=0).to_string())
    print(f"\nnon-monotone lines: {len(nm)}; with rise and fall each >= 1 agent: {int(nm.material.sum())}; "
          f"passing criteria 1-3 including size: {int(nm.passes_criteria_1_to_3.sum())}")
    big_fall = L[(L.outcome == "own_target") & (L.axis == "b") & (L.largest_significant_fall <= -1 / 24)]
    print(f"b-lines with a significant fall of at least one agent: {len(big_fall)} "
          f"(by setup/arm: {big_fall.groupby(['setup', 'arm']).size().to_dict()})")


if __name__ == "__main__":
    main()
