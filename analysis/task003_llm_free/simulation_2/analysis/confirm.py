"""Decide hypotheses H1-H6 of the analysis plan (§5) on the seed-2027 confirmation runs.

Needs: build_tables.py run on the confirmation grid with --tag confirm_ (the 52 main cells),
and the S0 confirmation run (sim2_s0_confirm_seed2027).

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/analysis/confirm.py [--date <date>]
Output: tables/confirm_hypotheses.csv; the verdicts printed.
"""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd

from common import SIZE, TARGET, boot_index, episode_values, mean_ci, out_dir, run_dir, verdict, write_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    a = ap.parse_args()
    d = out_dir(a.date)
    T = d / "tables"
    gains = pd.read_csv(T / "confirm_gains.csv")
    gate = pd.read_csv(T / "confirm_gate_effect.csv")
    cells = pd.read_csv(T / "confirm_cells.csv")
    ev = pd.read_parquet(T / "confirm_episode_values.parquet")
    V = {f: g.sort_values("episode").reset_index(drop=True) for f, g in ev.groupby("folder")}
    s0_run = run_dir("sim2_s0_confirm_seed2027")
    s0 = episode_values(s0_run, [p.name for p in (s0_run / "cells").iterdir() if p.is_dir()])
    S0 = {f: g.sort_values("episode").reset_index(drop=True) for f, g in s0.groupby("folder")}
    n = len(next(iter(V.values())))
    idx = boot_index(n, "confirm_hypotheses")
    rows = []

    def add(h, what, m, lo, hi, supported, note=""):
        rows.append({"hypothesis": h, "comparison": what, "estimate": m, "lo": lo, "hi": hi,
                     "verdict": verdict(m, lo, hi), "supported": bool(supported), "note": note})

    # H1: persistence within ±size in every main cell
    for r in gains.itertuples():
        ok = -SIZE <= r.persistence_lo and r.persistence_hi <= SIZE
        add("H1", f"G(40) - G(30): {r.cell}", r.persistence, r.persistence_lo, r.persistence_hi, ok)

    # H2-H4: core (rate 1, gate on, stop on) - S0, same seed
    core = cells[(cells.lambda_c == 1.0) & (cells.gate == "on") & (cells.stop == "on")]
    folder_of = dict(zip(cells.cell, cells.folder))

    def core_gain(setup, arm):
        c = core[(core.setup == setup) & (core.arm == arm)].iloc[0]
        k = TARGET[arm]
        x, s = V[c.folder], V[folder_of[c.silent_cell]]
        return (x[f"mean_p_A{k}_t30"] - s[f"mean_p_A{k}_t30"]).to_numpy()

    def s0_gain(setup, arm):
        k = TARGET[arm]
        x, s = S0[f"S0__{setup}__{arm}"], S0[f"S0__{setup}__silent"]
        return (x[f"mean_p_A{k}_t30"] - s[f"mean_p_A{k}_t30"]).to_numpy()

    for h, setup, arm in (("H2", "task003_nosolution", "truth"), ("H2", "task003_nosolution", "false"),
                          ("H3", "task003_symmetric", "false")):
        m, lo, hi = mean_ci(core_gain(setup, arm) - s0_gain(setup, arm), idx)
        add(h, f"G(30) core - S0: {setup} {arm}", m, lo, hi, m <= -SIZE and hi < 0)
    c = core[(core.setup == "task003_symmetric")].iloc[0]
    silent = V[folder_of[c.silent_cell]].mean_p_A0_t30.to_numpy()
    m, lo, hi = mean_ci(silent - S0["S0__task003_symmetric__silent"].mean_p_A0_t30.to_numpy(), idx)
    add("H4", "silent symmetric m_A0(30): core - S0", m, lo, hi, m <= -SIZE and hi < 0)

    # H5: gate off on nosolution at rate 1: messages rise, gain within ±size
    g1 = gate[(gate.lambda_c == 1.0) & (gate.stop == "on")]
    for r in g1[g1.setup == "task003_nosolution"].itertuples():
        ok = r.effect_messages_lo > 0 and -SIZE <= r.effect_G30_lo and r.effect_G30_hi <= SIZE
        add("H5", f"gate off - on, nosolution {r.arm}: G(30)", r.effect_G30, r.effect_G30_lo, r.effect_G30_hi, ok,
            note=f"messages {r.effect_messages:+.1f} [{r.effect_messages_lo:+.1f}, {r.effect_messages_hi:+.1f}]")
    # H6: gate off raises the symmetric truth gain
    r = g1[(g1.setup == "task003_symmetric") & (g1.arm == "truth")].iloc[0]
    add("H6", "gate off - on, symmetric truth: G(30)", r.effect_G30, r.effect_G30_lo, r.effect_G30_hi,
        r.effect_G30 >= SIZE and r.effect_G30_lo > 0,
        note=f"messages {r.effect_messages:+.1f} [{r.effect_messages_lo:+.1f}, {r.effect_messages_hi:+.1f}]")

    out = pd.DataFrame(rows)
    out.to_csv(T / "confirm_hypotheses.csv", index=False)
    write_index(d, [{"file": "tables/confirm_hypotheses.csv", "question": "H1-H6", "what": "hypotheses on seed 2027"}])
    summary = out.groupby("hypothesis").agg(rows=("supported", "size"), supported=("supported", "sum"))
    print(summary.to_string())
    print(out[out.hypothesis != "H1"].round(3).to_string(index=False))
    h1 = out[out.hypothesis == "H1"]
    print(f"\nH1: {h1.supported.sum()} of {len(h1)} cells within ±{SIZE:.3f}; largest |G(40)-G(30)| = {h1.estimate.abs().max():.3f}")


if __name__ == "__main__":
    main()
