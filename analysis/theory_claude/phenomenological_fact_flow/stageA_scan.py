"""Stage A scans of the fact-flow model V1 (z = 0): does the reading reversal exist at P0, does it depend on the
mechanism stated in advance (F_C << F_A), and how robust is it to each parameter (x0.5, x2, one at a time)?

Reversal (for one reading step q1 -> q2) = D > +tol at the smallest rho and D < -tol at some larger rho, with
tol = 1e-4 (and also reported relative to max |G|). Crossover = first rho where D changes sign from + to -.
Writes results/stageA_scan.csv and prints a table.

Usage, from analysis/theory_claude/phenomenological_fact_flow:
    ../../../.venv/bin/python stageA_scan.py
"""
from __future__ import annotations
import pathlib
from dataclasses import replace

import numpy as np
import pandas as pd

from fact_flow import P, G_at_T

HERE = pathlib.Path(__file__).resolve().parent
RHOS = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
TOL = 1e-4


def summarize(p, label):
    G = {q: np.array([G_at_T(p, q, r)[0] for r in RHOS]) for q in (3, 6, 9)}
    row = dict(label=label, Gmax_q6=float(np.max(np.abs(G[6]))))
    for q1, q2 in ((3, 6), (6, 9)):
        D = G[q2] - G[q1]
        pos_low = D[0] > TOL
        neg_high = bool(np.any(D[1:] < -TOL))
        cross = next((f"{RHOS[i]:g}-{RHOS[i+1]:g}" for i in range(len(D) - 1) if D[i] > 0 >= D[i + 1]), "none")
        row[f"rev_{q1}{q2}"] = bool(pos_low and neg_high)
        row[f"cross_{q1}{q2}"] = cross
        row[f"D{q1}{q2}_low"] = float(D[0])
        row[f"D{q1}{q2}_min"] = float(D.min())
    return row


def main():
    p0 = P()
    rows = [summarize(p0, "P0")]
    rows.append(summarize(replace(p0, F_C=16.0), "F_C = F_A = 16"))
    rows.append(summarize(replace(p0, F_A=4.0), "F_A = F_C = 4"))
    rows.append(summarize(replace(p0, selective=False), "neutral reposting"))
    for name in ("N_over_W", "F_A", "F_C", "beta_A", "beta_C", "kA", "kC", "k0", "lam0"):
        for f in (0.5, 2.0):
            rows.append(summarize(replace(p0, **{name: getattr(p0, name) * f}), f"{name} x{f:g}"))
    for h in (np.log(0.25), np.log(1.0)):
        rows.append(summarize(replace(p0, h0=float(h)), f"h0 = ln {np.exp(h):g}"))
    r = pd.DataFrame(rows)
    (HERE / "results").mkdir(exist_ok=True)
    r.to_csv(HERE / "results" / "stageA_scan.csv", index=False)
    pd.set_option("display.width", 220)
    print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
