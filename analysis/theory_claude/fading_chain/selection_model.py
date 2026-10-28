"""Step 2 (pre-registered in ../../theory_blackboard/2026-10-28_claude_13.md): coherence-dependent selection of
controller facts by support-based reposting. Engine-free mean field; the fact-flow model with each controller fact
tracked separately.

  dm_A/dt = (1 - m_A) r_A - gamma m_A,           r_A = 1 - (1 - b_A / F_A)^q
  dm_f/dt = (1 - m_f) r_f - gamma m_f,           r_f = 1 - (1 - b_f)^q                       (f = 1..F_C)
  db_A/dt = (N/W)(S_A - b_A) - (lam/W) b_A
  db_f/dt = (N/W)(S_f - b_f) + (lam/W)(1/F_C - b_f)
  x = h0 - beta_A F_A m_A + beta_C sum_f m_f (2 c_f - 1),   Phi2 = sigma(x)
  g_f = eps + (1 - eps)[c_f sigma(x + beta_C) + (1 - c_f)(1 - sigma(x - beta_C))]
  g_A = eps + (1 - eps)(1 - Phi2)
  w_f = kC m_f g_f,  w_A = kA F_A m_A g_A,  S_f = w_f / (w_A + sum w + k0),  S_A = w_A / (...)
Model susceptibility (add-one, encountered mixture):
  S = sum_f (b_f / b_C)(1 - m_f)[c_f (sigma(x + beta_C) - sigma(x)) + (1 - c_f)(sigma(x - beta_C) - sigma(x))]
Writes selection_grid.csv, selection_paths.csv.

Usage, from the repository root: .venv/bin/python analysis/theory_claude/fading_chain/selection_model.py
"""
from __future__ import annotations
import pathlib

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

HERE = pathlib.Path(__file__).resolve().parent
P0 = dict(N_over_W=0.5, F_A=16.0, beta_A=0.5, beta_C=0.5, h0=float(np.log(0.5)), kA=1.0, kC=1.0, k0=0.2, lam0=1.0,
          W=48.0, T=30.0)
SETS = {"coherent": np.array([0.97, 0.97, 0.97, 0.97]), "ambivalent": np.array([0.9, 0.7, 0.5, 0.3])}
TIMES = np.array([30.0, 40.0, 80.0, 130.0])


def sigma(x):
    return 1.0 / (1.0 + np.exp(-x))


def parts(y, c, p, eps):
    F = len(c)
    mA, mf, bA, bf = y[0], y[1:1 + F], y[1 + F], y[2 + F:2 + 2 * F]
    x = p["h0"] - p["beta_A"] * p["F_A"] * mA + p["beta_C"] * np.sum(mf * (2 * c - 1))
    bC = p["beta_C"]
    gf = eps + (1 - eps) * (c * sigma(x + bC) + (1 - c) * (1 - sigma(x - bC)))
    gA = eps + (1 - eps) * (1 - sigma(x))
    wf, wA = p["kC"] * mf * gf, p["kA"] * p["F_A"] * mA * gA
    tot = wA + wf.sum() + p["k0"]
    return mA, mf, bA, bf, x, wA / tot, wf / tot


def rhs(t, y, c, p, eps, q, gamma, controlled):
    mA, mf, bA, bf, x, SA, Sf = parts(y, c, p, eps)
    F = len(c)
    lam = p["lam0"] if (controlled and t < p["T"]) else 0.0
    rA = 1 - (1 - np.clip(bA, 0, 1) / p["F_A"]) ** q
    rf = 1 - (1 - np.clip(bf, 0, 1)) ** q
    dmA = (1 - mA) * rA - gamma * mA
    dmf = (1 - mf) * rf - gamma * mf
    nw = p["N_over_W"]
    dbA = nw * (SA - bA) - lam / p["W"] * bA
    dbf = nw * (Sf - bf) + lam / p["W"] * (1.0 / F - bf)
    if not controlled:
        dmf, dbf = 0 * dmf, 0 * dbf
    return np.concatenate([[dmA], dmf, [dbA], dbf])


def solve(c, p, eps, q, rho, controlled):
    F = len(c)
    y0 = np.concatenate([[1 / p["F_A"]], np.zeros(F), [0.0], np.zeros(F)])
    sol = solve_ivp(rhs, (0, TIMES[-1]), y0, args=(c, p, eps, q, -np.log(rho), controlled), t_eval=TIMES,
                    rtol=1e-9, atol=1e-12, max_step=0.25)
    return sol.y


def observables(yc, ys, c, p, eps):
    out = []
    for k, t in enumerate(TIMES):
        _, mf, _, bf, x, _, _ = parts(yc[:, k], c, p, eps)
        xs = parts(ys[:, k], c, p, eps)[4]
        bsum = bf.sum()
        wb = bf / bsum if bsum > 0 else np.zeros_like(bf)
        S = np.sum(wb * (1 - mf) * (c * (sigma(x + p["beta_C"]) - sigma(x)) + (1 - c) * (sigma(x - p["beta_C"]) - sigma(x))))
        out.append(dict(t=t, delta=sigma(x) - sigma(xs), b_C=bsum, cbar=float(np.sum(wb * c)), S=S, Phi2_c=sigma(x),
                        M_C=mf.sum()))
    return out


def main():
    rows, paths = [], []
    for h0name, h0 in (("prior_1/3", float(np.log(0.5))), ("majority_A2", float(np.log(2.0)))):
        p = dict(P0, h0=h0)
        for name, c in SETS.items():
            for eps in (1.0, 0.75, 0.5):
                for rho in (0.25, 0.75):
                    yc, ys = solve(c, p, eps, 6, rho, True), solve(c, p, eps, 6, rho, False)
                    ob = observables(yc, ys, c, p, eps)
                    for o in ob:
                        paths.append(dict(h0=h0name, set=name, eps=eps, rho=rho, **o))
                    a, b = ob[0], ob[-1]
                    rows.append(dict(h0=h0name, set=name, eps=eps, rho=rho, delta30=a["delta"], delta130=b["delta"],
                                     ret_delta=b["delta"] / a["delta"] if a["delta"] else np.nan,
                                     ret_bC=b["b_C"] / a["b_C"] if a["b_C"] else np.nan, cbar30=a["cbar"],
                                     cbar130=b["cbar"], ret_S=b["S"] / a["S"] if a["S"] else np.nan,
                                     Phi2_c30=a["Phi2_c"]))
    g = pd.DataFrame(rows)
    g["fading_beyond_exposure"] = g.ret_bC - g.ret_delta
    g.to_csv(HERE / "selection_grid.csv", index=False)
    pd.DataFrame(paths).to_csv(HERE / "selection_paths.csv", index=False)
    pd.set_option("display.width", 250)
    print(g.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
