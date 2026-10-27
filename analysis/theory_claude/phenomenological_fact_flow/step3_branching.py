"""Revised step 3: two-type (holder/board) branching approximation for the number of agent facts still circulating at T
(see decision_log.md). Background along the silent-arm mean-field trajectory. Compared with the finite model's silent
arm. Writes findings/step3_branching.csv."""
import pathlib
from dataclasses import replace
import numpy as np, pandas as pd
from scipy.integrate import solve_ivp
from fact_flow import P, run

HERE = pathlib.Path(__file__).resolve().parent
RHOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def feff(p, q, rho, n_init=(2,) * 8 + (1,) * 8):
    gamma = -np.log(rho)
    s = run(p, q, rho, False, t_eval=np.linspace(0, p.T, 601))
    ts, mA = s.t, s.y[0]
    MA = p.F_A * mA
    nu = p.N_over_W

    def rates(t):
        m = np.interp(t, ts, mA)
        M = np.interp(t, ts, MA)
        pi = p.kA / (p.kA * (1 + M) + p.k0)
        r = p.N_over_W * q * (1 - m)
        return pi, r

    def back(tau, u):            # tau = T - s, integrate forward in tau
        uH, uB = u
        pi, r = rates(p.T - tau)
        return [gamma * (1 - uH) + pi * (uH * uB - uH), nu * (1 - uB) + r * (uB * uH - uB)]

    sol = solve_ivp(back, (0, p.T), [0.0, 0.0], rtol=1e-9, atol=1e-11, max_step=0.1)
    uH0 = sol.y[0, -1]
    pi_T, r_T = rates(p.T)
    R_f = pi_T * r_T / (gamma * nu)
    return sum(1 - uH0 ** n for n in n_init), uH0, R_f


fin = pd.read_csv(HERE / "findings" / "step3_reversal.csv").set_index("rho")
rows = []
p = replace(P(), eps=1.0)
for rho in RHOS:
    F, u, R = feff(p, 6, rho)
    rows.append(dict(rho=rho, Feff_branching=F, extinction_one_holder=u, R_f_at_T=R,
                     Feff_finite_silent=fin.loc[rho, "distinct_A_silent"], Feff_finite_controlled=fin.loc[rho, "distinct_A_false"]))
r = pd.DataFrame(rows)
r["ratio"] = r.Feff_branching / r.Feff_finite_silent
r.to_csv(HERE / "findings" / "step3_branching.csv", index=False)
print(r.round(3).to_string(index=False))


# ---- revision 1: neutral drift at conserved total copies (diffusion approximation) ----
def feff_drift(p, q, rho):
    gamma = -np.log(rho)
    s = run(p, q, rho, False)
    mA, bA = s.y[0, -1], s.y[2, -1]
    W, N = p.W, p.N_over_W * p.W
    Ne = W * bA + N * p.F_A * mA
    mu = (W * bA * p.N_over_W + N * p.F_A * mA * gamma) / Ne
    F = p.F_A
    return F * (1 - np.exp(-2 * Ne / (F * mu * p.T))), Ne, mu


r["Feff_drift"], r["Ne"], r["mu"] = zip(*[feff_drift(p, 6, rho) for rho in RHOS])
r["ratio_drift"] = r.Feff_drift / r.Feff_finite_silent
r.to_csv(HERE / "findings" / "step3_branching.csv", index=False)
print(r[["rho", "Feff_drift", "Ne", "mu", "Feff_finite_silent", "ratio_drift"]].round(3).to_string(index=False))
