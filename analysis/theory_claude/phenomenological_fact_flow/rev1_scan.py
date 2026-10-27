"""Revision 1: beta_C / beta_A in {1, 2, 3, 4} (P0 otherwise). For each: reversal for 3->6 (D at rho 0.05 and min over
rho >= 0.3), G(q 6) range, normal-continuation retention delta(130)/delta(30) at rho 0.25 / 0.75 for selective
reposting, and the reproduction numbers R_lin (controller-free background, plan Eq. 8) and R_eff at the state at T.
Writes results/rev1_scan.csv."""
import pathlib
from dataclasses import replace
import numpy as np, pandas as pd
from fact_flow import P, G_at_T, run, phi2
from stageD_persistence import delta_path

HERE = pathlib.Path(__file__).resolve().parent
RH = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


def R_numbers(p, q, rho):
    gamma = -np.log(rho)
    s = run(p, q, rho, False, t_end=200.0)          # controller-free background (silent arm, near stationary by t=200)
    yS = s[1].y[:, -1]
    Phi0 = phi2(p, yS[0], 0.0)
    c = run(p, q, rho, True).y[:, -1]                # state at T
    PhiT = phi2(p, c[0], c[1])
    e = p.eps if p.selective else 1.0
    gC0, gA0 = e + (1 - e) * Phi0, e + (1 - e) * (1 - Phi0)
    gCT, gAT = e + (1 - e) * PhiT, e + (1 - e) * (1 - PhiT)
    H0 = p.kA * p.F_A * yS[0] * gA0 + p.k0
    HT = p.kA * p.F_A * c[0] * gAT + p.k0
    return q * p.kC * gC0 / (gamma * H0), q * p.kC * gCT / (gamma * HT), PhiT


def main():
    rows = []
    for ratio in (1.0, 2.0, 3.0, 4.0):
        p = replace(P(), beta_C=0.5 * ratio)
        D = np.array([G_at_T(p, 6, r)[0] - G_at_T(p, 3, r)[0] for r in RH])
        G6 = [G_at_T(p, 6, r)[0] for r in RH]
        row = dict(beta_ratio=ratio, D36_lowrho=D[0], D36_min_high=D[3:].min(), reversal=bool(D[0] > 1e-4 and D[3:].min() < -1e-4),
                   cross=next((f"{RH[i]}-{RH[i+1]}" for i in range(len(D) - 1) if D[i] > 0 >= D[i + 1]), "none"),
                   G6_min=min(G6), G6_max=max(G6))
        for rho in (0.25, 0.75):
            t, d, _ = delta_path(p, 6, rho, "normal")
            row[f"retained_{rho}"] = float(np.interp(130, t, d) / np.interp(30, t, d))
            Rl, Re, PhiT = R_numbers(p, 6, rho)
            row[f"R_lin_{rho}"], row[f"R_eff_{rho}"], row[f"PhiT_{rho}"] = Rl, Re, PhiT
        rows.append(row)
    r = pd.DataFrame(rows)
    r.to_csv(HERE / "results" / "rev1_scan.csv", index=False)
    pd.set_option("display.width", 250)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
