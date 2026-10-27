"""Revision 3: softened vote gating eps x beta_C/beta_A (kappa_C = kappa_A = 1, P0 otherwise). Writes findings/rev3_grid.csv."""
import pathlib
from dataclasses import replace
import numpy as np, pandas as pd
from fact_flow import P, G_at_T
from stageD_persistence import delta_path
from rev1_scan import R_numbers

HERE = pathlib.Path(__file__).resolve().parent
RH = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
rows = []
for eps in (0.0, 0.25, 0.5, 0.75, 1.0):
    for br in (1.0, 2.0):
        p = replace(P(), eps=eps, beta_C=0.5 * br)
        D = np.array([G_at_T(p, 6, r)[0] - G_at_T(p, 3, r)[0] for r in RH])
        row = dict(eps=eps, betaC_over_betaA=br, reversal=bool(D[0] > 1e-4 and D[3:].min() < -1e-4),
                   cross=next((f"{RH[i]}-{RH[i+1]}" for i in range(len(D) - 1) if D[i] > 0 >= D[i + 1]), "none"),
                   D_low=D[0], D_min_high=D[3:].min(), G6_rho025=G_at_T(p, 6, 0.25)[0])
        for rho in (0.25, 0.75):
            t, d, yc = delta_path(p, 6, rho, "normal")
            d30 = np.interp(30, t, d)
            row[f"ret_{rho}"] = float(np.interp(130, t, d) / d30)
            row[f"ret80_{rho}"] = float(np.interp(80, t, d) / d30)
            row[f"Reff_{rho}"] = R_numbers(p, 6, rho)[1]
        row["persist"] = bool(row["ret_0.25"] >= 0.5 and row["ret_0.75"] >= 0.5)
        row["both"] = row["reversal"] and row["persist"]
        rows.append(row)
r = pd.DataFrame(rows)
r.to_csv(HERE / "findings" / "rev3_grid.csv", index=False)
pd.set_option("display.width", 250)
print(r.round(3).to_string(index=False))
