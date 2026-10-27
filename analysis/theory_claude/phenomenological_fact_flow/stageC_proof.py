"""Stage C: complementarity variable z (plan Eq. 4) with proof sizes taken from the worlds (task003 l = 6, P = 12; OP
l = 3, P = 2), at the current best version (revision 3, eps = 1, P0). Compares z = 0 (no proof), task003-like and OP-like
proofs: G(q 6), D(3->6) and its crossover, the share of agents with a proof, and normal-continuation retention.
Writes findings/stageC_proof.csv and findings/stageC_curves.csv."""
import pathlib
from dataclasses import replace
import numpy as np, pandas as pd
from fact_flow import P, G_at_T, zproof
from stageD_persistence import delta_path

HERE = pathlib.Path(__file__).resolve().parent
RH = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
VARIANTS = {"no proof (z = 0)": dict(), "task003-like proof (l = 6, P = 12)": dict(proof_l=6, proof_P=12.0),
            "OP-like proof (l = 3, P = 2)": dict(proof_l=3, proof_P=2.0)}
rows, curves = [], []
for lab, kw in VARIANTS.items():
    p = replace(P(), eps=1.0, **kw)
    G3 = np.array([G_at_T(p, 3, r)[0] for r in RH])
    G6s = [G_at_T(p, 6, r) for r in RH]
    G6 = np.array([g[0] for g in G6s])
    D = G6 - G3
    zs = np.array([float(zproof(p, g[2][0])) for g in G6s])
    for r, g3, g6, d, z in zip(RH, G3, G6, D, zs):
        curves.append(dict(variant=lab, rho=r, G3=g3, G6=g6, D36=d, z_silent_q6=z))
    cross = next((f"{RH[i]:g}-{RH[i+1]:g}" for i in range(len(D) - 1) if D[i] > 0 >= D[i + 1]), "none")
    row = dict(variant=lab, reversal=bool(D[0] > 1e-4 and D.min() < -1e-4), cross_36=cross, D_low=D[0], D_min=D.min(),
               G6_rho05=float(np.interp(0.5, RH, G6)), G6_rho099=G6[-1], z_rho05=float(np.interp(0.5, RH, zs)),
               z_rho09=float(np.interp(0.9, RH, zs)))
    for rho in (0.25, 0.75):
        t, d, _ = delta_path(p, 6, rho, "normal")
        d30 = np.interp(30, t, d)
        row[f"ret_{rho}"] = float(np.interp(130, t, d) / d30) if abs(d30) > 1e-9 else np.nan
    rows.append(row)
r = pd.DataFrame(rows)
r.to_csv(HERE / "findings" / "stageC_proof.csv", index=False)
pd.DataFrame(curves).to_csv(HERE / "findings" / "stageC_curves.csv", index=False)
pd.set_option("display.width", 250)
print(r.round(4).to_string(index=False))
