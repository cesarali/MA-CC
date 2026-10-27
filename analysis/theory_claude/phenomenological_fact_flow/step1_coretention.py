"""Step 1: proofs with the exact co-retention moment h_l (equal copies) instead of the product m_A^l. Compares, at the
current best version (eps = 1, P0): no proof; task003-like proof (l 6, P 12) and OP-like proof (l 3, P 2), each with the
independence product and with co-retention. Reports the 3->6 crossover, G at long memory, the share holding a proof
(silent arm, q 6) at rho 0.25 / 0.5 / 0.75, the ratio h_l / m_A^l, and the check h_1 = m_A.
Writes findings/step1_coretention.csv and findings/step1_curves.csv."""
import pathlib
from dataclasses import replace
import numpy as np, pandas as pd
from fact_flow import P, G_at_T, zproof

HERE = pathlib.Path(__file__).resolve().parent
RH = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
V = {"no proof": dict(),
     "task003-like, independent": dict(proof_l=6, proof_P=12.0),
     "task003-like, co-retention": dict(proof_l=6, proof_P=12.0, coretention=True),
     "OP-like, independent": dict(proof_l=3, proof_P=2.0),
     "OP-like, co-retention": dict(proof_l=3, proof_P=2.0, coretention=True)}
rows, curves = [], []
for lab, kw in V.items():
    p = replace(P(), eps=1.0, **kw)
    res6 = [G_at_T(p, 6, r) for r in RH]
    G3 = np.array([G_at_T(p, 3, r)[0] for r in RH])
    G6 = np.array([g[0] for g in res6])
    D = G6 - G3
    z, ratio, chk = [], [], []
    for g in res6:
        s = g[2]
        if p.coretention:
            hl = s[3 + p.proof_l]
            z.append(1 - (1 - hl) ** p.proof_P)
            ratio.append(hl / max(s[0] ** p.proof_l, 1e-300))
            chk.append(abs(s[4] - s[0]))
        else:
            z.append(float(zproof(p, s[0])))
            ratio.append(1.0 if p.proof_l else np.nan)
            chk.append(0.0)
    z = np.array(z)
    for r, d, g6, zz, rr in zip(RH, D, G6, z, ratio):
        curves.append(dict(variant=lab, rho=r, D36=d, G6=g6, z=zz, h_over_product=rr))
    rows.append(dict(variant=lab, cross_36=next((f"{RH[i]:g}-{RH[i+1]:g}" for i in range(len(D) - 1) if D[i] > 0 >= D[i + 1]), "none"),
                     D_low=D[0], D_min=D.min(), G6_rho09=float(np.interp(0.9, RH, G6)),
                     z_025=float(np.interp(0.25, RH, z)), z_05=float(np.interp(0.5, RH, z)), z_075=float(np.interp(0.75, RH, z)),
                     ratio_025=float(np.interp(0.25, RH, ratio)), ratio_05=float(np.interp(0.5, RH, ratio)),
                     max_h1_minus_mA=float(max(chk))))
r = pd.DataFrame(rows)
r.to_csv(HERE / "findings" / "step1_coretention.csv", index=False)
pd.DataFrame(curves).to_csv(HERE / "findings" / "step1_curves.csv", index=False)
pd.set_option("display.width", 250)
print(r.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
