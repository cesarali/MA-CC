"""Stage A: chain-rule split of D(3->6) for model V1 (plan Eq. 9). With x = h0 + beta_C M_C - beta_A M_A and
G = sigma(x_ctrl) - sigma(x_silent), a finite step q1 -> q2 gives (exactly, with mean-value slopes s = dsigma/dx)
  D = s_c * beta_C * dM_C                  (controller evidence gained by reading, controlled arm)
    - s_c * beta_A * dM_A^ctrl             (agent evidence gained, controlled arm)
    + s_s * beta_A * dM_A^silent           (agent evidence gained, silent arm)
s_c, s_s: secant slopes of sigma between the q1 and q2 states in each arm. Prints the terms by rho for P0 and two
variants. Writes findings/stageA_mechanism.csv.
"""
import pathlib
from dataclasses import replace
import numpy as np, pandas as pd
from fact_flow import P, G_at_T, sigma

HERE = pathlib.Path(__file__).resolve().parent
RHOS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


def split(p, rho, q1=3, q2=6):
    out = {}
    for q in (q1, q2):
        _, c, s = G_at_T(p, q, rho)
        out[q] = dict(MA_c=p.F_A * c[0], MC=p.F_C * c[1], MA_s=p.F_A * s[0], bC=c[3])
    x = lambda MA, MC: p.h0 + p.beta_C * MC - p.beta_A * MA  # noqa: E731
    xc1, xc2 = x(out[q1]["MA_c"], out[q1]["MC"]), x(out[q2]["MA_c"], out[q2]["MC"])
    xs1, xs2 = x(out[q1]["MA_s"], 0), x(out[q2]["MA_s"], 0)
    sc = (sigma(xc2) - sigma(xc1)) / (xc2 - xc1) if xc2 != xc1 else sigma(xc1) * (1 - sigma(xc1))
    ss = (sigma(xs2) - sigma(xs1)) / (xs2 - xs1) if xs2 != xs1 else sigma(xs1) * (1 - sigma(xs1))
    dMC = out[q2]["MC"] - out[q1]["MC"]
    dMAc = out[q2]["MA_c"] - out[q1]["MA_c"]
    dMAs = out[q2]["MA_s"] - out[q1]["MA_s"]
    terms = dict(C_gain=sc * p.beta_C * dMC, A_gain_ctrl=-sc * p.beta_A * dMAc, A_gain_silent=ss * p.beta_A * dMAs)
    return dict(rho=rho, D=sum(terms.values()), **terms, slope_ctrl=sc, slope_silent=ss, dMC=dMC, dMA_ctrl=dMAc,
                dMA_silent=dMAs, MA_silent_q3=out[q1]["MA_s"], MC_q3=out[q1]["MC"], Phi_silent_q3=sigma(xs1))


rows = []
for label, p in (("P0", P()), ("F_C = F_A = 16", replace(P(), F_C=16.0)), ("neutral", replace(P(), selective=False))):
    for r in RHOS:
        rows.append(dict(model=label, **split(p, r)))
d = pd.DataFrame(rows)
(HERE / "findings").mkdir(exist_ok=True)
d.to_csv(HERE / "findings" / "stageA_mechanism.csv", index=False)
pd.set_option("display.width", 250)
print(d.round(4).to_string(index=False))


# --- the small-effect approximation: D ~ beta_C sigma'(x_s) [dM_C - (1 - 2 Phi_s) beta_A M_C dM_A]
def approx(p, rho, q1=3, q2=6):
    a, b = split(p, rho, q1, q2), None
    _, c1, s1 = G_at_T(p, q1, rho)
    _, c2, s2 = G_at_T(p, q2, rho)
    MC = 0.5 * p.F_C * (c1[1] + c2[1])
    xs = p.h0 - p.beta_A * p.F_A * 0.5 * (s1[0] + s2[0])
    Phi = sigma(xs)
    sp = Phi * (1 - Phi)
    dMC = p.F_C * (c2[1] - c1[1])
    dMA = p.F_A * (s2[0] - s1[0])
    return dict(rho=rho, D=a["D"], D_approx=p.beta_C * sp * (dMC - (1 - 2 * Phi) * p.beta_A * MC * dMA),
                dose_term=p.beta_C * sp * dMC, sensitivity_term=-p.beta_C * sp * (1 - 2 * Phi) * p.beta_A * MC * dMA)


rows = []
for label, p in (("P0", P()), ("neutral", replace(P(), selective=False)), ("beta_A x2", replace(P(), beta_A=1.0)),
                 ("beta_A x0.5", replace(P(), beta_A=0.25))):
    for r in RHOS:
        rows.append(dict(model=label, **approx(p, r)))
a = pd.DataFrame(rows)
a.to_csv(HERE / "findings" / "stageA_approx.csv", index=False)
print(a.round(4).to_string(index=False))
