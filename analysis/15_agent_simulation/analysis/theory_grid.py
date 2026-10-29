"""Theory on the campaign grid: the mean-field fact-flow model (theory_claude/phenomenological_fact_flow/fact_flow.py)
at the P0 parameters fixed on 27 Oct 2026, with W = 30 (as in the campaign; N/W = 0.5 is unchanged) and neutral
reposting eps = 1 (the version agreed in the final common statement). Nothing is fitted to the campaign.

For each q in {1,2,3,4,6,8,12,16} and rho in the campaign grid:
  g(30) = Phi_2(controlled) - Phi_2(silent) at T = 30 (the model's next-vote gap);
  D(q_i -> q_{i+1}) = g(30; q_{i+1}) - g(30; q_i);
  retained = g(130) / g(30) under normal continuation;
  R_C at the stop (local growth number of controller facts, R_C = q kC gC / (gamma H)).
Writes theory_grid.csv next to this file. Usage, from the repository root:
    .venv/bin/python analysis/15_agent_simulation/analysis/theory_grid.py
"""
import pathlib, sys
from dataclasses import replace

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "theory_claude" / "phenomenological_fact_flow"))
from fact_flow import P, phi2, run  # noqa: E402

Q = [1, 2, 3, 4, 6, 8, 12, 16]
RHO = [0.018, 0.135, 0.37, 0.61, 0.78, 0.88, 0.94, 1.0]


def main():
    p = replace(P(), W=30.0, eps=1.0)
    rows = []
    for q in Q:
        for rho in RHO:
            t_eval = np.array([30.0, 130.0])
            c1, c2 = run(p, q, rho, True, t_end=130.0, mode="normal", t_eval=t_eval)
            s1, s2 = run(p, q, rho, False, t_end=130.0, mode="normal", t_eval=t_eval)
            g = lambda ca, sa: phi2(p, ca[0], ca[1]) - phi2(p, sa[0], sa[1])
            g30 = g(c1.y[:, -1], s1.y[:, -1])
            g130 = g(c2.y[:, -1], s2.y[:, -1])
            mA, mC = c1.y[0, -1], c1.y[1, -1]
            gamma = -np.log(rho)
            H = p.kA * p.F_A * mA * 1.0 + p.k0
            R_C = q * p.kC * 1.0 / (gamma * H) if gamma > 0 else np.inf
            rows.append(dict(q=q, rho=rho, g30=g30, g130=g130, retained=g130 / g30 if g30 else np.nan, R_C=R_C,
                             phi_c30=phi2(p, mA, mC), phi_s30=phi2(p, s1.y[0, -1], 0.0)))
    d = pd.DataFrame(rows)
    d = d.sort_values(["rho", "q"])
    d["D_next"] = d.groupby("rho").g30.shift(-1) - d.g30          # D from this q to the next q in the grid
    d.to_csv(HERE / "theory_grid.csv", index=False)
    print(d.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
