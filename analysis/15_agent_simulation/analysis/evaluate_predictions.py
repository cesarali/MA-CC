"""Evaluate the pre-registered predictions P1-P7 (../00_preregistration.md) from the tables written by figures.py.
Writes predictions.csv and predictions.md into the report folder.
Usage, from the repository root:
    .venv/bin/python analysis/15_agent_simulation/analysis/evaluate_predictions.py <report folder>

Note on P5. The pre-registration says "rho <= 0.61" but counts "4 setups x 4 q x 3 rho = 48 cells"; the grid has four
values up to 0.61 (0.018, 0.135, 0.37, 0.61). Both readings are evaluated and reported: all four values (64 cells), and
the three values 0.135, 0.37, 0.61 (48 cells).
"""
import pathlib, sys

import numpy as np
import pandas as pd

Q = [1, 2, 3, 4, 6, 8, 12, 16]
RHO = [0.018, 0.135, 0.37, 0.61, 0.78, 0.88, 0.94, 1.0]
SETUPS = ["task003_symmetric_15", "task003_nosolution_uniform_15", "op_symmetric_15", "op_nosolution_uniform_15"]
TAU = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 64.0])


def reversal(d, read):
    """P1 test: a pair among 1->2 .. 4->6 with D > 2 se at some rho and D < -2 se at a longer rho."""
    hits = []
    for q1 in (1, 2, 3, 4):
        x = d[(d.read_mode == read) & (d.q == q1)].sort_values("rho")
        pos = x[(x.D > 2 * x.D_se)].rho
        neg = x[(x.D < -2 * x.D_se)].rho
        if len(pos) and len(neg) and neg.max() > pos.min():
            hits.append(f"{q1}->{Q[Q.index(q1) + 1]}")
    return hits


def main():
    out = pathlib.Path(sys.argv[1])
    g = pd.read_csv(out / "table_gain_t30.csv")
    cr = pd.read_csv(out / "table_crossovers.csv")
    ab = pd.read_csv(out / "table_absolute_t30.csv")
    have = [s for s in SETUPS if s in set(g.setup)]
    rows = []

    for read, name in (("posts", "P1"), ("distinct_facts", "P7")):
        res = {s: reversal(g[g.setup == s], read) for s in have}
        ok = all(res[s] for s in have)
        rows.append(dict(prediction=name, verdict="met" if ok else "failed",
                         detail="; ".join(f"{s}: {', '.join(v) or 'none'}" for s, v in res.items())))

    viol = {}
    for s in have:
        x = cr[cr.setup == s].tau_cross.to_numpy()
        x = x[np.isfinite(x)]
        viol[s] = int(np.sum(np.diff(x) > 0))
    rows.append(dict(prediction="P2", verdict="met" if all(v <= 1 for v in viol.values()) else "failed",
                     detail="; ".join(f"{s}: {v} increase(s) of tau* with q" for s, v in viol.items())))

    det, ok = [], True
    for task in ("task003", "op"):
        sym, nos = f"{task}_symmetric_15", f"{task}_nosolution_uniform_15"
        if sym in have and nos in have:
            for pair in ("3->4", "4->6"):
                a = cr[(cr.setup == sym) & (cr.pair == pair)].tau_cross.iloc[0]
                b = cr[(cr.setup == nos) & (cr.pair == pair)].tau_cross.iloc[0]
                good = np.isfinite(a) and np.isfinite(b) and a <= b
                ok &= bool(good)
                det.append(f"{task} {pair}: sym {a:.2f} vs no-sol {b:.2f}")
    rows.append(dict(prediction="P3", verdict="met" if ok else "failed", detail="; ".join(det)))

    if all(s in have for s in SETUPS):
        r = g[(g.read_mode == "posts") & (g.q == 6)]
        det, wins, t_ok = [], 0, True
        for kind in ("symmetric_15", "nosolution_uniform_15"):
            for rho in (0.37, 0.61, 0.78):
                t = r[(r.setup == f"task003_{kind}") & np.isclose(r.rho, rho)].retained.iloc[0]
                o = r[(r.setup == f"op_{kind}") & np.isclose(r.rho, rho)].retained.iloc[0]
                wins += int(t > o)
                t_ok &= bool(t >= 0.7)
                det.append(f"{kind} rho {rho}: task003 {t:.2f} vs OP {o:.2f}")
        rows.append(dict(prediction="P4", verdict="met" if (wins >= 4 and t_ok) else "failed",
                         detail=f"task003 higher in {wins}/6, task003 all >= 0.7: {t_ok}; " + "; ".join(det)))

    for label, rhos in (("P5 (4 rho values, 64 cells)", [0.018, 0.135, 0.37, 0.61]),
                        ("P5 (3 rho values, 48 cells)", [0.135, 0.37, 0.61])):
        m = g[g.q.isin([3, 4, 6, 8]) & g.rho.round(3).isin(rhos)]
        p = m.pivot_table(index=["setup", "q", "rho"], columns="read_mode", values="G")
        p = p.dropna()
        lower = (p.distinct_facts < p.posts).mean()
        change = ((p.distinct_facts - p.posts) / p.posts).mean()
        ok = lower >= 0.8 and change <= -0.15
        rows.append(dict(prediction=label, verdict="met" if ok else "failed",
                         detail=f"G lower under dedup in {lower:.0%} of {len(p)} cells; mean relative change {change:+.1%}"))

        a = ab[ab.q.isin([3, 4, 6, 8]) & ab.rho.round(3).isin(rhos)]
        worse = []
        for s in a.setup.unique():
            x = a[a.setup == s].pivot_table(index=["q", "rho"], columns="read_mode",
                                            values="vote_A0_silent" if "symmetric" in s else "vote_A2_silent")
            d = x.distinct_facts - x.posts
            bad = (d < -0.02) if "symmetric" in s else (d > 0.02)
            worse.append(f"{s}: {int(bad.sum())}/{len(d)} cells beyond 0.02")
        rows.append(dict(prediction=label.replace("P5", "P6"),
                         verdict="met" if all(w.split(": ")[1].startswith("0/") for w in worse) else "failed",
                         detail="; ".join(worse)))
    d = pd.DataFrame(rows)
    d.to_csv(out / "predictions.csv", index=False)
    md = ["| prediction | verdict | detail |", "|---|---|---|"]
    md += [f"| {r.prediction} | {r.verdict} | {r.detail} |" for r in d.itertuples()]
    (out / "predictions.md").write_text("\n".join(md) + "\n")
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()
