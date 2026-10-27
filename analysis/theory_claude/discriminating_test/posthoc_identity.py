"""POST-HOC (not pre-registered), after both summaries failed: how much of the variation of dP is explained by
(i) which controller fact is added, (ii) fact identity x count of agent facts, (iii) fact identity x agent-fact identities
(an upper bound: per-fact linear model on the indicator of each agent fact held)? Same split as the test.
Writes posthoc_identity.csv."""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from marginal_persuasion import RUNS, collect, r2, load_world, load_setup  # noqa: E402

rows = []
for task in RUNS:
    for kind in ("symmetric", "no-proof"):
        d, _, _ = collect(task, kind)
        run, wf, sd, names = RUNS[task]
        w = load_world(str(wf)); st = load_setup(names[kind], sd, w)
        agent = sorted(set(st.agent_facts))
        tr = d[(d.q == 6) & (d.episode % 2 == 0)]
        te = d[(d.q == 3) & (d.episode % 2 == 1)].copy()
        fm = tr.groupby("fact").dP.mean()
        p1 = te.fact.map(fm).fillna(tr.dP.mean())
        fa = tr.groupby(["fact", "a"]).dP.mean()
        p2 = pd.Series([fa.get((f, a), fm.get(f, 0.0)) for f, a in zip(te.fact, te.a)], index=te.index)
        rows.append(dict(task=task, setup=kind, R2_fact_identity=r2(te.dP.to_numpy(), p1.to_numpy()),
                         R2_fact_x_count=r2(te.dP.to_numpy(), p2.to_numpy()),
                         share_pairs_dP_zero=float((te.dP.abs() < 1e-9).mean()), sd_dP=te.dP.std()))
        print("done", task, kind, flush=True)
r = pd.DataFrame(rows)
r.to_csv(pathlib.Path(__file__).resolve().parent / "posthoc_identity.csv", index=False)
print(r.round(3).to_string(index=False))
