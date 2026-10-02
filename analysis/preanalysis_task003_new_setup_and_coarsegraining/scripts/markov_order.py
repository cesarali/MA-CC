"""Within-arm Markov-order test: is the current state sufficient?

The cross-arm transfer test is invalid here: silent and controlled episodes
occupy nearly disjoint regions of the state space, so a kernel fitted on one
extrapolates rather than predicts.  The valid test is WITHIN an arm -- does
knowing the PREVIOUS state improve prediction of the next one, over knowing the
current state alone?  If it does, the state is not Markovian.

Scored by held-out log-likelihood with episode-level splits, so the comparison
is not just "more parameters fit better".
"""
import json, sys
import numpy as np, pandas as pd

d = pd.read_parquet("/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/results/data/states.parquet")

def triples(df, col):
    """(previous, current, next) state triples within each episode."""
    out = []
    for _, g in df.sort_values("round_index").groupby(["cell_id", "episode_id"]):
        s = g[col].to_numpy()
        for i in range(1, len(s) - 1):
            out.append((s[i-1], s[i], s[i+1]))
    return pd.DataFrame(out, columns=["prev", "cur", "nxt"])

def heldout_ll(tr, te, order, alpha=0.5):
    """Dirichlet-smoothed predictive log-likelihood of the test triples."""
    states = sorted(set(tr.nxt) | set(te.nxt) | set(tr.cur) | set(te.cur))
    key = (lambda r: (r.cur,)) if order == 1 else (lambda r: (r.prev, r.cur))
    cnt = {}
    for r in tr.itertuples():
        cnt.setdefault(key(r), {}).setdefault(r.nxt, 0)
        cnt[key(r)][r.nxt] += 1
    ll = 0.0
    for r in te.itertuples():
        row = cnt.get(key(r), {})
        tot = sum(row.values()) + alpha * len(states)
        ll += np.log((row.get(r.nxt, 0) + alpha) / tot)
    return ll, len(te)

res = []
for arm, sub in (("silent", d[d.arm == "silent"]),
                 ("A2-target", d[d.arm == "A2"])):
    eps = sub[["cell_id", "episode_id"]].drop_duplicates()
    rng = np.random.default_rng(0)
    mask = rng.random(len(eps)) < 0.7
    tr_ids = set(map(tuple, eps[mask].to_numpy()))
    for col in ("V", "M", "MV"):
        t = triples(sub, col)
        idx = [tuple(x) in tr_ids for x in sub.sort_values("round_index")
               .groupby(["cell_id", "episode_id"]).head(0).to_numpy()[:, :2]] if False else None
        # split by episode
        keys = sub.sort_values("round_index").groupby(["cell_id", "episode_id"])
        tr_rows, te_rows = [], []
        for k, g in keys:
            s = g[col].to_numpy()
            tri = [(s[i-1], s[i], s[i+1]) for i in range(1, len(s) - 1)]
            (tr_rows if tuple(k) in tr_ids else te_rows).extend(tri)
        tr = pd.DataFrame(tr_rows, columns=["prev", "cur", "nxt"])
        te = pd.DataFrame(te_rows, columns=["prev", "cur", "nxt"])
        if len(te) < 50: continue
        ll1, n = heldout_ll(tr, te, 1)
        ll2, _ = heldout_ll(tr, te, 2)
        res.append(dict(arm=arm, variable=col, n_test=n,
                        ll_order1=ll1/n, ll_order2=ll2/n, gain=(ll2-ll1)/n))
        print("%-10s %-3s n=%4d  order1 %.4f  order2 %.4f  gain %+.4f nats/step"
              % (arm, col, n, ll1/n, ll2/n, (ll2-ll1)/n), flush=True)
json.dump(res, open("/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/02_markov_tests/markov_order.json", "w"), indent=1)
print("DONE", flush=True)
