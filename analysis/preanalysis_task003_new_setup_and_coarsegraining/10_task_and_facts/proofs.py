"""Enumerate minimal solving sets for task_003 and study who can assemble one."""
import sys, json, itertools, collections
sys.path.insert(0, "/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/10_task_and_facts")
import pandas as pd
from engine import World

w = World((3, 1, 1, 2, 2, 1, 1, 1, 2))
ID = w.ids
mn = w.minimal_sets(max_size=6)
tot = {}
for k in (4, 5, 6):
    tot[k] = sum(1 for c in itertools.combinations(range(len(w.facts)), k) if w.solves(c))
    print(f"size {k}: {tot[k]} solving, {len(mn[k])} minimal", flush=True)

allmin = [c for k in sorted(mn) for c in mn[k]]
cnt = collections.Counter(ID[i] for c in allmin for i in c)
cnt4 = collections.Counter(ID[i] for c in mn[4] for i in c)

df = pd.read_csv("/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/10_task_and_facts/task_003_facts.csv").set_index("fact")
pool = set(df[df.allocation.isin(["pool only", "both"])].index)
pkts = set(df[df.allocation.isin(["agent only", "both"])].index)
cover = lambda S: sum(1 for c in allmin if all(ID[i] in S for i in c))

pack = w.max_packing(mn)
out = dict(
    totals={str(k): tot[k] for k in tot},
    minimal={str(k): len(mn[k]) for k in mn},
    n_minimal=len(allmin),
    required_in_all_min4=[f for f in ID if cnt4[f] == len(mn[4])],
    appearance={f: cnt[f] for f in ID},
    appearance_min4={f: cnt4[f] for f in ID},
    coverage=dict(agents=cover(pkts), controller=cover(pool), both=cover(pool | pkts),
                  total=len(allmin)),
    packing=[[ID[i] for i in c] for c in pack],
    min4=[[ID[i] for i in c] for c in mn[4]],
)
json.dump(out, open("/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/10_task_and_facts/task_003_proofs.json", "w"), indent=1)
print("minimal total:", len(allmin), "| packing:", len(pack), "| coverage:", out["coverage"], flush=True)
print("DONE", flush=True)
