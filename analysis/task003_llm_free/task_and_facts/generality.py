"""Is task_003's proof structure typical, or special?

Samples random hidden worlds with a unique winner and measures, for each, the
smallest proof size, how many minimal proofs exist, and whether a single fact is
required by every shortest proof.
"""
import sys, random, json, itertools
sys.path.insert(0, "/Users/rsanchez/Projects/MA-CC/analysis/task003_llm_free/task_and_facts")
from engine import World

rng = random.Random(20260904)
rows = []
seen = set()
while len(rows) < 12:
    vec = tuple(rng.choice((1, 2, 3)) for _ in range(9))
    if vec in seen:
        continue
    seen.add(vec)
    w = World(vec)
    sc = sorted(w.scores, reverse=True)
    if sc[0] == sc[1]:
        continue                       # no unique winner
    mn = w.minimal_sets(max_size=5)
    smallest = next((k for k in sorted(mn) if mn[k]), None)
    if smallest is None:
        rows.append(dict(vector=vec, scores=w.scores, truth=w.truth, n_facts=len(w.facts),
                         smallest=None, n_smallest=0, required=None, margin=sc[0]-sc[1]))
        print(rows[-1], flush=True)
        continue
    small = mn[smallest]
    cnt = {}
    for c in small:
        for i in c:
            cnt[w.ids[i]] = cnt.get(w.ids[i], 0) + 1
    required = [f for f, n in cnt.items() if n == len(small)]
    rows.append(dict(vector=vec, scores=w.scores, truth=w.truth, n_facts=len(w.facts),
                     smallest=smallest, n_smallest=len(small), required=required,
                     margin=sc[0]-sc[1]))
    print(rows[-1], flush=True)
json.dump(rows, open("/Users/rsanchez/Projects/MA-CC/analysis/task003_llm_free/task_and_facts/generality.json", "w"), indent=1)
print("DONE", flush=True)
