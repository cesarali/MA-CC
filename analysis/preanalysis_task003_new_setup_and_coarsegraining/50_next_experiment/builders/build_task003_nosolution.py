"""Rebuild the task003-nosolution agents and its single 4/4/4 controller pool, and check them
against the frozen designs/task003_nosolution.json.

Constraints (reasoning in builders/README.md and designs/README.md): 24 agents, one fact each,
8 agents per allocation, none of the 6 decisive facts, so the population cannot prove any
allocation; one 12-fact pool, 4 facts favouring each allocation, that proves nothing and
shares no fact with the agents. Random search with a fixed seed, so it reproduces exactly.

Run from the repository root:
    .venv/bin/python analysis/preanalysis_task003_new_setup_and_coarsegraining/50_next_experiment/builders/build_task003_nosolution.py
"""
import json, collections, random, pathlib, sys, itertools
S = pathlib.Path(__file__).resolve().parents[4] / "results/studies/musr_truthful_selective_task_calibration_01/tasks"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "10_task_and_facts"))
from mas_cc.musr_team_allocation_generator.ambiguity import TeamAllocationCompletionIndex
from mas_cc.musr_team_allocation_generator.symbolic_facts import CanonicalFact
from engine import World
idx=TeamAllocationCompletionIndex(); prior=idx.metrics_for_facts(()).probabilities
facts={r["fact_id"]:CanonicalFact.from_dict(r) for r in json.loads((S/"task_003/facts/all_true_facts.json").read_text())}
dec={r["fact_id"] for r in json.loads((S/"task_003/facts/decisive_facts.json").read_text())}
LEAN,STR={},{}
for f in facts:
    p=idx.metrics_for_facts((facts[f],)).probabilities
    lift=[p[k]-prior[k] for k in range(3)]
    LEAN[f]=max(range(3),key=lambda k:(lift[k],-k)); STR[f]=lift[LEAN[f]]
BY={k:sorted([f for f in facts if LEAN[f]==k],key=lambda f:(STR[f],f)) for k in (0,1,2)}

w=World((3,1,1,2,2,1,1,1,2)); masks=[w.fact_mask[i] for i in range(49)]
bad=~w.win_mask[w.truth]; ALL=w.all_mask
idmap={f.fact_id:i for i,f in enumerate(w.facts)}
def solv(c):
    m=ALL
    for i in c:
        m&=masks[i]
        if m==0: return False
    return (m&bad)==0
def proofs_in(ids,maxk=6):
    h=sorted(idmap[f] for f in ids); n=0
    for k in range(1,maxk+1):
        for c in itertools.combinations(h,k):
            if solv(c) and not (k>1 and any(solv(c[:j]+c[j+1:]) for j in range(k))): n+=1
    return n
def jp(i): return [round(p,4) for p in idx.metrics_for_facts(tuple(facts[f] for f in sorted(i))).probabilities]
def lc(i):
    c=collections.Counter(LEAN[f] for f in i); return (c[0],c[1],c[2])
def w1(a,b):
    pts=sorted(set(a)|set(b)); t=0.0
    for l,r in zip(pts,pts[1:]):
        t+=abs(sum(1 for x in a if x<=l)/len(a)-sum(1 for x in b if x<=l)/len(b))*(r-l)
    return t
def fill(p,n,rng):
    o=list(p)
    while len(o)<n: o.append(rng.choice(p))
    return o

def best_design(k0,kr,c,need_dec,rng,tries=40000,target_pa0=None):
    best=None
    for _ in range(tries):
        p1=rng.sample(BY[1],c); p2=rng.sample(BY[2],c)
        p0=rng.sample([f for f in BY[0] if f not in dec],c)
        pool=set(p0+p1+p2); pj=jp(pool)
        if max(pj)==1.0: continue
        a1=[f for f in BY[1] if f not in pool]; a2=[f for f in BY[2] if f not in pool]
        if len(a1)<kr or len(a2)<kr: continue
        s1=rng.sample(a1,kr); s2=rng.sample(a2,kr)
        if need_dec:
            extra=[f for f in BY[0] if f not in dec and f not in pool]
            s0=sorted(dec)+(rng.sample(extra,k0-6) if k0>6 else [])
        else:
            a0=[f for f in BY[0] if f not in dec and f not in pool]
            if len(a0)<k0: continue
            s0=rng.sample(a0,k0)
        distinct=set(s0+s1+s2); pa0=jp(distinct)[0]
        if need_dec and pa0!=1.0: continue
        if not need_dec and pa0==1.0: continue
        # pool neutrality: joint posterior close to uniform, and A0 vs A2 symmetric
        neutral = abs(pj[0]-pj[2]) + 0.5*abs(pj[1]-1/3)
        pen = abs(pa0-target_pa0) if target_pa0 is not None else 0.0
        score=(round(pen,3), round(neutral,3), round(w1([STR[f] for f in p0],[STR[f] for f in p1+p2]),5))
        if best is None or score<best[0]:
            best=(score,sorted(s0),sorted(s1),sorted(s2),sorted(pool),distinct,pa0,pj,
                  fill(s0,8,rng)+fill(s1,8,rng)+fill(s2,8,rng))
    return best

def show(name,r):
    _,s0,s1,s2,pool,distinct,pa0,pj,slots=r
    print(f"\n{'='*74}\n{name}")
    print(f"  AGENTS  24 slots, {len(distinct)} distinct   slot lean {lc(slots)}   distinct lean {lc(distinct)}")
    print(f"    joint posterior  {jp(distinct)}")
    print(f"    decisive held    {len(distinct&set(dec))} of 6")
    print(f"    minimal proofs the population can assemble: {proofs_in(distinct)}")
    print(f"    A0-leaning facts: {s0}")
    print(f"    A1-leaning facts: {s1}")
    print(f"    A2-leaning facts: {s2}")
    print(f"  POOL    {len(pool)} facts  lean {lc(pool)}   joint {pj}")
    print(f"    proves nothing: {max(pj)<1.0}   proofs inside pool: {proofs_in(set(pool))}")
    print(f"    facts: {pool}")
    print(f"  OVERLAP agents<->pool: {len(distinct & set(pool))}")

rng=random.Random(2026)
r=best_design(5,5,4,False,rng,target_pa0=0.50)
show("task003-nosolution   k0=5 kr=5, pool 4/4/4", r)
_,s0,s1,s2,pool,distinct,pa0,pj,slots=r
frozen=json.loads((pathlib.Path(__file__).resolve().parents[1]/"designs"/"task003_nosolution.json").read_text())
assert sorted(pool)==sorted(frozen["controller_pool"]["fact_ids"]), "pool differs from the frozen design"
assert sorted(distinct)==sorted(frozen["agents"]["distinct_facts"]), "agents differ from the frozen design"
print("\nmatches designs/task003_nosolution.json: pool and agent facts identical")
