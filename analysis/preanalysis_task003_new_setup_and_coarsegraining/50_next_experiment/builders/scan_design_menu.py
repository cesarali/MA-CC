import json, collections, random, pathlib, sys
S = pathlib.Path("/private/tmp/claude-501/-Users-rsanchez-Projects-MA-CC/2624cbd7-c886-46ad-ba56-a68bf0495a6c/scratchpad/tasks")
sys.path.insert(0, "/Users/rsanchez/Projects/MA-CC/src")
from mas_cc.musr_team_allocation_generator.ambiguity import TeamAllocationCompletionIndex
from mas_cc.musr_team_allocation_generator.symbolic_facts import CanonicalFact
idx=TeamAllocationCompletionIndex(); prior=idx.metrics_for_facts(()).probabilities
facts={r["fact_id"]:CanonicalFact.from_dict(r) for r in json.loads((S/"task_003/facts/all_true_facts.json").read_text())}
dec={r["fact_id"] for r in json.loads((S/"task_003/facts/decisive_facts.json").read_text())}
LEAN,STR={},{}
for f in facts:
    p=idx.metrics_for_facts((facts[f],)).probabilities
    lift=[p[k]-prior[k] for k in range(3)]
    LEAN[f]=max(range(3),key=lambda k:(lift[k],-k)); STR[f]=lift[LEAN[f]]
BY={k:sorted([f for f in facts if LEAN[f]==k],key=lambda f:(STR[f],f)) for k in (0,1,2)}
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

def search(k0,kr,c,need_dec,rng,tries=9000,target_pa0=None):
    best=None
    for _ in range(tries):
        p1=rng.sample(BY[1],c); p2=rng.sample(BY[2],c)
        p0=rng.sample([f for f in BY[0] if f not in dec],c)
        pool=set(p0+p1+p2)
        if max(jp(pool))==1.0: continue
        a1=[f for f in BY[1] if f not in pool]; a2=[f for f in BY[2] if f not in pool]
        if len(a1)<kr or len(a2)<kr: continue
        s1=rng.sample(a1,kr); s2=rng.sample(a2,kr)
        if need_dec:
            if k0<6: continue
            extra=[f for f in BY[0] if f not in dec and f not in pool]
            s0=sorted(dec)+ (rng.sample(extra,k0-6) if k0>6 else [])
        else:
            a0=[f for f in BY[0] if f not in dec and f not in pool]
            if len(a0)<k0: continue
            s0=rng.sample(a0,k0)
        distinct=set(s0+s1+s2)
        pa0=jp(distinct)[0]
        if need_dec and pa0!=1.0: continue
        if not need_dec and pa0==1.0: continue
        slots=fill(s0,8,rng)+fill(s1,8,rng)+fill(s2,8,rng)
        d=w1([STR[f] for f in p0],[STR[f] for f in p1+p2])
        pen = abs(pa0-target_pa0) if target_pa0 is not None else 0.0
        score=(len(distinct&pool), round(pen,4), round(d,6))
        if best is None or score<best[0]: best=(score,slots,distinct,pool,pa0,d)
    return best

def row(label,k0,kr,c,need_dec,rng,target=None):
    r=search(k0,kr,c,need_dec,rng,target_pa0=target)
    if not r: print(f"  {label:34s} INFEASIBLE"); return
    _,slots,distinct,pool,pa0,d=r
    ov=len(distinct&pool)
    print(f"  {label:34s} agents {len(distinct):2d} distinct {lc(distinct)}  slots {lc(slots)}"
          f"  P(A0)={pa0:.3f}  pool {len(pool):2d} {lc(pool)} joint {jp(pool)}  W1={d:.4f}  overlap {ov}")
    return r

rng=random.Random(23)
print("TASK003-NOSOLUTION-v2   (no decisive facts; population must NOT be able to prove)")
print("  target: slot balance 8/8/8, zero overlap, pool proves nothing")
for k0,kr,c in ((4,4,5),(5,4,5),(6,4,5),(4,5,4),(5,5,4),(6,5,4),(8,4,5),(10,4,5)):
    row(f"k0={k0} kr={kr} pool c={c}",k0,kr,c,False,rng,target=0.50)
print()
print("TASK003-SYMMETRIC-v2    (all 6 decisive facts held; population CAN prove)")
for k0,kr,c in ((6,4,5),(6,5,4),(6,6,3),(7,4,5),(8,4,5),(10,4,5),(6,3,6)):
    row(f"k0={k0} kr={kr} pool c={c}",k0,kr,c,True,rng)
