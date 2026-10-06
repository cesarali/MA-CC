"""Build designs/task003_nosolution.json: 24 agents who cannot prove anything, and one pool.

Rules (the 2 October design, plus private eligibility, added 5 October):

  1. 24 agents, one fact each, each fact held by 1 or 2 agents.
  2. Every agent fact is private-eligible: read alone it leaves P <= 0.45 for every
     allocation and entropy >= 0.90 (the task generator's rule for what one agent may
     hold). The pool may contain ineligible facts.
  3. No decisive fact. 5 distinct facts favouring each allocation, 8 agents each.
     A fact favours the allocation it raises most; a tie goes to the lower-numbered
     allocation (the rule this design has always used).
  4. The agents' joint posterior is exactly (0.5, 0, 0.5): a coin flip between the
     truth and the false target, with no proof available.
  5. One 12-fact pool, 4 facts favouring each allocation, proving nothing, sharing
     no fact with the agents.
  6. Ranked by the sum of three imbalances, all in probability units:
       - pool strength match: Wasserstein-1 distance (how different two lists of
         numbers are) between the pool's A0 facts' strengths and its A2 facts'
         strengths. The pool serves both targets, so neither controller should
         get stronger facts;
       - pool neutrality: |P(A0) - P(A2)| + 0.5 |P(A1) - 1/3| for the whole pool;
       - agents' starting gap: |mean belief in A0 - mean belief in A2| over the
         24 agents, each reading only its own fact.

Only 9 facts favour A1 and 9 favour A2, so the agents' 5 + the pool's 4 use all of
them: choosing the agents' rival facts fixes the pool's rival facts. The search
therefore enumerates every rival choice, samples the agents' A0 facts with a fixed
seed, and for each agent set that meets rule 4 searches the pool's A0 facts
exhaustively.

Duplicates: in each group, the 3 facts held twice are chosen to minimise the
agents' starting gap.

Run from the repository root:
    .venv/bin/python analysis/task003_LLMfree/experimental_setup/builders/build_task003_nosolution.py
"""
from __future__ import annotations
import csv, itertools, json, math, pathlib, random, sys

HERE = pathlib.Path(__file__).resolve().parent
PRE = HERE.parents[1]
sys.path.insert(0, str(PRE / "task_and_facts"))
from engine import World  # noqa: E402

OUT = HERE.parent / "designs" / "task003_nosolution.json"
SAMPLES_PER_RIVAL_CHOICE = 400
SEED = 2026

R = {r["fact"]: r for r in csv.DictReader(open(PRE / "task_and_facts" / "task_003_facts.csv"))}
dP = lambda f, k: float(R[f][f"dP_A{k}"])
LEAN = {f: max(range(3), key=lambda k: (dP(f, k), -k)) for f in R}
STR = {f: dP(f, LEAN[f]) for f in R}
DEC = {f for f in R if R[f]["decisive"] == "True"}
ELIG = {f for f in R if R[f]["eligible"] == "True"}
BY = {k: sorted(f for f in R if LEAN[f] == k and f not in DEC) for k in range(3)}
W = World((3, 1, 1, 2, 2, 1, 1, 1, 2))
IX = {f: i for i, f in enumerate(W.ids)}
post = lambda fs: tuple(W.posterior([IX[f] for f in fs]))


def w1(a, b):
    pts = sorted(set(a) | set(b)); t = 0.0
    for lo, hi in zip(pts, pts[1:]):
        t += abs(sum(x <= lo for x in a) / len(a) - sum(x <= lo for x in b) / len(b)) * (hi - lo)
    return t


M = {f: W.fact_mask[IX[f]] for f in R}
pc = int.bit_count


def postm(m):
    n = pc(m)
    return tuple(pc(m & W.win_mask[k]) / n for k in range(3))


def andm(fs, m=None):
    m = W.all_mask if m is None else m
    for f in fs:
        m &= M[f]
    return m


BEL = {f: postm(M[f]) for f in R}


def agent_gap(slots):
    return abs(sum(BEL[f][0] - BEL[f][2] for f in slots)) / len(slots)


def best_duplicates(a0, a1, a2):
    return min(itertools.product(*(itertools.combinations(g, 3) for g in (a0, a1, a2))),
               key=lambda d: (round(agent_gap(list(a0) + list(a1) + list(a2) + [f for g in d for f in g]), 6), d))


assert len(BY[1]) == 9 and len(BY[2]) == 9
rng = random.Random(SEED)
a0_cands = [f for f in BY[0] if f in ELIG]
subsets0 = list(itertools.combinations(BY[0], 4))
best = None
hits = 0
for a1 in itertools.combinations([f for f in BY[1] if f in ELIG], 5):
    p1 = [f for f in BY[1] if f not in a1]
    m1 = andm(a1)
    for a2 in itertools.combinations([f for f in BY[2] if f in ELIG], 5):
        p2 = [f for f in BY[2] if f not in a2]
        if len(p1) != 4 or len(p2) != 4:
            continue
        m12 = andm(a2, m1)
        found = []
        tried = set()
        for _ in range(SAMPLES_PER_RIVAL_CHOICE):
            a0 = tuple(sorted(rng.sample(a0_cands, 5)))
            if a0 in tried:
                continue
            tried.add(a0)
            pa = postm(andm(a0, m12))
            if abs(pa[0] - 0.5) < 1e-12 and pa[1] < 1e-12:
                found.append(a0)
        if not found:
            continue
        hits += len(found)
        mr = andm(p1 + p2)
        s2 = [STR[f] for f in p2]
        pools = []
        for p0 in subsets0:
            pj = postm(andm(p0, mr))
            if max(pj) == 1.0:
                continue
            neutral = abs(pj[0] - pj[2]) + 0.5 * abs(pj[1] - 1 / 3)
            pools.append((w1([STR[f] for f in p0], s2) + neutral, w1([STR[f] for f in p0], s2), neutral, p0, pj))
        pools.sort(key=lambda x: (round(x[0], 6), x[3]))
        for a0 in found:
            held = set(a0) | set(a1) | set(a2)
            pool_entry = next(x for x in pools if not set(x[3]) & held)
            if best is not None and round(pool_entry[0], 6) > best[0][0]:
                continue                      # cannot win: the agents' gap is >= 0
            dups = best_duplicates(a0, a1, a2)
            gap = agent_gap(list(a0) + list(a1) + list(a2) + [f for g in dups for f in g])
            key = (round(pool_entry[0] + gap, 6), a0, a1, a2)
            if best is None or key < best[0]:
                best = (key, pool_entry, dups, gap)

(total, a0, a1, a2), (_, pool_w1, pool_neutral, p0, pj), best_dup, gap = best
pool = sorted(p0) + sorted(f for f in BY[1] if f not in a1) + sorted(f for f in BY[2] if f not in a2)
groups_distinct = {0: list(a0), 1: list(a1), 2: list(a2)}
groups = {k: groups_distinct[k] + list(best_dup[k]) for k in range(3)}
slots = groups[0] + groups[1] + groups[2]
held = set(slots)
mult = {f: slots.count(f) for f in sorted(held)}
beliefs = [W.posterior([IX[f]]) for f in slots]
pa = post(sorted(held))
score = (round(pool_neutral, 4), round(pool_w1, 5))


def slot_mean(g, dups):
    return (sum(STR[f] for f in g) + sum(STR[f] for f in dups)) / 8


design = {
    "name": "task003-nosolution",
    "status": "FROZEN 2026-10-05",
    "note": ("Population CANNOT prove anything: no decisive fact, joint posterior a coin flip "
             "between ALLOCATION_0 and ALLOCATION_2. 24 agents, 8 per allocation, every agent "
             "fact private-eligible. One 12-fact pool, 4 facts favouring each allocation, used "
             "for both targets, proving nothing, sharing no fact with the agents."),
    "built_by": "experimental_setup/builders/build_task003_nosolution.py",
    "task": "task_003",
    "world": list(W.vector),
    "gold_target": "ALLOCATION_0",
    "false_target": "ALLOCATION_2",
    "population_size": 24,
    "facts_per_agent": 1,
    "agents": {
        "distinct_facts": sorted(held),
        "n_distinct": len(held),
        "distinct_lean_A0_A1_A2": [5, 5, 5],
        "slot_lean_A0_A1_A2": [8, 8, 8],
        "by_lean": {f"A{k}": sorted(groups_distinct[k]) for k in range(3)},
        "slot_multiplicity": mult,
        "multiplicity_range": [min(mult.values()), max(mult.values())],
        "all_private_eligible": held <= ELIG,
        "joint_posterior": [round(p, 6) for p in pa],
        "decisive_held": sorted(held & DEC),
        "minimal_proofs_assemblable": 0 if max(pa) < 1.0 else None,
        "mean_strength_per_agent_A0_A1_A2": [round(slot_mean(groups_distinct[k], best_dup[k]), 4) for k in range(3)],
        "mean_individual_belief_A0_A1_A2": [round(sum(b[k] for b in beliefs) / 24, 4) for k in range(3)],
        "starting_gap_A0_vs_A2": round(gap, 4),
        "agent_assignments": {f"agent_{i + 1:03d}": [f] for i, f in enumerate(slots)},
    },
    "controller_pool": {
        "fact_ids": pool,
        "size": len(pool),
        "lean_A0_A1_A2": [4, 4, 4],
        "joint_posterior": [round(p, 6) for p in pj],
        "proves_any_allocation": max(pj) == 1.0,
        "neutrality": score[0],
        "strength_wasserstein1_A0_vs_A2": score[1],
        "sum_dP_A0_facts": round(sum(dP(f, 0) for f in pool if LEAN[f] == 0), 4),
        "sum_dP_A2_facts": round(sum(dP(f, 2) for f in pool if LEAN[f] == 2), 4),
        "ineligible_facts_in_pool": sorted(set(pool) - ELIG),
        "shared_by_both_targets": True,
    },
    "n_overlap": len(held & set(pool)),
    "overlap_agents_pool": sorted(held & set(pool)),
    "search": {"seed": SEED, "samples_per_rival_choice": SAMPLES_PER_RIVAL_CHOICE,
               "agent_sets_meeting_rule_4": hits, "total_imbalance": total},
}

assert design["agents"]["all_private_eligible"] and not design["agents"]["decisive_held"]
assert abs(pa[0] - 0.5) < 1e-12 and pa[1] < 1e-12 and design["n_overlap"] == 0
assert not design["controller_pool"]["proves_any_allocation"] and max(mult.values()) <= 2

OUT.write_text(json.dumps(design, indent=2) + "\n")
a, c = design["agents"], design["controller_pool"]
print(f"wrote {OUT.relative_to(PRE)}  ({hits} agent sets met the coin-flip rule)")
for k in range(3):
    print(f"  agents A{k}: {sorted(groups_distinct[k])}  held twice: {sorted(best_dup[k])}")
print(f"  joint {a['joint_posterior']}  mean strength per agent {a['mean_strength_per_agent_A0_A1_A2']}  mean belief {a['mean_individual_belief_A0_A1_A2']}")
print(f"  pool {pool}")
print(f"  pool joint {c['joint_posterior']}  neutrality {c['neutrality']}  W1 A0-vs-A2 {c['strength_wasserstein1_A0_vs_A2']}  sum dP A0 {c['sum_dP_A0_facts']} A2 {c['sum_dP_A2_facts']}  ineligible in pool {c['ineligible_facts_in_pool']}")
print(f"  agents' starting gap A0 vs A2 {a['starting_gap_A0_vs_A2']}")
