"""Choose the task003-nosolution agents and its single controller pool, together.

Outputs: agents.json (the 24 agents) and controller_pool.json (the one pool, used
for both targets), in this folder.

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

Run from the repository root (about 2 minutes; same result every run):
    .venv/bin/python analysis/task003_llm_free/experimental_setup/task003_nosolution/build.py
"""
from __future__ import annotations
import csv, itertools, json, math, pathlib, random, sys

HERE = pathlib.Path(__file__).resolve().parent
PRE = HERE.parents[1]                                    # analysis/task003_llm_free
sys.path.insert(0, str(PRE / "task_and_facts"))
from engine import World  # noqa: E402

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


A = lambda k: f"ALLOCATION_{k}"


def favours(f):
    v = [dP(f, k) for k in range(3)]
    return [A(k) for k in range(3) if abs(v[k] - max(v)) < 1e-12]


def fact_detail(f):
    return {
        "fact_id": f,
        "text": R[f]["text"],
        "dP": {A(k): round(dP(f, k), 4) for k in range(3)},
        "favours": favours(f),
        "eligible_for_one_agent": f in ELIG,
        "decisive": f in DEC,
    }


agents = {
    "setup": "task003-nosolution",
    "what_this_is": ("The 24 agents of task003-nosolution. Each holds one fact. None holds a "
                     "decisive fact; together they put ALLOCATION_0 and ALLOCATION_2 at exactly "
                     "0.5 each and cannot prove anything."),
    "status": "FROZEN 2026-10-05",
    "built_by": "build.py (this folder)",
    "task": "task_003",
    "world": list(W.vector),
    "truth": A(0),
    "false_target": A(2),
    "agent_assignments": {f"agent_{i + 1:03d}": [f] for i, f in enumerate(slots)},
    "facts": [
        {**fact_detail(f), "agent_group": A(LEAN[f]), "held_by_agents": mult[f]}
        for f in sorted(held, key=lambda f: (LEAN[f], f))
    ],
    "properties": {
        "agents": 24,
        "distinct_facts": len(held),
        "agents_per_group_A0_A1_A2": [8, 8, 8],
        "distinct_facts_per_group_A0_A1_A2": [5, 5, 5],
        "group_rule": "a fact belongs to the allocation it raises most; a tie goes to the lower-numbered allocation",
        "all_facts_eligible_for_one_agent": held <= ELIG,
        "decisive_facts_held": f"{len(held & DEC)} of 6",
        "joint_posterior_A0_A1_A2": [round(p, 6) for p in pa],
        "minimal_proofs_assemblable": 0 if max(pa) < 1.0 else None,
        "mean_individual_belief_A0_A1_A2": [round(sum(b[k] for b in beliefs) / 24, 4) for k in range(3)],
        "starting_gap_A0_vs_A2": round(gap, 4),
        "agent_sets_meeting_rule_4": hits,
        "search": {"seed": SEED, "samples_per_rival_choice": SAMPLES_PER_RIVAL_CHOICE, "total_imbalance": total},
    },
}

pool_file = {
    "setup": "task003-nosolution",
    "used_when_controller_targets": "ALLOCATION_0 and ALLOCATION_2 (the same pool for both)",
    "what_this_is": ("The controller's single pool. 4 facts favour each allocation, so the menu "
                     "itself is unbiased; the controller's target decides which facts it picks."),
    "status": "FROZEN 2026-10-05",
    "fact_ids": pool,
    "facts": [{**fact_detail(f), "pool_group": A(LEAN[f])} for f in pool],
    "properties": {
        "size": len(pool),
        "facts_per_group_A0_A1_A2": [4, 4, 4],
        "joint_posterior_A0_A1_A2": [round(p, 6) for p in pj],
        "proves_any_allocation": max(pj) == 1.0,
        "distance_from_prior": score[0],
        "strength_mismatch_A0_vs_A2_facts": score[1],
        "sum_dP_A0_facts": round(sum(dP(f, 0) for f in pool if LEAN[f] == 0), 4),
        "sum_dP_A2_facts": round(sum(dP(f, 2) for f in pool if LEAN[f] == 2), 4),
        "facts_not_eligible_for_one_agent": sorted(set(pool) - ELIG),
        "facts_the_agents_already_hold": sorted(held & set(pool)),
    },
}

assert agents["properties"]["all_facts_eligible_for_one_agent"] and not (held & DEC)
assert abs(pa[0] - 0.5) < 1e-12 and pa[1] < 1e-12 and not (held & set(pool))
assert not pool_file["properties"]["proves_any_allocation"] and max(mult.values()) <= 2

(HERE / "agents.json").write_text(json.dumps(agents, indent=2, ensure_ascii=False) + "\n")
(HERE / "controller_pool.json").write_text(json.dumps(pool_file, indent=2, ensure_ascii=False) + "\n")
pr, pp = agents["properties"], pool_file["properties"]
print(f"wrote agents.json and controller_pool.json  ({hits} agent sets met the coin-flip rule)")
for k in range(3):
    print(f"  agents {A(k)}: {sorted(groups_distinct[k])}  held twice: {sorted(best_dup[k])}")
print(f"  joint {pr['joint_posterior_A0_A1_A2']}  mean belief {pr['mean_individual_belief_A0_A1_A2']}  starting gap {pr['starting_gap_A0_vs_A2']}")
print(f"  pool {pool}")
print(f"  pool joint {pp['joint_posterior_A0_A1_A2']}  distance from prior {pp['distance_from_prior']}  "
      f"A0-vs-A2 mismatch {pp['strength_mismatch_A0_vs_A2_facts']}  sum dP A0 {pp['sum_dP_A0_facts']} A2 {pp['sum_dP_A2_facts']}")
