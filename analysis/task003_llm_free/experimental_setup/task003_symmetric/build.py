"""Choose the 24 task003-symmetric agents for the two fixed controller pools.

Inputs (fixed, designed by hand on 2 October; see pool_design_2026-10-02.md §2):
    controller_pool_A0.json, controller_pool_A2.json -- their "fact_ids" lists.
    The two lists are in pair order: entry i of the A0 pool is matched in
    strength with entry i of the A2 pool.

Outputs:
    agents.json            -- the 24 agents and their properties
    controller_pool_A0.json, controller_pool_A2.json
                           -- rewritten with the SAME fact_ids, plus each fact's
                              text, strength, partner and properties

Rules for the agents, in order:

  1. 24 agents, one fact each, every fact private-eligible, each fact held by
     1 or 2 agents.
  2. 8 agents per allocation. An agent counts for allocation k when its fact
     raises P(k) at least as much as any other allocation. A fact tied between
     two allocations may count for either.
  3. The A0 group is exactly the 6 decisive facts, so the agents can prove A0.
  4. Equal new evidence for both controllers: the pool facts the agents already
     hold must be matched PAIRS (an A0-pool fact together with its A2 partner).
  5. Equal duplication: decisive facts and rival facts are held by the same mean
     number of agents (8/6 = 16/12), so forgetting does not favour either side.
  6. Among what is left: fewest tied facts, then the smallest overlap, then rival
     groups of the most similar strength, then alphabetical order.

Duplicates: the 2 decisive facts in most of the agents' minimal proofs, and in
each rival group the 2 untied, non-overlapping facts in fewest of those proofs.

Run from the repository root (a few seconds; same result every run):
    .venv/bin/python analysis/task003_llm_free/experimental_setup/task003_symmetric/build.py
"""
from __future__ import annotations
import csv, itertools, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                   # analysis/task003_llm_free
sys.path.insert(0, str(ROOT / "task_and_facts"))
from engine import World  # noqa: E402

R = {r["fact"]: r for r in csv.DictReader(open(ROOT / "task_and_facts" / "task_003_facts.csv"))}
POOL_FILES = {0: HERE / "controller_pool_A0.json", 2: HERE / "controller_pool_A2.json"}
POOLS = {k: json.loads(p.read_text())["fact_ids"] for k, p in POOL_FILES.items()}
P0, P2 = POOLS[0], POOLS[2]
PAIR = dict(zip(P0, P2))
PAIR.update({b: a for a, b in zip(P0, P2)})

dP = lambda f, k: float(R[f][f"dP_A{k}"])
DEC = sorted(f for f in R if R[f]["decisive"] == "True")
ELIG = {f for f in R if R[f]["eligible"] == "True"}
W = World((3, 1, 1, 2, 2, 1, 1, 1, 2))
IX = {f: i for i, f in enumerate(W.ids)}


def favours(f):
    v = [dP(f, k) for k in range(3)]
    return {k for k in range(3) if abs(v[k] - max(v)) < 1e-12}


tied = lambda f: len(favours(f)) > 1
post = lambda fs: [round(p, 6) for p in W.posterior([IX[f] for f in fs])]


def minimal_proofs(fs, maxk=6):
    h = sorted(IX[f] for f in fs)
    memo, out = {}, []
    solves = lambda c: memo.setdefault(c, W.solves(c))
    for k in range(1, maxk + 1):
        for c in itertools.combinations(h, k):
            if solves(c) and not (k > 1 and any(solves(c[:j] + c[j + 1:]) for j in range(k))):
                out.append({W.ids[i] for i in c})
    return out


def candidates():
    C1 = sorted(f for f in ELIG if 1 in favours(f) and f not in DEC)
    C2 = sorted(f for f in ELIG if 2 in favours(f) and f not in DEC)
    for S1 in itertools.combinations(C1, 6):
        need2 = {PAIR[f] for f in set(S1) & set(P0)}
        if not (set(S1) & set(P2)) <= need2:
            continue
        forced = need2 - set(S1)
        if not forced <= set(C2) or len(forced) > 6:
            continue
        rest = [f for f in C2 if f not in P0 and f not in P2 and f not in S1]
        for R2 in itertools.combinations(rest, 6 - len(forced)):
            yield S1, tuple(sorted(forced | set(R2)))


def rank(c):
    S1, S2 = c
    overlap = len(set(S1 + S2) & set(P0))
    strength = abs(sum(dP(f, 1) for f in S1) - sum(dP(f, 2) for f in S2)) / 6
    return (sum(map(tied, S1 + S2)), overlap, round(strength, 6), S1, S2)


cands = list(candidates())
S1, S2 = min(cands, key=rank)
held = set(DEC) | set(S1) | set(S2)
proofs = minimal_proofs(held)
in_proofs = lambda f: sum(f in p for p in proofs)

dup_dec = sorted(DEC, key=lambda f: (-in_proofs(f), f))[:2]


def rival_dups(S):
    ok = [f for f in S if not tied(f) and f not in P0 and f not in P2]
    return sorted(ok, key=lambda f: (in_proofs(f), f))[:2]


groups = {0: list(DEC) + dup_dec, 1: list(S1) + rival_dups(S1), 2: list(S2) + rival_dups(S2)}
group_of = {f: k for k, g in groups.items() for f in g}
assert all(len(g) == 8 for g in groups.values())
slots = groups[0] + groups[1] + groups[2]
mult = {f: slots.count(f) for f in sorted(held)}
beliefs = [W.posterior([IX[f]]) for f in slots]
A = lambda k: f"ALLOCATION_{k}"


def fact_detail(f):
    return {
        "fact_id": f,
        "text": R[f]["text"],
        "dP": {A(k): round(dP(f, k), 4) for k in range(3)},
        "favours": [A(k) for k in sorted(favours(f))],
        "eligible_for_one_agent": f in ELIG,
        "decisive": f in DEC,
    }


agents = {
    "setup": "task003-symmetric",
    "what_this_is": ("The 24 agents of task003-symmetric. Each holds one fact. Together they "
                     "hold all 6 decisive facts, so together they can prove ALLOCATION_0."),
    "status": "FROZEN 2026-10-05",
    "built_by": "build.py (this folder)",
    "task": "task_003",
    "world": list(W.vector),
    "truth": A(0),
    "false_target": A(2),
    "agent_assignments": {f"agent_{i + 1:03d}": [f] for i, f in enumerate(slots)},
    "facts": [
        {**fact_detail(f),
         "agent_group": A(group_of[f]),
         "held_by_agents": mult[f],
         "also_in_controller_pool": (A(0) if f in P0 else A(2) if f in P2 else None)}
        for f in sorted(held, key=lambda f: (group_of[f], f))
    ],
    "properties": {
        "agents": 24,
        "distinct_facts": len(held),
        "agents_per_group_A0_A1_A2": [8, 8, 8],
        "decisive_facts_held": f"{len(set(DEC) & held)} of 6",
        "joint_posterior_A0_A1_A2": post(held),
        "minimal_proofs_assemblable_up_to_6_facts": len(proofs),
        "mean_agents_per_decisive_fact": round(sum(mult[f] for f in DEC) / 6, 4),
        "mean_agents_per_rival_fact": round(sum(mult[f] for f in set(S1) | set(S2)) / 12, 4),
        "tied_facts_in_rival_groups": sorted(f for f in set(S1) | set(S2) if tied(f)),
        "mean_individual_belief_A0_A1_A2": [round(sum(b[k] for b in beliefs) / 24, 4) for k in range(3)],
        "agent_sets_meeting_rules_1_to_5": len(cands),
        "equally_good_alternatives": sum(1 for c in cands if rank(c)[:3] == rank((S1, S2))[:3]) - 1,
    },
}


def pool_file(k):
    ids = POOLS[k]
    other = 2 if k == 0 else 0
    new = [f for f in ids if f not in held]
    return {
        "setup": "task003-symmetric",
        "used_when_controller_targets": A(k),
        "what_this_is": (f"The controller's pool when it steers toward {A(k)}. Never used for "
                         f"the other target. Designed by hand on 2 October (pool_design_2026-10-02.md "
                         f"§2) and unchanged since. Entry i is matched in strength with entry i of "
                         f"controller_pool_A{other}.json."),
        "status": "FROZEN 2026-10-02",
        "fact_ids": ids,
        "facts": [{**fact_detail(f), "partner_in_other_pool": PAIR[f], "held_by_agents": mult.get(f, 0)}
                  for f in ids],
        "properties": {
            "size": len(ids),
            "sum_dP_toward_target": round(sum(dP(f, k) for f in ids), 4),
            "joint_posterior_A0_A1_A2": post(ids),
            "proves_any_allocation": max(post(ids)) == 1.0,
            "contains_decisive_facts": bool(set(ids) & set(DEC)),
            "shares_facts_with_other_pool": bool(set(ids) & set(POOLS[other])),
            "facts_the_agents_already_hold": [f for f in ids if f in held],
            "new_facts_for_the_agents": len(new),
            "sum_dP_toward_target_of_new_facts": round(sum(dP(f, k) for f in new), 4),
        },
    }


pools = {k: pool_file(k) for k in POOL_FILES}

# invariants
pr, p0, p2 = agents["properties"], pools[0]["properties"], pools[2]["properties"]
assert pr["joint_posterior_A0_A1_A2"][0] == 1.0
assert not p0["proves_any_allocation"] and not p2["proves_any_allocation"]
assert not p0["contains_decisive_facts"] and not p2["contains_decisive_facts"]
assert not p0["shares_facts_with_other_pool"]
assert p0["new_facts_for_the_agents"] == p2["new_facts_for_the_agents"]
assert sum(mult[f] for f in held & set(P0)) == sum(mult[f] for f in held & set(P2))
assert held <= ELIG and max(mult.values()) <= 2
assert pr["mean_agents_per_decisive_fact"] == pr["mean_agents_per_rival_fact"]

(HERE / "agents.json").write_text(json.dumps(agents, indent=2, ensure_ascii=False) + "\n")
for k, p in POOL_FILES.items():
    p.write_text(json.dumps(pools[k], indent=2, ensure_ascii=False) + "\n")

print(f"wrote agents.json, controller_pool_A0.json, controller_pool_A2.json")
print(f"  {len(cands)} agent sets meet rules 1-5; {pr['equally_good_alternatives']} equally good alternatives")
for k, g in groups.items():
    print(f"  agents {A(k)}: {sorted(set(g))}  held twice: {sorted({f for f in g if g.count(f) > 1})}")
print(f"  joint {pr['joint_posterior_A0_A1_A2']}  proofs {pr['minimal_proofs_assemblable_up_to_6_facts']}  "
      f"agents per decisive/rival fact {pr['mean_agents_per_decisive_fact']}/{pr['mean_agents_per_rival_fact']}")
for k in POOL_FILES:
    q = pools[k]["properties"]
    print(f"  pool {A(k)}: {q['new_facts_for_the_agents']} new facts (sum dP {q['sum_dP_toward_target_of_new_facts']}), "
          f"agents already hold {q['facts_the_agents_already_hold']}")
