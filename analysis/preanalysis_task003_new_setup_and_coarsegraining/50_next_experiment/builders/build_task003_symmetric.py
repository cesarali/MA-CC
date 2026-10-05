"""Build designs/task003_symmetric.json: the 24 agents and the two controller pools.

The two pools are fixed inputs (10_task_and_facts/symmetric_pools.json, designed
2 October; reasoning in 20_controller_redesign/symmetric_controller_and_coarse_graining.md §2).
This script chooses the agents. Rules, in order:

  1. 24 agents, one fact each, every fact private-eligible, multiplicity 1 or 2.
  2. 8 agents per allocation. An agent belongs to allocation k when its fact raises
     P(k) at least as much as any other allocation (a fact tied between two
     allocations may serve either).
  3. The A0 group is exactly the 6 decisive facts, so the population can prove A0.
  4. Equal new evidence for both controllers: the pool facts the agents already
     hold must be weight-matched PAIRS (A0-pool fact together with its A2 partner).
  5. Equal duplication: decisive facts and rival facts are held by the same mean
     number of agents (8/6 = 16/12, so 6 distinct facts in each rival group).
  6. Among what survives: fewest tied facts, then smallest overlap, then rival
     groups of most similar strength, then lexicographic order.

Duplicates: the 2 decisive facts in most of the agents' minimal proofs, and in each
rival group the 2 untied, non-overlapping facts in fewest of those proofs. Pool
facts the agents hold are held once each, so both controllers face the same
number of agents holding their overlap.

Run from the repository root with the project venv:
    .venv/bin/python analysis/preanalysis_task003_new_setup_and_coarsegraining/50_next_experiment/builders/build_task003_symmetric.py
"""
from __future__ import annotations
import csv, itertools, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
PRE = HERE.parents[1]
sys.path.insert(0, str(PRE / "10_task_and_facts"))
from engine import World  # noqa: E402

OUT = HERE.parent / "designs" / "task003_symmetric.json"
R = {r["fact"]: r for r in csv.DictReader(open(PRE / "10_task_and_facts" / "task_003_facts.csv"))}
POOLS = json.loads((PRE / "10_task_and_facts" / "symmetric_pools.json").read_text())
P0, P2 = POOLS["A0"], POOLS["A2"]
PAIR = dict(zip(P0, P2))

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
overlap0 = sorted(held & set(P0))
overlap2 = sorted(held & set(P2))

dup_dec = sorted(DEC, key=lambda f: (-in_proofs(f), f))[:2]
def rival_dups(S):
    ok = [f for f in S if not tied(f) and f not in P0 and f not in P2]
    return sorted(ok, key=lambda f: (in_proofs(f), f))[:2]
dup1, dup2 = rival_dups(S1), rival_dups(S2)
groups = {0: list(DEC) + dup_dec, 1: list(S1) + dup1, 2: list(S2) + dup2}
assert all(len(g) == 8 for g in groups.values())

slots = groups[0] + groups[1] + groups[2]
assign = {f"agent_{i + 1:03d}": [f] for i, f in enumerate(slots)}
mult = {f: slots.count(f) for f in sorted(held)}
beliefs = [W.posterior([IX[f]]) for f in slots]


def pool_block(ids, target, own):
    new = [f for f in ids if f not in held]
    return {
        "target": target,
        "used_only_when_controller_targets": target,
        "fact_ids": ids,
        "size": len(ids),
        "sum_dP_target": round(sum(dP(f, own) for f in ids), 4),
        "joint_posterior": post(ids),
        "proves_any_allocation": max(post(ids)) == 1.0,
        "held_by_agents": [f for f in ids if f in held],
        "agents_holding_those": sum(mult.get(f, 0) for f in ids),
        "new_facts": new,
        "n_new_facts": len(new),
        "sum_dP_target_new_facts": round(sum(dP(f, own) for f in new), 4),
    }


design = {
    "name": "task003-symmetric",
    "status": "FROZEN 2026-10-05",
    "note": ("Population CAN prove the truth (holds all 6 decisive facts). 24 agents, 8 per allocation. "
             "Two controller pools: the A0 pool is used only when the controller targets ALLOCATION_0, "
             "the A2 pool only when it targets ALLOCATION_2. Both controllers bring the same number and "
             "weight of facts the agents do not already hold, and decisive and rival facts are duplicated "
             "at the same rate."),
    "built_by": "50_next_experiment/builders/build_task003_symmetric.py",
    "task": "task_003",
    "world": list(W.vector),
    "gold_target": "ALLOCATION_0",
    "false_target": "ALLOCATION_2",
    "population_size": 24,
    "facts_per_agent": 1,
    "agents": {
        "groups": {f"A{k}": {"facts": sorted(set(g)), "slots": 8,
                              "duplicated": sorted({f for f in g if g.count(f) > 1}),
                              "tied_facts": {f: [f"A{j}" for j in sorted(favours(f))] for f in sorted(set(g)) if tied(f)}}
                   for k, g in groups.items()},
        "distinct_facts": sorted(held),
        "n_distinct": len(held),
        "slot_multiplicity": mult,
        "mean_multiplicity_decisive": round(sum(mult[f] for f in DEC) / 6, 4),
        "mean_multiplicity_rival": round(sum(mult[f] for f in set(S1) | set(S2)) / 12, 4),
        "joint_posterior": post(held),
        "decisive_held": DEC,
        "minimal_proofs_assemblable": len(proofs),
        "mean_individual_belief_A0_A1_A2": [round(sum(b[k] for b in beliefs) / 24, 4) for k in range(3)],
        "agent_assignments": assign,
    },
    "controller_pools": {
        "ALLOCATION_0": pool_block(P0, "ALLOCATION_0", 0),
        "ALLOCATION_2": pool_block(P2, "ALLOCATION_2", 2),
    },
    "pool_pairs": [{"A0": a, "A2": b, "dP_A0": round(dP(a, 0), 4), "dP_A2": round(dP(b, 2), 4),
                    "held_by_agents": a in held} for a, b in PAIR.items()],
    "selection": {"candidates_satisfying_rules_1_to_5": len(cands),
                  "equivalent_alternatives": sum(1 for c in cands if rank(c)[:3] == rank((S1, S2))[:3]) - 1},
}

# invariants
cp = design["controller_pools"]
assert design["agents"]["joint_posterior"][0] == 1.0
assert not cp["ALLOCATION_0"]["proves_any_allocation"] and not cp["ALLOCATION_2"]["proves_any_allocation"]
assert cp["ALLOCATION_0"]["n_new_facts"] == cp["ALLOCATION_2"]["n_new_facts"]
assert cp["ALLOCATION_0"]["agents_holding_those"] == cp["ALLOCATION_2"]["agents_holding_those"]
assert not set(P0) & set(P2) and not (set(P0) | set(P2)) & set(DEC)
assert held <= ELIG and max(mult.values()) <= 2
assert design["agents"]["mean_multiplicity_decisive"] == design["agents"]["mean_multiplicity_rival"]

OUT.write_text(json.dumps(design, indent=2) + "\n")
a, c = design["agents"], cp
print(f"wrote {OUT.relative_to(PRE)}")
print(f"  {len(cands)} agent sets satisfy rules 1-5; {design['selection']['equivalent_alternatives']} equivalent alternatives")
for k, g in a["groups"].items():
    print(f"  {k}: {g['facts']}  duplicated={g['duplicated']}  tied={g['tied_facts']}")
print(f"  joint {a['joint_posterior']}  proofs {a['minimal_proofs_assemblable']}  mean belief {a['mean_individual_belief_A0_A1_A2']}")
print(f"  multiplicity decisive {a['mean_multiplicity_decisive']} rival {a['mean_multiplicity_rival']}")
for t in c:
    print(f"  pool {t}: new {c[t]['n_new_facts']} (sum dP {c[t]['sum_dP_target_new_facts']}), held {c[t]['held_by_agents']} by {c[t]['agents_holding_those']} agents, joint {c[t]['joint_posterior']}")
