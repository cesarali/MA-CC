"""Build the four 15-agent setups (29 October 2026; researcher's design).

  task003/symmetric, object_placement/symmetric
      15 agents, one distinct fact each, jointly proving A0 (P(A0) = 1). Chosen as a subset of the original 24-agent
      setup's 18 distinct agent facts. The two original controller pools (A0, A2) are kept unchanged.
  task003/no_solution, object_placement/no_solution
      15 agents, one distinct fact each, whose combined posterior is approximately uniform over A0/A1/A2, and one
      12-fact controller pool, used for both targets, also approximately uniform; no fact in both; agents and pool
      together prove nothing. Found by a seeded joint search (simulated annealing).

Rebuild rather than edit: the output is deterministic. Usage, from the repository root:
    caffeinate -i .venv/bin/python analysis/15_agent_simulation/build_setups.py
"""
from __future__ import annotations
import itertools, json, math, pathlib, random, sys

HERE = pathlib.Path(__file__).resolve().parent
A = HERE.parent
sys.path.insert(0, str(A / "task003_llm_free"))
from llmfree_core.kworld import load_world  # noqa: E402
from llmfree_core.setups import load_setup  # noqa: E402

TASKS = {
    "task003": dict(world=A / "task003_llm_free/llmfree_core/world_task003.json",
                    world_rel="task003_llm_free/llmfree_core/world_task003.json",
                    old_dir=A / "task003_llm_free/experimental_setup", sym="task003_symmetric", nos="task003_nosolution"),
    "object_placement": dict(world=A / "object_placements_llm_free/world/world_op1.json",
                             world_rel="object_placements_llm_free/world/world_op1.json",
                             old_dir=A / "object_placements_llm_free/experimental_setup", sym="op_symmetric",
                             nos="op_nosolution"),
}
N_AGENTS, POOL_SIZE = 15, 12
SEED = 20261029
STATUS = "BUILT 2026-10-29 by 15_agent_simulation/build_setups.py"
PROVED = 1 - 1e-9


class Eval:
    def __init__(self, w):
        self.w, self.cache = w, {}
        self.single = [self.post([f]) for f in range(w.n_facts)]

    def post(self, facts):
        m = 0
        for f in facts:
            m |= 1 << f
        if m not in self.cache:
            self.cache[m] = tuple(float(x) for x in self.w.posterior(m))
        return self.cache[m]

    def lean(self, f):
        p = self.single[f]
        return max(range(3), key=lambda a: p[a])

    def neutral(self, f):
        return max(self.single[f]) < 1 / 3 + 0.01


def dev(p):
    return max(abs(x - 1 / 3) for x in p)


def lean_counts(ev, facts):
    c = [0, 0, 0]
    for f in facts:
        if not ev.neutral(f):
            c[ev.lean(f)] += 1
    return c


# ------------------------------------------------------------------ symmetric: 15 of the original 18 agent facts
def build_symmetric(ev, old):
    distinct = sorted(set(old.agent_facts))
    pools = set(old.pools[0]) | set(old.pools[2])
    best = None
    for drop in itertools.combinations(distinct, len(distinct) - N_AGENTS):
        keep = [f for f in distinct if f not in drop]
        if ev.post(keep)[0] < PROVED:
            continue
        if any(ev.post([f])[0] >= PROVED for f in keep) or any(ev.post(list(pr))[0] >= PROVED for pr in itertools.combinations(keep, 2)):
            continue
        c = lean_counts(ev, keep)
        key = (len(set(keep) & pools), max(c) - min(c), sum(ev.neutral(f) for f in keep), tuple(keep))
        if best is None or key < best[0]:
            best = (key, keep)
    return best[1]


# ------------------------------------------------------------------ no-solution: joint uniform agents + pool
def cost(ev, A_, P_, lean_weight, neutral_weights):
    """Uniformity of agents and pool (dominant), no proof together (hard), few neutral facts (mostly in the agents), balanced single-fact
    leanings (normalised imbalance, weighted per task: see LEAN_WEIGHT)."""
    pa, pp, pu = ev.post(A_), ev.post(P_), ev.post(A_ + P_)
    c = 10 * dev(pa) + 10 * dev(pp)
    if max(pu) >= PROVED:
        c += 50
    w_neutral, w_split = neutral_weights
    na, npool = sum(ev.neutral(f) for f in A_), sum(ev.neutral(f) for f in P_)
    c += w_neutral * (na + npool) + w_split * abs(na / len(A_) - npool / len(P_))
    for S in (A_, P_):
        cnt = lean_counts(ev, S)
        n = sum(cnt)
        c += lean_weight * sum(abs(x - n / 3) for x in cnt) / max(n, 1)
    return c


# Leaning-balance weight. task003: 1.0 gives 5/5/5 agent leanings and 4/4/4 pool leanings at a small cost in uniformity
# (max deviation 0.017). Object Placements: its world has few informative A1-leaning facts; a strong weight does not
# balance the leanings and only adds neutral pool facts, so a mild weight is used (tested 29 Oct 2026).
LEAN_WEIGHT = {"task003": 1.0, "object_placement": 0.3}
# Neutral-fact penalties (per neutral fact, and for an unequal neutral fraction between agents and pool). task003 has no
# neutral facts. Object Placements has 18 neutral facts among its 34, so at least 11 of the 27 chosen facts must be
# neutral; they are split so that agents and pool carry the same neutral fraction. Without the split term the search was
# indifferent and put 9 to 12 neutral facts on one side (tested 29 Oct 2026).
NEUTRAL_WEIGHTS = {"task003": (0.05, 0.0), "object_placement": (0.05, 1.0)}


def build_nosolution(ev, lean_weight, neutral_weights, restarts=12, iters=25000):
    n = ev.w.n_facts
    best = None
    for r in range(restarts):
        rng = random.Random(SEED + r)
        facts = list(range(n))
        rng.shuffle(facts)
        A_, P_, R_ = facts[:N_AGENTS], facts[N_AGENTS:N_AGENTS + POOL_SIZE], facts[N_AGENTS + POOL_SIZE:]
        c, T = cost(ev, A_, P_, lean_weight, neutral_weights), 1.0
        for _ in range(iters):
            A2, P2, R2 = A_[:], P_[:], R_[:]
            u = rng.random()
            if u < 0.4 and R2:
                i, j = rng.randrange(N_AGENTS), rng.randrange(len(R2)); A2[i], R2[j] = R2[j], A2[i]
            elif u < 0.8 and R2:
                i, j = rng.randrange(POOL_SIZE), rng.randrange(len(R2)); P2[i], R2[j] = R2[j], P2[i]
            else:
                i, j = rng.randrange(N_AGENTS), rng.randrange(POOL_SIZE); A2[i], P2[j] = P2[j], A2[i]
            c2 = cost(ev, A2, P2, lean_weight, neutral_weights)
            if c2 < c or rng.random() < math.exp(-(c2 - c) / T):
                A_, P_, R_, c = A2, P2, R2, c2
            T *= 0.9996
        key = (c, tuple(sorted(A_)), tuple(sorted(P_)))
        if best is None or key < best:
            best = key
        print(f"  restart {r}: cost {c:.4f}", flush=True)
    return sorted(best[1]), sorted(best[2])


# ------------------------------------------------------------------ writing
def r4(p):
    return [round(x, 4) for x in p]


def fact_row(ev, f, pool_membership):
    w = ev.w
    p = ev.single[f]
    return {"fact_id": w.fact_ids[f], "text": w.fact_text[f],
            "dP": {f"ALLOCATION_{a}": round(p[a] - 1 / 3, 4) for a in range(3)},
            "favours": [] if ev.neutral(f) else [f"ALLOCATION_{a}" for a in range(3) if abs(p[a] - max(p)) < 1e-9],
            "also_in_controller_pool": pool_membership.get(f)}


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_setup(task, kind, ev, cfg, agents, pools):
    folder = HERE / task / kind
    folder.mkdir(parents=True, exist_ok=True)
    w = ev.w
    membership = {}
    for name, facts in pools.items():
        for f in facts:
            membership[f] = (membership.get(f, "") + "," + name).strip(",")
    agent_ids = [f"agent_{i + 1:03d}" for i in range(N_AGENTS)]
    order = sorted(agents, key=lambda f: w.fact_ids[f])
    joint = ev.post(order)
    singles = [ev.single[f] for f in order]
    props = {
        "agents": N_AGENTS, "facts_per_agent": 1, "distinct_facts": len(set(order)),
        "joint_posterior_A0_A1_A2": r4(joint),
        "proves": next((f"ALLOCATION_{a}" for a in range(3) if joint[a] >= PROVED), None),
        "max_deviation_from_uniform": round(dev(joint), 4),
        "single_fact_leanings_A0_A1_A2": lean_counts(ev, order),
        "neutral_single_facts": sum(ev.neutral(f) for f in order),
        "mean_individual_belief_A0_A1_A2": r4([sum(s[a] for s in singles) / len(singles) for a in range(3)]),
        "facts_also_in_a_controller_pool": [w.fact_ids[f] for f in order if f in membership],
    }
    write_json(folder / "agents.json", {
        "setup": f"{task}_{kind}_15", "status": STATUS, "world": cfg["world_rel"],
        "truth": "ALLOCATION_0", "false_target": "ALLOCATION_2",
        "agent_assignments": {a: [w.fact_ids[f]] for a, f in zip(agent_ids, order)},
        "facts": [fact_row(ev, f, membership) for f in order],
        "properties": props,
    })
    for name, facts in pools.items():
        pf = sorted(facts, key=lambda f: w.fact_ids[f])
        pj, uj = ev.post(pf), ev.post(pf + order)
        targets = {"controller_pool_A0.json": ["ALLOCATION_0"], "controller_pool_A2.json": ["ALLOCATION_2"],
                   "controller_pool.json": ["ALLOCATION_0", "ALLOCATION_2"]}[name]
        write_json(folder / name, {
            "setup": f"{task}_{kind}_15", "used_when_controller_targets": targets, "status": STATUS,
            "fact_ids": [w.fact_ids[f] for f in pf],
            "facts": [fact_row(ev, f, {}) for f in pf],
            "properties": {
                "size": len(pf), "joint_posterior_A0_A1_A2": r4(pj),
                "proves_any_allocation": max(pj) >= PROVED,
                "max_deviation_from_uniform": round(dev(pj), 4),
                "single_fact_leanings_A0_A1_A2": lean_counts(ev, pf),
                "neutral_single_facts": sum(ev.neutral(f) for f in pf),
                "facts_the_agents_already_hold": [w.fact_ids[f] for f in pf if f in set(order)],
                "new_facts_for_the_agents": sum(f not in set(order) for f in pf),
                "agents_plus_whole_pool_posterior_A0_A1_A2": r4(uj),
                "agents_plus_whole_pool_proves": next((f"ALLOCATION_{a}" for a in range(3) if uj[a] >= PROVED), None),
            },
        })
    return props


def main():
    summary = {}
    for task, cfg in TASKS.items():
        w = load_world(cfg["world"])
        ev = Eval(w)
        old_sym = load_setup(cfg["sym"], cfg["old_dir"], w)
        agents = build_symmetric(ev, old_sym)
        pools = {"controller_pool_A0.json": list(old_sym.pools[0]), "controller_pool_A2.json": list(old_sym.pools[2])}
        summary[f"{task}/symmetric"] = write_setup(task, "symmetric", ev, cfg, agents, pools)
        print(task, "symmetric done", flush=True)
        agents, pool = build_nosolution(ev, LEAN_WEIGHT[task], NEUTRAL_WEIGHTS[task])
        summary[f"{task}/no_solution"] = write_setup(task, "no_solution", ev, cfg, agents, {"controller_pool.json": pool})
        print(task, "no_solution done", flush=True)
    write_json(HERE / "build_summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
