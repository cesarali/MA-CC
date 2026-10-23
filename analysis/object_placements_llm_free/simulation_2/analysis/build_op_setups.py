"""Search for an Object Placements world (llmfree_core/op_world.py) that admits a symmetric and a no-proof setup built
with task003's rules, and write the world and both setups.

World: one actual story (3 people, 2 items, 3 locations, 3 moves), a question "where would person P look for item I"
(P is not the last mover of I) whose answer (P's final belief) differs from where the item really is. Answers:
A0 = P's belief (correct), A2 = the item's true final location (the controller's false target), A1 = the third location.

op_symmetric (mirrors task003_symmetric):
  24 agents, one fact each, 18 distinct facts (6 held twice); all agent facts together prove A0; no single fact and no
  pair of agent facts proves A0. A2 pool: 12 facts, 10 not held by agents, 2 held by agents, chosen greedily to raise
  P(A2) (the pool alone cannot prove A2: all facts are true). A0 pool: 12 facts, the 6 remaining true facts plus 6 held
  by agents, chosen to raise P(A0) without proving it. (task003 used 10 new facts in each pool; a story has only 34
  true facts, so the A0 pool reuses more agent facts. The A2 pool, the one the false controller uses, is unchanged.)
op_nosolution (mirrors task003_nosolution):
  24 agents, one fact each, 15 distinct facts (9 held twice); every agent fact read alone is at most 45% sure of any
  answer; all agent facts together sit as close as possible to (0.5, 0, 0.5); no proof. One pool of 12 facts, none
  held by agents, used for both targets; agents + the whole pool still prove nothing.

The search samples candidate stories (seeded) and, per story, random fact subsets; it keeps the first world that
admits both setups, preferring the closest no-proof balance and the strongest A2 pool. Writes (paths relative to analysis/object_placements_llm_free/)
  world/world_op1.json                             {"type": "object_placements", story, question, answer_order, names}
  experimental_setup/op_symmetric/                 agents.json, controller_pool_A0.json, controller_pool_A2.json
  experimental_setup/op_nosolution/                agents.json, controller_pool.json
and prints the measured properties. The text written inside these files ("world", "built_by", "what_this_is") still
names the locations at build time (22 Oct 2026, under task003_llm_free/); it is kept so that a rebuild gives the same
bytes, and so the same file hashes as recorded in the run manifest.

Usage, from the repository root:
    caffeinate -i .venv/bin/python analysis/object_placements_llm_free/simulation_2/analysis/build_op_setups.py
"""
from __future__ import annotations
import itertools, json, pathlib, random, sys

HERE = pathlib.Path(__file__).resolve().parent
OP = HERE.parents[1]                                              # analysis/object_placements_llm_free
sys.path.insert(0, str(OP.parent / "task003_llm_free"))          # shared llmfree_core
from llmfree_core import op_world as op  # noqa: E402

SEED, N_STORIES, N_SUBSETS = 20261022, 400, 3000
NAMES = {"people": ["Alex", "Blair", "Casey"], "items": ["notebook", "keys"], "locations": ["desk", "shelf", "drawer"]}


def final_locations(start, moves):
    loc = list(start)
    for (mv, it, d, s) in moves:
        loc[it] = d
    return loc


def candidates(rng):
    full = [s for s in op.stories() if len(s[1]) == 3]
    rng.shuffle(full)
    for st, mv, w in full:
        last = {it: m for (m, it, d, s) in mv}
        for (p, i) in itertools.product(range(3), range(2)):
            if i not in last or last[i] == p:
                continue
            belief, actual = op.answer_of(st, mv, (p, i)), final_locations(st, mv)[i]
            if belief == actual:
                continue
            other = [l for l in range(3) if l not in (belief, actual)][0]
            yield st, mv, (p, i), [belief, other, actual]


def bits_of(world, facts):
    return world.factset(facts)


def proves_any(world, s):
    return max(world.posterior(s)) == 1.0


def symmetric_design(world, rng):
    T = list(world.fact_ids)
    single = {f: world.posterior(world.factset([f])) for f in T}
    best = None
    for _ in range(N_SUBSETS):
        F = rng.sample(T, 18)
        if world.posterior(world.factset(F))[0] != 1.0:
            continue
        if any(max(single[f]) == 1.0 for f in F):
            continue
        if any(proves_any(world, world.factset(pair)) for pair in itertools.combinations(F, 2)):
            continue
        rest = [f for f in T if f not in F]
        a2 = greedy_pool(world, rest, F, 10, 2, target=2)
        p2 = world.posterior(world.factset(a2))[2]
        rest0 = [f for f in rest if f not in a2]
        a0 = greedy_pool(world, rest0, F, len(rest0), 12 - len(rest0), target=0, forbid_proof=True)
        if a0 is None:
            continue
        score = p2
        if best is None or score > best[0]:
            best = (score, F, a2, a0)
    return best


def greedy(world, pool, k, target):
    chosen = []
    for _ in range(k):
        cand = [f for f in pool if f not in chosen]
        chosen.append(max(cand, key=lambda f: world.posterior(world.factset(chosen + [f]))[target]))
    return chosen


def greedy_pool(world, new, held, n_new, n_held, target, forbid_proof=False):
    """n_new facts from `new` and n_held from `held`, greedily raising P(target); None if impossible."""
    chosen = []
    for src, n in ((new, n_new), (held, n_held)):
        for _ in range(n):
            cand = [f for f in src if f not in chosen]
            if forbid_proof:
                cand = [f for f in cand if world.posterior(world.factset(chosen + [f]))[target] < 1.0]
            if not cand:
                return None
            chosen.append(max(cand, key=lambda f: world.posterior(world.factset(chosen + [f]))[target]))
    return chosen


def nosolution_design(world, rng):
    T = list(world.fact_ids)
    eligible = [f for f in T if max(world.posterior(world.factset([f]))) <= 0.45]
    if len(eligible) < 15:
        return None
    best = None
    for _ in range(N_SUBSETS):
        F = rng.sample(eligible, 15)
        p = world.posterior(world.factset(F))
        if max(p) == 1.0:
            continue
        gap = abs(p[0] - 0.5) + abs(p[2] - 0.5) + p[1]
        if best is not None and gap >= best[0]:
            continue
        rest = [f for f in T if f not in F]
        for _ in range(40):
            pool = rng.sample(rest, 12) if len(rest) >= 12 else None
            if pool is None:
                break
            if proves_any(world, world.factset(F + pool)):
                continue
            best = (gap, F, pool)
            break
    return best


def main():
    rng = random.Random(SEED)
    op.table()
    for n, (st, mv, q, order) in enumerate(candidates(rng)):
        if n >= N_STORIES:
            break
        world = op.OPWorld(st, mv, q, order, NAMES)
        sym = symmetric_design(world, rng)
        if sym is None:
            print(n, "no symmetric design", flush=True)
            continue
        nos = nosolution_design(world, rng)
        print(n, "symmetric A2-pool P(A2) =", round(sym[0], 3), "| nosolution gap =",
              None if nos is None else round(nos[0], 3), flush=True)
        if nos is None or nos[0] > 0.05:
            continue
        write(world, sym, nos, st, mv, q, order)
        return
    print("no world found; enlarge N_STORIES / N_SUBSETS")


def agents_with_duplicates(F, n_agents, rng):
    facts = list(F) + rng.sample(list(F), n_agents - len(F))
    rng.shuffle(facts)
    return {f"agent_{k + 1:03d}": [f] for k, f in enumerate(facts)}


def write(world, sym, nos, st, mv, q, order):
    rng = random.Random(SEED + 1)
    desc = {"type": "object_placements", "start": list(st), "moves": [list(m[:3]) + [list(m[3])] for m in mv],
            "question": list(q), "answer_order": order, "names": NAMES,
            "what_this_is": "MuSR-style Object Placements world (llmfree_core/op_world.py); built by "
                            "key_phenomena_hypothesis/object_placements/build_op_setups.py"}
    (OP / "world" / "world_op1.json").write_text(json.dumps(desc, indent=1) + "\n")
    _, F, a2, a0 = sym
    out = OP / "experimental_setup" / "op_symmetric"
    out.mkdir(exist_ok=True)
    common = {"world": "llmfree_core/world_op1.json", "truth": "ALLOCATION_0", "false_target": "ALLOCATION_2",
              "status": "BUILT 2026-10-22", "built_by": "key_phenomena_hypothesis/object_placements/build_op_setups.py"}
    (out / "agents.json").write_text(json.dumps({"setup": "op-symmetric", **common,
                                                 "agent_assignments": agents_with_duplicates(F, 24, rng)}, indent=1))
    (out / "controller_pool_A2.json").write_text(json.dumps({"setup": "op-symmetric", **common, "fact_ids": a2}, indent=1))
    (out / "controller_pool_A0.json").write_text(json.dumps({"setup": "op-symmetric", **common, "fact_ids": a0}, indent=1))
    _, Fn, pool = nos
    out = OP / "experimental_setup" / "op_nosolution"
    out.mkdir(exist_ok=True)
    (out / "agents.json").write_text(json.dumps({"setup": "op-nosolution", **common,
                                                 "agent_assignments": agents_with_duplicates(Fn, 24, rng)}, indent=1))
    (out / "controller_pool.json").write_text(json.dumps({"setup": "op-nosolution", **common, "fact_ids": pool}, indent=1))
    w = world
    fs = lambda x: w.posterior(w.factset(x))  # noqa: E731
    print("story", st, mv, "question", q, "answers (A0, A1, A2) =", order)
    print("symmetric: agents", fs(F), "A2 pool", fs(a2), "A0 pool", fs(a0), "agents + A2 pool", fs(F + a2))
    print("nosolution: agents", fs(Fn), "pool", fs(pool), "agents + pool", fs(Fn + pool))


if __name__ == "__main__":
    main()
