"""Checks of task003_llm_free/llmfree_core/op_world.py against MuSR's own generator.

MuSR's `ObjectPlacementsDataset.create_sequence_v2` (github.com/Zayne-Sprague/MuSR, src/dataset_types/
object_placements_dataset.py, MIT licence) is extracted from a local clone (path in MUSR below; clone with
`git clone https://github.com/Zayne-Sprague/MuSR`) without its heavy imports and run many times with our settings.
The empirical frequencies of stories must match op_world's exact probabilities.

Run from the repository root:
    .venv/bin/python -m pytest analysis/object_placements_llm_free/simulation_2/analysis/test_op_world.py -q
"""
import ast, collections, os, pathlib, random, sys, textwrap
from copy import deepcopy

import pytest

OP = pathlib.Path(__file__).resolve().parents[2]                # analysis/object_placements_llm_free
sys.path.insert(0, str(OP.parent / "task003_llm_free"))        # shared llmfree_core
from llmfree_core import op_world as op  # noqa: E402

MUSR = pathlib.Path(os.environ.get("MUSR_CLONE", "/private/tmp/claude-501/-Users-rsanchez-Projects/"
                                   "341a8223-b37a-485a-9adb-4e95056830ca/scratchpad/musr"))
PEOPLE, ITEMS, LOCS = ["p0", "p1", "p2"], ["i0", "i1"], ["l0", "l1", "l2"]


def musr_generator():
    src = (MUSR / "src/dataset_types/object_placements_dataset.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "create_sequence_v2")
    code = textwrap.dedent(ast.get_source_segment(src, fn))
    ns = {"random": random, "deepcopy": deepcopy, "List": list, "Tuple": tuple}
    exec(code, ns)
    return ns["create_sequence_v2"]


@pytest.mark.skipif(not (MUSR / "src").exists(), reason="no local MuSR clone")
def test_story_frequencies_match_musr():
    gen = musr_generator()
    random.seed(7)
    n = 60000
    counts = collections.Counter()
    for _ in range(n):
        start = [(it, random.choice(LOCS)) for it in ITEMS]
        events, beliefs, actual, structure = gen(None, list(ITEMS), list(LOCS), list(PEOPLE), max_sequence_length=3,
                                                 chance_subject_sees=1 / 3, initial_starting_positions=start)
        moves = []
        for e in structure[1:]:
            w = e["event"].split(" ")
            mv, it, d = PEOPLE.index(w[0]), ITEMS.index(w[3]), LOCS.index(w[-1].rstrip("."))
            seen = tuple(sorted(PEOPLE.index(s.split(" ")[0]) for s in e["sequence"] if " saw " in s))
            moves.append((mv, it, d, seen))
        counts[(tuple(LOCS.index(l) for _, l in start), tuple(moves))] += 1
    exact = {(s, m): float(w) for s, m, w in op.stories()}
    # the 40 stories that are most probable EXACTLY (choosing by empirical counts would select lucky fluctuations)
    top = sorted(exact, key=exact.get, reverse=True)[:40]
    worst = max(abs(counts[k] / n - exact[k]) / (exact[k] * (1 - exact[k]) / n) ** 0.5 for k in top)
    assert worst < 4.0, worst
    # aggregates: number of moves seen by others, and the final location of each item
    for f in (lambda k: sum(len(m[3]) for m in k[1]), lambda k: tuple(next((m[2] for m in reversed(k[1]) if m[1] == i),
                                                                             k[0][i]) for i in range(2))):
        e, x = collections.Counter(), collections.Counter()
        for k, c in counts.items():
            e[f(k)] += c / n
        for k, w in exact.items():
            x[f(k)] += w
        for key in x:
            if x[key] > 0.01:
                assert abs(e[key] - x[key]) < 4 * (x[key] * (1 - x[key]) / n) ** 0.5 + 0.004, (key, e[key], x[key])
    assert set(counts) <= set(exact)


def test_world_basics():
    st, mv, w = next(s for s in op.stories() if len(s[1]) == 3)
    last = {it: m for (m, it, d, s) in mv}
    q = next((p, i) for p in range(3) for i in range(2) if last.get(i) != p)
    ans = op.answer_of(st, mv, q)
    others = [l for l in range(3) if l != ans]
    world = op.OPWorld(st, mv, q, [ans] + others)
    assert abs(sum(world.posterior(0)) - 1) < 1e-12
    assert world.posterior((1 << world.n_facts) - 1)[0] == 1.0          # all true facts determine the story


def test_fast_table_equals_slow_classes():
    """The table-based class construction equals a direct per-story computation (posteriors on random fact sets)."""
    from fractions import Fraction
    rng = random.Random(4)
    st, mv, w = rng.choice([s for s in op.stories() if len(s[1]) == 3])
    last = {it: m for (m, it, d, s) in mv}
    q = next((p, i) for p in range(3) for i in range(2) if last.get(i) != p)
    ans = op.answer_of(st, mv, q)
    world = op.OPWorld(st, mv, q, [ans] + [l for l in range(3) if l != ans])
    true = set(world.fact_ids)
    for _ in range(30):
        held = set(rng.sample(sorted(true), rng.randint(1, 6)))
        c = [Fraction(0)] * 3
        for (s2, m2, w2) in op.stories():
            if held <= op.propositions(s2, m2, q):
                c[world.answer_order.index(op.answer_of(s2, m2, q))] += w2
        tot = sum(c)
        direct = [float(x / tot) for x in c]
        fast = world.posterior(world.factset(held))
        assert all(abs(x - y) < 1e-9 for x, y in zip(direct, fast)), (held, direct, fast)


def test_cached_world_equals_fresh():
    import json
    from llmfree_core.op_world import OPWorld, load_op_world
    path = OP / "world" / "world_op1.json"
    d = json.loads(path.read_text())
    fresh = OPWorld(d["start"], d["moves"], d["question"], d["answer_order"], d.get("names"))
    cached = load_op_world(path)
    assert fresh.fact_ids == cached.fact_ids and fresh.sha256 == cached.sha256
    rng = random.Random(9)
    for _ in range(200):
        s = 0
        for j in rng.sample(range(fresh.n_facts), rng.randint(0, 10)):
            s |= 1 << j
        assert fresh.posterior(s) == cached.posterior(s)
