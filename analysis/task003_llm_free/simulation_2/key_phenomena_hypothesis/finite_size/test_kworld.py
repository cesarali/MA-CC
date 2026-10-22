"""Checks of the K-projects posterior engine (llmfree_core/kworld.py). Run from the repository root:
    .venv/bin/python -m pytest analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/finite_size/test_kworld.py -q
"""
import itertools, pathlib, random, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]          # analysis/task003_llm_free
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/rsanchez/Projects/MA-CC/src")
from llmfree_core.world import World  # noqa: E402
from llmfree_core.kworld import KWorld, scores  # noqa: E402

BASE = World(ROOT / "llmfree_core" / "world_task003.json")
K1 = KWorld(1)


def test_scores_match_generator():
    from mas_cc.musr_team_allocation_generator.latent_problem import problem_from_latent_values
    rng = random.Random(0)
    vecs = list(itertools.product((1, 2, 3), repeat=9))
    for v in rng.sample(vecs, 400) + [(3, 1, 1, 2, 2, 1, 1, 1, 2)]:
        assert tuple(problem_from_latent_values(v).candidate_scores) == scores(v)


def test_fact_predicates_match_world_masks():
    """Each base fact's truth set, restricted to unique-winner vectors, equals world_task003.json's mask."""
    vecs = list(itertools.product((1, 2, 3), repeat=9))
    uniq = [k for k, v in enumerate(vecs) if sorted(scores(v))[-1] != sorted(scores(v))[-2]]
    assert len(uniq) == 14388
    for j in range(BASE.n_facts):
        m = 0
        for bit, k in enumerate(uniq):
            if (K1._fact[j] >> k) & 1:
                m |= 1 << bit
        assert m == BASE._fact_mask[j], BASE.fact_ids[j]


def test_k1_equals_world_on_random_sets():
    rng = random.Random(1)
    for _ in range(300):
        s = 0
        for i in rng.sample(range(BASE.n_facts), rng.randint(0, 8)):
            s |= 1 << i
        a, b = BASE.posterior(s), K1.posterior(s)
        assert all(abs(x - y) < 1e-12 for x, y in zip(a, b)), (s, a, b)
        assert (a[0] == 1.0) == (b[0] == 1.0)


def test_k2_independence_and_symmetry():
    K2 = KWorld(2)
    assert K2.posterior(0) == K2.posterior(0)
    p = K2.posterior(0)
    assert abs(p[0] - p[1]) < 1e-12 and abs(p[1] - p[2]) < 1e-12        # the prior is symmetric
    # the same fact in project 1 or project 2 gives the same posterior
    for j in range(0, BASE.n_facts, 7):
        assert all(abs(x - y) < 1e-12 for x, y in zip(K2.posterior(1 << j), K2.posterior(1 << (BASE.n_facts + j))))


def test_fast_path_equals_direct():
    """The table-based combination (K = 2, 4) gives exactly the direct convolution's posterior."""
    rng = random.Random(2)
    for K in (2, 4):
        w = KWorld(K)
        for _ in range(250):
            s = 0
            for i in rng.sample(range(w.n_facts), rng.randint(0, 14)):
                s |= 1 << i
            fast = w.posterior(s)
            w._cache.clear()
            direct = w.posterior_direct(s)
            w._cache.clear()
            assert all(abs(x - y) < 1e-12 for x, y in zip(fast, direct)), (K, s, fast, direct)
            assert [x == 1.0 for x in fast] == [x == 1.0 for x in direct]
