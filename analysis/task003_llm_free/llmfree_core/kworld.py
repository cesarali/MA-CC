"""K independent projects with one split: an exact posterior engine with the same interface as world.World.

Task (decided 22 October 2026): the three people (Alice, Bruno, Chandra) work on K independent projects with the
SAME split: one person builds the pipeline alone in every project, the other two interview together. Each project p
has its own nine hidden numbers V_p (six skills, three cooperation values, each 1, 2 or 3, uniform and independent).
A split's score is the sum of its per-project scores; the answer (three possibilities) is the split with the highest
total. As in task_003, only completions with a unique winner count (here: a unique winner of the TOTAL score).

Facts: task_003's 49 true facts, one copy per project, ID "k<p>:<base id>" and index p * 49 + (base index). Project p's
facts constrain only V_p.

Exact posterior. For a fact set M, each project's surviving completions (those consistent with M's facts on that
project) are summarised by a histogram over the score differences (d1, d2) = (s0 - s1, s0 - s2), each in -8..8.
Projects are independent, so the histogram of the total differences is the 2-D convolution of the K histograms. A0
wins where D1 > 0 and D2 > 0; A1 where D1 < 0 and D2 > D1; A2 where D2 < 0 and D1 > D2; ties are excluded.
P(Aa | M) = (completions won by a) / (completions with a unique winner). Counts are combined in float64 by direct
convolution, so zero counts stay exactly zero and a proof gives exactly P = 1. K = 1 reproduces world.World exactly
(tested in finite_size/test_kworld.py).

Scores (verified against the task generator in the test): with V = (a_p, a_i, b_p, b_i, c_p, c_i, ab, ac, bc),
  s0 = a_p + b_i + c_i + bc,   s1 = b_p + a_i + c_i + ac,   s2 = c_p + a_i + b_i + ab.
"""
from __future__ import annotations
import hashlib, itertools, json, pathlib, re

import numpy as np
from scipy.signal import convolve2d

from .world import World

HERE = pathlib.Path(__file__).resolve().parent
_ID = re.compile(r"cf_x(\d\d)_(eq|ge|le)_(x(\d\d)|(\d))$")


def scores(v):
    return (v[0] + v[3] + v[5] + v[8], v[2] + v[1] + v[5] + v[7], v[4] + v[1] + v[3] + v[6])


def predicate(fid):
    """The truth function of a catalog fact on a nine-number vector, from its ID."""
    m = _ID.match(fid)
    if not m:
        raise ValueError(f"unknown fact id {fid}")
    i, op = int(m.group(1)), m.group(2)
    if m.group(4) is not None:
        j = int(m.group(4))
        return {"eq": lambda v: v[i] == v[j], "ge": lambda v: v[i] >= v[j], "le": lambda v: v[i] <= v[j]}[op]
    c = int(m.group(5))
    return {"eq": lambda v: v[i] == c, "ge": lambda v: v[i] >= c, "le": lambda v: v[i] <= c}[op]


class KWorld:
    """Same interface as world.World: fact_ids, fact_text, index, n_facts, factset, ids, posterior, proves."""

    def __init__(self, K: int, base_file: str | pathlib.Path = HERE / "world_task003.json"):
        base = World(base_file)
        self.K, self.base = K, base
        self.sha256 = hashlib.sha256(f"k_projects:{K}:{base.sha256}".encode()).hexdigest()
        self.nb = base.n_facts
        self.fact_ids = [f"k{p}:{fid}" for p in range(K) for fid in base.fact_ids]
        self.fact_text = [f"[project {p + 1}] {t}" for p in range(K) for t in base.fact_text]
        self.index = {f: i for i, f in enumerate(self.fact_ids)}
        self.n_facts = len(self.fact_ids)
        vecs = list(itertools.product((1, 2, 3), repeat=9))
        sc = np.array([scores(v) for v in vecs])
        d1, d2 = sc[:, 0] - sc[:, 1], sc[:, 0] - sc[:, 2]
        self.dmin = int(min(d1.min(), d2.min()))
        size = int(max(d1.max(), d2.max())) - self.dmin + 1
        self.size = size
        self._cls = {}                                   # (i, j) -> bitmask over all 3^9 vectors
        for k, (a, b) in enumerate(zip(d1 - self.dmin, d2 - self.dmin)):
            self._cls[(int(a), int(b))] = self._cls.get((int(a), int(b)), 0) | (1 << k)
        self._fact = []                                  # base fact -> bitmask over all 3^9 vectors
        for fid in base.fact_ids:
            pr = predicate(fid)
            m = 0
            for k, v in enumerate(vecs):
                if pr(v):
                    m |= 1 << k
            self._fact.append(m)
        self._all = (1 << len(vecs)) - 1
        self._hist_cache: dict[int, np.ndarray] = {}
        self._pair_cache: dict = {}
        self._table_cache: dict = {}
        self._cache: dict[int, tuple[float, float, float]] = {}
        self._vecs, self._d = vecs, (d1, d2)

    # --------------------------------------------------------------- interface
    def factset(self, fact_ids) -> int:
        s = 0
        for f in fact_ids:
            s |= 1 << self.index[f]
        return s

    def ids(self, s: int) -> list[str]:
        return [self.fact_ids[i] for i in range(self.n_facts) if (s >> i) & 1]

    def _hist(self, sub: int) -> np.ndarray:
        h = self._hist_cache.get(sub)
        if h is None:
            m = self._all
            i = 0
            x = sub
            while x:
                if x & 1:
                    m &= self._fact[i]
                x >>= 1
                i += 1
            h = np.zeros((self.size, self.size))
            for (a, b), cm in self._cls.items():
                h[a, b] = (m & cm).bit_count()
            if len(self._hist_cache) > 500_000:
                self._hist_cache.clear()
            self._hist_cache[sub] = h
        return h

    def total_hist(self, s: int) -> np.ndarray:
        full = (1 << self.nb) - 1
        h = None
        for p in range(self.K):
            hp = self._hist((s >> (p * self.nb)) & full)
            h = hp if h is None else convolve2d(h, hp, mode="full")
        return h

    # --------------------------------------------------------------- fast exact path
    def _pair(self, a: int, b: int) -> np.ndarray:
        """Convolved histogram of two projects (cached by their fact subsets)."""
        key = (a, b)
        h = self._pair_cache.get(key)
        if h is None:
            h = convolve2d(self._hist(a), self._hist(b), mode="full")
            if len(self._pair_cache) > 200_000:
                self._pair_cache.clear()
            self._pair_cache[key] = h
        return h

    def _tables(self, key, g: np.ndarray, off: int):
        """Cumulative tables of a histogram g over (v1, v2) (v = index + off) giving, for any shift u, the mass with
        u + v in each winner region: A0 (D1 > 0, D2 > 0), A1 (D1 < 0, D2 > D1), A2 (D2 < 0, D1 > D2)."""
        t = self._table_cache.get(key)
        if t is not None:
            return t
        n = g.shape[0]
        # A0: suffix sums in both coordinates
        s0 = g[::-1, ::-1].cumsum(0).cumsum(1)[::-1, ::-1]
        # A1: coordinates (v1, w = v2 - v1); prefix in v1, suffix in w
        wsz = 2 * n - 1
        g1 = np.zeros((n, wsz))
        g2 = np.zeros((n, wsz))
        i = np.arange(n)
        for j in range(n):
            g1[i, (j - i) + n - 1] += g[i, j]          # v1 index i, w index (j - i)
            g2[j, (i - j) + n - 1] += g[i, j]          # v2 index j, z index (i - j)  (z = v1 - v2)
        c1 = g1.cumsum(0)[:, ::-1].cumsum(1)[:, ::-1]  # prefix over v1, suffix over w
        c2 = g2.cumsum(0)[:, ::-1].cumsum(1)[:, ::-1]  # prefix over v2, suffix over z
        t = (s0, c1, c2, n, off)
        if len(self._table_cache) > 200_000:
            self._table_cache.clear()
        self._table_cache[key] = t
        return t

    def _combine(self, h: np.ndarray, hoff: int, t) -> tuple[float, float, float]:
        """Counts of A0, A1, A2 winners in the convolution of h (offset hoff) with the histogram behind tables t."""
        s0, c1, c2, n, goff = t
        m = h.shape[0]
        u = np.arange(m) + hoff
        U1, U2 = np.meshgrid(u, u, indexing="ij")
        nz = h > 0
        U1, U2, w = U1[nz], U2[nz], h[nz]
        # A0: v1 > -u1, v2 > -u2  ->  v1 >= -u1 + 1 ; index = value - goff
        a = np.clip(-U1 + 1 - goff, 0, n)
        b = np.clip(-U2 + 1 - goff, 0, n)
        S0 = np.pad(s0, ((0, 1), (0, 1)))
        c0 = float((w * S0[a, b]).sum())
        # A1: v1 <= -u1 - 1 (prefix up to index), w' = v2 - v1 >= u1 - u2 + 1
        P1 = np.pad(c1, ((1, 0), (0, 1)))              # row 0 = empty prefix, extra column = empty suffix
        r = np.clip(-U1 - 1 - goff + 1, 0, n)          # number of rows included = index + 1
        k = np.clip(U1 - U2 + 1 + (n - 1), 0, 2 * n - 1)
        cc1 = float((w * P1[r, k]).sum())
        P2 = np.pad(c2, ((1, 0), (0, 1)))
        r2 = np.clip(-U2 - 1 - goff + 1, 0, n)
        k2 = np.clip(U2 - U1 + 1 + (n - 1), 0, 2 * n - 1)
        cc2 = float((w * P2[r2, k2]).sum())
        return c0, cc1, cc2

    def posterior(self, s: int) -> tuple[float, float, float]:
        hit = self._cache.get(s)
        if hit is None and self.K in (2, 4):
            full = (1 << self.nb) - 1
            sub = [(s >> (p * self.nb)) & full for p in range(self.K)]
            if self.K == 2:
                h, hoff = self._hist(sub[0]), self.dmin
                t = self._tables(("h", sub[1]), self._hist(sub[1]), self.dmin)
            else:
                h, hoff = self._pair(sub[0], sub[1]), 2 * self.dmin
                t = self._tables(("p", sub[2], sub[3]), self._pair(sub[2], sub[3]), 2 * self.dmin)
            c0, c1, c2 = self._combine(h, hoff, t)
            tot = c0 + c1 + c2
            hit = (c0 / tot, c1 / tot, c2 / tot)
            if len(self._cache) >= 2_000_000:
                self._cache.clear()
            self._cache[s] = hit
            return hit
        return self.posterior_direct(s) if hit is None else hit

    def posterior_direct(self, s: int) -> tuple[float, float, float]:
        hit = self._cache.get(s)
        if hit is None:
            h = self.total_hist(s)
            off = self.K * self.dmin
            n = h.shape[0]
            D = np.arange(n) + off
            D1, D2 = D[:, None], D[None, :]
            c0 = h[(D1 > 0) & (D2 > 0)].sum()
            c1 = h[(D1 < 0) & (D2 > D1)].sum()
            c2 = h[(D2 < 0) & (D1 > D2)].sum()
            tot = c0 + c1 + c2
            hit = (c0 / tot, c1 / tot, c2 / tot)
            if len(self._cache) >= 2_000_000:
                self._cache.clear()
            self._cache[s] = hit
        return hit

    def proves(self, s: int, k: int) -> bool:
        return self.posterior(s)[k] == 1.0


def load_world(path: str | pathlib.Path):
    """world.World for a plain world file; KWorld for a K-projects description {"type": "k_projects", "K": K,
    "base": <base world file, relative to this description>}."""
    path = pathlib.Path(path)
    d = json.loads(path.read_text())
    if d.get("type") == "k_projects":
        return KWorld(int(d["K"]), (path.parent / d["base"]).resolve())
    return World(path)
