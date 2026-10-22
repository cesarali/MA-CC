"""MuSR Object Placements as an exact world (same interface as world.World).

Source task: Sprague et al., "MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning"
(arXiv 2310.16049; github.com/Zayne-Sprague/MuSR, MIT licence). Their generator
(`src/dataset_types/object_placements_dataset.py`, `create_sequence_v2`, with the settings of
`musr_dataset_scripts/create_object_placements.py`) builds a story of moves; a language model then writes the text.
This module keeps only the symbolic story and reproduces the generator's sampling rules exactly:

  3 people, 2 items, 3 locations (MuSR's most common number of answer options), 3 moves, chance 1/3 that each person
  other than the mover sees a move (MuSR: 0.33), at most 2 arrivals of an item at one location
  (max_location_use_per_item = 2), one mover per move. Starting positions: uniform and independent (in MuSR a language
  model chooses them). Per move: the mover is drawn uniformly among people who know where at least one item is; the
  item uniformly among the items the mover knows; the destination uniformly among the item's allowed locations other
  than its current one. The mover sees any other item already at the destination. Every other person sees the move
  with chance 1/3; whoever does not see it loses track of that item. A person's belief about an item is its true
  location while they know it, otherwise their last belief.

The question (as in MuSR) is "where would person P look for item I": the answer is P's belief after the last move
(P is not the last mover of I). The three answers are the three locations, in the order given by `answer_order`
(answer_order[0] is the correct location, answer_order[2] the controller's false target).

Facts (abstract propositions, each with an ID; the catalog is defined for every story):
  start:i:l      item i starts at location l           mover:k:p   move k is made by person p
  item:k:i       move k moves item i                     dest:k:l    move k takes the item to location l
  saw:k:p        person p (not the mover) saw move k     missed:k:p  person p (not the mover) missed move k
  belief:t:p:i:l after move t, person p believes item i is at location l (t = 1..3), except P's final belief about I,
                 which is the answer itself
The world's facts are the true propositions of the actual story (34 here). The prior over stories is the generator's
own sampling probability. Exact posterior: stories that agree on every true fact of the actual story and on the answer
cannot be told apart by any set of true facts, so they are merged (probabilities summed); a fact set's posterior is the
probability-weighted share of surviving classes per answer.
"""
from __future__ import annotations
import hashlib, itertools, json, pathlib
from fractions import Fraction

import numpy as np

P, I, L, M = 3, 2, 3, 3
P_SEE = Fraction(1, 3)
MAX_USE = 2


# ------------------------------------------------------------------ generator (exact enumeration)
def replay(start, moves):
    """Beliefs after each move: bel[t][p][i] (t = 0 is the start)."""
    loc, known = list(start), [[True] * P for _ in range(I)]
    bel = [[list(start) for _ in range(P)]]
    for (mv, it, d, seen) in moves:
        for j in range(I):
            if loc[j] == d and not known[j][mv]:
                known[j][mv] = True
        loc[it] = d
        for o in range(P):
            if o != mv:
                known[it][o] = o in seen
        b = [list(x) for x in bel[-1]]
        for p in range(P):
            for j in range(I):
                if known[j][p]:
                    b[p][j] = loc[j]
        bel.append(b)
    return bel


def enumerate_stories():
    """Every story the generator can produce, with its exact probability: list of (start, moves, prob)."""
    out = []

    def rec(loc, known, alive, hist, moves, w, start):
        if len(moves) == M:
            out.append((start, tuple(moves), w))
            return
        poss = {p: [i for i in range(I) if known[i][p]] for p in range(P)}
        movers = [p for p in range(P) if poss[p]]
        if not movers:
            out.append((start, tuple(moves), w))
            return
        for mv in movers:
            for it in poss[mv]:
                dests = [l for l in alive[it] if l != loc[it]]
                for d in dests:
                    others = [p for p in range(P) if p != mv]
                    for seen in itertools.product((True, False), repeat=len(others)):
                        loc2, known2 = list(loc), [list(r) for r in known]
                        alive2, hist2 = [list(a) for a in alive], [list(h) for h in hist]
                        for j in range(I):
                            if loc2[j] == d and not known2[j][mv]:
                                known2[j][mv] = True
                        loc2[it] = d
                        hist2[it][d] += 1
                        if hist2[it][d] == MAX_USE:
                            alive2[it].remove(d)
                        for o, sv in zip(others, seen):
                            known2[it][o] = sv
                        pw = Fraction(1, len(movers)) * Fraction(1, len(poss[mv])) * Fraction(1, len(dests))
                        for sv in seen:
                            pw *= P_SEE if sv else 1 - P_SEE
                        rec(loc2, known2, alive2, hist2,
                            moves + [(mv, it, d, tuple(o for o, sv in zip(others, seen) if sv))], w * pw, start)

    for start in itertools.product(range(L), repeat=I):
        rec(list(start), [[True] * P for _ in range(I)], [list(range(L)) for _ in range(I)],
            [[0] * L for _ in range(I)], [], Fraction(1, L ** I), start)
    return out


def propositions(start, moves, question):
    """The true propositions of a story (catalog in the module docstring)."""
    out = {f"start:{i}:{l}" for i, l in enumerate(start)}
    bel = replay(start, moves)
    for k, (mv, it, d, seen) in enumerate(moves):
        out |= {f"mover:{k}:{mv}", f"item:{k}:{it}", f"dest:{k}:{d}"}
        for o in range(P):
            if o != mv:
                out.add(f"saw:{k}:{o}" if o in seen else f"missed:{k}:{o}")
    for t in range(1, len(moves) + 1):
        for p in range(P):
            for i in range(I):
                if t == M and (p, i) == tuple(question):
                    continue
                out.add(f"belief:{t}:{p}:{i}:{bel[t][p][i]}")
    return out


def answer_of(start, moves, question):
    return replay(start, moves)[-1][question[0]][question[1]]


_STORIES = None
_TABLE = None


def stories():
    global _STORIES
    if _STORIES is None:
        _STORIES = enumerate_stories()
    return _STORIES


def catalog():
    """Every proposition of the catalog (any story), in a fixed order."""
    out = [f"start:{i}:{l}" for i in range(I) for l in range(L)]
    for k in range(M):
        out += [f"mover:{k}:{p}" for p in range(P)] + [f"item:{k}:{i}" for i in range(I)]
        out += [f"dest:{k}:{l}" for l in range(L)]
        out += [f"saw:{k}:{p}" for p in range(P)] + [f"missed:{k}:{p}" for p in range(P)]
    out += [f"belief:{t}:{p}:{i}:{l}" for t in range(1, M + 1) for p in range(P) for i in range(I) for l in range(L)]
    return out


def table():
    """(truth matrix stories x catalog, final belief per (person, item), probabilities), computed once."""
    global _TABLE
    if _TABLE is None:
        cat = catalog()
        col = {f: j for j, f in enumerate(cat)}
        S = stories()
        X = np.zeros((len(S), len(cat)), dtype=bool)
        final = np.zeros((len(S), P, I), dtype=np.int8)
        w = np.zeros(len(S))
        for r, (st, mv, pr) in enumerate(S):
            for f in propositions(st, mv, (-1, -1)):          # (-1, -1): keep every final belief
                X[r, col[f]] = True
            final[r] = np.array(replay(st, mv)[-1])
            w[r] = float(pr)
        _TABLE = (cat, col, X, final, w)
    return _TABLE


# ------------------------------------------------------------------ world
class OPWorld:
    """An Object Placements world: the actual story, the question and the answer order. Interface of world.World."""

    def __init__(self, start, moves, question, answer_order, names=None):
        self.start, self.moves = tuple(start), tuple(tuple(m[:3]) + (tuple(m[3]),) for m in moves)
        self.question, self.answer_order = tuple(question), list(answer_order)
        assert answer_of(self.start, self.moves, self.question) == self.answer_order[0], "answer_order[0] must be correct"
        self.fact_ids = sorted(propositions(self.start, self.moves, self.question))
        self.index = {f: i for i, f in enumerate(self.fact_ids)}
        self.n_facts = len(self.fact_ids)
        self.names = names or {"people": ["Alex", "Blair", "Casey"], "items": ["notebook", "keys"],
                               "locations": ["desk", "shelf", "drawer"]}
        self.fact_text = [self.text(f) for f in self.fact_ids]
        rank = np.zeros(L, dtype=np.int8)
        for a, loc in enumerate(self.answer_order):
            rank[loc] = a
        cat, col, X, final, w = table()
        cols = [col[f] for f in self.fact_ids]
        bitw = (np.uint64(1) << np.arange(self.n_facts, dtype=np.uint64))
        sig = (X[:, cols].astype(np.uint64) * bitw).sum(axis=1).astype(np.uint64)
        ans = rank[final[:, self.question[0], self.question[1]]]
        key = sig * np.uint64(4) + ans.astype(np.uint64)
        uniq, inv = np.unique(key, return_inverse=True)
        self.sig = uniq >> np.uint64(2)
        self.ans = (uniq & np.uint64(3)).astype(np.int8)
        self.w = np.bincount(inv, weights=w)
        self.support = [np.nonzero((self.sig >> np.uint64(j)) & np.uint64(1))[0] for j in range(self.n_facts)]
        self._cache: dict[int, tuple[float, float, float]] = {}
        desc = json.dumps([self.start, self.moves, self.question, self.answer_order])
        self.sha256 = hashlib.sha256(f"object_placements:{desc}".encode()).hexdigest()

    def text(self, f):
        n = self.names
        parts = f.split(":")
        kind, a = parts[0], [int(x) for x in parts[1:]]
        plural = kind == "start" and n["items"][a[0]].endswith("s")
        if kind == "start":
            return f"The {n['items'][a[0]]} {'start' if plural else 'starts'} on the {n['locations'][a[1]]}."
        if kind == "mover":
            return f"Move {a[0] + 1} is made by {n['people'][a[1]]}."
        if kind == "item":
            return f"Move {a[0] + 1} moves the {n['items'][a[1]]}."
        if kind == "dest":
            return f"Move {a[0] + 1} puts the item on the {n['locations'][a[1]]}."
        if kind == "saw":
            return f"{n['people'][a[1]]} saw move {a[0] + 1}."
        if kind == "missed":
            return f"{n['people'][a[1]]} did not see move {a[0] + 1}."
        if kind == "belief":
            plural = n["items"][a[2]].endswith("s")
            return (f"After move {a[0]}, {n['people'][a[1]]} thinks the {n['items'][a[2]]} {'are' if plural else 'is'} "
                    f"on the {n['locations'][a[3]]}.")
        return f

    def factset(self, fact_ids) -> int:
        s = 0
        for f in fact_ids:
            s |= 1 << self.index[f]
        return s

    def ids(self, s: int) -> list[str]:
        return [self.fact_ids[i] for i in range(self.n_facts) if (s >> i) & 1]

    def posterior(self, s: int) -> tuple[float, float, float]:
        hit = self._cache.get(s)
        if hit is None:
            if s == 0:
                idx = np.arange(len(self.sig))
            else:
                held = [j for j in range(self.n_facts) if (s >> j) & 1]
                j0 = min(held, key=lambda j: len(self.support[j]))
                idx = self.support[j0]
                m = np.uint64(s)
                idx = idx[(self.sig[idx] & m) == m]
            c = np.bincount(self.ans[idx], weights=self.w[idx], minlength=3)
            tot = c.sum()
            hit = (c[0] / tot, c[1] / tot, c[2] / tot)
            if len(self._cache) >= 2_000_000:
                self._cache.clear()
            self._cache[s] = hit
        return hit

    def proves(self, s: int, k: int) -> bool:
        return self.posterior(s)[k] == 1.0


def load_op_world(path: str | pathlib.Path) -> OPWorld:
    """Load a world description; its merged story classes are cached in <world>.classes.npz (checked against the
    world's checksum), so simulator workers do not re-enumerate the 617,472 stories."""
    path = pathlib.Path(path)
    d = json.loads(path.read_text())
    cache = path.with_suffix(".classes.npz")
    desc = json.dumps([tuple(d["start"]), tuple(tuple(m[:3]) + (tuple(m[3]),) for m in d["moves"]),
                       tuple(d["question"]), list(d["answer_order"])])
    sha = hashlib.sha256(f"object_placements:{desc}".encode()).hexdigest()
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["sha256"]) == sha:
            w = OPWorld.__new__(OPWorld)
            w.start, w.question, w.answer_order = tuple(d["start"]), tuple(d["question"]), list(d["answer_order"])
            w.moves = tuple(tuple(m[:3]) + (tuple(m[3]),) for m in d["moves"])
            w.fact_ids = [str(x) for x in z["fact_ids"]]
            w.index = {f: i for i, f in enumerate(w.fact_ids)}
            w.n_facts = len(w.fact_ids)
            w.names = d.get("names") or {"people": ["Alex", "Blair", "Casey"], "items": ["notebook", "keys"],
                                         "locations": ["desk", "shelf", "drawer"]}
            w.fact_text = [w.text(f) for f in w.fact_ids]
            w.sig, w.ans, w.w = z["sig"], z["ans"], z["w"]
            w.support = [np.nonzero((w.sig >> np.uint64(j)) & np.uint64(1))[0] for j in range(w.n_facts)]
            w._cache = {}
            w.sha256 = sha
            return w
    w = OPWorld(d["start"], d["moves"], d["question"], d["answer_order"], d.get("names"))
    assert w.sha256 == sha
    np.savez(cache, sha256=np.array(sha), fact_ids=np.array(w.fact_ids), sig=w.sig, ans=w.ans, w=w.w)
    return w
