"""Exact solving-set analysis for MuSR team-allocation worlds.

A world is nine hidden integers in {1,2,3}: six skills (person x task) and three
pairwise cooperation values.  A *fact* is one true proposition about that world.
A set of facts *solves* the task when every hidden world consistent with it is
won by the correct allocation, i.e. the posterior P(correct) is exactly 1.

Solving is monotone: adding facts never breaks a proof.  So the solving sets form
an up-set and are fully described by the MINIMAL ones, which is what this module
enumerates.  Worlds are represented as bitmasks so set operations are integer
ANDs and exhaustive search over small subsets is cheap.
"""
from __future__ import annotations
import itertools, sys
sys.path.insert(0, "/Users/rsanchez/Projects/MA-CC/src")
from mas_cc.musr_team_allocation_generator.latent_problem import problem_from_latent_values
from mas_cc.musr_team_allocation_generator.symbolic_facts import true_canonical_facts


class World:
    """One hidden world, its true facts, and bitmask machinery over completions."""

    def __init__(self, vector):
        self.vector = tuple(vector)
        self.problem = problem_from_latent_values(self.vector)
        self.scores = tuple(self.problem.candidate_scores)
        self.truth = max(range(3), key=lambda i: self.scores[i])
        self.facts = list(true_canonical_facts(self.problem))
        self.ids = [f.fact_id for f in self.facts]
        # keep only completions with a unique winner, matching the generator's index
        self.worlds = []
        for vec in itertools.product((1, 2, 3), repeat=9):
            sc = problem_from_latent_values(vec).candidate_scores
            top = max(sc)
            win = [i for i, s in enumerate(sc) if s == top]
            if len(win) == 1:
                self.worlds.append((vec, win[0]))
        self.n_worlds = len(self.worlds)
        self.all_mask = (1 << self.n_worlds) - 1
        self.win_mask = [0, 0, 0]
        for i, (_, w) in enumerate(self.worlds):
            self.win_mask[w] |= 1 << i
        self.fact_mask = [
            sum(1 << i for i, (vec, _) in enumerate(self.worlds) if f.holds(vec))
            for f in self.facts
        ]

    def surviving(self, idxs) -> int:
        m = self.all_mask
        for i in idxs:
            m &= self.fact_mask[i]
        return m

    def posterior(self, idxs) -> tuple[float, float, float]:
        m = self.surviving(idxs)
        n = bin(m).count("1")
        if n == 0:
            return (0.0, 0.0, 0.0)
        return tuple(bin(m & self.win_mask[a]).count("1") / n for a in range(3))

    def solves(self, idxs) -> bool:
        """Every surviving world is won by the correct allocation."""
        m = self.surviving(idxs)
        return m != 0 and (m & ~self.win_mask[self.truth]) == 0

    def minimal_sets(self, max_size: int = 6):
        """Exhaustively enumerate minimal solving sets up to ``max_size``."""
        n = len(self.facts)
        minimal, seen = {}, []
        for k in range(1, max_size + 1):
            found = []
            for c in itertools.combinations(range(n), k):
                if not self.solves(c):
                    continue
                if any(set(s) <= set(c) for s in seen):
                    continue            # a proper subset already solves
                found.append(c)
            minimal[k] = found
            seen.extend(found)
        return minimal

    def max_packing(self, minimal, trials: int = 4000, seed: int = 0):
        """Greedy lower bound on the largest family of pairwise DISJOINT proofs."""
        import random
        pool = [c for k in sorted(minimal) for c in minimal[k]]
        rng = random.Random(seed)
        best = []
        for _ in range(trials):
            order = pool[:]
            rng.shuffle(order)
            order.sort(key=len)
            used, pack = set(), []
            for c in order:
                if not (set(c) & used):
                    pack.append(c)
                    used |= set(c)
            if len(pack) > len(best):
                best = pack
        return best
