"""The task_003 world: exact posteriors over the three allocations.

A *fact set* is an int used as a bitmask: bit i set means fact i is in the set.
Example: facts 0 and 5 -> 0b100001 == 33. With 49 facts every set fits in 64 bits,
which keeps memory states cheap to store, compare and cache.
"""
from __future__ import annotations
import hashlib, json, pathlib

ALLOCATIONS = ("ALLOCATION_0", "ALLOCATION_1", "ALLOCATION_2")


def bits(s: int):
    """The fact indices in a fact set, in increasing order."""
    while s:
        low = s & -s
        yield low.bit_length() - 1
        s ^= low


class World:
    """Possible worlds and true facts of one task, with a cached exact posterior."""

    def __init__(self, path: str | pathlib.Path):
        raw = pathlib.Path(path).read_text()
        d = json.loads(raw)
        self.sha256 = hashlib.sha256(raw.rstrip("\n").encode()).hexdigest()
        self.fact_ids: list[str] = [f["id"] for f in d["facts"]]
        self.fact_text: list[str] = [f["text"] for f in d["facts"]]
        self.index: dict[str, int] = {f: i for i, f in enumerate(self.fact_ids)}
        self._fact_mask = [int(f["mask"], 16) for f in d["facts"]]
        self._all = int(d["all_mask"], 16)
        self._win = [int(d["win_masks"][a], 16) for a in ALLOCATIONS]
        self.n_facts = len(self.fact_ids)
        self._cache: dict[int, tuple[float, float, float]] = {}
        self.cache_limit = 2_000_000     # cleared when full, so long runs cannot exhaust memory

    def factset(self, fact_ids) -> int:
        """Fact IDs -> fact set."""
        s = 0
        for f in fact_ids:
            s |= 1 << self.index[f]
        return s

    def ids(self, s: int) -> list[str]:
        """Fact set -> fact IDs."""
        return [self.fact_ids[i] for i in bits(s)]

    def posterior(self, s: int) -> tuple[float, float, float]:
        """Exact P(A0), P(A1), P(A2) given the facts in set s. The empty set gives 1/3 each."""
        hit = self._cache.get(s)
        if hit is None:
            m = self._all
            for i in bits(s):
                m &= self._fact_mask[i]
            n = m.bit_count()
            # every fact is true, so the real world always survives and n > 0
            assert n > 0, "an inconsistent fact set: impossible with true facts"
            hit = tuple((m & w).bit_count() / n for w in self._win)
            if len(self._cache) >= self.cache_limit:
                self._cache.clear()
            self._cache[s] = hit
        return hit

    def proves(self, s: int, k: int) -> bool:
        return self.posterior(s)[k] == 1.0
