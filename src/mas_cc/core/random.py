"""Reproducible seed records and deterministic seed derivation."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

_MAX_SEED = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class Seed:
    """An immutable non-negative seed with stable child derivation."""

    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("Seed.value must be an integer")
        if not 0 <= self.value <= _MAX_SEED:
            raise ValueError(f"Seed.value must be between 0 and {_MAX_SEED}")

    def derive(self, namespace: str | int) -> "Seed":
        """Derive the same child seed for the same seed and namespace."""

        if not isinstance(namespace, (str, int)) or isinstance(namespace, bool):
            raise TypeError("Seed namespace must be a string or integer")
        payload = f"mas-cc-seed-v1\0{self.value}\0{namespace}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & _MAX_SEED
        return Seed(value)

    def create_random(self) -> random.Random:
        """Return an isolated standard-library random number generator."""

        return random.Random(self.value)

    def __int__(self) -> int:
        return self.value
