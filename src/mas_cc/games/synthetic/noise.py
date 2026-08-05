"""Pre-drawn, per-episode coin tapes shared by both run modes.

The agents in these games are "lookup tables plus coins". This module owns the
coins, and it owns them in one specific way: **every draw an episode will ever
need is made up front, from the episode seed, as a numpy array.**

That is not an optimization, it is what makes the two modes comparable. Speed
mode reads the tape and computes the whole ``(rounds, agents)`` action array in
one vectorized pass; fidelity mode reads the *same* tape one cell at a time and
sends each cell through prompts, provider, parser, validator, recorder. Same
seed therefore has to produce the same actions, exactly - so any disagreement
is a pipeline bug and not a sampling difference we would have to argue about
with error bars.

Streams are named and independent. Adding a quantity to a game (a control input,
a second noise source) takes a new stream name and leaves every existing
episode's draws bit-identical, so a config that ran last week still replays.
"""

from __future__ import annotations

import hashlib

import numpy as np

_STREAM_NAMESPACE = "mas-cc-synthetic-stream-v1"


def _stream_id(name: str) -> int:
    """A stable 32-bit id for a named stream, independent of declaration order."""

    payload = f"{_STREAM_NAMESPACE}\0{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def episode_generator(seed: int, stream: str) -> np.random.Generator:
    """One independent reproducible draw stream for one episode.

    ``SeedSequence`` mixes the episode seed with the stream id, so streams are
    independent of each other and of draw order *within* a stream - which is
    what lets fidelity mode index into a tape rather than having to consume it
    in exactly the order speed mode did.
    """

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("episode seed must be an integer")
    return np.random.default_rng(np.random.SeedSequence([int(seed), _stream_id(stream)]))


def bernoulli_draws(
    seed: int, stream: str, shape: tuple[int, ...], probability: np.ndarray | float
) -> np.ndarray:
    """A boolean tape of the given shape, each cell true with its own probability.

    ``probability`` broadcasts against ``shape``, so a per-agent noise vector
    of length N fills an ``(rounds, N)`` tape without a loop.
    """

    uniform = episode_generator(seed, stream).random(shape)
    return uniform < np.asarray(probability, dtype=float)
