"""Game 2 - Markov: real dynamics, structural zeros, exact chain."""

from .game import (
    MarkovDynamics,
    MarkovSpec,
    SyntheticMarkovGame,
    initial_distribution,
    pooled_laws,
    transition_matrix,
)

__all__ = [
    "MarkovDynamics",
    "MarkovSpec",
    "SyntheticMarkovGame",
    "initial_distribution",
    "pooled_laws",
    "transition_matrix",
]
