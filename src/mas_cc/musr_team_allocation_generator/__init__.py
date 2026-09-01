"""Native MuSR-style Team Allocation task generation for MAS-CC."""

from .generate import GenerationConfig, generate_dataset, generate_world
from .latent_problem import generate_latent_problem, latent_facts, score_allocation
from .schemas import FrozenTask, LatentProblem
from .validate import validate_frozen_task

__all__ = [
    "FrozenTask",
    "GenerationConfig",
    "LatentProblem",
    "generate_dataset",
    "generate_latent_problem",
    "generate_world",
    "latent_facts",
    "score_allocation",
    "validate_frozen_task",
]
