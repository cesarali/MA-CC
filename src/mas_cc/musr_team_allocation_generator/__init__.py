"""Native MuSR-style Team Allocation task generation for MAS-CC."""

from .generate import GenerationConfig, generate_dataset, generate_world
from .ambiguity import (
    PrivateViewMetrics,
    TeamAllocationCompletionIndex,
    choose_private_views,
    exact_private_view_metrics,
)
from .latent_problem import (
    LATENT_VALUE_PRIOR,
    LATENT_VALUE_SUPPORT,
    generate_latent_problem,
    latent_facts,
    problem_from_latent_values,
    score_allocation,
)
from .schemas import FrozenTask, LatentProblem
from .validate import validate_frozen_task

__all__ = [
    "FrozenTask",
    "GenerationConfig",
    "LatentProblem",
    "LATENT_VALUE_PRIOR",
    "LATENT_VALUE_SUPPORT",
    "PrivateViewMetrics",
    "TeamAllocationCompletionIndex",
    "choose_private_views",
    "exact_private_view_metrics",
    "generate_dataset",
    "generate_latent_problem",
    "generate_world",
    "latent_facts",
    "problem_from_latent_values",
    "score_allocation",
    "validate_frozen_task",
]
