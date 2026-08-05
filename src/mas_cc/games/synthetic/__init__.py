"""Synthetic games with analytically known answers.

These are not tests. There is no assertion, no red/green, nothing that fails a
build. They are a rehearsal of the full workflow with the answer key in hand:
the same `Game` contract, the same decision loop, the same recorder, the same
artifacts - run on problems where we derived the closed form ourselves, so that
when the system reports something else the discrepancy is unambiguous.

The agents are not LLMs. They are lookup tables plus coins, with dynamics we
specified, which is exactly why every information-theoretic quantity the system
reports here can be compared against a number we already know.

Nothing here says anything about emergent conventions between real models. It
is entirely about whether the machinery is behaving as it should, at a stage
where the code is fresh and mutual information is the most sensitive thing in
it.
"""

from .analysis import (
    ParityResult,
    calibration_curve,
    compare_modes,
    episode_summary,
    metric_check,
    null_distribution,
    pairwise_estimates,
    plot_calibration,
    read_action_series,
    read_final_metrics,
    read_streaming_metrics,
    simulate_estimates,
    with_truth,
)
from .protocols import (
    GroundTruth,
    GroundTruthQuantity,
    SimulatedEpisodes,
    SyntheticDecision,
    SyntheticGame,
    SyntheticGameResult,
    SyntheticInteractionRecord,
    SyntheticTransition,
)
from .provider import SyntheticAgentProvider, SyntheticPromptError
from .runtime import run_synthetic_game, run_synthetic_game_sync

__all__ = [
    "GroundTruth",
    "GroundTruthQuantity",
    "ParityResult",
    "SimulatedEpisodes",
    "SyntheticAgentProvider",
    "SyntheticDecision",
    "SyntheticGame",
    "SyntheticGameResult",
    "SyntheticInteractionRecord",
    "SyntheticPromptError",
    "SyntheticTransition",
    "calibration_curve",
    "compare_modes",
    "create_synthetic_provider_registry",
    "episode_summary",
    "metric_check",
    "null_distribution",
    "pairwise_estimates",
    "plot_calibration",
    "read_action_series",
    "read_final_metrics",
    "read_streaming_metrics",
    "run_synthetic_game",
    "run_synthetic_game_sync",
    "simulate_estimates",
    "with_truth",
]


def create_synthetic_provider_registry():
    """The default provider registry plus the synthetic agent.

    Registered here rather than in `llm_runtime` on purpose: that package is a
    portable kernel that ships no game-specific content, and the synthetic
    agent only means anything next to the games that render its prompts. Same
    application-boundary convention as `register_game_prompt_factories`.
    """

    from mas_cc.llm_runtime.providers import create_default_provider_registry

    registry = create_default_provider_registry()
    registry.register(
        "synthetic_agent", "mas_cc.games.synthetic.provider:SyntheticAgentProvider"
    )
    return registry
