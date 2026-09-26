from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class SimulationParameters:
    # Population / synthetic task
    N: int = 24
    F: int = 10
    q: int = 3
    rounds: int = 30

    # Initial fact distribution
    initial_fact_redundancy: int = 3
    truth_fact_fraction: float = 0.70

    # Memory / epistemic dynamics
    rho: float = 0.75

    # Decision strengths
    beta_evidence: float = 2.0
    beta_social: float = 1.0

    # Controller
    budget_fraction: float = 0.125
    sensing_fraction: float = 0.50
    controller_target: int = -1  # -1 false, +1 truth
    policy_beta: float = 8.0
    policy_threshold: float = 0.50

    # Logging
    save_micro: bool = False

    @property
    def budget(self) -> int:
        """Integer per-round budget induced by population-relative budget."""
        return int(round(self.budget_fraction * self.N))


@dataclass(frozen=True)
class Message:
    author: int
    vote: int
    fact_id: Optional[int]
    is_controller: bool = False


@dataclass
class AgentState:
    vote: int
    active_facts: set[int] = field(default_factory=set)


@dataclass
class EpisodeResult:
    seed: int
    params: dict
    fact_weights: list[int]
    rounds: list[dict]
    micro: list[dict]


