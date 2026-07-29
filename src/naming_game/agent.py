"""Independent Naming Game agent state."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

from .models import AgentSnapshot, Inventory, normalize_inventory


@dataclass
class Agent:
    """One agent with private mutable state and history."""

    agent_id: int
    inventory: Inventory
    history: list[dict[str, Any]] = field(default_factory=list)
    model_config: dict[str, Any] = field(default_factory=dict)
    rng_seed: int | None = None
    instance_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    provider_session_id: str | None = None
    _rng: random.Random = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.inventory = normalize_inventory(self.inventory)
        self._rng = random.Random(self.rng_seed)

    def snapshot(self, *, evidence: str | None = None) -> AgentSnapshot:
        return AgentSnapshot(self.agent_id, self.inventory, evidence=evidence)

    def set_inventory(self, inventory: Inventory) -> None:
        self.inventory = normalize_inventory(inventory)

    def record(self, event: dict[str, Any]) -> None:
        # Every participant gets its own dictionary; histories and entries are
        # never shared as mutable objects.
        self.history.append(dict(event))


def initial_inventories(num_agents: int, seed: int) -> tuple[Inventory, ...]:
    """Build an approximately even, seeded initial population."""

    if num_agents < 2:
        raise ValueError("num_agents must be at least 2.")
    rng = random.Random(seed)
    inventories: list[Inventory] = [frozenset({"A"})] * (num_agents // 2)
    inventories.extend([frozenset({"B"})] * (num_agents // 2))
    if num_agents % 2:
        inventories.append(frozenset({rng.choice(("A", "B"))}))
    rng.shuffle(inventories)
    return tuple(inventories)


def create_agents(
    num_agents: int,
    seed: int,
    *,
    inventories: Sequence[Inventory] | None = None,
) -> list[Agent]:
    selected = tuple(inventories) if inventories is not None else initial_inventories(num_agents, seed)
    if len(selected) != num_agents:
        raise ValueError("The number of initial inventories must equal num_agents.")
    return [
        Agent(
            agent_id=agent_id,
            inventory=normalize_inventory(inventory),
            rng_seed=(seed * 1_000_003) + agent_id,
        )
        for agent_id, inventory in enumerate(selected)
    ]
