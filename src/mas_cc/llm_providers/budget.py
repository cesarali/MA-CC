"""Optional preflight budget ceiling; never queries a provider implicitly."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BudgetCeiling:
    usd: float

    def __post_init__(self) -> None:
        if self.usd < 0:
            raise ValueError("budget ceiling cannot be negative")

    def permits(self, conservative_cost_usd: float | None) -> bool | None:
        return None if conservative_cost_usd is None else conservative_cost_usd <= self.usd
