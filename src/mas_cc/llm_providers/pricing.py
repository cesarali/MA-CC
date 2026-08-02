"""Versioned static token pricing used only for preflight estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ModelPricing:
    provider: str
    model: str
    input_usd_per_million_tokens: float
    output_usd_per_million_tokens: float
    source: str

    def __post_init__(self) -> None:
        if self.input_usd_per_million_tokens < 0 or self.output_usd_per_million_tokens < 0:
            raise ValueError("pricing rates cannot be negative")

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_usd_per_million_tokens
            + output_tokens * self.output_usd_per_million_tokens
        ) / 1_000_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_usd_per_million_tokens": self.input_usd_per_million_tokens,
            "output_usd_per_million_tokens": self.output_usd_per_million_tokens,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class PricingCatalog:
    version: str
    entries: tuple[ModelPricing, ...]

    def find(self, provider: str, model: str) -> ModelPricing | None:
        return next(
            (
                entry
                for entry in self.entries
                if entry.provider == provider and entry.model == model
            ),
            None,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PricingCatalog":
        version = value.get("version")
        entries = value.get("entries")
        if not isinstance(version, str) or not version or not isinstance(entries, list):
            raise ValueError("pricing catalog requires version and entries")
        return cls(
            version,
            tuple(
                ModelPricing(
                    provider=str(item["provider"]),
                    model=str(item["model"]),
                    input_usd_per_million_tokens=float(
                        item["input_usd_per_million_tokens"]
                    ),
                    output_usd_per_million_tokens=float(
                        item["output_usd_per_million_tokens"]
                    ),
                    source=str(item["source"]),
                )
                for item in entries
            ),
        )


def default_pricing_catalog() -> PricingCatalog:
    """Small audited catalog; unknown models deliberately produce unknown cost."""

    return PricingCatalog(
        version="2026-08-01-v1",
        entries=(
            ModelPricing("mock", "deterministic-v1", 0.0, 0.0, "local deterministic mock"),
            ModelPricing(
                "openai",
                "gpt-4o-mini",
                0.15,
                0.60,
                "https://openai.com/index/api-prompt-caching/",
            ),
            ModelPricing(
                "gemma_local",
                "google/gemma-4-12B-it",
                0.0,
                0.0,
                "marginal API cost only; hardware/energy excluded",
            ),
        ),
    )
