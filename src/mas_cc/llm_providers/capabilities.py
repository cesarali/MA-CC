"""Static provider capability declarations used by planning and validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    supports_seed: bool = False
    reports_usage: bool = False
    supports_system_messages: bool = True
    supports_parallel_requests: bool = True
    is_local: bool = False
    max_request_concurrency: int | None = None
    authorized_budget_query: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "supports_seed": self.supports_seed,
            "reports_usage": self.reports_usage,
            "supports_system_messages": self.supports_system_messages,
            "supports_parallel_requests": self.supports_parallel_requests,
            "is_local": self.is_local,
            "max_request_concurrency": self.max_request_concurrency,
            "authorized_budget_query": self.authorized_budget_query,
        }
