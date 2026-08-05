"""Provider-owned exception type."""

from __future__ import annotations

from ..exceptions import MasCCError


class ProviderError(MasCCError, RuntimeError):
    """A normalized, credential-safe LLM provider failure.

    Adapters retain the original exception as ``__cause__`` for local debugging,
    while this public record contains only fields that are safe to log.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        code: str = "provider_error",
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        if not message or "Bearer " in message:
            raise ValueError("ProviderError.message must be non-empty and credential-safe")
        self.provider = provider
        self.code = code
        self.retryable = retryable
        self.status_code = status_code
        super().__init__(message)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "type": type(self).__name__,
            "provider": self.provider,
            "code": self.code,
            "message": str(self),
            "retryable": self.retryable,
        }
        if self.status_code is not None:
            result["status_code"] = self.status_code
        return result
