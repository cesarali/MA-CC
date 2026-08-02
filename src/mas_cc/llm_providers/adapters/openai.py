"""Official OpenAI chat-completions adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mas_cc.config import LLMProviderConfig

from ._openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    def __init__(
        self,
        config: LLMProviderConfig,
        *,
        environment: Mapping[str, str] | None = None,
        session: Any | None = None,
    ) -> None:
        super().__init__(
            config,
            provider_name="openai",
            default_credentials_env="OPENAI_API_KEY",
            fixed_base_url="https://api.openai.com/v1",
            environment=environment,
            session=session,
        )
