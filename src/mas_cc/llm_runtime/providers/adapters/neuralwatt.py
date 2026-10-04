"""NeuralWatt OpenAI-compatible chat-completions adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from mas_cc.llm_runtime.config import LLMProviderConfig

from ._openai_compatible import OpenAICompatibleProvider
from ..load_control import SharedProviderCoordinator


class NeuralWattProvider(OpenAICompatibleProvider):
    """Use NeuralWatt with provider-scoped routing and JSON-object defaults."""

    def __init__(
        self,
        config: LLMProviderConfig,
        *,
        environment: Mapping[str, str] | None = None,
        session: Any | None = None,
        request_coordinator: SharedProviderCoordinator | None = None,
    ) -> None:
        options = dict(config.options)
        if (
            "response_format" not in options
            and "structured_output_tool" not in options
        ):
            options["response_format"] = {"type": "json_object"}
            config = replace(config, options=options)
        super().__init__(
            config,
            provider_name="neuralwatt",
            default_credentials_env="NEURALWATT_API_KEY",
            fixed_base_url="https://api.neuralwatt.com/v1",
            discover_endpoint=True,
            environment=environment,
            session=session,
            request_coordinator=request_coordinator,
        )
