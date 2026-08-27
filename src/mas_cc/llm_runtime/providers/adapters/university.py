"""University of Potsdam OpenAI-compatible proxy adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mas_cc.llm_runtime.config import LLMProviderConfig

from ._openai_compatible import OpenAICompatibleProvider
from ..load_control import SharedProviderCoordinator


class UniversityProvider(OpenAICompatibleProvider):
    def __init__(
        self,
        config: LLMProviderConfig,
        *,
        environment: Mapping[str, str] | None = None,
        session: Any | None = None,
        request_coordinator: SharedProviderCoordinator | None = None,
    ) -> None:
        self._windows_proxy_configured = False
        super().__init__(
            config,
            provider_name="university",
            default_credentials_env="POTSDAM_API_KEY",
            default_base_url_env="BASE_POTSDAM_LLM_URL",
            discover_endpoint=True,
            environment=environment,
            session=session,
            request_coordinator=request_coordinator,
        )

    def _get_session(self):
        session = super()._get_session()
        if not self._windows_proxy_configured:
            from ._potsdam_network import ensure_windows_vpn_bridge

            proxy = ensure_windows_vpn_bridge(self._base_url)
            if proxy is not None:
                session.proxies["https"] = proxy
            self._windows_proxy_configured = True
        return session
