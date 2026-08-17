"""Apply model parameter profiles before a request reaches an adapter."""

from __future__ import annotations

import warnings
from dataclasses import replace
from threading import Lock
from typing import Any

from ..messages import MessageRole
from .errors import ProviderError
from .model_profiles import ModelProfile, ModelProfileRegistry
from .requests import CompletionRequest
from .responses import CompletionResponse


OMIT_TEMPERATURE_METADATA_KEY = "_llm_runtime_omit_temperature"
PROFILE_ADJUSTMENTS_METADATA_KEY = "model_profile_adjustments"


class ModelProfileOverrideWarning(UserWarning):
    """A caller request was changed to satisfy a known model contract."""


_WARNED_OVERRIDES: set[tuple[str, str, str]] = set()
_WARNED_OVERRIDES_LOCK = Lock()


def _warn_override_once(profile: ModelProfile, rule_key: str, description: str) -> None:
    key = (profile.provider_type, profile.model, rule_key)
    with _WARNED_OVERRIDES_LOCK:
        if key in _WARNED_OVERRIDES:
            return
        _WARNED_OVERRIDES.add(key)
    warnings.warn(
        f"Adjusted request for {profile.provider_type}/{profile.model}: {description} "
        f"(profile source: {profile.probe_source}).",
        ModelProfileOverrideWarning,
        stacklevel=3,
    )


def apply_model_profile(request: CompletionRequest, profile: ModelProfile) -> CompletionRequest:
    """Return a provider-compatible request and expose every substitution.

    ``CompletionRequest.temperature`` is intentionally a required float.  The
    private metadata marker is therefore the bridge used to express an omitted
    wire parameter without weakening that provider-independent public type.

    A model requiring an output-token field other than ``max_tokens`` cannot
    be represented here because the adapter owns that wire-field choice.  Such
    profiles fail before dispatch and are the trigger for a generic adapter
    change rather than a silent best effort.
    """

    if not profile.supported:
        detail = profile.notes[0] if profile.notes else "the probe marked it unsupported"
        raise ProviderError(
            f"Model {profile.model!r} cannot be used by the chat runtime: {detail}",
            provider=profile.provider_type,
            code="unsupported_model_profile",
            retryable=False,
        )
    if profile.max_output_tokens_field != "max_tokens":
        raise ProviderError(
            f"Model {profile.model!r} requires {profile.max_output_tokens_field!r}, but "
            "this runtime can currently send only 'max_tokens'.",
            provider=profile.provider_type,
            code="unsupported_model_parameter",
            retryable=False,
        )
    if profile.supports_system_messages is False and any(
        message.role is MessageRole.SYSTEM for message in request.messages
    ):
        raise ProviderError(
            f"Model {profile.model!r} does not accept system messages.",
            provider=profile.provider_type,
            code="unsupported_model_parameter",
            retryable=False,
        )

    changes: dict[str, Any] = {}
    temperature = request.temperature
    metadata = dict(request.metadata)
    rule = profile.temperature
    if rule.mode == "fixed" and temperature != rule.value:
        changes["temperature"] = {"requested": temperature, "sent": rule.value}
        temperature = rule.value
    elif rule.mode == "omit":
        changes["temperature"] = {"requested": temperature, "sent": None}
        metadata[OMIT_TEMPERATURE_METADATA_KEY] = True

    seed = request.seed
    if seed is not None and profile.supports_seed is False:
        changes["seed"] = {"requested": seed, "sent": None}
        seed = None

    if not changes:
        return request

    metadata[PROFILE_ADJUSTMENTS_METADATA_KEY] = {
        "provider_type": profile.provider_type,
        "model": profile.model,
        "profile_source": profile.probe_source,
        "changes": changes,
    }
    if "temperature" in changes:
        values = changes["temperature"]
        _warn_override_once(
            profile,
            f"temperature:{rule.mode}:{rule.value}",
            f"temperature={values['requested']!r} -> {values['sent']!r}",
        )
    if "seed" in changes:
        values = changes["seed"]
        _warn_override_once(
            profile,
            "seed:unsupported",
            f"seed={values['requested']!r} -> {values['sent']!r}",
        )
    return replace(request, temperature=temperature, seed=seed, metadata=metadata)


class ProfiledLLMProvider:
    """Provider decorator that normalizes every request for its concrete model."""

    def __init__(
        self,
        provider: Any,
        provider_type: str,
        registry: ModelProfileRegistry,
    ) -> None:
        self._provider = provider
        self.name = provider.name
        self.model = provider.model
        self.capabilities = provider.capabilities
        self.profile = registry.get(provider_type, provider.model)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return await self._provider.complete(apply_model_profile(request, self.profile))

    def close(self) -> None:
        self._provider.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)
