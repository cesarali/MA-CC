"""Per-model parameter contracts loaded from the shipped probe catalogue."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping


TemperatureMode = Literal["any", "fixed", "omit"]
ProbeSource = Literal["probe", "manual", "inferred", "unprobed"]


@dataclass(frozen=True, slots=True)
class TemperatureRule:
    """Describe whether and how a model accepts ``temperature``."""

    mode: TemperatureMode
    value: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"any", "fixed", "omit"}:
            raise ValueError(f"unknown temperature rule {self.mode!r}")
        if self.mode == "fixed":
            if (
                isinstance(self.value, bool)
                or not isinstance(self.value, (int, float))
                or not math.isfinite(self.value)
                or self.value < 0
            ):
                raise ValueError("a fixed temperature rule needs a non-negative value")
            object.__setattr__(self, "value", float(self.value))
        elif self.value is not None:
            raise ValueError(f"temperature rule {self.mode!r} cannot carry a value")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TemperatureRule":
        if not isinstance(value, Mapping):
            raise TypeError("temperature must be an object")
        return cls(mode=str(value.get("mode", "")), value=value.get("value"))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"mode": self.mode}
        if self.mode == "fixed":
            result["value"] = self.value
        return result


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """The parameter contract known for one provider/model pair."""

    provider_type: str
    model: str
    family: str
    temperature: TemperatureRule
    supports_seed: bool | None = None
    supports_system_messages: bool | None = None
    max_output_tokens_field: str = "max_tokens"
    probed_at: str | None = None
    probe_source: ProbeSource = "unprobed"
    supported: bool = True
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("provider_type", "model", "family", "max_output_tokens_field"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ModelProfile.{name} must be non-empty")
        if self.probe_source not in {"probe", "manual", "inferred", "unprobed"}:
            raise ValueError(f"unknown probe source {self.probe_source!r}")
        if self.probed_at is not None and not isinstance(self.probed_at, str):
            raise TypeError("ModelProfile.probed_at must be str or None")
        for name in ("supports_seed", "supports_system_messages"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"ModelProfile.{name} must be bool or None")
        if not isinstance(self.supported, bool):
            raise TypeError("ModelProfile.supported must be bool")
        if isinstance(self.notes, str):
            raise TypeError("ModelProfile.notes must be a sequence of strings")
        notes = tuple(self.notes)
        if any(not isinstance(note, str) for note in notes):
            raise TypeError("ModelProfile.notes must contain strings")
        object.__setattr__(self, "notes", notes)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelProfile":
        if not isinstance(value, Mapping):
            raise TypeError("model profile entry must be an object")
        temperature = value.get("temperature")
        if not isinstance(temperature, Mapping):
            raise TypeError("model profile temperature must be an object")
        return cls(
            provider_type=str(value.get("provider_type", "")),
            model=str(value.get("model", "")),
            family=str(value.get("family", "")),
            temperature=TemperatureRule.from_mapping(temperature),
            supports_seed=value.get("supports_seed"),
            supports_system_messages=value.get("supports_system_messages"),
            max_output_tokens_field=str(value.get("max_output_tokens_field", "max_tokens")),
            probed_at=value.get("probed_at"),
            probe_source=str(value.get("probe_source", "unprobed")),  # type: ignore[arg-type]
            supported=value.get("supported", True),
            notes=tuple(value.get("notes", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_type": self.provider_type,
            "model": self.model,
            "family": self.family,
            "temperature": self.temperature.to_dict(),
            "supports_seed": self.supports_seed,
            "supports_system_messages": self.supports_system_messages,
            "max_output_tokens_field": self.max_output_tokens_field,
            "probed_at": self.probed_at,
            "probe_source": self.probe_source,
            "supported": self.supported,
            "notes": list(self.notes),
        }


def infer_model_family(model: str) -> str:
    """Return a deliberately broad family name from an advertised model id."""

    name = model.rsplit("/", 1)[-1].lower()
    if name.startswith("gpt-image"):
        return "gpt-image"
    if name.startswith("gpt-5"):
        return "gpt-5"
    if name.startswith(("gpt-4", "chatgpt-4")):
        return "gpt-4"
    for prefix, family in (
        ("claude", "claude"),
        ("deepseek", "deepseek"),
        ("qwen", "qwen"),
        ("gemma", "gemma"),
        ("llama", "llama"),
        ("mistral", "mistral"),
    ):
        if name.startswith(prefix):
            return family
    return "unknown"


def _inferred_profile(provider_type: str, model: str) -> ModelProfile:
    family = infer_model_family(model)
    if family == "gpt-5":
        temperature = TemperatureRule("fixed", 1.0)
        notes = ("Family fallback: GPT-5 chat deployments commonly require temperature=1.",)
    else:
        temperature = TemperatureRule("any")
        notes = ("Permissive family fallback; run the parameter probe to verify this model.",)
    supported = family != "gpt-image"
    if not supported:
        notes = ("Image-family models are not supported by the chat completion runtime.",)
    return ModelProfile(
        provider_type=provider_type,
        model=model,
        family=family,
        temperature=temperature,
        probe_source="inferred",
        supported=supported,
        notes=notes,
    )


class ModelProfileRegistry:
    """Exact profile lookup with explicit, visibly inferred fallbacks."""

    def __init__(
        self,
        profiles: Iterable[ModelProfile] | None = None,
        *,
        catalogue_path: str | Path | None = None,
    ) -> None:
        if profiles is not None and catalogue_path is not None:
            raise ValueError("pass profiles or catalogue_path, not both")
        loaded = (
            tuple(profiles)
            if profiles is not None
            else self._load_catalogue(catalogue_path)
        )
        entries: dict[tuple[str, str], ModelProfile] = {}
        for profile in loaded:
            key = (profile.provider_type.strip().lower(), profile.model)
            if key in entries:
                raise ValueError(f"duplicate model profile for {key!r}")
            entries[key] = profile
        self._entries = entries

    @staticmethod
    def _load_catalogue(catalogue_path: str | Path | None) -> tuple[ModelProfile, ...]:
        if catalogue_path is None:
            resource = files(__package__).joinpath("model_profiles.json")
            payload = json.loads(resource.read_text(encoding="utf-8"))
        else:
            payload = json.loads(Path(catalogue_path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("model profile catalogue must be an object")
        if payload.get("schema_version") != 1 or not isinstance(payload.get("profiles"), list):
            raise ValueError("model profile catalogue must use schema_version 1")
        return tuple(ModelProfile.from_mapping(item) for item in payload["profiles"])

    def get(self, provider_type: str, model: str) -> ModelProfile:
        normalized_provider = provider_type.strip().lower()
        try:
            return self._entries[(normalized_provider, model)]
        except KeyError:
            return _inferred_profile(normalized_provider, model)

    def known(self) -> tuple[ModelProfile, ...]:
        return tuple(
            sorted(self._entries.values(), key=lambda item: (item.provider_type, item.model))
        )


def default_model_profile_registry() -> ModelProfileRegistry:
    return ModelProfileRegistry()
