"""Reading and freezing one probe configuration file.

The probe deliberately does **not** reuse the experiment ``RunConfig``: it has
no game, no population, no control policy and no trajectory, so most of that
schema would be inapplicable fields carrying misleading defaults.  What it does
share is the provider configuration, so a probe model spec compiles straight
into the same :class:`LLMProviderConfig` the games use and reaches the provider
through the same adapter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import json

from mas_cc.llm_runtime.config import LLMProviderConfig

from .design import DesignSpec, TARGET_SEMANTICS


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One configured LLM and the generation settings used to call it."""

    label: str
    provider: str
    model: str
    credentials_env: str | None = None
    base_url_env: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 4096
    timeout_seconds: float = 180.0
    max_retries: int = 2
    request_concurrency: int = 1
    seed: int | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def provider_config(self) -> LLMProviderConfig:
        return LLMProviderConfig(
            type=self.provider,
            model=self.model,
            credentials_env=self.credentials_env,
            base_url_env=self.base_url_env,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            request_concurrency=self.request_concurrency,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            options=dict(self.options),
        )

    @property
    def generation_settings_hash(self) -> str:
        """Fingerprint of everything that could change an answer but is not the
        prompt: sampling temperature, output cap, and any provider seed."""

        payload = json.dumps(
            {
                "provider": self.provider,
                "model": self.model,
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "seed": self.seed,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def call_identity(self) -> str:
        """Stable call namespace for this labeled model specification."""

        return f"{self.label}:{self.generation_settings_hash}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "provider": self.provider,
            "model": self.model,
            "credentials_env": self.credentials_env,
            "base_url_env": self.base_url_env,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "request_concurrency": self.request_concurrency,
            "seed": self.seed,
            "options": dict(self.options),
            "generation_settings_hash": self.generation_settings_hash,
        }


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """How the calls are dispatched.  Never what the calls *are*."""

    workers: int = 4
    backend: str = "process_pool"
    resume: bool = True
    max_retries: int = 2
    provider_concurrency_caps: Mapping[str, int] = field(default_factory=dict)
    """Optional per-provider ceilings, e.g. ``{"university": 8}``.  A cap
    reduces how many calls of one provider are in flight; it never removes a
    cell from the scientific grid."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "workers": self.workers,
            "backend": self.backend,
            "resume": self.resume,
            "max_retries": self.max_retries,
            "provider_concurrency_caps": dict(self.provider_concurrency_caps),
        }


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    models: tuple[ModelSpec, ...]
    dataset_dirs: Mapping[int, str]
    design: DesignSpec
    execution: ExecutionSpec
    output_dir: Path
    source_path: str

    def model(self, label: str) -> ModelSpec:
        for spec in self.models:
            if spec.label == label:
                return spec
        raise KeyError(label)

    def to_dict(self) -> dict[str, Any]:
        design = self.design
        return {
            "probe": "controller_retention",
            "source_config": self.source_path,
            "output_dir": str(self.output_dir),
            "models": [spec.to_dict() for spec in self.models],
            "tasks": {
                "dataset_dirs": {str(k): v for k, v in self.dataset_dirs.items()},
                "tasks_per_depth": design.tasks_per_depth,
                "task_ids": {str(k): list(v) for k, v in design.tasks.items()},
            },
            "design": {
                "seed": design.seed,
                "reasoning_depths": list(design.reasoning_depths),
                "q_values": list(design.q_values),
                "receivers": list(design.receivers),
                "targets": list(design.targets),
                "replicates": design.replicates,
            },
            "execution": self.execution.to_dict(),
        }


class ProbeConfigError(ValueError):
    """The probe configuration is missing something or is self-contradictory."""


def _tuple(value: Any, name: str, allowed: Sequence[Any] | None = None) -> tuple:
    if value is None:
        raise ProbeConfigError(f"{name} must be a non-empty list")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProbeConfigError(f"{name} must be a list")
    items = tuple(value)
    if not items:
        raise ProbeConfigError(f"{name} must be a non-empty list")
    if allowed is not None:
        unknown = [item for item in items if item not in allowed]
        if unknown:
            raise ProbeConfigError(f"{name}: {unknown[0]!r} is not one of {list(allowed)}")
    return items


def load_probe_config(path: str | Path) -> ProbeConfig:
    """Read, validate, and freeze one probe YAML file."""

    import yaml

    config_path = Path(path)
    if not config_path.is_file():
        raise ProbeConfigError(f"probe config does not exist: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProbeConfigError(f"{config_path} must contain a YAML mapping")
    return build_probe_config(payload, source_path=str(config_path))


def build_probe_config(
    payload: Mapping[str, Any], *, source_path: str = "<memory>"
) -> ProbeConfig:
    raw_models = payload.get("models")
    if not isinstance(raw_models, Sequence) or isinstance(raw_models, (str, bytes)):
        raise ProbeConfigError("models must be a list of provider/model specifications")
    if not raw_models:
        raise ProbeConfigError("models must contain at least one specification")
    models: list[ModelSpec] = []
    for index, entry in enumerate(raw_models):
        if not isinstance(entry, Mapping):
            raise ProbeConfigError(f"models[{index}] must be a mapping")
        for key in ("provider", "model", "label"):
            if not entry.get(key):
                raise ProbeConfigError(f"models[{index}].{key} is required")
        model = ModelSpec(
                label=str(entry["label"]),
                provider=str(entry["provider"]),
                model=str(entry["model"]),
                credentials_env=_optional_str(entry.get("credentials_env")),
                base_url_env=_optional_str(entry.get("base_url_env")),
                temperature=float(entry.get("temperature", 0.0)),
                max_output_tokens=int(entry.get("max_output_tokens", 4096)),
                timeout_seconds=float(entry.get("timeout_seconds", 180.0)),
                max_retries=int(entry.get("max_retries", 2)),
                request_concurrency=int(entry.get("request_concurrency", 1)),
                seed=_optional_int(entry.get("seed")),
                options=dict(entry.get("options") or {}),
            )
        if model.max_output_tokens < 1:
            raise ProbeConfigError(f"models[{index}].max_output_tokens must be positive")
        if model.timeout_seconds <= 0:
            raise ProbeConfigError(f"models[{index}].timeout_seconds must be positive")
        if model.max_retries < 0:
            raise ProbeConfigError(f"models[{index}].max_retries must be non-negative")
        if model.request_concurrency < 1:
            raise ProbeConfigError(f"models[{index}].request_concurrency must be positive")
        models.append(model)
    labels = [spec.label for spec in models]
    if len(set(labels)) != len(labels):
        raise ProbeConfigError("model labels must be unique")

    tasks_section = payload.get("tasks") or {}
    if not isinstance(tasks_section, Mapping):
        raise ProbeConfigError("tasks must be a mapping")
    raw_dirs = tasks_section.get("dataset_dirs")
    if not isinstance(raw_dirs, Mapping) or not raw_dirs:
        raise ProbeConfigError("tasks.dataset_dirs must map reasoning depth to a directory")
    dataset_dirs = {int(key): str(value) for key, value in raw_dirs.items()}
    explicit = {
        int(key): tuple(str(item) for item in value)
        for key, value in (tasks_section.get("task_ids") or {}).items()
    }

    design_section = payload.get("design") or {}
    if not isinstance(design_section, Mapping):
        raise ProbeConfigError("design must be a mapping")
    depths = tuple(
        int(value)
        for value in _tuple(
            design_section.get("reasoning_depths", [1, 2]), "design.reasoning_depths"
        )
    )
    if depths != (1, 2):
        raise ProbeConfigError("design.reasoning_depths must be exactly [1, 2]")
    missing = [depth for depth in depths if depth not in dataset_dirs]
    if missing:
        raise ProbeConfigError(
            f"tasks.dataset_dirs has no directory for reasoning depth {missing[0]}"
        )
    design = DesignSpec(
        seed=int(design_section.get("seed", 20260828)),
        reasoning_depths=depths,
        q_values=tuple(
            int(value) for value in _tuple(design_section.get("q_values", [2, 3]), "design.q_values")
        ),
        receivers=tuple(
            str(value)
            for value in _tuple(
                design_section.get("receivers", ["naive"]),
                "design.receivers",
                ("naive",),
            )
        ),
        targets=tuple(
            str(value)
            for value in _tuple(
                design_section.get("targets", list(TARGET_SEMANTICS)),
                "design.targets",
                TARGET_SEMANTICS,
            )
        ),
        tasks_per_depth=int(tasks_section.get("tasks_per_depth", 12)),
        replicates=int(design_section.get("replicates", 1)),
        tasks=explicit,
    )
    if design.q_values != (2, 3):
        raise ProbeConfigError("design.q_values must be exactly [2, 3]")
    if design.receivers != ("naive",):
        raise ProbeConfigError("design.receivers must be exactly [naive]")
    if design.targets != TARGET_SEMANTICS:
        raise ProbeConfigError('design.targets must be exactly [truth, "false"]')
    if design.replicates != 1:
        raise ProbeConfigError("design.replicates must be exactly 1")
    if design.tasks_per_depth != 12:
        raise ProbeConfigError("tasks.tasks_per_depth must be exactly 12")

    execution_section = payload.get("execution") or {}
    if not isinstance(execution_section, Mapping):
        raise ProbeConfigError("execution must be a mapping")
    workers = int(execution_section.get("workers", 4))
    if workers < 1:
        raise ProbeConfigError("execution.workers must be at least 1")
    backend = str(execution_section.get("backend", "process_pool"))
    if backend not in ("process_pool", "thread_pool", "serial"):
        raise ProbeConfigError(
            "execution.backend must be 'process_pool', 'thread_pool', or 'serial'"
        )
    execution = ExecutionSpec(
        workers=workers,
        backend=backend,
        resume=bool(execution_section.get("resume", True)),
        max_retries=int(execution_section.get("max_retries", 2)),
        provider_concurrency_caps={
            str(key): int(value)
            for key, value in (execution_section.get("provider_concurrency_caps") or {}).items()
        },
    )
    if execution.max_retries != 0:
        raise ProbeConfigError(
            "execution.max_retries must be 0; provider adapters own transport retries"
        )
    if any(value < 1 for value in execution.provider_concurrency_caps.values()):
        raise ProbeConfigError("execution.provider_concurrency_caps values must be positive")

    storage = payload.get("storage") or {}
    output_dir = Path(str(storage.get("output_dir", "results/probes/controller_retention")))
    return ProbeConfig(
        models=tuple(models),
        dataset_dirs=dataset_dirs,
        design=design,
        execution=execution,
        output_dir=output_dir,
        source_path=source_path,
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProbeConfigError("seed must be an integer")
    return value


__all__ = [
    "ExecutionSpec",
    "ModelSpec",
    "ProbeConfig",
    "ProbeConfigError",
    "build_probe_config",
    "load_probe_config",
]
