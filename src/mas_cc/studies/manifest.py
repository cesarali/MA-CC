"""Discovery and validation of lightweight ``study.yaml`` manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


_RESERVED = frozenset({"study.yaml", "analysis.yaml"})
_PREFLIGHT_CONTRACTS = frozenset(
    {
        "relational_false_takeover_v1",
        "relational_persistence_exploratory_v1",
        "relational_persistence_refinement_v1",
        "relational_persistence_truth_refinement_v1",
        "relational_persistence_q1_l2_false_v1",
        "relational_persistence_q1_l2_truth_v1",
        "relational_persistence_high_statistics_false_v1",
        "relational_persistence_high_statistics_truth_v1",
        "relational_persistence_large_population_false_v1",
        "relational_persistence_large_population_truth_v1",
        "musr_blackboard_population_01_v1",
        "musr_blackboard_population_01_v2",
        "musr_blackboard_false_q3_companion_v1",
        "musr_blackboard_population_scout_r1_v1",
    }
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


@dataclass(frozen=True, slots=True)
class StudySpec:
    """A normalized orchestration manifest; experiment YAMLs remain authoritative."""

    name: str
    config_dir: Path
    configs: tuple[Path, ...]
    execution: Mapping[str, Any] = field(default_factory=dict)
    analysis_recipe: Path | None = None
    manifest_path: Path | None = None
    preflight: Mapping[str, Any] = field(default_factory=dict)


def discover_study(config_dir: str | Path) -> StudySpec:
    """Discover experiment configs in stable order, honoring ``study.yaml`` when present."""

    root = Path(config_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"study config directory does not exist: {root}")
    manifest_path = root / "study.yaml"
    raw: Mapping[str, Any] = {}
    if manifest_path.is_file():
        try:
            loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML in {manifest_path}: {exc}") from exc
        raw = _mapping(loaded, "study.yaml")

    study_section = _mapping(raw.get("study"), "study")
    name = str(study_section.get("name") or root.name).strip()
    if not name:
        raise ValueError("study.name must be non-empty")

    listed = raw.get("configs")
    if listed is None:
        paths = sorted(
            (
                path.resolve()
                for pattern in ("*.yaml", "*.yml")
                for path in root.glob(pattern)
                if path.name not in _RESERVED
            ),
            key=lambda path: path.name,
        )
    else:
        if isinstance(listed, (str, bytes)) or not isinstance(listed, list):
            raise ValueError("study.yaml configs must be a list of YAML paths")
        paths = []
        for item in listed:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("study.yaml configs entries must be non-empty paths")
            candidate = (root / item).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"study config must stay within {root}: {item}"
                ) from exc
            paths.append(candidate)

    if not paths:
        raise ValueError(f"no experiment YAML configs found in {root}")
    if len(set(paths)) != len(paths):
        raise ValueError("study.yaml contains duplicate experiment configs")
    for path in paths:
        if path.name in _RESERVED:
            raise ValueError(
                f"reserved orchestration YAML cannot be an experiment config: {path.name}"
            )
        if not path.is_file():
            raise ValueError(f"listed experiment config does not exist: {path}")
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"experiment config is not YAML: {path}")

    execution = dict(_mapping(raw.get("execution"), "execution"))
    mode = execution.get("mode", "config_array")
    if mode not in {"config_array", "cell_array", "auto"}:
        raise ValueError(
            f"unsupported study execution.mode {mode!r}; expected "
            "'config_array', 'cell_array', or 'auto'"
        )
    provider_load_control = execution.get("provider_load_control")
    if provider_load_control is not None:
        from mas_cc.llm_runtime.providers.load_control import ProviderLoadControlConfig

        ProviderLoadControlConfig.from_mapping(
            _mapping(provider_load_control, "execution.provider_load_control")
        )

    analysis = _mapping(raw.get("analysis"), "analysis")
    recipe_value = analysis.get("recipe")
    recipe = None
    if recipe_value is not None:
        if not isinstance(recipe_value, str) or not recipe_value.strip():
            raise ValueError("analysis.recipe must be a non-empty path")
        recipe = (root / recipe_value).resolve()
        try:
            recipe.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"analysis recipe must stay within {root}: {recipe_value}"
            ) from exc
        if not recipe.is_file():
            raise ValueError(f"analysis recipe does not exist: {recipe}")
        try:
            _mapping(
                yaml.safe_load(recipe.read_text(encoding="utf-8")), "analysis recipe"
            )
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML in {recipe}: {exc}") from exc
    elif (root / "analysis.yaml").is_file():
        recipe = (root / "analysis.yaml").resolve()
        try:
            _mapping(
                yaml.safe_load(recipe.read_text(encoding="utf-8")), "analysis recipe"
            )
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML in {recipe}: {exc}") from exc

    preflight = dict(_mapping(raw.get("preflight"), "preflight"))
    contract = preflight.get("contract")
    if contract is not None and contract not in _PREFLIGHT_CONTRACTS:
        raise ValueError(
            "unsupported study preflight.contract; expected one of "
            + ", ".join(sorted(_PREFLIGHT_CONTRACTS))
        )

    return StudySpec(
        name=name,
        config_dir=root,
        configs=tuple(paths),
        execution=execution,
        analysis_recipe=recipe,
        manifest_path=manifest_path if manifest_path.is_file() else None,
        preflight=preflight,
    )
