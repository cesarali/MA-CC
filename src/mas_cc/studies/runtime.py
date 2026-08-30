"""Execution-only runtime context installed by generic study workers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from mas_cc.llm_runtime.providers.load_control import (
    LOAD_CONTROL_CONFIG_ENV,
    LOAD_CONTROL_DIR_ENV,
    ProviderLoadControlConfig,
)


EXECUTION_SITE_ENV = "MAS_CC_EXECUTION_SITE"


def _mapping_file(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, Mapping) else {}


def validate_study_execution_site(manifest_path: str | Path) -> None:
    """Reject a prepared study launched by the wrong site adapter.

    Direct/local worker calls leave ``MAS_CC_EXECUTION_SITE`` unset. The two
    scheduler launchers always set it, so real cluster execution fails closed
    if a Potsdam preparation reaches NERSC or vice versa.
    """

    actual = os.environ.get(EXECUTION_SITE_ENV, "").strip()
    if actual and actual not in {"potsdam", "nersc"}:
        raise ValueError(f"unsupported execution site: {actual!r}")
    study_root = Path(manifest_path).expanduser().resolve().parent
    preparation = _mapping_file(study_root / "preparation.json")
    expected = str(preparation.get("execution_site", "unspecified"))
    if expected == "unspecified" and not actual:
        return
    if expected not in {"potsdam", "nersc"} or expected != actual:
        raise ValueError(
            f"study was prepared for execution site {expected!r}, "
            f"not {actual or 'unset'!r}: {study_root}"
        )


def configure_study_provider_load_control(manifest_path: str | Path) -> None:
    """Expose one study-wide adaptive policy to every provider factory call."""

    study_root = Path(manifest_path).expanduser().resolve().parent
    study = _mapping_file(study_root / "study_manifest.json")
    execution = study.get("execution", {})
    raw = execution.get("provider_load_control") if isinstance(execution, Mapping) else None
    plan = _mapping_file(study_root / "execution_plan.json")
    resolved = plan.get("provider_load_control") if isinstance(plan, Mapping) else None
    if resolved is None:
        total = int(plan.get("total_request_concurrency", 24)) if plan else 24
        target = int(plan.get("target_rpm", 900)) if plan else 900
        resolved = ProviderLoadControlConfig.from_mapping(
            raw if isinstance(raw, Mapping) else None,
            defaults={
                "initial_concurrency": min(24, total),
                "minimum_concurrency": min(4, total),
                "maximum_concurrency": total,
                "target_rpm": target,
            },
        ).to_dict()
    else:
        resolved = ProviderLoadControlConfig.from_mapping(resolved).to_dict()
    if resolved["mode"] == "off":
        os.environ.pop(LOAD_CONTROL_CONFIG_ENV, None)
        os.environ.pop(LOAD_CONTROL_DIR_ENV, None)
        return

    control_root = study_root / "runtime" / "provider-control"
    control_root.mkdir(parents=True, exist_ok=True)
    settings = control_root / "settings.json"
    if not settings.exists():
        fd, temporary = tempfile.mkstemp(prefix="settings-", suffix=".tmp", dir=control_root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(resolved, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, settings)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    os.environ[LOAD_CONTROL_CONFIG_ENV] = json.dumps(resolved, sort_keys=True)
    os.environ[LOAD_CONTROL_DIR_ENV] = str(control_root)
