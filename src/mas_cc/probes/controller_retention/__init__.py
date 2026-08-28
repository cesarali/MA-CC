"""Local controller-retention and exposure probe.

Asks one question with one provider call at a time: **when a single LLM
decision is held fixed in every other respect, does the controller still move
the answer as the visible social group ``q`` grows?**

The probe is provider-backed but game-free.  It reuses the relational reasoning
game's prompt blocks, social-source renderer, option shuffle, response contract
and parser verbatim, and runs none of its population dynamics - no rounds, no
sensor, no controller policy, no budget, no trajectory.  See
``docs/tdd/misselaneous/28082026_controller_retention_local_probe_tdd_v3.md``.
"""

from __future__ import annotations

from .config import ModelSpec, ProbeConfig, load_probe_config
from .design import PROBE_VERSION, DesignSpec, Vignette, build_vignettes
from .preflight import run_preflight

__all__ = [
    "PROBE_VERSION",
    "DesignSpec",
    "ModelSpec",
    "ProbeConfig",
    "Vignette",
    "build_vignettes",
    "load_probe_config",
    "run_preflight",
]
