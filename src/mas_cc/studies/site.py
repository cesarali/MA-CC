"""Resolve generic study launchers for the active SLURM cluster."""

from __future__ import annotations

import os
import re
import socket
import subprocess
from pathlib import Path


def active_cluster() -> str | None:
    declared = os.environ.get("SLURM_CLUSTER_NAME", "").strip()
    if declared:
        return declared.lower()
    try:
        result = subprocess.run(
            ["scontrol", "show", "config"], check=True,
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        host = socket.gethostname().lower()
        return "cygnus" if host.startswith("slurm-cluster-") or host.startswith("cygnus-") else None
    match = re.search(r"^ClusterName\s*=\s*(\S+)", result.stdout, flags=re.MULTILINE)
    return match.group(1).lower() if match else None


def default_study_launcher(*, cell_array: bool) -> Path:
    site = "Cygnus" if active_cluster() == "cygnus" else "Potsdam"
    name = "run_study_cell_array.job" if cell_array else "run_config_array.job"
    return Path("scripts") / site / "SLURM" / name
