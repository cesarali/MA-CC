"""Publish an aggregated study to a bucket and prove it arrived.

The publish step is the last node of a study: copy the package zip to
``<remote>/aggregation_results/`` and mirror the analysis directory to
``<remote>/<prefix>/<study>/analysis``, then read the remote back
(``rclone size``, ``rclone lsjson``) and write ``publish_receipt.json`` with
local and remote object counts and byte totals. A receipt whose ``verified``
is false is a failed publish, whatever rclone's exit code said.

Transport is ``rclone`` (present on the Cygnus login node as ``~/bin/rclone``);
credentials stay in the rclone config file named by ``--rclone-config`` or the
``RCLONE_CONFIG`` environment variable and never touch this module. The
``remote`` is an rclone path such as ``r2tmp:agent-swarm-control-research``;
a plain directory works too (rclone's local backend), which is what the tests
use.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

RECEIPT = "publish_receipt.json"


class PublishError(RuntimeError):
    pass


def find_analysis_dir(study_dir: Path) -> Path:
    """``<study>/analysis`` or the newest ``analysis-runs/<run>/output`` that holds a package."""
    candidates = [study_dir / "analysis"]
    runs = study_dir / "analysis-runs"
    if runs.is_dir():
        candidates += sorted((run / "output" for run in runs.iterdir() if (run / "output").is_dir()),
                             key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*_analysis.zip")):
            return candidate
    raise PublishError(f"no analysis directory with a *_analysis.zip under {study_dir}")


def _local_inventory(directory: Path) -> tuple[int, int]:
    files = [path for path in directory.rglob("*") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files)


class Rclone:
    def __init__(self, binary: str | None = None, config: Path | None = None, *, transfers: int = 16) -> None:
        self.binary = binary or os.environ.get("MA_CC_RCLONE") or shutil.which("rclone") or str(Path.home() / "bin" / "rclone")
        self.config = Path(config) if config else (Path(os.environ["RCLONE_CONFIG"]) if os.environ.get("RCLONE_CONFIG") else None)
        self.transfers = transfers

    def _run(self, *arguments: str) -> str:
        command = [self.binary]
        if self.config is not None:
            command += ["--config", str(self.config)]
        command += list(arguments)
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise PublishError(f"rclone {arguments[0]} failed ({completed.returncode}): {completed.stderr.strip()[-400:]}")
        return completed.stdout

    def copyto(self, source: Path, destination: str) -> None:
        self._run("copyto", str(source), destination)

    def sync(self, source: Path, destination: str) -> None:
        self._run("sync", str(source), destination, "--transfers", str(self.transfers))

    def size(self, target: str) -> tuple[int, int]:
        info = json.loads(self._run("size", "--json", target))
        return int(info["count"]), int(info["bytes"])

    def object_size(self, target: str) -> int | None:
        entries = json.loads(self._run("lsjson", target))
        return int(entries[0]["Size"]) if entries else None


def publish(study_dir: Path, *, remote: str, prefix: str, rclone: Rclone | None = None,
            analysis_dir: Path | None = None, study_name: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    study_dir = Path(study_dir)
    analysis = Path(analysis_dir) if analysis_dir else find_analysis_dir(study_dir)
    zips = sorted(analysis.glob("*_analysis.zip"))
    if not zips:
        raise PublishError(f"no *_analysis.zip in {analysis}")
    package = zips[-1]
    name = study_name or study_dir.name
    remote = remote.rstrip("/")
    package_target = f"{remote}/aggregation_results/{package.name}"
    analysis_target = f"{remote}/{prefix.strip('/')}/{name}/analysis"
    local_count, local_bytes = _local_inventory(analysis)
    receipt: dict[str, Any] = {
        "schema_version": 1, "study_dir": str(study_dir), "analysis_dir": str(analysis), "package": package.name,
        "package_bytes": package.stat().st_size, "remote": remote, "package_target": package_target,
        "analysis_target": analysis_target, "local_files": local_count, "local_bytes": local_bytes,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "dry_run": dry_run,
    }
    if dry_run:
        receipt.update(verified=None, status="dry_run")
        return receipt
    client = rclone or Rclone()
    client.copyto(package, package_target)
    client.sync(analysis, analysis_target)
    remote_count, remote_bytes = client.size(analysis_target)
    remote_package_bytes = client.object_size(package_target)
    receipt.update(
        remote_files=remote_count, remote_bytes=remote_bytes, remote_package_bytes=remote_package_bytes,
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    receipt["verified"] = (remote_count == local_count and remote_bytes == local_bytes
                           and remote_package_bytes == receipt["package_bytes"])
    receipt["status"] = "ok" if receipt["verified"] else "mismatch"
    receipt_path = study_dir / RECEIPT
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    client.copyto(receipt_path, f"{remote}/{prefix.strip('/')}/{name}/{RECEIPT}")
    if not receipt["verified"]:
        raise PublishError(f"publish did not verify: {json.dumps({k: receipt[k] for k in ('local_files', 'remote_files', 'local_bytes', 'remote_bytes', 'package_bytes', 'remote_package_bytes')})}")
    return receipt


__all__ = ["PublishError", "RECEIPT", "Rclone", "find_analysis_dir", "publish"]
