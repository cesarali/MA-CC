"""Transient aggregation heartbeat and compact stage measurements."""
from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import json
import os
from pathlib import Path
import resource
from threading import Event, RLock, Thread
from time import monotonic


active_profile: ContextVar["AggregationProfile | None"] = ContextVar(
    "aggregation_profile", default=None
)


def _rss_mib() -> float:
    try:
        resident = int(Path("/proc/self/statm").read_text().split()[1])
        return resident * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class AggregationProfile:
    def __init__(self, path: Path, mirror: Path | None = None):
        self.path = path
        self.mirror = mirror
        self.metadata: dict = {}
        if mirror is not None and mirror.is_file():
            try:
                previous = json.loads(mirror.read_text())
                self.metadata = {
                    key: previous[key]
                    for key in (
                        "generation_id", "scientific_input_identity", "analysis_recipe_hash",
                        "jobs", "job_states", "final_archive",
                    )
                    if key in previous
                }
                if "job_states" in self.metadata:
                    self.metadata["job_states"]["finalizer"] = "RUNNING"
            except (OSError, ValueError):
                pass
        self.started = self.stage_started = monotonic()
        self.stage_name = "canonical_discovery_read"
        self.start_rss = self.peak_rss = _rss_mib()
        self.records: list[dict] = []
        self.details: dict = {}
        self.lock = RLock()
        self.stop = Event()
        self.thread = Thread(target=self._heartbeat, daemon=True)

    def _record(self) -> dict:
        rss = _rss_mib()
        self.peak_rss = max(self.peak_rss, rss)
        return {
            "details": dict(self.details),
            "stage": self.stage_name,
            "elapsed_seconds": monotonic() - self.stage_started,
            "rss_delta_mib": rss - self.start_rss,
            "sampled_peak_rss_mib": self.peak_rss,
            "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "child_peak_rss_mib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
        }

    def stage(self, name: str, **details) -> None:
        with self.lock:
            self.records.append(self._record())
            self.stage_name, self.stage_started = name, monotonic()
            self.start_rss = self.peak_rss = _rss_mib()
            self.details = details
            self._write()

    def update(self, update: dict) -> None:
        with self.lock:
            name = update.get("substage", update.get("stage", self.stage_name))
            if name != self.stage_name:
                self.stage(str(name))
            self.details.update(update)
            self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **self.metadata, **self.details, **self._record(),
            "total_elapsed_seconds": monotonic() - self.started,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if payload.get("status") == "failed" and "job_states" in payload:
            payload["job_states"]["finalizer"] = "FAILED"
        for destination in (self.path, self.mirror):
            if destination is None:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n")
            temporary.replace(destination)

    def _heartbeat(self) -> None:
        while not self.stop.wait(30):
            with self.lock:
                self._write()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "stages": [*self.records, self._record()],
                "elapsed_seconds": monotonic() - self.started,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "child_peak_rss_mib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
            }


def measured_aggregation(function):
    @wraps(function)
    def measured(study_dir, **kwargs):
        directory = kwargs.get("analysis_output_dir") or Path(study_dir).expanduser().resolve() / "analysis"
        profile = AggregationProfile(
            Path(directory) / "progress.json", kwargs.get("progress_path")
        )
        token = active_profile.set(profile)
        profile.thread.start()
        succeeded = False
        try:
            result = function(study_dir, **kwargs)
            succeeded = True
            return result
        except Exception as exc:
            try:
                profile.update({"status": "failed", "error": str(exc)})
            except OSError:
                pass  # Diagnostics must not hide the original failure.
            raise
        finally:
            profile.stop.set()
            profile.thread.join()
            active_profile.reset(token)
            if succeeded:
                profile.path.unlink(missing_ok=True)
    return measured
