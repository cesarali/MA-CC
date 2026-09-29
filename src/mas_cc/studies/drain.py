"""Job-scoped cooperative drain control for study workers."""

from __future__ import annotations

import contextvars
import csv
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def manifest_hash(root: Path) -> str:
    return hashlib.sha256((root / "study_manifest.json").read_bytes()).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class Drained(RuntimeError):
    """Scientific execution stopped at a durable, resumable boundary."""

    def __init__(self, boundary: str) -> None:
        super().__init__(f"drained at {boundary}")
        self.boundary = boundary


class DrainController:
    def __init__(self, root: Path, *, job_id: str, attempt: str, array_index: int,
                 plan_path: Path | None = None) -> None:
        self.root = root.resolve()
        self.job_id = str(job_id)
        self.attempt = str(attempt)
        self.array_index = array_index
        self.plan_path = plan_path or self.root / "execution_plan.json"
        manifest = json.loads((self.root / "study_manifest.json").read_text())
        self.study_id = str(manifest["study_id"])
        self.study_manifest_hash = manifest_hash(self.root)
        self.directory = self.root / "runtime" / "drain" / f"job-{self.job_id}"
        self._signaled = False
        self._request: dict[str, Any] | None = None
        self._last_poll = 0.0
        self.last_boundary: str | None = None
        self.last_boundary_at: str | None = None
        self.final_state: str | None = None
        self.deadline: str | None = os.environ.get("MAS_CC_DRAIN_DEADLINE")
        self._safe_point = "episode"
        self.active_episodes = 0
        self.active_branches = 0
        self._work_lock = threading.Lock()

    @property
    def requested(self) -> bool:
        if self._signaled and self._request is None:
            self.request(reason="walltime_signal", requested_by="launcher")
        if self._request is not None:
            return True
        if time.monotonic() - self._last_poll < 0.25:
            return False
        self._last_poll = time.monotonic()
        path = self.directory / "request.json"
        if not path.exists():
            return False
        value = json.loads(path.read_text(encoding="utf-8"))
        if all(value.get(key) == expected for key, expected in self._identity().items()):
            self._request = value
        return self._request is not None

    @property
    def reason(self) -> str | None:
        return None if not self.requested else str(self._request["reason"])

    @property
    def requested_at(self) -> str | None:
        return None if not self.requested else str(self._request["requested_at"])

    @property
    def strongest_supported_safe_point(self) -> str:
        return self._safe_point

    def set_safe_point(self, capability: str) -> None:
        if capability not in {"episode", "branch", "round_replay"}:
            raise ValueError("unsupported drain safe point")
        self._safe_point = capability

    def _identity(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "study_id": self.study_id,
            "study_manifest_hash": self.study_manifest_hash,
            "submission_attempt": self.attempt,
            "job_id": self.job_id,
        }

    def request(self, *, reason: str, requested_by: str) -> dict[str, Any]:
        path = self.directory / "request.json"
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".request.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                value = json.loads(path.read_text(encoding="utf-8"))
                if any(value.get(key) != expected for key, expected in self._identity().items()):
                    raise ValueError("drain request belongs to another submission")
            else:
                value = {**self._identity(), "requested_at": _now(), "reason": reason,
                         "requested_by": requested_by}
                _atomic_json(path, value)
        self._request = value
        self._refresh_state()
        return value

    def acknowledge(self, state: str, metadata: Mapping[str, Any] | None = None,
                    *, boundary: str | None = None) -> None:
        if metadata:
            if "active_episodes" in metadata:
                self.active_episodes = int(metadata["active_episodes"])
            if "active_branches" in metadata:
                self.active_branches = int(metadata["active_branches"])
            boundary = boundary or metadata.get("last_durable_boundary")
        if boundary is not None:
            self.last_boundary = boundary
            self.last_boundary_at = _now()
        _atomic_json(self.directory / "shards" / f"{self.array_index}.json", {
            **self._identity(), "array_index": self.array_index, "pid": os.getpid(),
            "state": state, "active_episodes": self.active_episodes,
            "active_branches": self.active_branches, "last_durable_boundary": self.last_boundary,
            "last_durable_boundary_at": self.last_boundary_at,
            "strongest_supported_safe_point": self._safe_point,
            "updated_at": _now(),
        })
        self._refresh_state()

    def _refresh_state(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".state.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            rows = [json.loads(path.read_text(encoding="utf-8"))
                    for path in (self.directory / "shards").glob("*.json")]
            plan_path = self.plan_path
            expected = (int(json.loads(plan_path.read_text())["shard_count"])
                        if plan_path.exists() else max(1, len(rows)))
            requested = (self.directory / "request.json").exists()
            states = [row.get("state") for row in rows]
            if "failed" in states:
                phase = "failed"
            elif len(rows) >= expected and all(state == "scientifically_complete" for state in states):
                phase = "scientifically_complete"
            elif requested and len(rows) >= expected and all(
                state in {"drained", "scientifically_complete"} for state in states
            ):
                phase = "drained_incomplete"
            elif "draining" in states or "drained" in states:
                phase = "draining"
            elif requested:
                phase = "drain_requested"
            else:
                phase = "running"
            _atomic_json(self.directory / "state.json", {
                **self._identity(), "phase": phase, "updated_at": _now(),
                "expected_shards": expected,
                "active_episodes": sum(int(row.get("active_episodes", 0)) for row in rows),
                "active_branches": sum(int(row.get("active_branches", 0)) for row in rows),
            })

    def work_started(self, kind: str) -> None:
        with self._work_lock:
            if kind == "episode":
                self.active_episodes += 1
            else:
                self.active_branches += 1
            self.acknowledge("draining" if self.requested else "running")

    def work_finished(self, kind: str) -> None:
        with self._work_lock:
            if kind == "episode":
                self.active_episodes -= 1
            else:
                self.active_branches -= 1
            self.acknowledge("draining" if self.requested else "running")

    def raise_if_safe(self, boundary: str) -> None:
        if self.requested:
            self.acknowledge("draining", boundary=boundary)
            raise Drained(boundary)

    def signal(self, _number: int, _frame: Any) -> None:
        self._signaled = True


_CURRENT: contextvars.ContextVar[DrainController | None] = contextvars.ContextVar(
    "mas_cc_drain_controller", default=None
)


def current_controller() -> DrainController | None:
    return _CURRENT.get()


def _study_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if (root / "study_manifest.json").is_file():
        return root
    if root.name.startswith("extension-") and root.parent.name == "extensions":
        candidate = root.parent.parent
        if (candidate / "study_manifest.json").is_file():
            return candidate
    raise ValueError(f"not a study root or extension directory: {root}")


def _submission_for_job(root: Path, job_id: str) -> tuple[Path, dict[str, Any]]:
    candidates = [root / "submission.json"]
    candidates.extend(sorted((root / "extensions").glob(
        "extension-*/submissions/attempt-*.json"
    )))
    matches = []
    for path in candidates:
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if (str(value.get("status", "")).lower() == "submitted"
                    and str(value.get("job_id")) == str(job_id)):
                matches.append((path, value))
    if len(matches) != 1:
        raise ValueError("job does not match exactly one study submission")
    path, submission = matches[0]
    if submission.get("study_manifest_hash") != manifest_hash(root):
        raise ValueError("study manifest changed since submission")
    return path, submission


def _plan_path(root: Path, submission_path: Path) -> Path:
    return (root / "execution_plan.json" if submission_path == root / "submission.json"
            else submission_path.parent.parent / "execution_plan.json")


def request_study_drain(root: str | Path, job_id: str, *, reason: str = "manual",
                        verify_scheduler: bool = True) -> dict[str, Any]:
    root = _study_root(root)
    if reason not in {"manual", "operator_signal", "walltime_signal"}:
        raise ValueError("unsupported drain reason")
    submission_path, submission = _submission_for_job(root, job_id)
    if verify_scheduler:
        result = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%A"],
                                check=True, capture_output=True, text=True)
        if str(job_id) not in result.stdout.split():
            raise ValueError(f"job {job_id} is no longer active")
    controller = DrainController(
        root,
        job_id=str(job_id),
        attempt=str(submission["submission_attempt"]),
        array_index=0,
        plan_path=_plan_path(root, submission_path),
    )
    return controller.request(reason=reason, requested_by="cli")


def drain_status(root: str | Path, job_id: str) -> dict[str, Any]:
    directory = Path(root).expanduser().resolve() / "runtime" / "drain" / f"job-{job_id}"
    request = directory / "request.json"
    shards = {}
    if (directory / "shards").exists():
        for path in (directory / "shards").glob("*.json"):
            shards[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return {"request": json.loads(request.read_text(encoding="utf-8")) if request.exists() else None,
            "shards": shards}


def study_status(root: str | Path, job_id: str | None = None) -> dict[str, Any]:
    root = _study_root(root)
    if job_id is None:
        submission_path = root / "submission.json"
        submission = json.loads(submission_path.read_text(encoding="utf-8"))
    else:
        submission_path, submission = _submission_for_job(root, job_id)
    job_id = str(submission.get("job_id", ""))
    drain = drain_status(root, job_id) if job_id else {"request": None, "shards": {}}
    shards = list(drain["shards"].values())
    counts = {state: sum(row["state"] == state for row in shards) for state in (
        "running", "draining", "drained", "failed", "scientifically_complete", "finished"
    )}
    plan_path = _plan_path(root, submission_path)
    plan = (json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file()
            else submission.get("execution_plan") or {})
    expected = int(plan.get("shard_count", max(1, len(shards))))
    remaining = None
    scheduler_state = None
    if job_id:
        try:
            live = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%L"],
                                  check=True, capture_output=True, text=True, timeout=5)
            remaining = live.stdout.strip().splitlines()[0] if live.stdout.strip() else None
            if remaining is None:
                history = subprocess.run(["sacct", "-n", "-j", job_id, "-o", "State"],
                                         check=True, capture_output=True, text=True, timeout=5)
                scheduler_state = history.stdout.strip().splitlines()[0].strip() if history.stdout.strip() else None
        except (OSError, subprocess.SubprocessError):
            pass
    if counts["failed"] or submission.get("status") == "failed":
        phase = "failed"
    elif len(shards) >= expected and all(row["state"] == "scientifically_complete" for row in shards):
        phase = "scientifically_complete"
    elif scheduler_state and scheduler_state.startswith("TIMEOUT"):
        phase = "scheduler_timeout"
    elif drain["request"] and len(shards) >= expected and all(
        row["state"] in {"drained", "scientifically_complete"} for row in shards
    ):
        phase = "drained_incomplete"
    elif counts["draining"] or counts["drained"]:
        phase = "draining"
    elif drain["request"]:
        phase = "drain_requested"
    else:
        phase = "running"
    return {"study_dir": str(root), "job_id": job_id, "phase": phase,
            "submission_path": str(submission_path),
            "drain_request": drain["request"], "expected_shards": expected,
            "shards": counts,
            "active_episodes": sum(int(row.get("active_episodes", 0)) for row in shards),
            "active_branches": sum(int(row.get("active_branches", 0)) for row in shards),
            "hard_walltime_remaining": remaining,
            "scheduler_state": scheduler_state,
            "last_durable_safe_point_time": max(
                (str(row["last_durable_boundary_at"]) for row in shards
                 if row.get("last_durable_boundary_at")),
                default=None,
            )}


@contextmanager
def worker_drain(manifest: str | Path, index: int) -> Iterator[DrainController | None]:
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    attempt = os.environ.get("MAS_CC_SUBMISSION_ATTEMPT")
    if not job_id:
        yield None
        return
    manifest_path = Path(manifest).resolve()
    root = manifest_path.parent
    extension_manifest = not (root / "study_manifest.json").is_file()
    if extension_manifest:
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            entries = list(csv.DictReader(stream))
        study_roots = {str(row.get("study_root", "")).strip() for row in entries}
        study_roots.discard("")
        if len(study_roots) != 1:
            raise ValueError("extension execution manifest has no unique study root")
        root = Path(study_roots.pop()).resolve()
        if not (root / "study_manifest.json").is_file():
            raise ValueError("extension study root has no study manifest")
    until = time.monotonic() + 30
    while True:
        try:
            submission_path, submission = _submission_for_job(root, job_id)
            break
        except ValueError as exc:
            if "manifest changed" in str(exc) or time.monotonic() >= until:
                raise
            time.sleep(0.2)
    if extension_manifest and submission_path.parent.parent != manifest_path.parent:
        raise ValueError("extension manifest does not match the submitted job")
    if not extension_manifest and submission_path != root / "submission.json":
        raise ValueError("base manifest does not match the submitted job")
    recorded_attempt = str(submission["submission_attempt"])
    if attempt and str(attempt) != recorded_attempt:
        raise ValueError("worker attempt does not match the submitted job")
    if extension_manifest:
        selected = [row for row in entries if int(row["array_index"]) == index]
        if len(selected) != 1 or not selected[0].get("episode_plan_path"):
            raise ValueError("extension shard has no unique episode plan")
        with Path(selected[0]["episode_plan_path"]).open(
            newline="", encoding="utf-8"
        ) as stream:
            episode = next(csv.DictReader(stream), None)
        if episode is None or str(episode.get("submission_attempt")) != recorded_attempt:
            raise ValueError("extension episode plan does not match the submitted attempt")
    attempt = recorded_attempt
    controller = DrainController(root, job_id=job_id,
                                 attempt=attempt, array_index=index,
                                 plan_path=_plan_path(root, submission_path))
    token = _CURRENT.set(controller)
    old = signal.getsignal(signal.SIGUSR1)
    try:
        import threading
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGUSR1, controller.signal)
        controller.acknowledge("running")
        yield controller
        controller.acknowledge(controller.final_state or
                               ("drained" if controller.requested else "finished"))
    except BaseException:
        controller.acknowledge("failed")
        raise
    finally:
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGUSR1, old)
        _CURRENT.reset(token)
