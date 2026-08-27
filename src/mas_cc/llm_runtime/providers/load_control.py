"""Cross-process adaptive load control for remote provider attempts.

The coordinator deliberately depends only on a POSIX shared filesystem.  Study
workers on different SLURM nodes transact against one small JSON state file
under the study result root; scientific code and games never see this layer.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import socket
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

try:  # The coordinator is used on POSIX clusters; ordinary Windows runs stay importable.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on Windows
    fcntl = None  # type: ignore[assignment]


LOAD_CONTROL_CONFIG_ENV = "MAS_CC_PROVIDER_LOAD_CONTROL"
LOAD_CONTROL_DIR_ENV = "MAS_CC_PROVIDER_CONTROL_DIR"


def _number(raw: Mapping[str, Any], key: str, default: float, *, minimum: float) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
        raise ValueError(f"execution.provider_load_control.{key} must be >= {minimum}")
    return float(value)


def _integer(raw: Mapping[str, Any], key: str, default: int, *, minimum: int) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"execution.provider_load_control.{key} must be >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class ProviderLoadControlConfig:
    """Execution-only policy shared by every remote request in one study."""

    mode: str = "shared_adaptive"
    initial_concurrency: int = 24
    minimum_concurrency: int = 4
    maximum_concurrency: int = 144
    target_rpm: int = 900
    lease_seconds: float = 600.0
    polling_seconds: float = 0.25
    event_window_seconds: float = 60.0
    local_failure_threshold: int = 3
    local_cooldown_seconds: float = 30.0
    global_min_samples: int = 12
    global_failure_ratio: float = 0.25
    global_cooldown_seconds: float = 60.0
    decrease_factor: float = 0.5
    increase_step: int = 1
    increase_interval_seconds: float = 30.0

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any] | None, *, defaults: Mapping[str, Any] | None = None
    ) -> "ProviderLoadControlConfig":
        values = {**dict(defaults or {}), **dict(raw or {})}
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(
                "unknown execution.provider_load_control field(s): " + ", ".join(unknown)
            )
        mode = str(values.get("mode", "shared_adaptive"))
        if mode not in {"off", "shared_adaptive"}:
            raise ValueError("execution.provider_load_control.mode must be 'off' or 'shared_adaptive'")
        config = cls(
            mode=mode,
            initial_concurrency=_integer(values, "initial_concurrency", 24, minimum=1),
            minimum_concurrency=_integer(values, "minimum_concurrency", 4, minimum=1),
            maximum_concurrency=_integer(values, "maximum_concurrency", 144, minimum=1),
            target_rpm=_integer(values, "target_rpm", 900, minimum=1),
            lease_seconds=_number(values, "lease_seconds", 600.0, minimum=1.0),
            polling_seconds=_number(values, "polling_seconds", 0.25, minimum=0.01),
            event_window_seconds=_number(values, "event_window_seconds", 60.0, minimum=1.0),
            local_failure_threshold=_integer(values, "local_failure_threshold", 3, minimum=1),
            local_cooldown_seconds=_number(values, "local_cooldown_seconds", 30.0, minimum=0.0),
            global_min_samples=_integer(values, "global_min_samples", 12, minimum=1),
            global_failure_ratio=_number(values, "global_failure_ratio", 0.25, minimum=0.0),
            global_cooldown_seconds=_number(values, "global_cooldown_seconds", 60.0, minimum=0.0),
            decrease_factor=_number(values, "decrease_factor", 0.5, minimum=0.01),
            increase_step=_integer(values, "increase_step", 1, minimum=1),
            increase_interval_seconds=_number(
                values, "increase_interval_seconds", 30.0, minimum=0.1
            ),
        )
        if not config.minimum_concurrency <= config.initial_concurrency <= config.maximum_concurrency:
            raise ValueError(
                "provider load concurrency must satisfy minimum <= initial <= maximum"
            )
        if config.global_failure_ratio > 1 or config.decrease_factor > 1:
            raise ValueError("global_failure_ratio and decrease_factor must be <= 1")
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RequestLease:
    token: str


class SharedProviderCoordinator:
    """A leased semaphore, rolling RPM gate, and AIMD circuit breaker."""

    def __init__(
        self,
        root: str | Path,
        config: ProviderLoadControlConfig,
        *,
        worker_id: str | None = None,
        node_id: str | None = None,
        clock: Any = time.time,
    ) -> None:
        if fcntl is None:
            raise RuntimeError("shared provider load control requires POSIX file locking")
        self.root = Path(root)
        self.config = config
        self.node_id = node_id or socket.gethostname()
        self.worker_id = worker_id or (
            f"{self.node_id}:{os.getpid()}:{os.environ.get('SLURM_ARRAY_TASK_ID', 'local')}"
        )
        self._clock = clock
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.root / "state.lock"
        self._state_path = self.root / "state.json"

    def _initial_state(self, now: float) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "limit": self.config.initial_concurrency,
            "leases": {},
            "dispatches": [],
            "events": [],
            "node_pauses": {},
            "global_pause_until": 0.0,
            "last_decrease_at": 0.0,
            "last_increase_at": now,
            "updated_at": now,
        }

    def _read(self, now: float) -> dict[str, Any]:
        if not self._state_path.is_file():
            return self._initial_state(now)
        try:
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"provider load-control state is unreadable: {self._state_path}") from exc

    def _write(self, state: Mapping[str, Any]) -> None:
        fd, temporary = tempfile.mkstemp(prefix="state-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(state, stream, sort_keys=True, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._state_path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _transaction(self, operation: Any) -> Any:
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            now = float(self._clock())
            state = self._read(now)
            cutoff = now - self.config.event_window_seconds
            state["leases"] = {
                token: item
                for token, item in dict(state.get("leases", {})).items()
                if float(item.get("expires_at", 0)) > now
            }
            state["dispatches"] = [
                value for value in state.get("dispatches", []) if float(value) > now - 60.0
            ]
            state["events"] = [
                item for item in state.get("events", []) if float(item.get("at", 0)) > cutoff
            ]
            state["node_pauses"] = {
                key: value
                for key, value in dict(state.get("node_pauses", {})).items()
                if float(value) > now
            }
            result = operation(state, now)
            state["updated_at"] = now
            self._write(state)
            return result

    def _try_acquire(self) -> tuple[RequestLease | None, float]:
        def operation(state: dict[str, Any], now: float) -> tuple[RequestLease | None, float]:
            pause_until = max(
                float(state.get("global_pause_until", 0)),
                float(state.get("node_pauses", {}).get(self.node_id, 0)),
            )
            if pause_until > now:
                return None, min(self.config.polling_seconds, pause_until - now)
            if len(state["leases"]) >= int(state.get("limit", self.config.initial_concurrency)):
                return None, self.config.polling_seconds
            if len(state["dispatches"]) >= self.config.target_rpm:
                return None, max(
                    self.config.polling_seconds, 60.0 - (now - float(state["dispatches"][0]))
                )
            token = uuid.uuid4().hex
            state["leases"][token] = {
                "worker": self.worker_id,
                "node": self.node_id,
                "acquired_at": now,
                "expires_at": now + self.config.lease_seconds,
            }
            state["dispatches"].append(now)
            return RequestLease(token), 0.0

        return self._transaction(operation)

    async def acquire(self) -> RequestLease:
        while True:
            lease, delay = await asyncio.to_thread(self._try_acquire)
            if lease is not None:
                return lease
            await asyncio.sleep(max(0.01, delay))

    def _release(
        self,
        lease: RequestLease,
        *,
        success: bool,
        retryable: bool,
        status_code: int | None,
        latency_seconds: float,
    ) -> None:
        def operation(state: dict[str, Any], now: float) -> None:
            state["leases"].pop(lease.token, None)
            event = {
                "at": now,
                "worker": self.worker_id,
                "node": self.node_id,
                "success": bool(success),
                "retryable": bool(retryable),
                "status_code": status_code,
                "latency_seconds": max(0.0, float(latency_seconds)),
            }
            state["events"].append(event)
            failures = [item for item in state["events"] if item.get("retryable")]
            local_failures = [item for item in failures if item.get("node") == self.node_id]
            if retryable and len(local_failures) >= self.config.local_failure_threshold:
                state["node_pauses"][self.node_id] = now + self.config.local_cooldown_seconds
            if (
                retryable
                and len(state["events"]) >= self.config.global_min_samples
                and len(failures) / len(state["events"]) >= self.config.global_failure_ratio
            ):
                state["global_pause_until"] = max(
                    float(state.get("global_pause_until", 0)),
                    now + self.config.global_cooldown_seconds,
                )
                if (
                    now - float(state.get("last_decrease_at", 0))
                    >= self.config.global_cooldown_seconds
                ):
                    state["limit"] = max(
                        self.config.minimum_concurrency,
                        math.floor(int(state["limit"]) * self.config.decrease_factor),
                    )
                    state["last_decrease_at"] = now
            elif (
                success
                and not failures
                and now - float(state.get("last_increase_at", 0))
                >= self.config.increase_interval_seconds
            ):
                state["limit"] = min(
                    self.config.maximum_concurrency,
                    int(state["limit"]) + self.config.increase_step,
                )
                state["last_increase_at"] = now

        self._transaction(operation)

    async def release(
        self,
        lease: RequestLease,
        *,
        success: bool,
        retryable: bool,
        status_code: int | None,
        latency_seconds: float,
    ) -> None:
        await asyncio.to_thread(
            self._release,
            lease,
            success=success,
            retryable=retryable,
            status_code=status_code,
            latency_seconds=latency_seconds,
        )

    def snapshot(self) -> dict[str, Any]:
        return self._transaction(lambda state, now: dict(state))


def coordinator_from_environment(
    provider_config: Any,
    *,
    environment: Mapping[str, str] | None = None,
) -> SharedProviderCoordinator | None:
    env = os.environ if environment is None else environment
    raw_config = env.get(LOAD_CONTROL_CONFIG_ENV, "").strip()
    root = env.get(LOAD_CONTROL_DIR_ENV, "").strip()
    if not raw_config or not root:
        return None
    try:
        loaded = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {LOAD_CONTROL_CONFIG_ENV}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{LOAD_CONTROL_CONFIG_ENV} must contain a JSON object")
    config = ProviderLoadControlConfig.from_mapping(loaded)
    if config.mode == "off":
        return None
    minimum_lease = float(provider_config.timeout_seconds) * (provider_config.max_retries + 1) + 60
    if config.lease_seconds < minimum_lease:
        config = replace(config, lease_seconds=minimum_lease)
    return SharedProviderCoordinator(root, config)
