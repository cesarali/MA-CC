"""Reliable cross-process provider load coordination on a shared filesystem."""

from __future__ import annotations
import asyncio, json, logging, math, os, random, socket, tempfile, time, uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore

LOAD_CONTROL_CONFIG_ENV = "MAS_CC_PROVIDER_LOAD_CONTROL"
LOAD_CONTROL_DIR_ENV = "MAS_CC_PROVIDER_CONTROL_DIR"
LOGGER = logging.getLogger(__name__)


class ProviderCoordinationUnavailable(RuntimeError):
    pass


class ProviderCoordinationStateError(RuntimeError):
    pass


class _IncompleteStateRead(OSError):
    pass


def _num(raw, key, default, minimum):
    value = raw.get(key, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < minimum
    ):
        raise ValueError(f"execution.provider_load_control.{key} must be >= {minimum}")
    return float(value)


def _int(raw, key, default, minimum):
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"execution.provider_load_control.{key} must be >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class ProviderLoadControlConfig:
    mode: str = "shared_adaptive"
    initial_concurrency: int = 144
    minimum_concurrency: int = 4
    maximum_concurrency: int = 144
    target_rpm: int = 900
    lease_seconds: float = 30.0
    heartbeat_seconds: float = 5.0
    transaction_retry_attempts: int = 6
    transaction_backoff_initial_seconds: float = 0.05
    transaction_backoff_max_seconds: float = 2.0
    polling_seconds: float = 0.25
    event_window_seconds: float = 60.0
    local_failure_threshold: int = 3
    local_cooldown_seconds: float = 30.0
    global_min_samples: int = 12
    global_failure_ratio: float = 0.1
    global_cooldown_seconds: float = 60.0
    decrease_factor: float = 0.5
    increase_step: int = 1
    increase_interval_seconds: float = 30.0
    retry_max_elapsed_seconds: float = 300.0
    retry_backoff_initial_seconds: float = 2.0
    retry_backoff_max_seconds: float = 60.0

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any] | None, *, defaults: Mapping[str, Any] | None = None
    ):
        v = {**dict(defaults or {}), **dict(raw or {})}
        unknown = sorted(set(v) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(
                "unknown execution.provider_load_control field(s): "
                + ", ".join(unknown)
            )
        mode = str(v.get("mode", "shared_adaptive"))
        if mode not in {"off", "shared_adaptive"}:
            raise ValueError(
                "execution.provider_load_control.mode must be 'off' or 'shared_adaptive'"
            )
        lease = _num(v, "lease_seconds", 30, 1)
        c = cls(
            mode=mode,
            initial_concurrency=_int(v, "initial_concurrency", 144, 1),
            minimum_concurrency=_int(v, "minimum_concurrency", 4, 1),
            maximum_concurrency=_int(v, "maximum_concurrency", 144, 1),
            target_rpm=_int(v, "target_rpm", 900, 1),
            lease_seconds=lease,
            heartbeat_seconds=_num(v, "heartbeat_seconds", min(5.0, lease / 3), 0.1),
            transaction_retry_attempts=_int(v, "transaction_retry_attempts", 6, 1),
            transaction_backoff_initial_seconds=_num(
                v, "transaction_backoff_initial_seconds", 0.05, 0.001
            ),
            transaction_backoff_max_seconds=_num(
                v, "transaction_backoff_max_seconds", 2, 0.001
            ),
            polling_seconds=_num(v, "polling_seconds", 0.25, 0.01),
            event_window_seconds=_num(v, "event_window_seconds", 60, 1),
            local_failure_threshold=_int(v, "local_failure_threshold", 3, 1),
            local_cooldown_seconds=_num(v, "local_cooldown_seconds", 30, 0),
            global_min_samples=_int(v, "global_min_samples", 12, 1),
            global_failure_ratio=_num(v, "global_failure_ratio", 0.1, 0),
            global_cooldown_seconds=_num(v, "global_cooldown_seconds", 60, 0),
            decrease_factor=_num(v, "decrease_factor", 0.5, 0.01),
            increase_step=_int(v, "increase_step", 1, 1),
            increase_interval_seconds=_num(v, "increase_interval_seconds", 30, 0.1),
            retry_max_elapsed_seconds=_num(v, "retry_max_elapsed_seconds", 300, 1),
            retry_backoff_initial_seconds=_num(
                v, "retry_backoff_initial_seconds", 2, 0.01
            ),
            retry_backoff_max_seconds=_num(v, "retry_backoff_max_seconds", 60, 0.01),
        )
        if not c.minimum_concurrency <= c.initial_concurrency <= c.maximum_concurrency:
            raise ValueError(
                "provider load concurrency must satisfy minimum <= initial <= maximum"
            )
        if c.global_failure_ratio > 1 or c.decrease_factor > 1:
            raise ValueError("global_failure_ratio and decrease_factor must be <= 1")
        if (
            c.retry_backoff_initial_seconds > c.retry_backoff_max_seconds
            or c.transaction_backoff_initial_seconds > c.transaction_backoff_max_seconds
        ):
            raise ValueError(
                "retry backoff initial seconds must not exceed its maximum"
            )
        if c.heartbeat_seconds >= c.lease_seconds:
            raise ValueError(
                "provider heartbeat interval must be shorter than lease TTL"
            )
        if c.heartbeat_seconds * 3 > c.lease_seconds:
            raise ValueError(
                "provider lease TTL must be at least three heartbeat intervals"
            )
        if c.lease_seconds >= c.retry_max_elapsed_seconds / 2:
            raise ValueError(
                "provider stale-capacity recovery must be materially shorter than the logical retry window"
            )
        return c

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RequestLease:
    token: str


class SharedProviderCoordinator:
    def __init__(self, root, config, *, worker_id=None, node_id=None, clock=time.time):
        if fcntl is None:
            raise RuntimeError(
                "shared provider load control requires POSIX file locking"
            )
        self.root = Path(root)
        self.config = config
        self.node_id = node_id or socket.gethostname()
        self.worker_id = (
            worker_id
            or f"{self.node_id}:{os.getpid()}:{os.environ.get('SLURM_ARRAY_TASK_ID','local')}"
        )
        self._clock = clock
        self._jitter = random.Random()
        self._pending = {}
        self._retries = {}
        self._last_error = None
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.root / "state.lock"
        self._state_path = self.root / "state.json"
        # Stable across state.json replacements. Once this exists, a missing
        # state file is a transient/corrupt shared-filesystem observation, not
        # permission to recreate an empty semaphore and erase live leases.
        self._initialized_path = self.root / "state.initialized"

    def _initial(self, now):
        return {
            "schema_version": 2,
            "limit": self.config.initial_concurrency,
            "leases": {},
            "dispatches": [],
            "events": [],
            "event_ids": [],
            "node_pauses": {},
            "global_pause_until": 0.0,
            "last_decrease_at": 0.0,
            "last_increase_at": now,
            "updated_at": now,
            "health": {
                "transaction_successes": {},
                "transaction_failures": {},
                "retry_counts": {},
                "expired_leases": 0,
                "last_error": None,
            },
        }

    def _read(self, now):
        try:
            state = json.loads(self._state_path.read_text())
        except FileNotFoundError as e:
            if self._initialized_path.exists():
                raise _IncompleteStateRead(
                    "initialized provider load-control state is temporarily "
                    f"missing at {self._state_path}"
                ) from e
            return self._initial(now)
        except json.JSONDecodeError as e:
            raise _IncompleteStateRead(
                f"incomplete provider load-control JSON at {self._state_path}"
            ) from e
        if not isinstance(state, dict) or state.get("schema_version") != 2:
            raise ProviderCoordinationStateError(
                f"unsupported provider load-control state at {self._state_path}; preserve it for diagnosis"
            )
        for key, kind in (
            ("leases", dict),
            ("dispatches", list),
            ("events", list),
            ("event_ids", list),
            ("node_pauses", dict),
            ("health", dict),
        ):
            if not isinstance(state.get(key), kind):
                raise ProviderCoordinationStateError(
                    f"invalid provider load-control field {key} at {self._state_path}"
                )
        return state

    def _write(self, state):
        fd, tmp = tempfile.mkstemp(prefix="state-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(state, stream, sort_keys=True, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, self._state_path)
            self._initialized_path.touch(exist_ok=True)
            dfd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        finally:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass

    def _once(self, name, op):
        with self._lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            now = float(self._clock())
            s = self._read(now)
            expired = sum(
                float(x.get("expires_at", 0)) <= now for x in s["leases"].values()
            )
            s["leases"] = {
                k: v
                for k, v in s["leases"].items()
                if float(v.get("expires_at", 0)) > now
            }
            s["dispatches"] = [x for x in s["dispatches"] if float(x) > now - 60]
            s["events"] = [
                x
                for x in s["events"]
                if float(x.get("at", 0)) > now - self.config.event_window_seconds
            ]
            ids = {x.get("event_id") for x in s["events"]}
            s["event_ids"] = [x for x in s["event_ids"] if x in ids]
            s["node_pauses"] = {
                k: v for k, v in s["node_pauses"].items() if float(v) > now
            }
            h = s["health"]
            h["expired_leases"] = int(h.get("expired_leases", 0)) + expired
            for k, v in self._pending.items():
                h.setdefault("transaction_failures", {})[k] = (
                    int(h["transaction_failures"].get(k, 0)) + v
                )
            for k, v in self._retries.items():
                h.setdefault("retry_counts", {})[k] = (
                    int(h["retry_counts"].get(k, 0)) + v
                )
            if self._last_error:
                h["last_error"] = {
                    "type": type(self._last_error).__name__,
                    "message": str(self._last_error),
                    "time": now,
                    "node": self.node_id,
                }
            result = op(s, now)
            h.setdefault("transaction_successes", {})[name] = (
                int(h["transaction_successes"].get(name, 0)) + 1
            )
            s["updated_at"] = now
            self._write(s)
            self._pending.clear()
            self._retries.clear()
            self._last_error = None
            return result

    def _transaction(self, name, op, *, deadline=None):
        last = None
        for attempt in range(1, self.config.transaction_retry_attempts + 1):
            try:
                return self._once(name, op)
            except ProviderCoordinationStateError:
                raise
            except OSError as e:
                last = e
                self._last_error = e
                self._pending[name] = self._pending.get(name, 0) + 1
                if attempt == self.config.transaction_retry_attempts:
                    break
                self._retries[name] = self._retries.get(name, 0) + 1
                delay = self._jitter.uniform(
                    0,
                    min(
                        self.config.transaction_backoff_max_seconds,
                        self.config.transaction_backoff_initial_seconds
                        * 2 ** (attempt - 1),
                    ),
                )
                if deadline and time.monotonic() + delay >= deadline:
                    break
                time.sleep(delay)
        if isinstance(last, _IncompleteStateRead):
            raise ProviderCoordinationStateError(
                f"provider load-control JSON remained invalid after {attempt} reads at "
                f"{self._state_path}; preserve it for diagnosis"
            ) from last
        raise ProviderCoordinationUnavailable(
            f"coordinator operation={name} root={self.root} node={self.node_id} worker={self.worker_id} attempts={attempt}: {type(last).__name__}: {last}"
        ) from last

    def _try_acquire(self, token=None, *, deadline=None):
        token = token or uuid.uuid4().hex

        def op(s, now):
            if token in s["leases"]:
                return RequestLease(token), 0.0
            pause = max(
                float(s.get("global_pause_until", 0)),
                float(s["node_pauses"].get(self.node_id, 0)),
            )
            if pause > now:
                return None, min(self.config.polling_seconds, pause - now)
            if len(s["leases"]) >= int(s["limit"]):
                return None, self.config.polling_seconds
            if len(s["dispatches"]) >= self.config.target_rpm:
                return None, max(
                    self.config.polling_seconds, 60 - (now - float(s["dispatches"][0]))
                )
            s["leases"][token] = {
                "worker": self.worker_id,
                "node": self.node_id,
                "acquired_at": now,
                "renewed_at": now,
                "expires_at": now + self.config.lease_seconds,
            }
            s["dispatches"].append(now)
            return RequestLease(token), 0.0

        return self._transaction("acquire", op, deadline=deadline)

    async def acquire(self, *, deadline=None):
        token = uuid.uuid4().hex
        while deadline is None or time.monotonic() < deadline:
            try:
                lease, delay = await asyncio.to_thread(
                    self._try_acquire, token, deadline=deadline
                )
            except ProviderCoordinationUnavailable:
                if deadline is None:
                    raise
                delay = self.config.polling_seconds
                lease = None
            if lease:
                return lease
            await asyncio.sleep(
                min(
                    max(0.01, delay),
                    (
                        max(0, deadline - time.monotonic())
                        if deadline
                        else max(0.01, delay)
                    ),
                )
            )
        raise ProviderCoordinationUnavailable(
            f"coordinator acquire deadline expired root={self.root} node={self.node_id} worker={self.worker_id}"
        )

    async def renew(self, lease, *, deadline=None):
        def op(s, now):
            item = s["leases"].get(lease.token)
            if item is None:
                return False
            item["renewed_at"] = now
            item["expires_at"] = now + self.config.lease_seconds
            return True

        return await asyncio.to_thread(
            self._transaction, "renew", op, deadline=deadline
        )

    async def release(
        self,
        lease,
        *,
        success,
        retryable,
        status_code,
        latency_seconds,
        event_id=None,
        deadline=None,
    ):
        event_id = event_id or lease.token

        def op(s, now):
            s["leases"].pop(lease.token, None)
            if event_id in s["event_ids"]:
                return
            e = {
                "event_id": event_id,
                "at": now,
                "worker": self.worker_id,
                "node": self.node_id,
                "success": bool(success),
                "retryable": bool(retryable),
                "status_code": status_code,
                "latency_seconds": max(0.0, float(latency_seconds)),
            }
            s["events"].append(e)
            s["event_ids"].append(event_id)
            fail = [x for x in s["events"] if x.get("retryable")]
            local = [x for x in fail if x.get("node") == self.node_id]
            if retryable and len(local) >= self.config.local_failure_threshold:
                s["node_pauses"][self.node_id] = (
                    now + self.config.local_cooldown_seconds
                )
            if (
                retryable
                and len(s["events"]) >= self.config.global_min_samples
                and len(fail) / len(s["events"]) >= self.config.global_failure_ratio
            ):
                s["global_pause_until"] = max(
                    float(s.get("global_pause_until", 0)),
                    now + self.config.global_cooldown_seconds,
                )
                if (
                    now - float(s.get("last_decrease_at", 0))
                    >= self.config.global_cooldown_seconds
                ):
                    s["limit"] = max(
                        self.config.minimum_concurrency,
                        math.floor(int(s["limit"]) * self.config.decrease_factor),
                    )
                    s["last_decrease_at"] = now
            elif (
                success
                and not fail
                and now - float(s.get("last_increase_at", 0))
                >= self.config.increase_interval_seconds
            ):
                s["limit"] = min(
                    self.config.maximum_concurrency,
                    int(s["limit"]) + self.config.increase_step,
                )
                s["last_increase_at"] = now

        await asyncio.to_thread(self._transaction, "release", op, deadline=deadline)

    def snapshot(self):
        def op(s, now):
            out = dict(s)
            leases = s["leases"]
            out["health"] = {
                **s["health"],
                "active_leases": len(leases),
                "oldest_lease_age_seconds": max(
                    (now - float(x["acquired_at"]) for x in leases.values()),
                    default=0.0,
                ),
                "dispatches_in_window": len(s["dispatches"]),
                "provider_events_in_window": len(s["events"]),
                "current_limit": s["limit"],
                "global_pause_until": s["global_pause_until"],
                "node_pause_until": s["node_pauses"].get(self.node_id, 0.0),
            }
            return out

        return self._transaction("snapshot", op)


def coordinator_from_environment(provider_config, *, environment=None):
    env = os.environ if environment is None else environment
    raw = env.get(LOAD_CONTROL_CONFIG_ENV, "").strip()
    root = env.get(LOAD_CONTROL_DIR_ENV, "").strip()
    if not raw or not root:
        return None
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid {LOAD_CONTROL_CONFIG_ENV}: {e}") from e
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{LOAD_CONTROL_CONFIG_ENV} must contain a JSON object")
    config = ProviderLoadControlConfig.from_mapping(loaded)
    return None if config.mode == "off" else SharedProviderCoordinator(root, config)
