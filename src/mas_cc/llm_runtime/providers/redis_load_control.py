"""Cross-process provider load coordination backed by Redis.

This is the ``redis_adaptive`` counterpart of ``SharedProviderCoordinator``
(``shared_adaptive``). It enforces the same policy: one concurrency limit
shared by every worker, a rolling requests-per-minute gate, per-node pauses,
a global breaker that halves the limit on retryable failures, and slow growth
after clean successes. The difference is where the shared state lives.

``SharedProviderCoordinator`` keeps one JSON file on a shared filesystem and
rewrites it under a directory lock for every acquire, renew and release. On a
network filesystem with hundreds of waiting requests that lock becomes the
bottleneck. Here each operation is one Lua script that Redis runs as a single
atomic step, so no lock file, no directory scan and no fsync are involved.

The Redis client (``redis`` on PyPI) is optional and imported only when this
coordinator is constructed. Redis must keep these keys: run it without key
eviction (``maxmemory-policy noeviction``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import socket
import time
import uuid
from typing import Any

from .load_control import (
    ProviderAdmissionTimeout,
    ProviderCoordinationUnavailable,
    ProviderCoordinationStateError,
    ProviderLoadControlConfig,
    RequestLease,
    provider_load_control_policy_hash,
)

_KEY_NAMES = (
    "limit",  # 1  string: current shared concurrency limit
    "leases",  # 2  hash: token -> lease JSON
    "lease_expiry",  # 3  sorted set: token -> expires_at
    "dispatches",  # 4  sorted set: token -> dispatch time (rolling RPM window)
    "events",  # 5  sorted set: event_id -> time (outcome window)
    "event_data",  # 6  hash: event_id -> event JSON
    "fail_events",  # 7  sorted set: retryable event_id -> time
    "fail_nodes",  # 8  hash: retryable event_id -> node
    "node_pauses",  # 9  hash: node -> paused until
    "meta",  # 10 hash: global_pause_until, last_decrease_at, last_increase_at, updated_at
    "health",  # 11 hash: counters
)

# Shared prelude for every script. ARGV[1] is always ``now`` and ARGV[2..7]
# the maintenance settings, so the purge and initialisation below behave the
# same for acquire, renew, release and snapshot.
#   ARGV[1] now
#   ARGV[2] initial_concurrency
#   ARGV[3] event_window_seconds
#   ARGV[4] namespace (stored for diagnosis only)
#   ARGV[5] policy_hash
#   ARGV[6] minimum_concurrency
#   ARGV[7] maximum_concurrency
_PRELUDE = r"""
local now = tonumber(ARGV[1])
local initial = tonumber(ARGV[2])
local event_window = tonumber(ARGV[3])
local policy_hash = ARGV[5]
local minimum = tonumber(ARGV[6])
local maximum = tonumber(ARGV[7])

if redis.call('EXISTS', KEYS[1]) == 0 then
  redis.call('SET', KEYS[1], initial)
  redis.call('HSET', KEYS[10],
    'global_pause_until', 0, 'last_decrease_at', 0,
    'last_increase_at', now, 'updated_at', now, 'namespace', ARGV[4],
    'policy_hash', policy_hash)
else
  local persisted_policy = redis.call('HGET', KEYS[10], 'policy_hash')
  if not persisted_policy or persisted_policy ~= policy_hash then
    return redis.error_reply('MAS_CC_POLICY_MISMATCH persisted=' ..
      tostring(persisted_policy) .. ' worker=' .. policy_hash)
  end
end

local live_limit = tonumber(redis.call('GET', KEYS[1]))
if not live_limit or live_limit < minimum or live_limit > maximum then
  return redis.error_reply('MAS_CC_LIMIT_OUT_OF_POLICY limit=' ..
    tostring(live_limit) .. ' allowed=[' .. tostring(minimum) .. ',' ..
    tostring(maximum) .. ']')
end

local expired = redis.call('ZRANGEBYSCORE', KEYS[3], '-inf', now)
for _, token in ipairs(expired) do
  redis.call('HDEL', KEYS[2], token)
end
if #expired > 0 then
  redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', now)
  redis.call('HINCRBY', KEYS[11], 'expired_leases', #expired)
end

redis.call('ZREMRANGEBYSCORE', KEYS[4], '-inf', now - 60)

local old_events = redis.call('ZRANGEBYSCORE', KEYS[5], '-inf', now - event_window)
for _, event_id in ipairs(old_events) do
  redis.call('HDEL', KEYS[6], event_id)
  redis.call('HDEL', KEYS[8], event_id)
end
if #old_events > 0 then
  redis.call('ZREMRANGEBYSCORE', KEYS[5], '-inf', now - event_window)
  redis.call('ZREMRANGEBYSCORE', KEYS[7], '-inf', now - event_window)
end

local pauses = redis.call('HGETALL', KEYS[9])
for i = 1, #pauses, 2 do
  if tonumber(pauses[i + 1]) <= now then
    redis.call('HDEL', KEYS[9], pauses[i])
  end
end

local function meta_number(field)
  return tonumber(redis.call('HGET', KEYS[10], field) or '0') or 0
end
"""

# Returns {granted (0/1), delay_seconds as a string}. Redis converts Lua
# numbers to integers in replies, so fractional delays travel as strings.
#   ARGV[8] token  ARGV[9] node  ARGV[10] worker
#   ARGV[11] lease_seconds  ARGV[12] polling_seconds  ARGV[13] target_rpm
_ACQUIRE = _PRELUDE + r"""
local token = ARGV[8]
local node = ARGV[9]
local worker = ARGV[10]
local lease_seconds = tonumber(ARGV[11])
local polling = tonumber(ARGV[12])
local target_rpm = tonumber(ARGV[13])

if redis.call('HEXISTS', KEYS[2], token) == 1 then
  return {1, '0'}
end

local pause = meta_number('global_pause_until')
local node_pause = tonumber(redis.call('HGET', KEYS[9], node) or '0') or 0
if node_pause > pause then pause = node_pause end
if pause > now then
  return {0, tostring(math.min(polling, pause - now))}
end

local limit = tonumber(redis.call('GET', KEYS[1]))
if redis.call('HLEN', KEYS[2]) >= limit then
  return {0, tostring(polling)}
end

if redis.call('ZCARD', KEYS[4]) >= target_rpm then
  local oldest = redis.call('ZRANGE', KEYS[4], 0, 0, 'WITHSCORES')
  local wait = 60 - (now - tonumber(oldest[2]))
  return {0, tostring(math.max(polling, wait))}
end

local lease = cjson.encode({
  worker = worker, node = node,
  acquired_at = now, renewed_at = now, expires_at = now + lease_seconds,
})
redis.call('HSET', KEYS[2], token, lease)
redis.call('ZADD', KEYS[3], now + lease_seconds, token)
redis.call('ZADD', KEYS[4], now, token)
redis.call('HSET', KEYS[10], 'updated_at', now)
redis.call('HINCRBY', KEYS[11], 'transaction_successes:acquire', 1)
return {1, '0'}
"""

# Returns 1 if the lease was still live and is now extended, else 0.
#   ARGV[8] token  ARGV[9] lease_seconds
_RENEW = _PRELUDE + r"""
local token = ARGV[8]
local lease_seconds = tonumber(ARGV[9])
local raw = redis.call('HGET', KEYS[2], token)
if not raw then
  return 0
end
local lease = cjson.decode(raw)
lease['renewed_at'] = now
lease['expires_at'] = now + lease_seconds
redis.call('HSET', KEYS[2], token, cjson.encode(lease))
redis.call('ZADD', KEYS[3], now + lease_seconds, token)
redis.call('HSET', KEYS[10], 'updated_at', now)
redis.call('HINCRBY', KEYS[11], 'transaction_successes:renew', 1)
return 1
"""

# Remove a lease without recording a provider outcome. This is used when an
# acquire completes after its waiting coroutine has been cancelled.
#   ARGV[8] token
_ABANDON = _PRELUDE + r"""
local token = ARGV[8]
local removed = redis.call('HDEL', KEYS[2], token)
redis.call('ZREM', KEYS[3], token)
redis.call('HSET', KEYS[10], 'updated_at', now)
if removed > 0 then
  redis.call('HINCRBY', KEYS[11], 'transaction_successes:abandon', 1)
end
return removed
"""

# Returns 1 if a new outcome event was recorded, 0 if this event_id was
# already recorded (a retried release after a lost reply).
#   ARGV[8] token  ARGV[9] event_id  ARGV[10] node  ARGV[11] worker
#   ARGV[12] success (0/1)  ARGV[13] retryable (0/1)  ARGV[14] status_code ('' = none)
#   ARGV[15] latency_seconds
#   ARGV[16] local_failure_threshold  ARGV[17] local_cooldown_seconds
#   ARGV[18] global_min_samples  ARGV[19] global_failure_ratio
#   ARGV[20] global_cooldown_seconds  ARGV[21] decrease_factor
#   ARGV[22] minimum_concurrency  ARGV[23] maximum_concurrency
#   ARGV[24] increase_step  ARGV[25] increase_interval_seconds
_RELEASE = _PRELUDE + r"""
local token = ARGV[8]
local event_id = ARGV[9]
local node = ARGV[10]
local success = ARGV[12] == '1'
local retryable = ARGV[13] == '1'

redis.call('HDEL', KEYS[2], token)
redis.call('ZREM', KEYS[3], token)

if redis.call('ZSCORE', KEYS[5], event_id) then
  redis.call('HSET', KEYS[10], 'updated_at', now)
  return 0
end

local status = cjson.null
if ARGV[14] ~= '' then status = tonumber(ARGV[14]) end
redis.call('ZADD', KEYS[5], now, event_id)
redis.call('HSET', KEYS[6], event_id, cjson.encode({
  event_id = event_id, at = now, worker = ARGV[11], node = node,
  success = success, retryable = retryable, status_code = status,
  latency_seconds = math.max(0, tonumber(ARGV[15])),
}))
if retryable then
  redis.call('ZADD', KEYS[7], now, event_id)
  redis.call('HSET', KEYS[8], event_id, node)
end

local total = redis.call('ZCARD', KEYS[5])
local fails = redis.call('ZCARD', KEYS[7])
local local_fails = 0
if fails > 0 then
  local fail_nodes = redis.call('HVALS', KEYS[8])
  for _, n in ipairs(fail_nodes) do
    if n == node then local_fails = local_fails + 1 end
  end
end

if retryable and local_fails >= tonumber(ARGV[16]) then
  redis.call('HSET', KEYS[9], node, now + tonumber(ARGV[17]))
end

local limit = tonumber(redis.call('GET', KEYS[1]))
if retryable and total >= tonumber(ARGV[18]) and fails / total >= tonumber(ARGV[19]) then
  local cooldown = tonumber(ARGV[20])
  local pause_until = math.max(meta_number('global_pause_until'), now + cooldown)
  redis.call('HSET', KEYS[10], 'global_pause_until', pause_until)
  if now - meta_number('last_decrease_at') >= cooldown then
    local reduced = math.max(tonumber(ARGV[22]), math.floor(limit * tonumber(ARGV[21])))
    redis.call('SET', KEYS[1], reduced)
    redis.call('HSET', KEYS[10], 'last_decrease_at', now)
  end
elseif success and fails == 0 and now - meta_number('last_increase_at') >= tonumber(ARGV[25]) then
  local grown = math.min(tonumber(ARGV[23]), limit + tonumber(ARGV[24]))
  redis.call('SET', KEYS[1], grown)
  redis.call('HSET', KEYS[10], 'last_increase_at', now)
end

redis.call('HSET', KEYS[10], 'updated_at', now)
redis.call('HINCRBY', KEYS[11], 'transaction_successes:release', 1)
return 1
"""

# Returns one JSON string with the raw pieces; Python assembles the same
# snapshot shape SharedProviderCoordinator returns.
_SNAPSHOT = _PRELUDE + r"""
return cjson.encode({
  limit = tonumber(redis.call('GET', KEYS[1])),
  leases = redis.call('HGETALL', KEYS[2]),
  dispatches = redis.call('ZRANGE', KEYS[4], 0, -1, 'WITHSCORES'),
  events = redis.call('HVALS', KEYS[6]),
  node_pauses = redis.call('HGETALL', KEYS[9]),
  meta = redis.call('HGETALL', KEYS[10]),
  health = redis.call('HGETALL', KEYS[11]),
})
"""


def _pairs(flat: Any) -> dict[str, str]:
    # cjson encodes an empty Lua table as {} rather than [].
    if not isinstance(flat, list):
        return {}
    return {str(flat[i]): flat[i + 1] for i in range(0, len(flat) - 1, 2)}


class RedisProviderCoordinator:
    """Same public contract as ``SharedProviderCoordinator``, state in Redis."""

    supports_admission_deadline = True

    def __init__(
        self,
        namespace,
        config: ProviderLoadControlConfig,
        *,
        url: str | None = None,
        client: Any = None,
        worker_id: str | None = None,
        node_id: str | None = None,
        clock=time.time,
    ):
        if client is None:
            if not url:
                raise ValueError("RedisProviderCoordinator needs a Redis URL or client")
            try:
                import redis  # optional dependency
            except ImportError as e:  # pragma: no cover - exercised without redis
                raise RuntimeError(
                    "provider load-control mode 'redis_adaptive' needs the "
                    "'redis' package (pip install 'llm-naming-game[redis]')"
                ) from e
            client = redis.Redis.from_url(
                url, socket_timeout=5, socket_connect_timeout=5, health_check_interval=30
            )
        self.config = config
        # ``root`` keeps the attribute adapters log for SharedProviderCoordinator.
        self.root = str(namespace)
        self.node_id = node_id or socket.gethostname()
        self.worker_id = (
            worker_id
            or f"{self.node_id}:{os.getpid()}:{os.environ.get('SLURM_ARRAY_TASK_ID', 'local')}"
        )
        self._client = client
        self._clock = clock
        self.policy_hash = provider_load_control_policy_hash(config)
        self._jitter = random.Random()
        digest = hashlib.sha256(self.root.encode("utf-8")).hexdigest()[:16]
        # The {...} hash tag keeps every key on one Redis Cluster slot, which
        # multi-key scripts require.
        prefix = f"mas_cc:provider_control:{{{digest}}}:"
        self._keys = [prefix + name for name in _KEY_NAMES]
        self._acquire_script = client.register_script(_ACQUIRE)
        self._renew_script = client.register_script(_RENEW)
        self._abandon_script = client.register_script(_ABANDON)
        self._release_script = client.register_script(_RELEASE)
        self._snapshot_script = client.register_script(_SNAPSHOT)
        # Same rule as the file coordinator: one coroutine per worker polls for
        # capacity; renewals and releases bypass the gate.
        self._acquire_gate = asyncio.Lock()
        self._pending_failures: dict[str, int] = {}
        self._pending_retries: dict[str, int] = {}

    # -- plumbing ---------------------------------------------------------

    def _base_args(self) -> list[Any]:
        return [
            repr(float(self._clock())),
            self.config.initial_concurrency,
            repr(float(self.config.event_window_seconds)),
            self.root,
            self.policy_hash,
            self.config.minimum_concurrency,
            self.config.maximum_concurrency,
        ]

    def _run(self, name: str, script, args: list[Any], *, deadline: float | None = None):
        """Run one script, retrying connection-level failures with backoff."""

        try:
            from redis.exceptions import ConnectionError as RedisConnectionError
            from redis.exceptions import ResponseError as RedisResponseError
            from redis.exceptions import TimeoutError as RedisTimeoutError

            transient: tuple[type[BaseException], ...] = (
                RedisConnectionError,
                RedisTimeoutError,
                OSError,
            )
        except ImportError:  # pragma: no cover - client injected without redis
            transient = (OSError,)
            RedisResponseError = ()
        last: BaseException | None = None
        attempts = self.config.transaction_retry_attempts
        for attempt in range(1, attempts + 1):
            try:
                result = script(keys=self._keys, args=self._base_args() + args)
            except RedisResponseError as e:
                message = str(e)
                if "MAS_CC_POLICY_MISMATCH" in message:
                    raise ProviderCoordinationStateError(
                        "Redis provider load-control policy mismatch for "
                        f"namespace={self.root}: {message}"
                    ) from e
                if "MAS_CC_LIMIT_OUT_OF_POLICY" in message:
                    raise ProviderCoordinationStateError(
                        "Redis provider load-control limit is outside its policy for "
                        f"namespace={self.root}: {message}"
                    ) from e
                raise ProviderCoordinationUnavailable(
                    f"coordinator operation={name} backend=redis namespace={self.root} "
                    f"returned an invalid script response: {message}"
                ) from e
            except transient as e:
                last = e
                self._pending_failures[name] = self._pending_failures.get(name, 0) + 1
                if attempt == attempts:
                    break
                self._pending_retries[name] = self._pending_retries.get(name, 0) + 1
                delay = self._jitter.uniform(
                    0,
                    min(
                        self.config.transaction_backoff_max_seconds,
                        self.config.transaction_backoff_initial_seconds * 2 ** (attempt - 1),
                    ),
                )
                if deadline and time.monotonic() + delay >= deadline:
                    break
                time.sleep(delay)
                continue
            self._flush_health(last)
            return result
        raise ProviderCoordinationUnavailable(
            f"coordinator operation={name} backend=redis namespace={self.root} "
            f"node={self.node_id} worker={self.worker_id} attempts={attempt}: "
            f"{type(last).__name__}: {last}"
        ) from last

    def _flush_health(self, last_error: BaseException | None) -> None:
        if not (self._pending_failures or self._pending_retries):
            return
        health = self._keys[10]
        try:
            pipe = self._client.pipeline(transaction=False)
            for name, count in self._pending_failures.items():
                pipe.hincrby(health, f"transaction_failures:{name}", count)
            for name, count in self._pending_retries.items():
                pipe.hincrby(health, f"retry_counts:{name}", count)
            if last_error is not None:
                pipe.hset(
                    health,
                    "last_error",
                    json.dumps(
                        {
                            "type": type(last_error).__name__,
                            "message": str(last_error),
                            "time": float(self._clock()),
                            "node": self.node_id,
                        }
                    ),
                )
            pipe.execute()
        except Exception:  # health reporting must never fail the request
            return
        self._pending_failures.clear()
        self._pending_retries.clear()

    # -- public contract -------------------------------------------------

    def _try_acquire(self, token=None, *, deadline=None):
        token = token or uuid.uuid4().hex
        granted, delay = self._run(
            "acquire",
            self._acquire_script,
            [
                token,
                self.node_id,
                self.worker_id,
                repr(float(self.config.lease_seconds)),
                repr(float(self.config.polling_seconds)),
                self.config.target_rpm,
            ],
            deadline=deadline,
        )
        if int(granted) == 1:
            return RequestLease(token), 0.0
        if isinstance(delay, bytes):
            delay = delay.decode("ascii")
        return None, float(delay)

    async def acquire(self, *, deadline=None, admission=False):
        token = uuid.uuid4().hex
        async with self._acquire_gate:
            contention_delay = self.config.polling_seconds
            while deadline is None or time.monotonic() < deadline:
                try:
                    attempt = asyncio.create_task(
                        asyncio.to_thread(
                            self._try_acquire, token, deadline=deadline
                        )
                    )
                    try:
                        lease, delay = await asyncio.shield(attempt)
                    except asyncio.CancelledError:
                        # A Redis call already running in a worker thread cannot
                        # be cancelled. Reconcile a late grant before propagating
                        # cancellation so capacity is never stranded.
                        try:
                            granted, _ = await asyncio.shield(attempt)
                        except Exception:
                            pass
                        else:
                            if granted is not None:
                                try:
                                    await asyncio.shield(self.abandon(granted))
                                except Exception:
                                    pass
                        raise
                except ProviderCoordinationUnavailable:
                    if deadline is None:
                        raise
                    delay = contention_delay
                    lease = None
                if lease:
                    return lease
                if delay <= self.config.polling_seconds:
                    contention_delay = min(
                        max(1.0, self.config.polling_seconds * 8),
                        max(self.config.polling_seconds, contention_delay * 1.5),
                    )
                    delay = self._jitter.uniform(
                        0.75 * contention_delay, 1.25 * contention_delay
                    )
                await asyncio.sleep(
                    min(
                        max(0.01, delay),
                        max(0, deadline - time.monotonic())
                        if deadline
                        else max(0.01, delay),
                    )
                )
        error_type = (
            ProviderAdmissionTimeout
            if admission
            else ProviderCoordinationUnavailable
        )
        raise error_type(
            f"coordinator acquire deadline expired backend=redis namespace={self.root} "
            f"node={self.node_id} worker={self.worker_id}"
        )

    async def renew(self, lease, *, deadline=None):
        renewed = await asyncio.to_thread(
            self._run,
            "renew",
            self._renew_script,
            [lease.token, repr(float(self.config.lease_seconds))],
            deadline=deadline,
        )
        return int(renewed) == 1

    async def abandon(self, lease, *, deadline=None):
        removed = await asyncio.to_thread(
            self._run,
            "abandon",
            self._abandon_script,
            [lease.token],
            deadline=deadline,
        )
        return int(removed) == 1

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
        c = self.config
        await asyncio.to_thread(
            self._run,
            "release",
            self._release_script,
            [
                lease.token,
                event_id,
                self.node_id,
                self.worker_id,
                "1" if success else "0",
                "1" if retryable else "0",
                "" if status_code is None else int(status_code),
                repr(max(0.0, float(latency_seconds))),
                c.local_failure_threshold,
                repr(float(c.local_cooldown_seconds)),
                c.global_min_samples,
                repr(float(c.global_failure_ratio)),
                repr(float(c.global_cooldown_seconds)),
                repr(float(c.decrease_factor)),
                c.minimum_concurrency,
                c.maximum_concurrency,
                c.increase_step,
                repr(float(c.increase_interval_seconds)),
            ],
            deadline=deadline,
        )

    def snapshot(self):
        raw = self._run("snapshot", self._snapshot_script, [])
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        now = float(self._clock())
        leases = {k: json.loads(v) for k, v in _pairs(data.get("leases")).items()}
        meta = {k: v for k, v in _pairs(data.get("meta")).items()}
        node_pauses = {k: float(v) for k, v in _pairs(data.get("node_pauses")).items()}
        dispatch_pairs = data.get("dispatches")
        dispatches = (
            [float(dispatch_pairs[i + 1]) for i in range(0, len(dispatch_pairs) - 1, 2)]
            if isinstance(dispatch_pairs, list)
            else []
        )
        events_raw = data.get("events")
        events = sorted(
            (json.loads(x) for x in events_raw) if isinstance(events_raw, list) else [],
            key=lambda e: float(e.get("at", 0)),
        )
        counters = _pairs(data.get("health"))
        health: dict[str, Any] = {
            "transaction_successes": {},
            "transaction_failures": {},
            "retry_counts": {},
            "expired_leases": int(counters.get("expired_leases", 0)),
            "last_error": json.loads(counters["last_error"]) if "last_error" in counters else None,
        }
        for key, value in counters.items():
            group, _, name = key.partition(":")
            if name and group in health and isinstance(health[group], dict):
                health[group][name] = int(value)
        global_pause_until = float(meta.get("global_pause_until", 0))
        health.update(
            {
                "active_leases": len(leases),
                "oldest_lease_age_seconds": max(
                    (now - float(x["acquired_at"]) for x in leases.values()), default=0.0
                ),
                "dispatches_in_window": len(dispatches),
                "provider_events_in_window": len(events),
                "current_limit": int(data.get("limit") or self.config.initial_concurrency),
                "global_pause_until": global_pause_until,
                "node_pause_until": node_pauses.get(self.node_id, 0.0),
            }
        )
        return {
            "schema_version": 2,
            "backend": "redis",
            "namespace": self.root,
            "policy_hash": meta.get("policy_hash"),
            "limit": int(data.get("limit") or self.config.initial_concurrency),
            "leases": leases,
            "dispatches": dispatches,
            "events": events,
            "event_ids": [e.get("event_id") for e in events],
            "node_pauses": node_pauses,
            "global_pause_until": global_pause_until,
            "last_decrease_at": float(meta.get("last_decrease_at", 0)),
            "last_increase_at": float(meta.get("last_increase_at", 0)),
            "updated_at": float(meta.get("updated_at", 0)),
            "health": health,
        }
