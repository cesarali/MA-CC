"""Behaviour tests for the Redis-backed provider coordinator.

They mirror tests/mas_cc/test_provider_load_control.py for every rule that does
not depend on the shared-filesystem mechanics (lock files, torn JSON). The Redis
server is fakeredis with Lua support, so the real Lua scripts run.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

fakeredis = pytest.importorskip("fakeredis")
pytest.importorskip("lupa")  # fakeredis needs lupa to run Lua scripts

from mas_cc.llm_runtime.providers.load_control import (  # noqa: E402
    LOAD_CONTROL_CONFIG_ENV,
    LOAD_CONTROL_DIR_ENV,
    LOAD_CONTROL_REDIS_URL_ENV,
    ProviderAdmissionTimeout,
    ProviderCoordinationUnavailable,
    ProviderCoordinationStateError,
    ProviderLoadControlConfig,
    coordinator_from_environment,
)
from mas_cc.llm_runtime.providers.redis_load_control import (  # noqa: E402
    RedisProviderCoordinator,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 1_000.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _config(**changes):
    values = {
        "mode": "redis_adaptive",
        "initial_concurrency": 2,
        "minimum_concurrency": 1,
        "maximum_concurrency": 4,
        "target_rpm": 20,
        "polling_seconds": 0.01,
        "local_failure_threshold": 2,
        "global_min_samples": 3,
        "global_failure_ratio": 0.5,
        "local_cooldown_seconds": 10,
        "global_cooldown_seconds": 20,
        "increase_interval_seconds": 5,
    }
    values.update(changes)
    return ProviderLoadControlConfig.from_mapping(values)


@pytest.fixture
def server():
    return fakeredis.FakeServer()


def _coordinator(server, namespace="study/job-1", config=None, **kwargs):
    client = fakeredis.FakeStrictRedis(server=server)
    return RedisProviderCoordinator(namespace, config or _config(), client=client, **kwargs)


def _ok(coordinator, lease, **outcome):
    values = dict(success=True, retryable=False, status_code=200, latency_seconds=0.1)
    values.update(outcome)
    asyncio.run(coordinator.release(lease, **values))


def test_redis_mode_is_accepted_and_unknown_modes_rejected():
    assert _config().mode == "redis_adaptive"
    with pytest.raises(ValueError, match="redis_adaptive"):
        ProviderLoadControlConfig.from_mapping({"mode": "local"})


def test_shared_workers_use_one_leased_concurrency_limit(server):
    first = _coordinator(server, worker_id="worker-a", node_id="node-a")
    second = _coordinator(server, worker_id="worker-b", node_id="node-b")

    lease_a = asyncio.run(first.acquire())
    lease_b = asyncio.run(second.acquire())
    state = first.snapshot()

    assert state["limit"] == 2
    assert len(state["leases"]) == 2
    assert first._try_acquire()[0] is None
    _ok(first, lease_a)
    _ok(second, lease_b)
    assert first.snapshot()["leases"] == {}


def test_concurrent_workers_cannot_exceed_shared_limit(server):
    active = 0
    observed_max = 0
    guard = threading.Lock()
    errors: list[BaseException] = []

    def worker():
        nonlocal active, observed_max
        try:
            coordinator = _coordinator(
                server, config=_config(initial_concurrency=2, maximum_concurrency=2)
            )
            lease = asyncio.run(coordinator.acquire())
            with guard:
                active += 1
                observed_max = max(observed_max, active)
            time.sleep(0.03)
            with guard:
                active -= 1
            _ok(coordinator, lease, latency_seconds=0.03)
        except BaseException as e:  # surface thread failures in the test
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not errors
    assert observed_max == 2
    matching = _config(initial_concurrency=2, maximum_concurrency=2)
    assert _coordinator(server, config=matching).snapshot()["leases"] == {}


def test_pending_acquisition_gate_never_blocks_lease_renewal(server):
    coordinator = _coordinator(
        server,
        config=_config(
            initial_concurrency=1, maximum_concurrency=1, lease_seconds=3, heartbeat_seconds=1
        ),
    )

    async def exercise():
        lease = await coordinator.acquire()
        await coordinator._acquire_gate.acquire()
        try:
            assert await asyncio.wait_for(coordinator.renew(lease), timeout=1)
            await asyncio.wait_for(
                coordinator.release(
                    lease, success=True, retryable=False, status_code=200, latency_seconds=0.01
                ),
                timeout=1,
            )
        finally:
            coordinator._acquire_gate.release()

    asyncio.run(exercise())


def test_local_pause_and_global_breaker_are_distinct(server):
    clock = _Clock()
    bad = _coordinator(server, worker_id="worker-bad", node_id="node-bad", clock=clock)
    healthy = _coordinator(server, worker_id="worker-good", node_id="node-good", clock=clock)

    for _ in range(2):
        lease = asyncio.run(bad.acquire())
        _ok(bad, lease, success=False, retryable=True, status_code=500)
    assert bad._try_acquire()[0] is None
    healthy_lease, _ = healthy._try_acquire()
    assert healthy_lease is not None
    _ok(healthy, healthy_lease, success=False, retryable=True, status_code=429)
    state = healthy.snapshot()
    assert state["global_pause_until"] > clock()
    assert state["limit"] == 1
    assert healthy._try_acquire()[0] is None


def test_stale_leases_are_recovered_and_success_increases_limit(server):
    clock = _Clock()
    coordinator = _coordinator(
        server, config=_config(lease_seconds=2, initial_concurrency=1), clock=clock
    )
    asyncio.run(coordinator.acquire())
    clock.advance(3)
    assert len(coordinator.snapshot()["leases"]) == 0

    clock.advance(3)
    lease = asyncio.run(coordinator.acquire())
    _ok(coordinator, lease, latency_seconds=0.2)
    assert coordinator.snapshot()["limit"] == 2


def test_rolling_rpm_gate_counts_dispatches_including_released_attempts(server):
    clock = _Clock()
    coordinator = _coordinator(
        server, config=_config(target_rpm=2), node_id="node-a", clock=clock
    )
    for _ in range(2):
        lease = asyncio.run(coordinator.acquire())
        _ok(coordinator, lease)
    lease, delay = coordinator._try_acquire()
    assert lease is None
    assert delay > 0  # fractional delays survive the Redis reply
    clock.advance(61)
    assert coordinator._try_acquire()[0] is not None


def test_denied_acquire_does_not_take_capacity_or_count_a_dispatch(server):
    coordinator = _coordinator(
        server, config=_config(initial_concurrency=1, maximum_concurrency=1)
    )
    lease = asyncio.run(coordinator.acquire())
    before = coordinator.snapshot()

    assert coordinator._try_acquire()[0] is None

    after = coordinator.snapshot()
    assert after["leases"] == before["leases"]
    assert after["dispatches"] == before["dispatches"]
    _ok(coordinator, lease)


def test_release_is_idempotent_and_records_one_event(server):
    coordinator = _coordinator(server)
    lease = asyncio.run(coordinator.acquire())
    _ok(coordinator, lease)
    _ok(coordinator, lease)
    state = coordinator.snapshot()
    assert state["leases"] == {}
    assert len(state["events"]) == 1


def test_renewed_lease_survives_and_dead_lease_is_reaped(server):
    clock = _Clock()
    coordinator = _coordinator(
        server, config=_config(lease_seconds=3, heartbeat_seconds=1), clock=clock
    )
    lease = asyncio.run(coordinator.acquire())
    clock.advance(2)
    assert asyncio.run(coordinator.renew(lease))
    clock.advance(2)
    assert lease.token in coordinator.snapshot()["leases"]
    clock.advance(2)
    state = coordinator.snapshot()
    assert lease.token not in state["leases"]
    assert state["health"]["expired_leases"] == 1
    assert asyncio.run(coordinator.renew(lease)) is False


def test_studies_with_different_namespaces_do_not_share_capacity(server):
    config = _config(initial_concurrency=1, maximum_concurrency=1)
    study_a = _coordinator(server, namespace="study-a/job-1", config=config)
    study_b = _coordinator(server, namespace="study-b/job-1", config=config)
    lease_a = asyncio.run(study_a.acquire())
    assert study_a._try_acquire()[0] is None
    lease_b, _ = study_b._try_acquire()
    assert lease_b is not None
    _ok(study_a, lease_a)
    _ok(study_b, lease_b)


def test_unreachable_redis_raises_coordination_unavailable():
    import redis

    client = redis.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.05)
    coordinator = RedisProviderCoordinator(
        "study/job-1",
        _config(
            transaction_retry_attempts=2,
            transaction_backoff_initial_seconds=0.001,
            transaction_backoff_max_seconds=0.001,
        ),
        client=client,
    )
    with pytest.raises(ProviderCoordinationUnavailable, match="backend=redis"):
        coordinator._try_acquire()


def test_transient_failure_is_retried_and_reported(server, monkeypatch):
    import redis

    coordinator = _coordinator(server, config=_config(transaction_backoff_initial_seconds=0.001))
    original = coordinator._acquire_script
    calls = 0

    def flaky(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise redis.exceptions.ConnectionError("injected connection reset")
        return original(*args, **kwargs)

    monkeypatch.setattr(coordinator, "_acquire_script", flaky)
    lease = asyncio.run(coordinator.acquire())
    state = coordinator.snapshot()
    assert lease.token in state["leases"]
    assert state["health"]["transaction_failures"]["acquire"] == 1
    assert state["health"]["retry_counts"]["acquire"] == 1
    assert state["health"]["last_error"]["type"] == "ConnectionError"


def test_environment_builds_redis_coordinator_and_requires_url(monkeypatch):
    settings = '{"mode": "redis_adaptive"}'
    with pytest.raises(ValueError, match=LOAD_CONTROL_REDIS_URL_ENV):
        coordinator_from_environment(
            None,
            environment={LOAD_CONTROL_CONFIG_ENV: settings, LOAD_CONTROL_DIR_ENV: "/tmp/study"},
        )
    built = coordinator_from_environment(
        None,
        environment={
            LOAD_CONTROL_CONFIG_ENV: settings,
            LOAD_CONTROL_DIR_ENV: "/tmp/study",
            LOAD_CONTROL_REDIS_URL_ENV: "redis://127.0.0.1:1/0",
        },
    )
    assert isinstance(built, RedisProviderCoordinator)
    assert built.root == "/tmp/study"


def test_admission_timeout_is_distinct_and_records_no_provider_event(server):
    coordinator = _coordinator(
        server,
        config=_config(initial_concurrency=1, maximum_concurrency=1),
    )

    async def exercise():
        lease = await coordinator.acquire()
        with pytest.raises(ProviderAdmissionTimeout):
            await coordinator.acquire(
                deadline=time.monotonic() + 0.03,
                admission=True,
            )
        state = coordinator.snapshot()
        assert state["events"] == []
        await coordinator.release(
            lease,
            success=True,
            retryable=False,
            status_code=200,
            latency_seconds=0.01,
        )

    asyncio.run(exercise())


def test_policy_mismatch_and_out_of_range_limit_fail_closed(server):
    original = _coordinator(server)
    lease = asyncio.run(original.acquire())
    mismatched = _coordinator(server, config=_config(maximum_concurrency=5))

    with pytest.raises(ProviderCoordinationStateError, match="policy mismatch"):
        mismatched.snapshot()

    original._client.set(original._keys[0], 100)
    with pytest.raises(ProviderCoordinationStateError, match="outside its policy"):
        original.snapshot()

    # Preserve the invalid state and live lease for diagnosis.
    assert original._client.hget(original._keys[1], lease.token) is not None


def test_cancelled_late_grant_is_abandoned(server, monkeypatch):
    coordinator = _coordinator(
        server,
        config=_config(initial_concurrency=1, maximum_concurrency=1),
    )
    original = coordinator._try_acquire

    def delayed(*args, **kwargs):
        time.sleep(0.05)
        return original(*args, **kwargs)

    monkeypatch.setattr(coordinator, "_try_acquire", delayed)

    async def exercise():
        pending = asyncio.create_task(
            coordinator.acquire(
                deadline=time.monotonic() + 1,
                admission=True,
            )
        )
        await asyncio.sleep(0.01)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(exercise())
    state = coordinator.snapshot()
    assert state["leases"] == {}
    assert state["events"] == []
