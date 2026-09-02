from __future__ import annotations

import asyncio
import multiprocessing
import os
import time

import pytest

from mas_cc.llm_runtime.providers.load_control import (
    ProviderLoadControlConfig,
    SharedProviderCoordinator,
    ProviderCoordinationStateError,
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


def _contending_worker(root, active, observed_max):
    coordinator = SharedProviderCoordinator(
        root,
        _config(initial_concurrency=2, maximum_concurrency=2),
    )
    lease = asyncio.run(coordinator.acquire())
    with active.get_lock(), observed_max.get_lock():
        active.value += 1
        observed_max.value = max(observed_max.value, active.value)
    time.sleep(0.03)
    with active.get_lock():
        active.value -= 1
    asyncio.run(
        coordinator.release(
            lease,
            success=True,
            retryable=False,
            status_code=200,
            latency_seconds=0.03,
        )
    )


def _die_while_holding_owner_lock(root, ready):
    coordinator = SharedProviderCoordinator(
        root,
        _config(
            lease_seconds=1,
            heartbeat_seconds=0.2,
            lock_stale_seconds=0.1,
        ),
    )
    with coordinator._exclusive_lock():
        ready.set()
        os._exit(0)


def test_shared_workers_use_one_leased_concurrency_limit(tmp_path):
    first = SharedProviderCoordinator(
        tmp_path, _config(), worker_id="worker-a", node_id="node-a"
    )
    second = SharedProviderCoordinator(
        tmp_path, _config(), worker_id="worker-b", node_id="node-b"
    )

    lease_a = asyncio.run(first.acquire())
    lease_b = asyncio.run(second.acquire())
    state = first.snapshot()

    assert state["limit"] == 2
    assert len(state["leases"]) == 2
    asyncio.run(
        first.release(
            lease_a, success=True, retryable=False, status_code=200, latency_seconds=1
        )
    )
    asyncio.run(
        second.release(
            lease_b, success=True, retryable=False, status_code=200, latency_seconds=1
        )
    )
    assert first.snapshot()["leases"] == {}


def test_concurrent_processes_cannot_exceed_shared_limit(tmp_path):
    context = multiprocessing.get_context("fork")
    active = context.Value("i", 0)
    observed_max = context.Value("i", 0)
    workers = [
        context.Process(
            target=_contending_worker, args=(tmp_path, active, observed_max)
        )
        for _ in range(6)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)
        assert worker.exitcode == 0
    assert observed_max.value == 2
    assert SharedProviderCoordinator(tmp_path, _config()).snapshot()["leases"] == {}


def test_process_death_inside_owner_lock_is_recovered(tmp_path):
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    worker = context.Process(
        target=_die_while_holding_owner_lock,
        args=(tmp_path, ready),
    )
    worker.start()
    assert ready.wait(timeout=3)
    worker.join(timeout=3)
    assert worker.exitcode == 0

    owner = tmp_path / "state.lock.owner"
    assert owner.is_dir()
    abandoned_time = time.time() - 5
    os.utime(owner, (abandoned_time, abandoned_time))
    marker = owner / "ticket"
    os.utime(marker, (abandoned_time, abandoned_time))
    for ticket in (tmp_path / "state.lock.queue").iterdir():
        os.utime(ticket, (abandoned_time, abandoned_time))

    coordinator = SharedProviderCoordinator(
        tmp_path,
        _config(
            lease_seconds=1,
            heartbeat_seconds=0.2,
            lock_stale_seconds=0.1,
        ),
    )
    lease = asyncio.run(coordinator.acquire(deadline=time.monotonic() + 3))
    asyncio.run(
        coordinator.release(
            lease,
            success=True,
            retryable=False,
            status_code=200,
            latency_seconds=0.01,
        )
    )
    assert not owner.exists()


def test_pending_acquisition_gate_never_blocks_lease_renewal(tmp_path):
    coordinator = SharedProviderCoordinator(
        tmp_path,
        _config(
            initial_concurrency=1,
            maximum_concurrency=1,
            lease_seconds=3,
            heartbeat_seconds=1,
        ),
    )

    async def exercise():
        lease = await coordinator.acquire()
        await coordinator._acquire_gate.acquire()
        try:
            assert await asyncio.wait_for(coordinator.renew(lease), timeout=1)
            await asyncio.wait_for(
                coordinator.release(
                    lease,
                    success=True,
                    retryable=False,
                    status_code=200,
                    latency_seconds=0.01,
                ),
                timeout=1,
            )
        finally:
            coordinator._acquire_gate.release()

    asyncio.run(exercise())


def test_local_pause_and_global_breaker_are_distinct(tmp_path):
    clock = _Clock()
    config = _config()
    bad = SharedProviderCoordinator(
        tmp_path, config, worker_id="worker-bad", node_id="node-bad", clock=clock
    )
    healthy = SharedProviderCoordinator(
        tmp_path, config, worker_id="worker-good", node_id="node-good", clock=clock
    )

    for coordinator in (bad, bad):
        lease = asyncio.run(coordinator.acquire())
        asyncio.run(
            coordinator.release(
                lease,
                success=False,
                retryable=True,
                status_code=500,
                latency_seconds=0.1,
            )
        )
    assert bad._try_acquire()[0] is None
    healthy_lease, _ = healthy._try_acquire()
    assert healthy_lease is not None
    asyncio.run(
        healthy.release(
            healthy_lease,
            success=False,
            retryable=True,
            status_code=429,
            latency_seconds=0.1,
        )
    )
    state = healthy.snapshot()
    assert state["global_pause_until"] > clock()
    assert state["limit"] == 1
    assert healthy._try_acquire()[0] is None


def test_stale_leases_are_recovered_and_success_increases_limit(tmp_path):
    clock = _Clock()
    config = _config(lease_seconds=2, initial_concurrency=1)
    coordinator = SharedProviderCoordinator(tmp_path, config, clock=clock)
    asyncio.run(coordinator.acquire())
    clock.advance(3)
    assert len(coordinator.snapshot()["leases"]) == 0

    clock.advance(3)
    lease = asyncio.run(coordinator.acquire())
    asyncio.run(
        coordinator.release(
            lease, success=True, retryable=False, status_code=200, latency_seconds=0.2
        )
    )
    assert coordinator.snapshot()["limit"] == 2


def test_rolling_rpm_gate_counts_dispatches_including_released_attempts(tmp_path):
    clock = _Clock()
    coordinator = SharedProviderCoordinator(
        tmp_path, _config(target_rpm=2), node_id="node-a", clock=clock
    )
    for _ in range(2):
        lease = asyncio.run(coordinator.acquire())
        asyncio.run(
            coordinator.release(
                lease,
                success=True,
                retryable=False,
                status_code=200,
                latency_seconds=0.1,
            )
        )
    assert coordinator._try_acquire()[0] is None
    clock.advance(61)
    assert coordinator._try_acquire()[0] is not None


def test_denied_acquire_does_not_rewrite_shared_state(tmp_path):
    coordinator = SharedProviderCoordinator(
        tmp_path, _config(initial_concurrency=1, maximum_concurrency=1)
    )
    lease = asyncio.run(coordinator.acquire())
    before = coordinator._state_path.read_bytes()
    before_mtime = coordinator._state_path.stat().st_mtime_ns

    assert coordinator._try_acquire()[0] is None

    assert coordinator._state_path.read_bytes() == before
    assert coordinator._state_path.stat().st_mtime_ns == before_mtime
    asyncio.run(
        coordinator.release(
            lease, success=True, retryable=False, status_code=200, latency_seconds=0.1
        )
    )


def test_abandoned_queue_ticket_is_safely_recovered(tmp_path):
    coordinator = SharedProviderCoordinator(
        tmp_path, _config(lease_seconds=2, heartbeat_seconds=0.5)
    )
    abandoned = coordinator._lock_path / "abandoned"
    abandoned.touch()
    old = time.time() - 60
    import os

    os.utime(abandoned, (old, old))

    lease = asyncio.run(coordinator.acquire())

    assert lease.token in coordinator.snapshot()["leases"]
    assert not abandoned.exists()


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError, match="minimum <= initial <= maximum"):
        ProviderLoadControlConfig.from_mapping(
            {"minimum_concurrency": 5, "initial_concurrency": 4}
        )


def test_release_is_idempotent_and_records_one_event(tmp_path):
    coordinator = SharedProviderCoordinator(tmp_path, _config())
    lease = asyncio.run(coordinator.acquire())
    outcome = dict(success=True, retryable=False, status_code=200, latency_seconds=0.1)
    asyncio.run(coordinator.release(lease, **outcome))
    asyncio.run(coordinator.release(lease, **outcome))
    state = coordinator.snapshot()
    assert state["leases"] == {}
    assert len(state["events"]) == 1


def test_transient_replace_failure_recovers_and_is_reported(tmp_path, monkeypatch):
    coordinator = SharedProviderCoordinator(
        tmp_path,
        _config(transaction_backoff_initial_seconds=0.001),
    )
    original = __import__(
        "mas_cc.llm_runtime.providers.load_control", fromlist=["os"]
    ).os.replace
    calls = 0

    def flaky_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected replace visibility race")
        return original(source, destination)

    monkeypatch.setattr(
        "mas_cc.llm_runtime.providers.load_control.os.replace", flaky_replace
    )
    lease = asyncio.run(coordinator.acquire())
    state = coordinator.snapshot()
    assert lease.token in state["leases"]
    assert state["health"]["transaction_failures"]["acquire"] == 1
    assert state["health"]["retry_counts"]["acquire"] == 1


def test_uncertain_release_commit_does_not_duplicate_outcome(tmp_path, monkeypatch):
    coordinator = SharedProviderCoordinator(
        tmp_path, _config(transaction_backoff_initial_seconds=0.001)
    )
    lease = asyncio.run(coordinator.acquire())
    original_write = coordinator._write
    calls = 0

    def uncertain_write(state):
        nonlocal calls
        original_write(state)
        calls += 1
        if calls == 1:
            raise OSError("injected post-replace acknowledgement loss")

    monkeypatch.setattr(coordinator, "_write", uncertain_write)
    asyncio.run(
        coordinator.release(
            lease, success=False, retryable=True, status_code=500, latency_seconds=0.1
        )
    )
    assert len(coordinator.snapshot()["events"]) == 1


def test_initialized_coordinator_never_resets_when_state_is_transiently_missing(
    tmp_path,
):
    coordinator = SharedProviderCoordinator(
        tmp_path,
        _config(
            transaction_retry_attempts=2,
            transaction_backoff_initial_seconds=0.001,
            transaction_backoff_max_seconds=0.001,
        ),
    )
    lease = asyncio.run(coordinator.acquire())
    assert coordinator._initialized_path.is_file()
    coordinator._state_path.unlink()

    with pytest.raises(ProviderCoordinationStateError, match="remained invalid"):
        coordinator.snapshot()

    # Never manufacture an empty state after the coordinator has owned leases.
    assert not coordinator._state_path.exists()


def test_renewed_lease_survives_and_dead_lease_is_reaped(tmp_path):
    clock = _Clock()
    coordinator = SharedProviderCoordinator(
        tmp_path, _config(lease_seconds=3, heartbeat_seconds=1), clock=clock
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


def test_corrupt_state_is_preserved_and_fails_explicitly(tmp_path):
    (tmp_path / "state.json").write_text("{bad json", encoding="utf-8")
    coordinator = SharedProviderCoordinator(tmp_path, _config())
    with pytest.raises(ProviderCoordinationStateError, match="preserve"):
        coordinator.snapshot()
    assert (tmp_path / "state.json").read_text(encoding="utf-8") == "{bad json"


def test_timing_invariants_are_validated():
    with pytest.raises(ValueError, match="three heartbeat"):
        ProviderLoadControlConfig.from_mapping(
            {"lease_seconds": 10, "heartbeat_seconds": 4}
        )
    with pytest.raises(ValueError, match="materially shorter"):
        ProviderLoadControlConfig.from_mapping(
            {"lease_seconds": 160, "heartbeat_seconds": 10}
        )
