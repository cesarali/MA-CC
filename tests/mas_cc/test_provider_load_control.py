from __future__ import annotations

import asyncio
import pytest

from mas_cc.llm_runtime.providers.load_control import (
    ProviderLoadControlConfig,
    SharedProviderCoordinator,
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


def test_shared_workers_use_one_leased_concurrency_limit(tmp_path):
    clock = _Clock()
    config = _config()
    first = SharedProviderCoordinator(
        tmp_path, config, worker_id="worker-a", node_id="node-a", clock=clock
    )
    second = SharedProviderCoordinator(
        tmp_path, config, worker_id="worker-b", node_id="node-b", clock=clock
    )

    lease_a = asyncio.run(first.acquire())
    clock.advance(60 / config.target_rpm)
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
                lease, success=False, retryable=True,
                status_code=500, latency_seconds=0.1,
            )
        )
        clock.advance(60 / config.target_rpm)
    assert bad._try_acquire()[0] is None
    healthy_lease, _ = healthy._try_acquire()
    assert healthy_lease is not None
    asyncio.run(
        healthy.release(
            healthy_lease, success=False, retryable=True,
            status_code=429, latency_seconds=0.1,
        )
    )
    state = healthy.snapshot()
    assert state["global_pause_until"] > clock()
    assert state["limit"] == 1
    assert healthy._try_acquire()[0] is None


def test_successful_retries_cannot_hide_a_failure_burst_below_min_samples(tmp_path):
    """Recovery samples must still trip the rolling global failure ratio."""

    clock = _Clock()
    config = _config(
        initial_concurrency=4,
        global_min_samples=4,
        global_failure_ratio=0.25,
        local_failure_threshold=10,
    )
    coordinator = SharedProviderCoordinator(
        tmp_path, config, worker_id="worker-a", node_id="node-a", clock=clock
    )

    for status_code in (None, 429):
        lease = asyncio.run(coordinator.acquire())
        asyncio.run(
            coordinator.release(
                lease,
                success=False,
                retryable=True,
                status_code=status_code,
                latency_seconds=120,
            )
        )
        clock.advance(60 / config.target_rpm)
    # The burst is real but there are not enough rolling samples yet.
    assert coordinator.snapshot()["limit"] == 4

    for index in range(2):
        lease = asyncio.run(coordinator.acquire())
        asyncio.run(
            coordinator.release(
                lease,
                success=True,
                retryable=False,
                status_code=200,
                latency_seconds=1,
            )
        )
        clock.advance(60 / config.target_rpm)

    state = coordinator.snapshot()
    assert state["global_pause_until"] > clock()
    assert state["limit"] == 2


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
    for index in range(2):
        lease = asyncio.run(coordinator.acquire())
        asyncio.run(
            coordinator.release(
                lease, success=True, retryable=False,
                status_code=200, latency_seconds=0.1,
            )
        )
        if index == 0:
            clock.advance(30)
    assert coordinator._try_acquire()[0] is None
    clock.advance(31)
    assert coordinator._try_acquire()[0] is not None


def test_rpm_gate_smoothly_paces_dispatches_instead_of_bursting(tmp_path):
    clock = _Clock()
    coordinator = SharedProviderCoordinator(
        tmp_path,
        _config(target_rpm=20, initial_concurrency=4),
        node_id="node-a",
        clock=clock,
    )

    first, _ = coordinator._try_acquire()
    assert first is not None
    second, delay = coordinator._try_acquire()
    assert second is None
    assert delay == pytest.approx(3.0)

    clock.advance(2.9)
    assert coordinator._try_acquire()[0] is None
    clock.advance(0.1)
    assert coordinator._try_acquire()[0] is not None


def test_dispatch_pacing_recovers_transaction_overhead_without_a_burst(tmp_path):
    clock = _Clock()
    coordinator = SharedProviderCoordinator(
        tmp_path,
        _config(target_rpm=20, initial_concurrency=4),
        node_id="node-a",
        clock=clock,
    )

    assert coordinator._try_acquire()[0] is not None
    clock.advance(3.5)
    assert coordinator._try_acquire()[0] is not None

    # The half-second delay is recovered from the virtual schedule: the next
    # dispatch remains due at t=6 rather than drifting to t=6.5.
    clock.advance(2.4)
    assert coordinator._try_acquire()[0] is None
    clock.advance(0.1)
    assert coordinator._try_acquire()[0] is not None


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError, match="minimum <= initial <= maximum"):
        ProviderLoadControlConfig.from_mapping(
            {"minimum_concurrency": 5, "initial_concurrency": 4}
        )
