from serving_platform.domain import DeviceType, HealthStatus, WorkerState
from serving_platform.registry import InMemoryWorkerRegistry


def state(worker_id: str = "worker-a") -> WorkerState:
    return WorkerState(worker_id, DeviceType.CPU, "cpu", None, None)


def test_registry_copies_snapshots_and_expires_stale_workers():
    now = [10.0]
    registry = InMemoryWorkerRegistry(heartbeat_timeout_seconds=5, clock=lambda: now[0])
    worker = state()
    registry.register(worker)

    worker.draining = True
    assert registry.get("worker-a").draining is False
    now[0] = 16.0
    assert registry.expire_stale() == ("worker-a",)
    assert registry.get("worker-a").health_status == HealthStatus.UNHEALTHY


def test_expired_worker_must_reregister_before_heartbeating():
    now = [0.0]
    registry = InMemoryWorkerRegistry(1, clock=lambda: now[0])
    worker = state()
    registry.register(worker)
    now[0] = 2.0
    registry.expire_stale()

    try:
        registry.heartbeat(worker)
    except RuntimeError as exc:
        assert "re-register" in str(exc)
    else:
        raise AssertionError("stale worker heartbeat unexpectedly succeeded")

    registry.register(worker)
    assert registry.get(worker.worker_id).health_status == HealthStatus.HEALTHY
