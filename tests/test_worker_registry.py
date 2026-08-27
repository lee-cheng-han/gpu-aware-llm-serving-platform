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


def test_delayed_heartbeat_enters_suspect_state_then_recovers():
    now = [0.0]
    registry = InMemoryWorkerRegistry(
        heartbeat_timeout_seconds=10,
        suspect_after_seconds=4,
        clock=lambda: now[0],
    )
    worker = state()
    registry.register(worker)
    now[0] = 5
    assert registry.expire_stale() == ()
    suspect = registry.get(worker.worker_id)
    assert suspect.health_status == HealthStatus.SUSPECT
    assert suspect.last_error_code == "heartbeat_delayed"
    assert suspect.consecutive_failures == 1

    registry.heartbeat(worker)
    recovered = registry.get(worker.worker_id)
    assert recovered.health_status == HealthStatus.HEALTHY
    assert recovered.heartbeat_sequence == 1
    assert recovered.consecutive_failures == 0
    assert recovered.last_error_code is None


def test_explicit_failure_marks_worker_suspect_and_routing_excludes_it():
    registry = InMemoryWorkerRegistry()
    registry.register(state())
    registry.mark_suspect("worker-a", "device_error")
    snapshot = registry.get("worker-a")
    assert snapshot.health_status == HealthStatus.SUSPECT
    assert snapshot.last_error_code == "device_error"
