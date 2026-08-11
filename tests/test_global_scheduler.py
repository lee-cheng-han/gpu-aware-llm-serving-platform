import pytest

from apps.control_plane import GlobalScheduler, RequestDeadlineExceeded
from serving_platform.domain import (
    DeviceType,
    HealthStatus,
    RequestRecord,
    RequestState,
    WorkerState,
)
from serving_platform.registry import InMemoryWorkerRegistry
from serving_platform.routing import LeastQueueDepthPolicy, NoEligibleWorker


def request(deadline: float = 100) -> RequestRecord:
    item = RequestRecord("req", "tenant", "model", "hello", 1, 4, 0, deadline, False)
    item.transition(RequestState.VALIDATED)
    item.transition(RequestState.ADMITTED)
    return item


def worker(worker_id: str, queue_depth: int) -> WorkerState:
    return WorkerState(
        worker_id,
        DeviceType.SIMULATED_GPU,
        "simulated",
        1000,
        500,
        resident_models={"model"},
        queue_depth=queue_depth,
        health_status=HealthStatus.HEALTHY,
    )


def test_global_scheduler_assigns_and_records_explanation():
    registry = InMemoryWorkerRegistry(clock=lambda: 10)
    registry.register(worker("busy", 3))
    registry.register(worker("idle", 0))
    item = request()

    assignment = GlobalScheduler(
        registry, LeastQueueDepthPolicy(), clock=lambda: 20
    ).assign(item)

    assert assignment.worker_id == "idle"
    assert assignment.explanation["scores"] == {"busy": 3.0, "idle": 0.0}
    assert assignment.explanation["estimated_request_tokens"] == 5
    assert item.status == RequestState.ASSIGNED
    assert item.assigned_worker_id == "idle"
    assert item.attempt_count == 1


def test_global_scheduler_times_out_before_policy_selection():
    item = request(deadline=20)
    scheduler = GlobalScheduler(
        InMemoryWorkerRegistry(), LeastQueueDepthPolicy(), clock=lambda: 20
    )
    with pytest.raises(RequestDeadlineExceeded):
        scheduler.assign(item)
    assert item.status == RequestState.TIMED_OUT
    assert item.assigned_worker_id is None


def test_global_scheduler_leaves_request_admitted_when_capacity_is_unavailable():
    item = request()
    scheduler = GlobalScheduler(
        InMemoryWorkerRegistry(), LeastQueueDepthPolicy(), clock=lambda: 20
    )
    with pytest.raises(NoEligibleWorker):
        scheduler.assign(item)
    assert item.status == RequestState.ADMITTED
    assert item.attempt_count == 0
