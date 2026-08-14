import asyncio

from apps.control_plane import (
    ControlPlane,
    GlobalScheduler,
    ReliabilitySupervisor,
    WorkerDirectory,
)
from serving_platform.domain import (
    DeviceType,
    HealthStatus,
    RequestRecord,
    RequestState,
    TenantLimits,
    WorkerState,
)
from serving_platform.registry import InMemoryWorkerRegistry
from serving_platform.request_state import InMemoryRequestStateStore
from serving_platform.routing import LeastQueueDepthPolicy
from serving_platform.scheduling import WeightedFairRequestQueue


def worker() -> WorkerState:
    return WorkerState(
        "failed-worker", DeviceType.SIMULATED_GPU, "sim", 1000, 900,
        resident_models={"model"}, health_status=HealthStatus.HEALTHY,
    )


def request(request_id: str, state: RequestState, attempts: int = 1) -> RequestRecord:
    item = RequestRecord(
        request_id, "tenant", "model", "hello", 1, 1, 0, 100, False
    )
    item.transition(RequestState.VALIDATED, 1)
    item.transition(RequestState.ADMITTED, 2)
    item.transition(RequestState.ASSIGNED, 3)
    item.assigned_worker_id = "failed-worker"
    item.attempt_count = attempts
    if state == RequestState.QUEUED:
        item.transition(RequestState.QUEUED, 4)
    elif state == RequestState.RUNNING:
        item.transition(RequestState.RUNNING, 4)
    return item


def control_plane(now, registry, store):
    fair_queue = WeightedFairRequestQueue(
        [TenantLimits("tenant", 2, 10, 100)], clock=lambda: now[0]
    )
    plane = ControlPlane(
        GlobalScheduler(registry, LeastQueueDepthPolicy(), clock=lambda: now[0]),
        WorkerDirectory(),
        requests=store,
        global_queue=fair_queue,
    )
    return plane, fair_queue


def test_recovery_requeues_only_work_that_never_started():
    now = [10.0]
    registry = InMemoryWorkerRegistry(clock=lambda: now[0])
    store = InMemoryRequestStateStore()
    queued = request("queued", RequestState.QUEUED)
    running = request("running", RequestState.RUNNING)
    store.create(queued)
    store.create(running)
    plane, fair_queue = control_plane(now, registry, store)

    result = plane.recover_failed_worker("failed-worker", max_attempts=3)

    assert result == {"requeued": 1, "failed": 1, "untouched": 0}
    recovered = store.get("queued")
    assert recovered.status == RequestState.QUEUED
    assert recovered.assigned_worker_id is None
    assert recovered.retry_reasons == ["assigned_worker_heartbeat_expired"]
    assert fair_queue.depth() == 1
    failed = store.get("running")
    assert failed.status == RequestState.FAILED
    assert failed.retry_reasons == ["worker_lost_after_execution_started"]


def test_recovery_enforces_retry_budget():
    now = [10.0]
    registry = InMemoryWorkerRegistry(clock=lambda: now[0])
    store = InMemoryRequestStateStore()
    exhausted = request("exhausted", RequestState.QUEUED, attempts=3)
    store.create(exhausted)
    plane, fair_queue = control_plane(now, registry, store)
    plane.recover_failed_worker("failed-worker", max_attempts=3)
    restored = store.get("exhausted")
    assert restored.status == RequestState.FAILED
    assert restored.retry_reasons == ["retry_budget_exhausted"]
    assert fair_queue.depth() == 0


def test_recovery_never_schedules_a_record_without_its_prompt_payload():
    now = [10.0]
    registry = InMemoryWorkerRegistry(clock=lambda: now[0])
    store = InMemoryRequestStateStore()
    metadata_only = request("metadata-only", RequestState.QUEUED)
    metadata_only.prompt = "[redacted]"
    metadata_only.payload_available = False
    store.create(metadata_only)
    plane, fair_queue = control_plane(now, registry, store)

    plane.recover_failed_worker("failed-worker", max_attempts=3)

    restored = store.get("metadata-only")
    assert restored.status == RequestState.FAILED
    assert restored.retry_reasons == ["request_payload_unavailable_after_restart"]
    assert fair_queue.depth() == 0


async def test_supervisor_automatically_expires_worker_and_recovers_request():
    now = [0.0]
    registry = InMemoryWorkerRegistry(heartbeat_timeout_seconds=1, clock=lambda: now[0])
    registry.register(worker())
    store = InMemoryRequestStateStore()
    store.create(request("queued", RequestState.QUEUED))
    plane, fair_queue = control_plane(now, registry, store)
    supervisor = ReliabilitySupervisor(registry, plane, check_interval_seconds=0.001)
    task = asyncio.create_task(supervisor.run())
    now[0] = 2
    deadline = asyncio.get_running_loop().time() + 1
    while fair_queue.depth() == 0:
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.001)
    supervisor.stop()
    await asyncio.wait_for(task, 1)
    assert registry.get("failed-worker").health_status == HealthStatus.UNHEALTHY
    assert supervisor.failure is None
