import pytest

from apps.control_plane import (
    ControlPlane,
    GlobalScheduler,
    WorkerDirectory,
    WorkerDispatchError,
)
from apps.worker import ManagedWorker
from runtime.simulated_gpu import SimulatedGpuConfig, SimulatedGpuRuntime
from serving_platform.domain import (
    ModelDefinition,
    RequestRecord,
    RequestState,
    RuntimeType,
    TenantLimits,
)
from serving_platform.registry import InMemoryModelRegistry, InMemoryWorkerRegistry
from serving_platform.request_state import InMemoryRequestStateStore
from serving_platform.routing import LeastQueueDepthPolicy, ModelResidencyAwarePolicy
from serving_platform.scheduling import WeightedFairRequestQueue


def model() -> ModelDefinition:
    return ModelDefinition(
        "model", "test", RuntimeType.SIMULATED, 100, ("float16",), "float16",
        1024, 2048, True, True, 10, 60,
    )


def request(request_id: str) -> RequestRecord:
    item = RequestRecord(
        request_id, "tenant", "model", "hello", 1, 2, 0, float("inf"), False
    )
    item.transition(RequestState.VALIDATED)
    item.transition(RequestState.ADMITTED)
    return item


def managed(worker_id: str, registry: InMemoryWorkerRegistry) -> ManagedWorker:
    definition = model()
    runtime = SimulatedGpuRuntime(
        [definition], SimulatedGpuConfig(total_memory_bytes=1000), sleeper=lambda _: None
    )
    worker = ManagedWorker(worker_id, runtime, registry)
    worker.register()
    worker.load_model(definition)
    worker.heartbeat()
    return worker


def test_control_plane_routes_into_selected_workers_local_queue():
    registry = InMemoryWorkerRegistry()
    busy = managed("busy", registry)
    idle = managed("idle", registry)
    existing = request("existing")
    existing.transition(RequestState.ASSIGNED)
    busy.enqueue_request(existing)
    busy.heartbeat()
    directory = WorkerDirectory()
    directory.add(busy)
    directory.add(idle)
    control_plane = ControlPlane(
        GlobalScheduler(registry, LeastQueueDepthPolicy()), directory
    )
    item = request("new")

    assignment = control_plane.dispatch(item)

    assert assignment.worker_id == "idle"
    assert item.status == RequestState.QUEUED
    assert idle.execute_batch()[0].request_id == "new"
    assert item.status == RequestState.COMPLETED


def test_control_plane_returns_request_to_global_queue_if_worker_disappears():
    registry = InMemoryWorkerRegistry()
    worker = managed("worker", registry)
    control_plane = ControlPlane(
        GlobalScheduler(registry, LeastQueueDepthPolicy()), WorkerDirectory()
    )
    item = request("new")

    with pytest.raises(WorkerDispatchError, match="no longer reachable"):
        control_plane.dispatch(item)
    assert item.status == RequestState.QUEUED
    assert item.assigned_worker_id is None
    assert item.attempt_count == 1
    worker.shutdown()


def test_control_plane_loads_registered_model_after_cold_placement():
    definition = model()
    models = InMemoryModelRegistry([definition])
    registry = InMemoryWorkerRegistry()
    runtime = SimulatedGpuRuntime(
        [definition], SimulatedGpuConfig(total_memory_bytes=1000), sleeper=lambda _: None
    )
    worker = ManagedWorker("cold", runtime, registry)
    worker.register()
    directory = WorkerDirectory()
    directory.add(worker)
    control_plane = ControlPlane(
        GlobalScheduler(registry, ModelResidencyAwarePolicy(models)), directory, models
    )
    item = request("cold-request")

    assignment = control_plane.dispatch(item)

    assert assignment.worker_id == "cold"
    assert runtime.is_model_loaded("model")
    assert item.status == RequestState.QUEUED
    assert worker.model_cache_metrics().cold_starts == 1
    assert registry.get("cold").resident_models == {"model"}


def test_control_plane_cancels_request_waiting_in_global_fair_queue():
    requests = InMemoryRequestStateStore()
    fair_queue = WeightedFairRequestQueue([TenantLimits("tenant", 1, 2, 100)])
    item = request("queued")
    requests.create(item)
    fair_queue.enqueue(item)
    requests.save(item)
    control_plane = ControlPlane(
        GlobalScheduler(InMemoryWorkerRegistry(), LeastQueueDepthPolicy()),
        WorkerDirectory(),
        requests=requests,
        global_queue=fair_queue,
    )
    assert control_plane.cancel(item.request_id)
    assert requests.get(item.request_id).status == RequestState.CANCELLED
    assert fair_queue.depth() == 0
