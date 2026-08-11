import pytest

from apps.control_plane import (
    ControlPlane,
    GlobalScheduler,
    WorkerDirectory,
    WorkerDispatchError,
)
from apps.worker import ManagedWorker
from runtime.simulated_gpu import SimulatedGpuConfig, SimulatedGpuRuntime
from serving_platform.domain import ModelDefinition, RequestRecord, RequestState, RuntimeType
from serving_platform.registry import InMemoryWorkerRegistry
from serving_platform.routing import LeastQueueDepthPolicy


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
