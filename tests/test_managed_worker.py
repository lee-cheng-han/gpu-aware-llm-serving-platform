from apps.worker import ManagedWorker
from runtime.simulated_gpu import SimulatedGpuConfig, SimulatedGpuRuntime
from serving_platform.domain import (
    HealthStatus,
    ModelDefinition,
    RequestRecord,
    RequestState,
    RuntimeType,
)
from serving_platform.registry import InMemoryWorkerRegistry


def model() -> ModelDefinition:
    return ModelDefinition(
        "sim-model", "test", RuntimeType.SIMULATED, 100, ("float16",), "float16",
        1024, 2048, True, True, 10, 60,
    )


def request(request_id: str = "request-1") -> RequestRecord:
    item = RequestRecord(request_id, "tenant", "sim-model", "hello", 1, 2, 0, 100, False)
    item.transition(RequestState.VALIDATED)
    item.transition(RequestState.ADMITTED)
    item.transition(RequestState.ASSIGNED)
    return item


def worker() -> tuple[ManagedWorker, InMemoryWorkerRegistry]:
    definition = model()
    registry = InMemoryWorkerRegistry()
    runtime = SimulatedGpuRuntime(
        [definition], SimulatedGpuConfig(total_memory_bytes=1000), sleeper=lambda _: None
    )
    return ManagedWorker("sim-worker", runtime, registry, max_queue_depth=1), registry


def test_managed_worker_registration_loading_queueing_and_cancellation():
    managed, registry = worker()
    managed.register()
    managed.load_model(model())
    managed.warmup_model("sim-model")
    item = request()
    managed.enqueue_request(item)
    snapshot = managed.heartbeat()

    assert snapshot.queue_depth == 1
    assert snapshot.resident_models == {"sim-model"}
    assert registry.get("sim-worker").queue_depth == 1
    assert managed.cancel_request(item.request_id)
    assert item.status == RequestState.CANCELLED


def test_draining_worker_rejects_work_and_shutdown_clears_registration():
    managed, registry = worker()
    managed.register()
    managed.load_model(model())
    managed.drain()

    try:
        managed.enqueue_request(request())
    except RuntimeError as exc:
        assert "not accepting" in str(exc)
    else:
        raise AssertionError("draining worker accepted a request")

    managed.shutdown()
    assert managed.health() == HealthStatus.STOPPED
    assert registry.get("sim-worker") is None
