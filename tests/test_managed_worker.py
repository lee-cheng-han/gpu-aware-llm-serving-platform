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


def request(
    request_id: str = "request-1",
    *,
    max_new_tokens: int = 2,
    deadline: float = float("inf"),
    temperature: float = 0,
) -> RequestRecord:
    item = RequestRecord(
        request_id,
        "tenant",
        "sim-model",
        "hello",
        1,
        max_new_tokens,
        0,
        deadline,
        False,
        temperature,
    )
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
    return ManagedWorker("sim-worker", runtime, registry, max_queue_depth=8), registry


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


def test_worker_executes_compatible_local_batches_and_defers_incompatible_work():
    managed, _ = worker()
    managed.register()
    managed.load_model(model())
    first = request("first")
    second = request("second")
    incompatible = request("later", max_new_tokens=3)
    for item in (first, second, incompatible):
        managed.enqueue_request(item)

    results = managed.execute_batch()
    snapshot = managed.heartbeat()

    assert [result.request_id for result in results] == ["first", "second"]
    assert first.status == second.status == RequestState.COMPLETED
    assert incompatible.status == RequestState.QUEUED
    assert snapshot.queue_depth == 1
    assert snapshot.active_batch_count == 0
    assert snapshot.recent_tokens_per_second > 0
    assert managed.execute_batch()[0].request_id == "later"


def test_worker_removes_expired_requests_before_execution():
    definition = model()
    registry = InMemoryWorkerRegistry(clock=lambda: 20)
    runtime = SimulatedGpuRuntime(
        [definition], SimulatedGpuConfig(total_memory_bytes=1000), sleeper=lambda _: None
    )
    managed = ManagedWorker("sim-worker", runtime, registry, clock=lambda: 20)
    managed.register()
    managed.load_model(definition)
    expired = request(deadline=20)
    managed.enqueue_request(expired)

    assert managed.execute_batch() == ()
    assert expired.status == RequestState.TIMED_OUT


def test_worker_marks_entire_batch_failed_when_runtime_fails():
    definition = model()
    registry = InMemoryWorkerRegistry()
    runtime = SimulatedGpuRuntime(
        [definition],
        SimulatedGpuConfig(total_memory_bytes=1000, failure_every_n_calls=1),
        sleeper=lambda _: None,
    )
    managed = ManagedWorker("sim-worker", runtime, registry)
    managed.register()
    managed.load_model(definition)
    first, second = request("first"), request("second")
    managed.enqueue_request(first)
    managed.enqueue_request(second)

    try:
        managed.execute_batch()
    except RuntimeError as exc:
        assert "controlled simulated" in str(exc)
    else:
        raise AssertionError("controlled runtime failure did not propagate")
    assert first.status == second.status == RequestState.FAILED
    assert managed.heartbeat().active_batch_count == 0


def test_worker_skips_externally_cancelled_queued_request():
    managed, _ = worker()
    managed.register()
    managed.load_model(model())
    item = request()
    managed.enqueue_request(item)
    item.transition(RequestState.CANCELLED)

    assert managed.execute_batch() == ()
    assert item.status == RequestState.CANCELLED
