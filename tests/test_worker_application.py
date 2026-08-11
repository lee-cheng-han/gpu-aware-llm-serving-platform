import asyncio

import pytest

from apps.worker import ManagedWorker, WorkerApplication
from runtime.simulated_gpu import SimulatedGpuConfig, SimulatedGpuRuntime
from serving_platform.domain import ModelDefinition, RequestRecord, RequestState, RuntimeType
from serving_platform.registry import InMemoryWorkerRegistry


def model() -> ModelDefinition:
    return ModelDefinition(
        "sim-model", "test", RuntimeType.SIMULATED, 100, ("float16",), "float16",
        1024, 2048, True, True, 10, 60,
    )


def request() -> RequestRecord:
    item = RequestRecord(
        "request", "tenant", "sim-model", "hello", 1, 2, 0, float("inf"), False
    )
    item.transition(RequestState.VALIDATED)
    item.transition(RequestState.ADMITTED)
    item.transition(RequestState.ASSIGNED)
    return item


@pytest.mark.asyncio
async def test_worker_application_supervises_execution_and_shutdown():
    definition = model()
    registry = InMemoryWorkerRegistry()
    runtime = SimulatedGpuRuntime(
        [definition], SimulatedGpuConfig(total_memory_bytes=1000), sleeper=lambda _: None
    )
    worker = ManagedWorker("worker", runtime, registry)
    completed = asyncio.Event()
    application: WorkerApplication

    def receive(results):
        assert results[0].request_id == "request"
        completed.set()
        application.stop()

    application = WorkerApplication(
        worker,
        [definition],
        heartbeat_interval_seconds=0.01,
        idle_poll_seconds=0.001,
        result_sink=receive,
    )
    task = asyncio.create_task(application.run())
    while registry.get("worker") is None:
        await asyncio.sleep(0)
    item = request()
    worker.enqueue_request(item)

    await asyncio.wait_for(completed.wait(), 1)
    await asyncio.wait_for(task, 1)
    assert item.status == RequestState.COMPLETED
    assert registry.get("worker") is None


@pytest.mark.asyncio
async def test_worker_application_propagates_execution_failure_and_unregisters():
    definition = model()
    registry = InMemoryWorkerRegistry()
    runtime = SimulatedGpuRuntime(
        [definition],
        SimulatedGpuConfig(total_memory_bytes=1000, failure_every_n_calls=1),
        sleeper=lambda _: None,
    )
    worker = ManagedWorker("worker", runtime, registry)
    application = WorkerApplication(worker, [definition], idle_poll_seconds=0.001)
    task = asyncio.create_task(application.run())
    while registry.get("worker") is None:
        await asyncio.sleep(0)
    item = request()
    worker.enqueue_request(item)

    with pytest.raises(ExceptionGroup) as caught:
        await asyncio.wait_for(task, 1)
    assert "controlled simulated GPU failure" in str(caught.value.exceptions[0])
    assert item.status == RequestState.FAILED
    assert registry.get("worker") is None
