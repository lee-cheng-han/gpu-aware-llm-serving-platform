import asyncio
import time
from collections import deque
from dataclasses import replace

from conftest import FakeWorker

from apps.gateway.metrics import Metrics
from scheduler.dynamic_batch import collect_batch, process_batch
from scheduler.queue import RequestQueue
from scheduler.request import InferenceRequest, RequestStatus
from scheduler.scheduler_loop import run_scheduler


def item(prompt="x"):
    request = InferenceRequest(prompt, 1, .7, 1, "dynamic_batch")
    request.queued_at = time.monotonic()
    request.future = asyncio.get_running_loop().create_future()
    return request


async def test_batch_respects_max_size(settings):
    settings = replace(settings, max_batch_size=2)
    queue = asyncio.Queue()
    await queue.put(item("b"))
    await queue.put(item("c"))
    batch = await collect_batch(queue, item("a"), settings, Metrics())
    assert [r.prompt for r in batch] == ["a", "b"]


async def test_batch_waits_for_window(settings):
    settings = replace(settings, max_wait_ms=15)
    started = time.monotonic()
    batch = await collect_batch(asyncio.Queue(), item(), settings, Metrics())
    assert len(batch) == 1
    assert time.monotonic() - started >= .010


async def test_timeout_skipped_and_batch_recorded(settings):
    metrics, worker, queue = Metrics(), FakeWorker(), asyncio.Queue()
    expired = item("old")
    expired.queued_at = time.monotonic() - 2
    await queue.put(expired)
    batch = await collect_batch(queue, item("good"), settings, metrics)
    await process_batch(batch, worker, settings, metrics)
    assert expired.status == RequestStatus.TIMEOUT
    assert batch[0].batch_size == 1
    assert metrics.snapshot()["avg_batch_size"] == 1


async def test_collection_defers_incompatible_and_keeps_scanning(settings):
    queue, deferred = asyncio.Queue(), deque()
    incompatible = item("different")
    incompatible.temperature = 0.5
    compatible = item("compatible")
    await queue.put(incompatible)
    await queue.put(compatible)
    first = item("first")
    batch = await collect_batch(
        queue, first, settings, Metrics(), deferred=deferred
    )
    assert [request.prompt for request in batch] == ["first", "compatible"]
    assert [request.prompt for request in deferred] == ["different"]
    queue.task_done()  # compatible remains owned by the collected batch


async def test_collection_reuses_compatible_deferred_requests(settings):
    queue, deferred = asyncio.Queue(), deque()
    compatible = item("compatible")
    incompatible = item("incompatible")
    incompatible.temperature = 0.1
    deferred.extend([incompatible, compatible])
    batch = await collect_batch(
        queue, item("first"), settings, Metrics(), deferred=deferred
    )
    assert [request.prompt for request in batch] == ["first", "compatible"]
    assert [request.prompt for request in deferred] == ["incompatible"]
    assert batch.queued_items == 0


async def test_collection_respects_total_token_budget(settings):
    settings = replace(settings, max_total_batch_tokens=5)
    queue, deferred = asyncio.Queue(), deque()
    candidate = item("too-large")
    candidate.estimated_tokens = 3  # cost 4; first costs 2
    await queue.put(candidate)
    batch = await collect_batch(
        queue, item("first"), settings, Metrics(), deferred=deferred
    )
    assert [request.prompt for request in batch] == ["first"]
    assert list(deferred) == [candidate]
    assert batch.total_estimated_tokens == 2


async def test_scheduler_executes_one_real_micro_batch(settings):
    settings = replace(
        settings,
        scheduler_policy="dynamic_batch",
        max_batch_size=3,
        max_wait_ms=5,
    )
    queue, worker, metrics = RequestQueue(3), FakeWorker(), Metrics()
    requests = [item(prompt) for prompt in ("one", "two", "three")]
    for request in requests:
        request.future = None  # submit creates the scheduler-owned result handle
        await queue.submit(request)

    scheduler = asyncio.create_task(run_scheduler(queue, worker, settings, metrics))
    results = await asyncio.gather(*(request.future for request in requests))
    await asyncio.wait_for(queue.queue.join(), timeout=1)
    queue.close(metrics)
    await asyncio.wait_for(scheduler, timeout=1)

    assert all(result.status == RequestStatus.COMPLETED for result in results)
    assert worker.batch_calls == [["one", "two", "three"]]
    snapshot = metrics.snapshot()
    assert snapshot["model_invocations"] == 1
    assert snapshot["batch_size_histogram"] == {"3": 3}
