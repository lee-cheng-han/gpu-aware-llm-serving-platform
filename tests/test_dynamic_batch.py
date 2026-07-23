import asyncio
from dataclasses import replace
import time

from app.metrics import Metrics
from scheduler.dynamic_batch import collect_batch, process_batch
from scheduler.request import InferenceRequest, RequestStatus
from conftest import FakeWorker


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
