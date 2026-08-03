import asyncio
import time
from dataclasses import replace

import pytest
from conftest import FakeWorker

from app.metrics import Metrics
from scheduler.no_batching import process_one
from scheduler.queue import QueueClosed, RequestQueue
from scheduler.request import InferenceRequest, RequestStatus
from scheduler.scheduler_loop import run_scheduler


def make_request(prompt="hello"):
    return InferenceRequest(prompt, 1, 0, 1, "no_batching")


async def test_queue_full_marks_request_rejected():
    queue = RequestQueue(1)
    await queue.submit(make_request("first"))
    rejected = make_request("second")
    with pytest.raises(asyncio.QueueFull):
        await queue.submit(rejected)
    assert rejected.status == RequestStatus.REJECTED
    assert rejected.future.done()
    assert queue.max_observed_size == 1


async def test_closed_queue_rejects_new_submission():
    queue, metrics = RequestQueue(1), Metrics()
    queue.close(metrics)
    rejected = make_request()
    with pytest.raises(QueueClosed):
        await queue.submit(rejected)
    assert rejected.status == RequestStatus.REJECTED


async def test_cancelled_queued_request_is_not_dispatched(settings):
    worker, metrics = FakeWorker(), Metrics()
    request = make_request()
    request.queued_at = time.monotonic()
    request.future = asyncio.get_running_loop().create_future()
    request.cancel()
    await process_one(request, worker, settings, metrics)
    assert worker.batch_calls == []
    assert metrics.snapshot()["cancelled_requests"] == 1


async def test_shutdown_finishes_active_and_rejects_queued(settings, monkeypatch):
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()

    async def controlled_to_thread(function, *args):
        generation_started.set()
        await release_generation.wait()
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", controlled_to_thread)
    settings = replace(settings, request_timeout_seconds=1)
    queue, worker, metrics = RequestQueue(2), FakeWorker(), Metrics()
    active, queued = make_request("active"), make_request("queued")
    await queue.submit(active)
    await queue.submit(queued)
    scheduler = asyncio.create_task(run_scheduler(queue, worker, settings, metrics))
    await generation_started.wait()

    assert queue.close(metrics) == 1
    release_generation.set()
    await asyncio.wait_for(scheduler, timeout=1)

    assert active.status == RequestStatus.COMPLETED
    assert queued.status == RequestStatus.REJECTED
    assert queued.future.done()
    assert metrics.snapshot()["rejected_requests"] == 1
