import asyncio
from dataclasses import replace

from conftest import FakeWorker

from app.metrics import Metrics
from scheduler.no_batching import process_one
from scheduler.request import InferenceRequest, RequestStatus


async def test_no_batching_processes_one(settings):
    worker, metrics = FakeWorker(), Metrics()
    item = InferenceRequest("hello", 2, .7, 1, "no_batching")
    item.queued_at = __import__("time").monotonic()
    item.future = asyncio.get_running_loop().create_future()
    await process_one(item, worker, settings, metrics)
    assert item.status == RequestStatus.COMPLETED
    assert worker.batch_calls == [["hello"]]
    assert item.batch_size == 1
    snapshot = metrics.snapshot()
    assert snapshot["model_invocations"] == 1
    assert snapshot["batch_size_histogram"] == {"1": 1}
    assert snapshot["avg_generation_ms"] == 2


async def test_scheduler_never_overlaps_no_batching_dispatches(settings, monkeypatch):
    from scheduler.queue import RequestQueue
    from scheduler.request import InferenceRequest
    from scheduler.scheduler_loop import run_scheduler

    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()
    calls = 0

    async def controlled_to_thread(function, *args):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            await release_first.wait()
        else:
            second_started.set()
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", controlled_to_thread)
    queue, worker, metrics = RequestQueue(2), FakeWorker(), Metrics()
    for prompt in ("first", "second"):
        await queue.submit(InferenceRequest(prompt, 1, 0, 1, "no_batching"))
    task = asyncio.create_task(run_scheduler(
        queue, worker, replace(settings, scheduler_policy="no_batching"), metrics
    ))
    await first_started.wait()
    await asyncio.sleep(0)
    assert not second_started.is_set()
    release_first.set()
    await second_started.wait()
    queue.close(metrics)
    await asyncio.wait_for(task, 1)
    assert worker.batch_calls == [["first"], ["second"]]
