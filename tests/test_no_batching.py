import asyncio

from app.metrics import Metrics
from scheduler.no_batching import process_one
from scheduler.request import InferenceRequest, RequestStatus
from conftest import FakeWorker


async def test_no_batching_processes_one(settings):
    worker, metrics = FakeWorker(), Metrics()
    item = InferenceRequest("hello", 2, .7, 1, "no_batching")
    item.queued_at = __import__("time").monotonic()
    item.future = asyncio.get_running_loop().create_future()
    await process_one(item, worker, settings, metrics)
    assert item.status == RequestStatus.COMPLETED
    assert worker.batch_calls == [["hello"]]
    assert item.batch_size == 1
