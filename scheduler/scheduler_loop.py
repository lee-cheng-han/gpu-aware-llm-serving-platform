from collections import deque

from scheduler.dynamic_batch import collect_batch, process_batch
from scheduler.no_batching import process_one
from scheduler.request import InferenceRequest, RequestStatus


async def run_scheduler(request_queue, worker, settings, metrics):
    queue = request_queue.queue
    deferred: deque[InferenceRequest] = deque()
    while True:
        if not request_queue.accepting and deferred:
            while deferred:
                request = deferred.popleft()
                request.finish(RequestStatus.REJECTED, "server is shutting down")
                metrics.rejected(reason="service_shutting_down")
            continue
        from_queue = not deferred
        first = await queue.get() if from_queue else deferred.popleft()
        if first is None:
            queue.task_done()
            break
        try:
            if settings.scheduler_policy == "dynamic_batch":
                collection = await collect_batch(
                    queue, first, settings, metrics, deferred=deferred
                )
                try:
                    await process_batch(collection, worker, settings, metrics)
                finally:
                    # Every additional item taken from asyncio.Queue stays unfinished
                    # until its shared model call reaches a terminal outcome.
                    for _ in range(collection.queued_items):
                        queue.task_done()
            else:
                await process_one(first, worker, settings, metrics)
        finally:
            if from_queue:
                queue.task_done()
