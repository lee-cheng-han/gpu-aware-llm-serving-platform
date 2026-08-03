from collections import deque

from scheduler.dynamic_batch import collect_batch, process_batch
from scheduler.no_batching import process_one
from scheduler.request import RequestStatus


async def run_scheduler(request_queue, worker, settings, metrics):
    queue = request_queue.queue
    deferred = deque()
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
                batch = await collect_batch(
                    queue, first, settings, metrics, deferred=deferred
                )
                await process_batch(batch, worker, settings, metrics)
                for _ in batch[1:]:
                    queue.task_done()
            else:
                await process_one(first, worker, settings, metrics)
        finally:
            if from_queue:
                queue.task_done()
