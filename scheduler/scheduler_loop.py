import asyncio
from scheduler.dynamic_batch import collect_batch, process_batch
from scheduler.no_batching import process_one


async def run_scheduler(request_queue, worker, settings, metrics):
    queue = request_queue.queue
    while True:
        first = await queue.get()
        try:
            if settings.scheduler_policy == "dynamic_batch":
                batch = await collect_batch(queue, first, settings, metrics)
                await process_batch(batch, worker, settings, metrics)
                for _ in batch[1:]:
                    queue.task_done()
            else:
                await process_one(first, worker, settings, metrics)
        finally:
            queue.task_done()
