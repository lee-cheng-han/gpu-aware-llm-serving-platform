import asyncio
import time
from scheduler.request import InferenceRequest, RequestStatus


class QueueClosed(RuntimeError):
    pass


class RequestQueue:
    def __init__(self, maximum: int):
        self.queue: asyncio.Queue[InferenceRequest | None] = asyncio.Queue(maxsize=maximum)
        self.accepting = True
        self.max_observed_size = 0

    async def submit(self, request: InferenceRequest) -> None:
        request.future = asyncio.get_running_loop().create_future()
        if not self.accepting:
            request.finish(RequestStatus.REJECTED, "scheduler is shutting down")
            raise QueueClosed("scheduler is shutting down")
        request.queued_at = time.monotonic()
        request.status = RequestStatus.QUEUED
        try:
            self.queue.put_nowait(request)
            self.max_observed_size = max(self.max_observed_size, self.queue.qsize())
        except asyncio.QueueFull:
            request.finish(RequestStatus.REJECTED, "scheduler queue is full")
            raise

    def close(self, metrics) -> int:
        """Reject queued work and wake the scheduler without cancelling active work."""
        if not self.accepting:
            return 0
        self.accepting = False
        rejected = 0
        while True:
            try:
                request = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if request is not None:
                request.finish(RequestStatus.REJECTED, "server is shutting down")
                metrics.rejected()
                rejected += 1
            self.queue.task_done()
        # Draining guarantees room for the sentinel even when the queue was full.
        self.queue.put_nowait(None)
        return rejected
