import asyncio
from scheduler.request import InferenceRequest, RequestStatus


class RequestQueue:
    def __init__(self, maximum: int):
        self.queue: asyncio.Queue[InferenceRequest] = asyncio.Queue(maxsize=maximum)

    async def submit(self, request: InferenceRequest) -> None:
        request.queued_at = __import__("time").monotonic()
        request.status = RequestStatus.QUEUED
        request.future = asyncio.get_running_loop().create_future()
        self.queue.put_nowait(request)
