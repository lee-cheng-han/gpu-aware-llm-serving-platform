import asyncio
from contextlib import asynccontextmanager

from app.errors import APIError


class ConcurrencyLimiter:
    def __init__(self, maximum: int):
        self.maximum = maximum
        self.active = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            if self.active >= self.maximum:
                raise APIError(
                    429, "concurrency_limit_exceeded",
                    "maximum concurrent requests exceeded",
                )
            self.active += 1

    async def release(self) -> None:
        async with self._lock:
            if self.active <= 0:
                raise RuntimeError("concurrency limiter released without an active slot")
            self.active -= 1

    @asynccontextmanager
    async def slot(self):
        await self.acquire()
        try:
            yield
        finally:
            await self.release()


class StreamTracker:
    """Tracks streams across response-body lifetime and coordinates shutdown."""

    def __init__(self):
        self.accepting = True
        self.active = 0
        self._lock = asyncio.Lock()
        self._idle = asyncio.Event()
        self._idle.set()

    async def start(self) -> None:
        async with self._lock:
            if not self.accepting:
                raise APIError(503, "service_shutting_down", "server is shutting down")
            self.active += 1
            self._idle.clear()

    async def finish(self) -> None:
        async with self._lock:
            if self.active <= 0:
                raise RuntimeError("stream tracker finished without an active stream")
            self.active -= 1
            if self.active == 0:
                self._idle.set()

    async def close(self, timeout: float) -> bool:
        async with self._lock:
            self.accepting = False
        try:
            await asyncio.wait_for(self._idle.wait(), timeout)
            return True
        except TimeoutError:
            return False


def validate_request(
    prompt: str,
    max_new_tokens: int,
    prompt_tokens: int,
    context_window_tokens: int,
    settings,
) -> None:
    if not prompt.strip():
        raise APIError(400, "empty_prompt", "prompt must not be empty")
    if prompt_tokens > settings.max_prompt_tokens:
        raise APIError(
            400, "prompt_too_long",
            f"prompt exceeds {settings.max_prompt_tokens} tokens",
            {"input_tokens": prompt_tokens, "max_prompt_tokens": settings.max_prompt_tokens},
        )
    if max_new_tokens > settings.max_new_tokens:
        raise APIError(
            400, "output_too_long",
            f"max_new_tokens exceeds {settings.max_new_tokens}",
            {"max_new_tokens": max_new_tokens, "limit": settings.max_new_tokens},
        )
    if prompt_tokens + max_new_tokens > context_window_tokens:
        raise APIError(
            400, "context_window_exceeded",
            "prompt and requested output exceed the model context window",
            {
                "input_tokens": prompt_tokens,
                "max_new_tokens": max_new_tokens,
                "context_window_tokens": context_window_tokens,
            },
        )
