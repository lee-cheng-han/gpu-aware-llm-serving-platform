import asyncio
from contextlib import asynccontextmanager

from app.errors import APIError


class ConcurrencyLimiter:
    def __init__(self, maximum: int):
        self.maximum = maximum
        self.active = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def slot(self):
        async with self._lock:
            if self.active >= self.maximum:
                raise APIError(
                    429, "concurrency_limit_exceeded",
                    "maximum concurrent requests exceeded",
                )
            self.active += 1
        try:
            yield
        finally:
            async with self._lock:
                self.active -= 1


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
