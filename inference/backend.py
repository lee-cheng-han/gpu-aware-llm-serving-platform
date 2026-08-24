from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol, TypeVar

from inference.token_types import GeneratedToken, SamplingParameters, TokenizedPrompt
from runtime.base import RuntimeCapacity

T = TypeVar("T")


class InferenceBackend(Protocol):
    def load_model(self) -> None: ...
    def unload_model(self) -> None: ...
    def tokenize(self, prompt: str) -> TokenizedPrompt: ...
    def prefill(
        self,
        sequence_id: str,
        prompt: TokenizedPrompt,
        parameters: SamplingParameters,
    ) -> GeneratedToken: ...
    def decode(
        self,
        sequence_id: str,
        parameters: SamplingParameters,
        index: int,
    ) -> GeneratedToken: ...
    def release_sequence(self, sequence_id: str) -> None: ...
    def capacity(self) -> RuntimeCapacity: ...
    def shutdown(self) -> None: ...
    def vocabulary_size(self) -> int: ...
    def eos_token_ids(self) -> frozenset[int]: ...


class BackendExecutor:
    """One command lane per backend; model calls cannot overlap."""

    def __init__(self, thread_name_prefix: str = "inference-backend"):
        self._pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=thread_name_prefix,
        )
        self._closed = False

    def submit(self, function: Callable[..., T], *args: object) -> Future[T]:
        if self._closed:
            raise RuntimeError("backend executor is shut down")
        return self._pool.submit(function, *args)

    def run(self, function: Callable[..., T], *args: object) -> T:
        return self.submit(function, *args).result()

    async def run_async(self, function: Callable[..., T], *args: object) -> T:
        if self._closed:
            raise RuntimeError("backend executor is shut down")
        future = self.submit(function, *args)
        try:
            while not future.done():
                await asyncio.sleep(0.0005)
            return future.result()
        except asyncio.CancelledError:
            # The device call is not preemptible. Wait for it before callers release
            # sequence state, preventing a cache use-after-free race.
            while not future.done():
                await asyncio.sleep(0.0005)
            raise

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=False)


@dataclass(frozen=True)
class BackendBundle:
    backend: InferenceBackend
    executor: BackendExecutor
