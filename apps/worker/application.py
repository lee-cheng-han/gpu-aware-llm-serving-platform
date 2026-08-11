from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

from apps.worker.service import ManagedWorker, WorkerExecutionResult
from serving_platform.domain import ModelDefinition


class WorkerApplication:
    """Supervises worker heartbeats and local batch execution as one lifecycle."""

    def __init__(
        self,
        worker: ManagedWorker,
        startup_models: Sequence[ModelDefinition] = (),
        heartbeat_interval_seconds: float = 5,
        idle_poll_seconds: float = 0.01,
        result_sink: Callable[[tuple[WorkerExecutionResult, ...]], None] | None = None,
    ):
        if min(heartbeat_interval_seconds, idle_poll_seconds) <= 0:
            raise ValueError("worker loop intervals must be positive")
        self.worker = worker
        self.startup_models = tuple(startup_models)
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.idle_poll_seconds = idle_poll_seconds
        self.result_sink = result_sink or (lambda results: None)
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self.worker.drain()
        self._stop.set()

    async def _wait_or_stop(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout)
        except TimeoutError:
            return False
        return True

    async def _heartbeat_loop(self) -> None:
        while not await self._wait_or_stop(self.heartbeat_interval_seconds):
            self.worker.heartbeat()

    async def _execute_uninterruptibly(self) -> tuple[WorkerExecutionResult, ...]:
        execution = asyncio.create_task(asyncio.to_thread(self.worker.execute_batch))
        try:
            return await asyncio.shield(execution)
        except asyncio.CancelledError:
            # A Python thread cannot be force-cancelled safely. Wait for the active
            # runtime call before allowing shutdown to unload its model.
            await execution
            raise

    async def _execution_loop(self) -> None:
        while not self._stop.is_set():
            results = await self._execute_uninterruptibly()
            if results:
                self.result_sink(results)
            else:
                await self._wait_or_stop(self.idle_poll_seconds)

    async def run(self) -> None:
        self.worker.register()
        try:
            for model in self.startup_models:
                self.worker.load_model(model)
                self.worker.warmup_model(model.model_id)
            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(self._heartbeat_loop(), name="worker-heartbeat")
                tasks.create_task(self._execution_loop(), name="worker-execution")
        finally:
            self.worker.shutdown()
