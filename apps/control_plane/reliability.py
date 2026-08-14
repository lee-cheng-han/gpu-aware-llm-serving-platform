from __future__ import annotations

import asyncio

from apps.control_plane.service import ControlPlane
from serving_platform.registry import InMemoryWorkerRegistry


class ReliabilitySupervisor:
    """Periodically expires heartbeats and applies safe request recovery rules."""

    def __init__(
        self,
        registry: InMemoryWorkerRegistry,
        control_plane: ControlPlane,
        check_interval_seconds: float = 1,
        max_request_attempts: int = 3,
    ):
        if check_interval_seconds <= 0 or max_request_attempts <= 0:
            raise ValueError("reliability supervisor settings must be positive")
        self.registry = registry
        self.control_plane = control_plane
        self.check_interval_seconds = check_interval_seconds
        self.max_request_attempts = max_request_attempts
        self._stop = asyncio.Event()
        self.failure: BaseException | None = None

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), self.check_interval_seconds
                    )
                except TimeoutError:
                    pass
                if self._stop.is_set():
                    break
                for worker_id in self.registry.expire_stale():
                    self.control_plane.recover_failed_worker(
                        worker_id, self.max_request_attempts
                    )
        except BaseException as exc:
            self.failure = exc
            raise
