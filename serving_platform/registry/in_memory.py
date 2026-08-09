from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from copy import deepcopy
from threading import RLock

from serving_platform.domain import HealthStatus, WorkerState


class InMemoryWorkerRegistry:
    """Thread-safe authoritative worker snapshots with explicit liveness expiry."""

    def __init__(
        self,
        heartbeat_timeout_seconds: float = 30,
        clock: Callable[[], float] = time.monotonic,
    ):
        if heartbeat_timeout_seconds <= 0:
            raise ValueError("heartbeat timeout must be positive")
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self._clock = clock
        self._workers: dict[str, WorkerState] = {}
        self._lock = RLock()

    def get(self, worker_id: str) -> WorkerState | None:
        with self._lock:
            worker = self._workers.get(worker_id)
            return deepcopy(worker) if worker else None

    def list(self) -> Sequence[WorkerState]:
        with self._lock:
            return tuple(deepcopy(self._workers[key]) for key in sorted(self._workers))

    def register(self, worker: WorkerState) -> None:
        with self._lock:
            snapshot = deepcopy(worker)
            snapshot.health_status = HealthStatus.HEALTHY
            snapshot.last_heartbeat = self._clock()
            self._workers[worker.worker_id] = snapshot

    def heartbeat(self, worker: WorkerState) -> None:
        with self._lock:
            current = self._workers.get(worker.worker_id)
            if current is None:
                raise KeyError(f"worker must register before heartbeat: {worker.worker_id}")
            if current.health_status in {HealthStatus.UNHEALTHY, HealthStatus.STOPPED}:
                raise RuntimeError(f"worker must re-register before recovery: {worker.worker_id}")
            snapshot = deepcopy(worker)
            snapshot.health_status = HealthStatus.HEALTHY
            snapshot.last_heartbeat = self._clock()
            self._workers[worker.worker_id] = snapshot

    def unregister(self, worker_id: str) -> None:
        with self._lock:
            self._workers.pop(worker_id, None)

    def expire_stale(self) -> Sequence[str]:
        now = self._clock()
        expired: list[str] = []
        with self._lock:
            for worker in self._workers.values():
                if (
                    worker.health_status == HealthStatus.HEALTHY
                    and now - worker.last_heartbeat > self.heartbeat_timeout_seconds
                ):
                    worker.health_status = HealthStatus.UNHEALTHY
                    expired.append(worker.worker_id)
        return tuple(sorted(expired))
