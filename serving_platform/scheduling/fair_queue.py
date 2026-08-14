from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Sequence
from threading import RLock

from serving_platform.domain import RequestRecord, RequestState, TenantLimits


class WeightedFairRequestQueue:
    """Token-cost deficit round robin with bounded priority aging per tenant."""

    def __init__(
        self,
        tenants: Sequence[TenantLimits],
        base_quantum_tokens: int = 128,
        priority_aging_seconds: float = 1,
        clock: Callable[[], float] = time.monotonic,
    ):
        if base_quantum_tokens <= 0 or priority_aging_seconds <= 0:
            raise ValueError("fair queue parameters must be positive")
        self.tenants = {tenant.tenant_id: tenant for tenant in tenants}
        self.base_quantum_tokens = base_quantum_tokens
        self.priority_aging_seconds = priority_aging_seconds
        self._clock = clock
        self._queues: dict[str, list[RequestRecord]] = {
            tenant_id: [] for tenant_id in self.tenants
        }
        self._deficits = {tenant_id: 0 for tenant_id in self.tenants}
        self._active: deque[str] = deque()
        self._lock = RLock()

    def enqueue(self, request: RequestRecord) -> None:
        with self._lock:
            queue = self._queues.get(request.tenant_id)
            if queue is None:
                raise ValueError(f"tenant is not configured: {request.tenant_id}")
            if request.status not in {RequestState.ADMITTED, RequestState.QUEUED}:
                raise ValueError("only admitted or retry-queued requests enter the global fair queue")
            if any(
                existing.request_id == request.request_id
                for existing in queue
            ):
                raise ValueError(f"request is already queued: {request.request_id}")
            if request.status == RequestState.ADMITTED:
                request.transition(RequestState.QUEUED, self._clock())
            queue.append(request)
            if request.tenant_id not in self._active:
                self._active.append(request.tenant_id)

    def _next_for_tenant(self, tenant_id: str) -> RequestRecord:
        now = self._clock()
        queue = self._queues[tenant_id]
        return max(
            queue,
            key=lambda request: (
                request.priority
                + (now - request.transition_timestamps[RequestState.QUEUED])
                / self.priority_aging_seconds,
                -request.created_at,
            ),
        )

    def pop(self) -> RequestRecord | None:
        with self._lock:
            visits_remaining = len(self._active)
            while self._active and visits_remaining > 0:
                tenant_id = self._active.popleft()
                queue = self._queues[tenant_id]
                queue[:] = [request for request in queue if not request.terminal]
                if not queue:
                    self._deficits[tenant_id] = 0
                    visits_remaining -= 1
                    continue
                self._deficits[tenant_id] += (
                    self.base_quantum_tokens * self.tenants[tenant_id].scheduling_weight
                )
                request = self._next_for_tenant(tenant_id)
                if request.estimated_tokens <= self._deficits[tenant_id]:
                    queue.remove(request)
                    self._deficits[tenant_id] -= request.estimated_tokens
                    if queue:
                        self._active.append(tenant_id)
                    return request
                self._active.append(tenant_id)
                visits_remaining -= 1
            return None

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            for queue in self._queues.values():
                for request in tuple(queue):
                    if request.request_id == request_id:
                        queue.remove(request)
                        request.transition(RequestState.CANCELLED, self._clock())
                        return True
        return False

    def depth(self, tenant_id: str | None = None) -> int:
        with self._lock:
            if tenant_id is not None:
                return len(self._queues[tenant_id])
            return sum(map(len, self._queues.values()))
