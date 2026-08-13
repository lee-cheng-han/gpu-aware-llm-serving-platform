from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from threading import RLock

from serving_platform.admission.interfaces import AdmissionDecision
from serving_platform.domain import (
    HealthStatus,
    ModelDefinition,
    RequestRecord,
    TenantLimits,
    WorkerState,
)


class TenantAdmissionController:
    """Atomic tenant/global reservations; callers release once work is terminal."""

    def __init__(
        self,
        tenants: Sequence[TenantLimits],
        global_queue_capacity: int,
        clock: Callable[[], float] = time.monotonic,
    ):
        if global_queue_capacity <= 0:
            raise ValueError("global queue capacity must be positive")
        self.tenants = {tenant.tenant_id: tenant for tenant in tenants}
        if len(self.tenants) != len(tenants):
            raise ValueError("tenant identifiers must be unique")
        self.global_queue_capacity = global_queue_capacity
        self._clock = clock
        self._queued: Counter[str] = Counter()
        self._running: Counter[str] = Counter()
        self._tokens: Counter[str] = Counter()
        self._reserved: dict[str, tuple[str, int, bool]] = {}
        self._lock = RLock()

    def decide(
        self,
        request: RequestRecord,
        model: ModelDefinition | None,
        workers: Sequence[WorkerState],
    ) -> AdmissionDecision:
        with self._lock:
            if request.request_id in self._reserved:
                return AdmissionDecision(True, "already_admitted", "reservation already exists")
            tenant = self.tenants.get(request.tenant_id)
            if tenant is None:
                return AdmissionDecision(False, "unknown_tenant", "tenant is not configured")
            if model is None:
                return AdmissionDecision(False, "unsupported_model", "model is not registered")
            if request.estimated_tokens > model.max_context_tokens:
                return AdmissionDecision(False, "context_window_exceeded", "context is too large")
            if request.deadline <= self._clock():
                return AdmissionDecision(False, "deadline_exceeded", "deadline has expired")
            healthy = [
                worker for worker in workers
                if worker.health_status == HealthStatus.HEALTHY and not worker.draining
            ]
            if not healthy:
                return AdmissionDecision(False, "no_healthy_capacity", "no healthy worker")
            if sum(self._queued.values()) >= self.global_queue_capacity:
                return AdmissionDecision(False, "global_queue_full", "global queue is full")
            if self._queued[tenant.tenant_id] >= tenant.max_queued_requests:
                return AdmissionDecision(False, "tenant_queue_full", "tenant queue is full")
            if self._running[tenant.tenant_id] >= tenant.max_concurrent_requests:
                return AdmissionDecision(
                    False, "tenant_concurrency_limit", "tenant concurrency limit reached"
                )
            if self._tokens[tenant.tenant_id] + request.estimated_tokens > tenant.token_quota:
                return AdmissionDecision(False, "tenant_token_quota", "tenant token quota exceeded")
            self._queued[tenant.tenant_id] += 1
            self._tokens[tenant.tenant_id] += request.estimated_tokens
            self._reserved[request.request_id] = (
                tenant.tenant_id,
                request.estimated_tokens,
                False,
            )
            return AdmissionDecision(True, "admitted", "request capacity reserved")

    def mark_running(self, request_id: str) -> None:
        with self._lock:
            tenant_id, tokens, running = self._reserved[request_id]
            if running:
                return
            limits = self.tenants[tenant_id]
            if self._running[tenant_id] >= limits.max_concurrent_requests:
                raise RuntimeError("tenant concurrency limit reached")
            if self._queued[tenant_id] > 0:
                self._queued[tenant_id] -= 1
            self._running[tenant_id] += 1
            self._reserved[request_id] = (tenant_id, tokens, True)

    def release(self, request_id: str) -> None:
        with self._lock:
            reservation = self._reserved.pop(request_id, None)
            if reservation is None:
                return
            tenant_id, tokens, running = reservation
            if running:
                self._running[tenant_id] -= 1
            else:
                self._queued[tenant_id] -= 1
            self._tokens[tenant_id] -= tokens

    def snapshot(self, tenant_id: str) -> dict[str, int]:
        with self._lock:
            return {
                "queued": self._queued[tenant_id],
                "running": self._running[tenant_id],
                "reserved_tokens": self._tokens[tenant_id],
            }
