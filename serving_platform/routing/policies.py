from __future__ import annotations

from collections.abc import Sequence
from threading import Lock

from serving_platform.domain import HealthStatus, RequestRecord, WorkerState
from serving_platform.routing.interfaces import RoutingDecision


class NoEligibleWorker(RuntimeError):
    def __init__(self, rejected: dict[str, str]):
        super().__init__("no eligible worker")
        self.rejected = rejected


def filter_workers(
    request: RequestRecord,
    workers: Sequence[WorkerState],
) -> tuple[tuple[WorkerState, ...], dict[str, str]]:
    eligible: list[WorkerState] = []
    rejected: dict[str, str] = {}
    for worker in sorted(workers, key=lambda item: item.worker_id):
        reason: str | None = None
        if worker.health_status != HealthStatus.HEALTHY:
            reason = f"health:{worker.health_status.value}"
        elif worker.draining:
            reason = "worker_draining"
        elif request.model_id not in worker.resident_models:
            reason = "model_not_resident"
        elif worker.active_batch_count >= worker.max_concurrency:
            reason = "worker_concurrency_saturated"
        if reason:
            rejected[worker.worker_id] = reason
        else:
            eligible.append(worker)
    return tuple(eligible), rejected


class RoundRobinPolicy:
    name = "round_robin"

    def __init__(self) -> None:
        self._next = 0
        self._lock = Lock()

    def select(
        self,
        request: RequestRecord,
        workers: Sequence[WorkerState],
    ) -> RoutingDecision:
        eligible, rejected = filter_workers(request, workers)
        if not eligible:
            raise NoEligibleWorker(rejected)
        with self._lock:
            selected = eligible[self._next % len(eligible)]
            self._next += 1
        return RoutingDecision(
            selected_worker_id=selected.worker_id,
            policy=self.name,
            candidates=tuple(worker.worker_id for worker in eligible),
            rejected=rejected,
            scores={worker.worker_id: 0.0 for worker in eligible},
        )


class LeastQueueDepthPolicy:
    name = "least_queue_depth"

    def select(
        self,
        request: RequestRecord,
        workers: Sequence[WorkerState],
    ) -> RoutingDecision:
        eligible, rejected = filter_workers(request, workers)
        if not eligible:
            raise NoEligibleWorker(rejected)
        scores = {worker.worker_id: float(worker.queue_depth) for worker in eligible}
        selected = min(eligible, key=lambda worker: (scores[worker.worker_id], worker.worker_id))
        return RoutingDecision(
            selected_worker_id=selected.worker_id,
            policy=self.name,
            candidates=tuple(worker.worker_id for worker in eligible),
            rejected=rejected,
            scores=scores,
        )
