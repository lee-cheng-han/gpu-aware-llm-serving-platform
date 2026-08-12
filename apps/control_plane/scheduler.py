from __future__ import annotations

import time
from collections.abc import Callable

from serving_platform.domain import Assignment, RequestRecord, RequestState
from serving_platform.registry.interfaces import WorkerRegistry
from serving_platform.routing import NoEligibleWorker, RoutingPolicy


class RequestDeadlineExceeded(TimeoutError):
    pass


class GlobalScheduler:
    """Assigns admitted requests using immutable worker-registry snapshots."""

    def __init__(
        self,
        registry: WorkerRegistry,
        policy: RoutingPolicy,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.registry = registry
        self.policy = policy
        self._clock = clock

    def assign(self, request: RequestRecord) -> Assignment:
        if request.status not in {RequestState.ADMITTED, RequestState.QUEUED}:
            raise ValueError("only admitted or globally queued requests can be assigned")
        now = self._clock()
        if request.deadline <= now:
            request.transition(RequestState.TIMED_OUT, now)
            raise RequestDeadlineExceeded("request deadline expired before assignment")

        decision = self.policy.select(request, self.registry.list())
        request.assigned_worker_id = decision.selected_worker_id
        request.attempt_count += 1
        request.transition(RequestState.ASSIGNED, now)
        return Assignment(
            request_id=request.request_id,
            worker_id=decision.selected_worker_id,
            policy=decision.policy,
            explanation={
                "candidates": decision.candidates,
                "rejected": decision.rejected,
                "scores": decision.scores,
                "scoring_inputs": decision.scoring_inputs,
                "estimated_request_tokens": request.estimated_tokens,
            },
            assigned_at=now,
        )


__all__ = ["GlobalScheduler", "NoEligibleWorker", "RequestDeadlineExceeded"]
