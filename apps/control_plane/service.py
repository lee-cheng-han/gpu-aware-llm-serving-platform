from __future__ import annotations

from threading import RLock

from apps.control_plane.scheduler import GlobalScheduler
from apps.worker import ManagedWorker
from serving_platform.domain import Assignment, RequestRecord, RequestState
from serving_platform.registry.interfaces import ModelRegistry
from serving_platform.request_state.interfaces import RequestStateStore
from serving_platform.scheduling import WeightedFairRequestQueue


class WorkerDispatchError(RuntimeError):
    pass


class WorkerDirectory:
    """Process-local handles corresponding to worker-registry state snapshots."""

    def __init__(self) -> None:
        self._workers: dict[str, ManagedWorker] = {}
        self._lock = RLock()

    def add(self, worker: ManagedWorker) -> None:
        with self._lock:
            self._workers[worker.worker_id] = worker

    def remove(self, worker_id: str) -> None:
        with self._lock:
            self._workers.pop(worker_id, None)

    def get(self, worker_id: str) -> ManagedWorker | None:
        with self._lock:
            return self._workers.get(worker_id)


class ControlPlane:
    def __init__(
        self,
        scheduler: GlobalScheduler,
        workers: WorkerDirectory,
        models: ModelRegistry | None = None,
        requests: RequestStateStore | None = None,
        global_queue: WeightedFairRequestQueue | None = None,
    ):
        self.scheduler = scheduler
        self.workers = workers
        self.models = models
        self.requests = requests
        self.global_queue = global_queue

    def dispatch(self, request: RequestRecord) -> Assignment:
        assignment = self.scheduler.assign(request)
        if self.requests is not None:
            self.requests.save(request)
        worker = self.workers.get(assignment.worker_id)
        try:
            if worker is None:
                raise WorkerDispatchError("selected worker is no longer reachable")
            if not worker.runtime.is_model_loaded(request.model_id):
                if self.models is None:
                    raise WorkerDispatchError("selected worker does not host the model")
                model = self.models.get(request.model_id)
                if model is None:
                    raise WorkerDispatchError("request model is not registered")
                worker.load_model(model)
                worker.heartbeat()
            if request.deadline <= self.scheduler.now():
                request.transition(RequestState.TIMED_OUT)
                raise WorkerDispatchError("request deadline expired during model loading")
            worker.enqueue_request(request)
            if self.requests is not None:
                self.requests.save(request)
        except (MemoryError, OverflowError, RuntimeError) as exc:
            if not request.terminal:
                request.assigned_worker_id = None
                request.transition(RequestState.QUEUED)
            if isinstance(exc, WorkerDispatchError):
                raise
            raise WorkerDispatchError("selected worker rejected the handoff") from exc
        return assignment

    def cancel(self, request_id: str) -> bool:
        if self.requests is None:
            raise RuntimeError("request state store is required for global cancellation")
        request = self.requests.get(request_id)
        if request is None or request.terminal:
            return False
        cancelled = False
        if self.global_queue is not None:
            cancelled = self.global_queue.cancel(request_id)
        if not cancelled and request.assigned_worker_id:
            worker = self.workers.get(request.assigned_worker_id)
            if worker is not None:
                cancelled = worker.cancel_request(request_id)
        if cancelled:
            request.transition(RequestState.CANCELLED)
            self.requests.save(request)
        return cancelled

    def recover_failed_worker(self, worker_id: str, max_attempts: int) -> dict[str, int]:
        if max_attempts <= 0:
            raise ValueError("maximum attempts must be positive")
        if self.requests is None:
            raise RuntimeError("request state store is required for worker recovery")
        self.workers.remove(worker_id)
        recovered = failed = untouched = 0
        for request in self.requests.list():
            if request.assigned_worker_id != worker_id or request.terminal:
                untouched += 1
                continue
            started = RequestState.RUNNING in request.transition_timestamps
            streamed = RequestState.STREAMING in request.transition_timestamps
            if request.status in {RequestState.RUNNING, RequestState.STREAMING} or started or streamed:
                request.transition(RequestState.FAILED)
                request.retry_reasons.append("worker_lost_after_execution_started")
                failed += 1
            elif not request.payload_available:
                request.transition(RequestState.FAILED)
                request.retry_reasons.append("request_payload_unavailable_after_restart")
                failed += 1
            elif request.attempt_count >= max_attempts or request.deadline <= self.scheduler.now():
                target = (
                    RequestState.TIMED_OUT
                    if request.deadline <= self.scheduler.now()
                    else RequestState.FAILED
                )
                request.transition(target)
                request.retry_reasons.append(
                    "deadline_exhausted" if target == RequestState.TIMED_OUT
                    else "retry_budget_exhausted"
                )
                failed += 1
            else:
                if request.status == RequestState.ASSIGNED:
                    request.transition(RequestState.QUEUED)
                request.assigned_worker_id = None
                request.retry_reasons.append("assigned_worker_heartbeat_expired")
                if self.global_queue is None:
                    request.transition(RequestState.FAILED)
                    request.retry_reasons.append("global_retry_queue_unavailable")
                    failed += 1
                else:
                    self.global_queue.enqueue(request)
                    recovered += 1
            self.requests.save(request)
        return {"requeued": recovered, "failed": failed, "untouched": untouched}
