from __future__ import annotations

from threading import RLock

from apps.control_plane.scheduler import GlobalScheduler
from apps.worker import ManagedWorker
from serving_platform.domain import Assignment, RequestRecord, RequestState
from serving_platform.registry.interfaces import ModelRegistry


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
    ):
        self.scheduler = scheduler
        self.workers = workers
        self.models = models

    def dispatch(self, request: RequestRecord) -> Assignment:
        assignment = self.scheduler.assign(request)
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
        except (MemoryError, OverflowError, RuntimeError) as exc:
            if not request.terminal:
                request.assigned_worker_id = None
                request.transition(RequestState.QUEUED)
            if isinstance(exc, WorkerDispatchError):
                raise
            raise WorkerDispatchError("selected worker rejected the handoff") from exc
        return assignment
