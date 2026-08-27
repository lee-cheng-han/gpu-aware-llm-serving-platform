from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from serving_platform.domain import RequestRecord, WorkerState
from serving_platform.registry.interfaces import ModelRegistry, WorkerRegistry
from serving_platform.routing.interfaces import RoutingDecision, RoutingPolicy


@dataclass(frozen=True)
class PlacementReservation:
    request_id: str
    worker_id: str
    model_id: str
    memory_bytes: int


@dataclass(frozen=True)
class PlacementReservationSnapshot:
    active_reservations: int
    reserved_memory_bytes: int
    demand_by_model: dict[str, int]


class AtomicPlacementReservations:
    """Serializes placement decisions and accounts for handoffs not in heartbeats yet."""

    def __init__(self, registry: WorkerRegistry, models: ModelRegistry | None = None) -> None:
        self.registry = registry
        self.models = models
        self._reservations: dict[str, PlacementReservation] = {}
        self._lock = RLock()

    def _effective_workers(self) -> tuple[WorkerState, ...]:
        workers = {worker.worker_id: worker for worker in self.registry.list()}
        reserved_models: set[tuple[str, str]] = set()
        for reservation in self._reservations.values():
            worker = workers.get(reservation.worker_id)
            if worker is None:
                continue
            worker.queue_depth += 1
            model_key = (reservation.worker_id, reservation.model_id)
            if reservation.memory_bytes and model_key not in reserved_models:
                if worker.available_memory_bytes is not None:
                    worker.available_memory_bytes = max(
                        0, worker.available_memory_bytes - reservation.memory_bytes
                    )
                reserved_models.add(model_key)
        return tuple(workers[key] for key in sorted(workers))

    def select_and_reserve(
        self,
        request: RequestRecord,
        policy: RoutingPolicy,
    ) -> RoutingDecision:
        with self._lock:
            if request.request_id in self._reservations:
                raise ValueError(f"request already has a placement reservation: {request.request_id}")
            decision = policy.select(request, self._effective_workers())
            worker = self.registry.get(decision.selected_worker_id)
            if worker is None:
                raise RuntimeError("selected worker disappeared before reservation")
            memory_bytes = 0
            if request.model_id not in worker.resident_models and self.models is not None:
                model = self.models.get(request.model_id)
                if model is not None and not any(
                    reservation.worker_id == worker.worker_id
                    and reservation.model_id == request.model_id
                    for reservation in self._reservations.values()
                ):
                    memory_bytes = model.estimated_memory_bytes
            self._reservations[request.request_id] = PlacementReservation(
                request.request_id,
                worker.worker_id,
                request.model_id,
                memory_bytes,
            )
            return decision

    def release(self, request_id: str) -> bool:
        with self._lock:
            return self._reservations.pop(request_id, None) is not None

    def demand(self, model_id: str) -> int:
        with self._lock:
            return sum(
                reservation.model_id == model_id
                for reservation in self._reservations.values()
            )

    def snapshot(self) -> PlacementReservationSnapshot:
        with self._lock:
            demand: dict[str, int] = {}
            for reservation in self._reservations.values():
                demand[reservation.model_id] = demand.get(reservation.model_id, 0) + 1
            return PlacementReservationSnapshot(
                len(self._reservations),
                sum(item.memory_bytes for item in self._reservations.values()),
                dict(sorted(demand.items())),
            )
