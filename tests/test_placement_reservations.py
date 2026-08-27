import pytest

from apps.control_plane import (
    ControlPlane,
    GlobalScheduler,
    WorkerDirectory,
    WorkerDispatchError,
)
from serving_platform.domain import (
    DeviceType,
    HealthStatus,
    ModelDefinition,
    RequestRecord,
    RequestState,
    RuntimeType,
    WorkerState,
)
from serving_platform.registry import InMemoryModelRegistry, InMemoryWorkerRegistry
from serving_platform.routing import (
    AtomicPlacementReservations,
    DemandAwarePlacementPolicy,
    LeastQueueDepthPolicy,
    ModelResidencyAwarePolicy,
)


def model():
    return ModelDefinition(
        "model", "main", RuntimeType.SIMULATED, 200, ("float16",), "float16",
        100, 200, True, True, 5, 60,
    )


def request(request_id):
    item = RequestRecord(
        request_id, "tenant", "model", "hello", 1, 2, 0, 100, False
    )
    item.transition(RequestState.VALIDATED)
    item.transition(RequestState.ADMITTED)
    return item


def worker(worker_id, *, resident=True, available=500):
    return WorkerState(
        worker_id,
        DeviceType.SIMULATED_GPU,
        "sim",
        1000,
        available,
        resident_models={"model"} if resident else set(),
        health_status=HealthStatus.HEALTHY,
    )


def test_concurrent_assignments_observe_uncommitted_queue_reservations():
    registry = InMemoryWorkerRegistry()
    registry.register(worker("a"))
    registry.register(worker("b"))
    reservations = AtomicPlacementReservations(registry)
    scheduler = GlobalScheduler(
        registry,
        LeastQueueDepthPolicy(),
        clock=lambda: 10,
        reservations=reservations,
    )

    first = scheduler.assign(request("first"))
    second = scheduler.assign(request("second"))

    assert {first.worker_id, second.worker_id} == {"a", "b"}
    assert reservations.snapshot().active_reservations == 2
    assert scheduler.release_assignment("first")
    assert scheduler.release_assignment("second")
    assert reservations.snapshot().active_reservations == 0


def test_cold_model_memory_is_reserved_until_handoff_finishes():
    definition = model()
    models = InMemoryModelRegistry([definition])
    registry = InMemoryWorkerRegistry()
    registry.register(worker("a", resident=False, available=250))
    registry.register(worker("b", resident=False, available=250))
    reservations = AtomicPlacementReservations(registry, models)
    scheduler = GlobalScheduler(
        registry,
        ModelResidencyAwarePolicy(models),
        clock=lambda: 10,
        reservations=reservations,
    )

    first = scheduler.assign(request("first"))
    second = scheduler.assign(request("second"))

    assert {first.worker_id, second.worker_id} == {"a", "b"}
    assert reservations.snapshot().reserved_memory_bytes == 400


def test_demand_policy_requests_a_new_replica_after_threshold():
    models = InMemoryModelRegistry([model()])
    policy = DemandAwarePlacementPolicy(
        models,
        demand=lambda model_id: 4,
        requests_per_replica=4,
    )
    decision = policy.select(
        request("scale"),
        [worker("warm"), worker("cold", resident=False)],
    )

    assert decision.selected_worker_id == "cold"
    assert decision.scoring_inputs["cold"]["scale_out_requested"] is True
    assert decision.scoring_inputs["cold"]["desired_replicas"] == 2


def test_failed_worker_handoff_always_releases_reservation():
    registry = InMemoryWorkerRegistry()
    registry.register(worker("gone"))
    reservations = AtomicPlacementReservations(registry)
    plane = ControlPlane(
        GlobalScheduler(
            registry,
            LeastQueueDepthPolicy(),
            clock=lambda: 10,
            reservations=reservations,
        ),
        WorkerDirectory(),
    )

    with pytest.raises(WorkerDispatchError):
        plane.dispatch(request("handoff"))

    assert reservations.snapshot().active_reservations == 0
