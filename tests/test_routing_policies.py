import pytest

from serving_platform.domain import (
    DeviceType,
    HealthStatus,
    RequestRecord,
    RequestState,
    WorkerState,
)
from serving_platform.routing import (
    LeastQueueDepthPolicy,
    NoEligibleWorker,
    RoundRobinPolicy,
)


def request() -> RequestRecord:
    item = RequestRecord("req", "tenant", "model", "hello", 1, 4, 0, 100, False)
    item.transition(RequestState.VALIDATED)
    item.transition(RequestState.ADMITTED)
    return item


def worker(worker_id: str, queue_depth: int = 0, **changes) -> WorkerState:
    values = {
        "worker_id": worker_id,
        "device_type": DeviceType.SIMULATED_GPU,
        "device_name": "simulated",
        "total_memory_bytes": 1000,
        "available_memory_bytes": 500,
        "resident_models": {"model"},
        "queue_depth": queue_depth,
        "health_status": HealthStatus.HEALTHY,
    }
    values.update(changes)
    return WorkerState(**values)


def test_round_robin_rotates_over_only_eligible_workers():
    policy = RoundRobinPolicy()
    workers = [
        worker("a"),
        worker("b"),
        worker("draining", draining=True),
        worker("missing", resident_models=set()),
        worker("unhealthy", health_status=HealthStatus.UNHEALTHY),
        worker("saturated", active_batch_count=1, max_concurrency=1),
    ]

    decisions = [policy.select(request(), workers) for _ in range(3)]
    assert [decision.selected_worker_id for decision in decisions] == ["a", "b", "a"]
    assert decisions[0].candidates == ("a", "b")
    assert decisions[0].rejected == {
        "draining": "worker_draining",
        "missing": "model_not_resident",
        "saturated": "worker_concurrency_saturated",
        "unhealthy": "health:unhealthy",
    }


def test_least_queue_depth_has_deterministic_tie_breaking_and_scores():
    decision = LeastQueueDepthPolicy().select(
        request(), [worker("c", 4), worker("b", 1), worker("a", 1)]
    )
    assert decision.selected_worker_id == "a"
    assert decision.scores == {"a": 1.0, "b": 1.0, "c": 4.0}


def test_policy_reports_why_every_worker_was_rejected():
    with pytest.raises(NoEligibleWorker) as caught:
        RoundRobinPolicy().select(
            request(),
            [worker("draining", draining=True), worker("missing", resident_models=set())],
        )
    assert caught.value.rejected == {
        "draining": "worker_draining",
        "missing": "model_not_resident",
    }
