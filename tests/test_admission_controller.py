from concurrent.futures import ThreadPoolExecutor

from serving_platform.admission import TenantAdmissionController
from serving_platform.domain import (
    DeviceType,
    HealthStatus,
    ModelDefinition,
    RequestRecord,
    RuntimeType,
    TenantLimits,
    WorkerState,
)


def model() -> ModelDefinition:
    return ModelDefinition(
        "model", "main", RuntimeType.SIMULATED, 100, ("float16",), "float16",
        100, 200, True, True, 5, 60,
    )


def worker() -> WorkerState:
    return WorkerState(
        "worker", DeviceType.SIMULATED_GPU, "simulated", 1000, 900,
        health_status=HealthStatus.HEALTHY,
    )


def request(request_id: str, tenant: str = "tenant", tokens: int = 10) -> RequestRecord:
    return RequestRecord(
        request_id, tenant, "model", "hello", tokens - 1, 1, 0, 100, False
    )


def test_admission_limits_are_enforced_atomically():
    controller = TenantAdmissionController(
        [TenantLimits("tenant", 1, 1, 10)], 10, clock=lambda: 0
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        decisions = list(pool.map(
            lambda item: controller.decide(item, model(), [worker()]),
            [request("one"), request("two")],
        ))
    assert sum(decision.admitted for decision in decisions) == 1
    assert {decision.code for decision in decisions} == {"admitted", "tenant_queue_full"}
    admitted_id = "one" if decisions[0].admitted else "two"
    controller.mark_running(admitted_id)
    assert controller.snapshot("tenant") == {
        "queued": 0, "running": 1, "reserved_tokens": 10
    }
    controller.release(admitted_id)
    assert controller.snapshot("tenant") == {
        "queued": 0, "running": 0, "reserved_tokens": 0
    }


def test_admission_rejection_codes_cover_model_context_deadline_and_capacity():
    limits = [TenantLimits("tenant", 1, 2, 1000)]
    controller = TenantAdmissionController(limits, 2, clock=lambda: 10)
    assert controller.decide(request("unknown"), None, [worker()]).code == "unsupported_model"
    too_large = request("large", tokens=101)
    assert controller.decide(too_large, model(), [worker()]).code == "context_window_exceeded"
    expired = request("expired")
    expired.deadline = 10
    assert controller.decide(expired, model(), [worker()]).code == "deadline_exceeded"
    unhealthy = worker()
    unhealthy.health_status = HealthStatus.UNHEALTHY
    assert controller.decide(request("unhealthy"), model(), [unhealthy]).code == "no_healthy_capacity"


def test_releasing_queued_reservation_does_not_decrement_running_request():
    controller = TenantAdmissionController(
        [TenantLimits("tenant", 2, 2, 100)], 2, clock=lambda: 0
    )
    assert controller.decide(request("running"), model(), [worker()]).admitted
    assert controller.decide(request("queued"), model(), [worker()]).admitted
    controller.mark_running("running")
    controller.release("queued")
    assert controller.snapshot("tenant") == {
        "queued": 0, "running": 1, "reserved_tokens": 10
    }
