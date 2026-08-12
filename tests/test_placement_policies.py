import pytest

from serving_platform.domain import (
    DeviceType,
    HealthStatus,
    ModelDefinition,
    RequestRecord,
    RequestState,
    RuntimeType,
    WorkerState,
)
from serving_platform.registry import InMemoryModelRegistry
from serving_platform.routing import (
    EstimatedCompletionTimePolicy,
    MemoryAwareLeastLoadedPolicy,
    ModelResidencyAwarePolicy,
    NoEligibleWorker,
    UnknownModel,
)


def model(
    runtime_type: RuntimeType = RuntimeType.SIMULATED,
) -> ModelDefinition:
    return ModelDefinition(
        "model", "main", runtime_type, 200, ("float16",), "float16",
        100, 200, True, True, 5, 60,
    )


def request(deadline: float = 100, prompt_tokens: int = 5) -> RequestRecord:
    item = RequestRecord(
        "request", "tenant", "model", "hello", prompt_tokens, 5, 0, deadline, False
    )
    item.transition(RequestState.VALIDATED)
    item.transition(RequestState.ADMITTED)
    return item


def worker(worker_id: str, **changes) -> WorkerState:
    values = {
        "worker_id": worker_id,
        "device_type": DeviceType.SIMULATED_GPU,
        "device_name": "simulated",
        "total_memory_bytes": 1000,
        "available_memory_bytes": 500,
        "resident_models": set(),
        "health_status": HealthStatus.HEALTHY,
    }
    values.update(changes)
    return WorkerState(**values)


def registry(definition: ModelDefinition | None = None) -> InMemoryModelRegistry:
    return InMemoryModelRegistry([definition or model()])


def test_residency_policy_prefers_warm_worker_over_shorter_cold_queue():
    decision = ModelResidencyAwarePolicy(registry()).select(
        request(),
        [worker("cold", queue_depth=0), worker("warm", queue_depth=5, resident_models={"model"})],
    )
    assert decision.selected_worker_id == "warm"
    assert decision.scores == {"cold": 1.0, "warm": 0.0}
    assert decision.scoring_inputs["warm"]["model_resident"] is True


def test_cold_worker_must_fit_model_after_memory_safety_reserve():
    policy = ModelResidencyAwarePolicy(registry(), memory_safety_reserve_bytes=100)
    with pytest.raises(NoEligibleWorker) as caught:
        policy.select(request(), [worker("small", available_memory_bytes=250)])
    assert caught.value.rejected == {
        "small": "insufficient_memory_after_safety_reserve"
    }


def test_unknown_cpu_memory_only_allows_already_resident_model():
    definition = model(RuntimeType.HUGGINGFACE)
    policy = MemoryAwareLeastLoadedPolicy(registry(definition))
    cold_cpu = worker(
        "cold-cpu", device_type=DeviceType.CPU, total_memory_bytes=None,
        available_memory_bytes=None,
    )
    warm_cpu = worker(
        "warm-cpu", device_type=DeviceType.CPU, total_memory_bytes=None,
        available_memory_bytes=None, resident_models={"model"},
    )
    decision = policy.select(request(), [cold_cpu, warm_cpu])
    assert decision.selected_worker_id == "warm-cpu"
    assert decision.rejected == {"cold-cpu": "memory_unknown_for_cold_load"}


def test_memory_policy_uses_projected_memory_queue_and_concurrency_load():
    decision = MemoryAwareLeastLoadedPolicy(registry(), 50).select(
        request(),
        [
            worker("loaded", available_memory_bytes=800, resident_models={"model"}),
            worker("cold", available_memory_bytes=900),
            worker("queued", available_memory_bytes=950, queue_depth=1),
        ],
    )
    assert decision.selected_worker_id == "loaded"
    assert decision.scoring_inputs["cold"]["projected_model_bytes"] == 200
    assert decision.scoring_inputs["loaded"]["memory_safety_reserve_bytes"] == 50


def test_estimated_completion_exposes_terms_and_filters_impossible_deadline():
    policy = EstimatedCompletionTimePolicy(registry(), clock=lambda: 10)
    decision = policy.select(
        request(deadline=11),
        [
            worker("fast", recent_tokens_per_second=100, queue_depth=1),
            worker("slow", recent_tokens_per_second=10, resident_models={"model"}),
        ],
    )
    assert decision.selected_worker_id == "slow"
    assert decision.rejected == {"fast": "deadline_not_feasible"}
    assert decision.scoring_inputs["slow"] == {
        "estimated_queue_delay_seconds": 0.0,
        "model_load_penalty_seconds": 0.0,
        "estimated_generation_seconds": 1.0,
        "batching_penalty_seconds": 0.0,
        "recent_tokens_per_second": 10,
        "fallback_throughput_used": False,
    }


def test_placement_rejects_context_runtime_and_unknown_model():
    policy = ModelResidencyAwarePolicy(registry())
    with pytest.raises(NoEligibleWorker) as context_error:
        policy.select(request(prompt_tokens=100), [worker("worker")])
    assert context_error.value.rejected == {"worker": "context_window_exceeded"}

    cpu = worker(
        "cpu", device_type=DeviceType.CPU, total_memory_bytes=None,
        available_memory_bytes=None, resident_models={"model"},
    )
    with pytest.raises(NoEligibleWorker) as runtime_error:
        policy.select(request(), [cpu])
    assert runtime_error.value.rejected == {"cpu": "runtime_incompatible"}

    unknown = request()
    unknown.model_id = "unknown"
    with pytest.raises(UnknownModel):
        policy.select(unknown, [worker("worker")])
