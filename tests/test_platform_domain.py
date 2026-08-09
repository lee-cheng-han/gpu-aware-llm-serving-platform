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
from serving_platform.domain.models import InvalidStateTransition
from serving_platform.lifecycle import RequestLifecycle


def model_definition() -> ModelDefinition:
    return ModelDefinition(
        model_id="sshleifer/tiny-gpt2",
        revision="main",
        runtime_type=RuntimeType.HUGGINGFACE,
        estimated_memory_bytes=10,
        supported_dtypes=("float32",),
        default_dtype="float32",
        max_context_tokens=1024,
        max_batch_tokens=1024,
        supports_streaming=True,
        supports_cancellation=False,
        load_timeout_seconds=60,
        idle_eviction_seconds=300,
    )


def request_record() -> RequestRecord:
    return RequestRecord(
        request_id="req_1",
        tenant_id="development",
        model_id="sshleifer/tiny-gpt2",
        prompt="hello",
        prompt_tokens=1,
        max_new_tokens=2,
        priority=0,
        deadline=100,
        stream=False,
    )


def test_request_state_machine_accepts_valid_path():
    request = request_record()
    lifecycle = RequestLifecycle()
    for state in (
        RequestState.VALIDATED,
        RequestState.ADMITTED,
        RequestState.QUEUED,
        RequestState.ASSIGNED,
        RequestState.RUNNING,
        RequestState.COMPLETED,
    ):
        lifecycle.transition(request, state)
    assert request.terminal
    assert set(request.transition_timestamps) == {
        RequestState.RECEIVED,
        RequestState.VALIDATED,
        RequestState.ADMITTED,
        RequestState.QUEUED,
        RequestState.ASSIGNED,
        RequestState.RUNNING,
        RequestState.COMPLETED,
    }


def test_terminal_request_cannot_transition():
    request = request_record()
    request.transition(RequestState.REJECTED)
    with pytest.raises(InvalidStateTransition):
        request.transition(RequestState.VALIDATED)


def test_request_never_skips_required_admission_states():
    request = request_record()
    with pytest.raises(InvalidStateTransition):
        request.transition(RequestState.RUNNING)


def test_worker_memory_invariant():
    with pytest.raises(ValueError, match="memory accounting"):
        WorkerState(
            worker_id="worker-1",
            device_type=DeviceType.CUDA,
            device_name="test-gpu",
            total_memory_bytes=10,
            available_memory_bytes=11,
        )
    worker = WorkerState(
        worker_id="worker-1",
        device_type=DeviceType.CPU,
        device_name="cpu",
        total_memory_bytes=100,
        available_memory_bytes=50,
        health_status=HealthStatus.HEALTHY,
    )
    assert worker.available_memory_bytes <= worker.total_memory_bytes


def test_worker_allows_explicitly_unknown_cpu_memory():
    worker = WorkerState(
        worker_id="cpu-worker",
        device_type=DeviceType.CPU,
        device_name="cpu",
        total_memory_bytes=None,
        available_memory_bytes=None,
    )
    assert worker.available_memory_bytes is None


def test_model_default_dtype_must_be_supported():
    definition = model_definition()
    assert definition.default_dtype == "float32"
    with pytest.raises(ValueError, match="default_dtype"):
        ModelDefinition(**{
            **vars(definition),
            "default_dtype": "float16",
        })
