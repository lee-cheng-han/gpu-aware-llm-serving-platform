from types import SimpleNamespace

import pytest
from conftest import FakeWorker

from runtime.huggingface import HuggingFaceRuntime
from serving_platform.domain import DeviceType, ModelDefinition, RuntimeType


def definition() -> ModelDefinition:
    return ModelDefinition(
        model_id="registered-model",
        revision="main",
        runtime_type=RuntimeType.HUGGINGFACE,
        estimated_memory_bytes=1,
        supported_dtypes=("float32",),
        default_dtype="float32",
        max_context_tokens=128,
        max_batch_tokens=128,
        supports_streaming=True,
        supports_cancellation=False,
        load_timeout_seconds=30,
        idle_eviction_seconds=300,
    )


def test_huggingface_adapter_preserves_existing_worker_results():
    runtime = HuggingFaceRuntime(definition(), worker=FakeWorker())
    results = runtime.generate("registered-model", ["hello world"], 2, 0)
    assert results[0].input_tokens == 2
    assert results[0].output_tokens == 2
    assert list(runtime.stream("registered-model", "hello", 2, 0))[0].output_tokens == 2
    assert runtime.capacity().total_memory_bytes is None


def test_huggingface_adapter_rejects_unregistered_model_before_loading():
    runtime = HuggingFaceRuntime(definition(), worker=FakeWorker())
    with pytest.raises(ValueError, match="not registered"):
        runtime.count_prompt_tokens("arbitrary/path", "hello")


def test_cuda_capacity_uses_pytorch_device_apis(monkeypatch):
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda index: (6_000, 8_000),
        get_device_name=lambda index: f"fake-cuda-{index}",
        memory_allocated=lambda index: 1_000,
        memory_reserved=lambda index: 1_500,
    )
    monkeypatch.setattr(
        "runtime.huggingface.runtime.import_module",
        lambda name: SimpleNamespace(cuda=fake_cuda),
    )
    runtime = HuggingFaceRuntime(
        definition(), worker=FakeWorker(), device_type=DeviceType.CUDA, cuda_device_index=2
    )

    capacity = runtime.capacity()
    assert capacity.device_name == "fake-cuda-2"
    assert capacity.total_memory_bytes == 8_000
    assert capacity.available_memory_bytes == 6_000
    assert capacity.allocated_memory_bytes == 1_000
    assert capacity.reserved_memory_bytes == 1_500


def test_cuda_capacity_fails_when_cuda_is_unavailable(monkeypatch):
    fake_cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setattr(
        "runtime.huggingface.runtime.import_module",
        lambda name: SimpleNamespace(cuda=fake_cuda),
    )
    runtime = HuggingFaceRuntime(
        definition(), worker=FakeWorker(), device_type=DeviceType.CUDA
    )
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        runtime.capacity()
