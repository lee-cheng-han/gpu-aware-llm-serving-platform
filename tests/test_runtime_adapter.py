import pytest
from conftest import FakeWorker

from runtime.huggingface import HuggingFaceRuntime
from serving_platform.domain import ModelDefinition, RuntimeType


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
