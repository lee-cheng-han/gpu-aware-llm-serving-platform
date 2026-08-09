import pytest

from runtime.simulated_gpu import SimulatedGpuConfig, SimulatedGpuRuntime
from serving_platform.domain import ModelDefinition, RuntimeType


def model(model_id: str = "sim-model", memory: int = 100) -> ModelDefinition:
    return ModelDefinition(
        model_id=model_id,
        revision="test",
        runtime_type=RuntimeType.SIMULATED,
        estimated_memory_bytes=memory,
        supported_dtypes=("float16",),
        default_dtype="float16",
        max_context_tokens=1024,
        max_batch_tokens=2048,
        supports_streaming=True,
        supports_cancellation=True,
        load_timeout_seconds=10,
        idle_eviction_seconds=60,
    )


def test_simulated_gpu_is_deterministic_and_models_batch_efficiency():
    delays: list[float] = []
    definition = model()
    runtime = SimulatedGpuRuntime(
        [definition],
        SimulatedGpuConfig(
            total_memory_bytes=1000,
            tokens_per_second=10,
            batching_efficiency=1,
            seed=42,
        ),
        sleeper=delays.append,
    )
    runtime.load_model(definition)
    first = runtime.generate(definition.model_id, ["hello", "world"], 10, 0)
    second = runtime.generate(definition.model_id, ["hello", "world"], 10, 0)

    assert [result.text for result in first] == [result.text for result in second]
    assert delays == [0, 1.0, 1.0]
    assert runtime.capacity().available_memory_bytes == 900


def test_simulated_gpu_enforces_memory_and_controlled_failures():
    definition = model(memory=101)
    runtime = SimulatedGpuRuntime(
        [definition],
        SimulatedGpuConfig(total_memory_bytes=100, failure_every_n_calls=1),
        sleeper=lambda _: None,
    )
    with pytest.raises(MemoryError):
        runtime.load_model(definition)

    runnable = model(memory=100)
    runtime = SimulatedGpuRuntime(
        [runnable],
        SimulatedGpuConfig(total_memory_bytes=100, failure_every_n_calls=1),
        sleeper=lambda _: None,
    )
    runtime.load_model(runnable)
    with pytest.raises(RuntimeError, match="controlled simulated"):
        runtime.generate(runnable.model_id, ["hello"], 1, 0)


def test_simulated_gpu_enforces_context_and_batch_token_limits():
    definition = model()
    runtime = SimulatedGpuRuntime([definition], sleeper=lambda _: None)
    runtime.load_model(definition)

    with pytest.raises(ValueError, match="context window"):
        runtime.generate(definition.model_id, ["word " * 1024], 1, 0)
    with pytest.raises(ValueError, match="batch.*token budget"):
        runtime.generate(definition.model_id, ["word " * 700] * 3, 1, 0)
