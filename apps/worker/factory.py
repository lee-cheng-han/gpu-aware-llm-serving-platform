from __future__ import annotations

from apps.worker.service import ManagedWorker
from runtime.simulated_gpu import SimulatedGpuConfig, SimulatedGpuRuntime
from serving_platform.domain import ModelDefinition, RuntimeType
from serving_platform.registry import InMemoryWorkerRegistry


def local_simulated_model() -> ModelDefinition:
    """Return the fixed, zero-cost model contract exposed by local worker processes."""
    return ModelDefinition(
        model_id="local-simulated-model",
        revision="deterministic-v1",
        runtime_type=RuntimeType.SIMULATED,
        estimated_memory_bytes=256 * 1024**2,
        supported_dtypes=("float16",),
        default_dtype="float16",
        max_context_tokens=1024,
        max_batch_tokens=1024,
        supports_streaming=False,
        supports_cancellation=True,
        load_timeout_seconds=5,
        idle_eviction_seconds=300,
    )


def create_local_simulated_worker(
    worker_id: str,
    tokens_per_second: float = 10_000,
) -> tuple[ManagedWorker, ModelDefinition]:
    model = local_simulated_model()
    registry = InMemoryWorkerRegistry()
    runtime = SimulatedGpuRuntime(
        [model],
        SimulatedGpuConfig(
            total_memory_bytes=4 * 1024**3,
            tokens_per_second=tokens_per_second,
            device_name=f"{worker_id}-simulated-gpu",
        ),
    )
    return ManagedWorker(worker_id, runtime, registry, max_batch_tokens=1024), model
