from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from runtime.base import RuntimeCapacity, RuntimeResult, RuntimeStreamChunk
from serving_platform.domain import DeviceType, ModelDefinition, RuntimeType


@dataclass(frozen=True)
class SimulatedGpuConfig:
    total_memory_bytes: int = 16 * 1024**3
    tokens_per_second: float = 100
    model_load_seconds: float = 0
    batching_efficiency: float = 0.75
    failure_every_n_calls: int | None = None
    seed: int = 0
    device_name: str = "deterministic-simulated-gpu"

    def __post_init__(self) -> None:
        if self.total_memory_bytes <= 0 or self.tokens_per_second <= 0:
            raise ValueError("simulated GPU memory and throughput must be positive")
        if not 0 <= self.batching_efficiency <= 1:
            raise ValueError("batching efficiency must be between zero and one")
        if self.failure_every_n_calls is not None and self.failure_every_n_calls <= 0:
            raise ValueError("failure interval must be positive")


class SimulatedGpuRuntime:
    """Deterministic capacity and timing model; it never performs real GPU inference."""

    runtime_type = RuntimeType.SIMULATED
    device_type = DeviceType.SIMULATED_GPU

    def __init__(
        self,
        models: Iterable[ModelDefinition],
        config: SimulatedGpuConfig | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        definitions = tuple(models)
        if any(model.runtime_type != RuntimeType.SIMULATED for model in definitions):
            raise ValueError("simulated runtime only accepts simulated model definitions")
        self.models = {model.model_id: model for model in definitions}
        if len(self.models) != len(definitions):
            raise ValueError("simulated model identifiers must be unique")
        self.config = config or SimulatedGpuConfig()
        self._sleep = sleeper
        self._loaded: set[str] = set()
        self._calls = 0

    def _definition(self, model_id: str) -> ModelDefinition:
        try:
            return self.models[model_id]
        except KeyError as exc:
            raise ValueError(f"model is not registered: {model_id}") from exc

    def load_model(self, model: ModelDefinition) -> None:
        registered = self._definition(model.model_id)
        if registered != model:
            raise ValueError("model definition does not match the registered definition")
        if model.model_id in self._loaded:
            return
        available_memory = self.capacity().available_memory_bytes
        if available_memory is None:
            raise RuntimeError("simulated GPU did not report memory capacity")
        if model.estimated_memory_bytes > available_memory:
            raise MemoryError(f"insufficient simulated GPU memory for {model.model_id}")
        self._sleep(self.config.model_load_seconds)
        self._loaded.add(model.model_id)

    def unload_model(self, model_id: str) -> None:
        self._definition(model_id)
        self._loaded.discard(model_id)

    def warmup_model(self, model_id: str) -> None:
        self._require_loaded(model_id)

    def is_model_loaded(self, model_id: str) -> bool:
        self._definition(model_id)
        return model_id in self._loaded

    def _require_loaded(self, model_id: str) -> ModelDefinition:
        model = self._definition(model_id)
        if model_id not in self._loaded:
            raise RuntimeError(f"model is not loaded: {model_id}")
        return model

    def count_prompt_tokens(self, model_id: str, prompt: str) -> int:
        self._definition(model_id)
        return len(prompt.split())

    def _before_generation(self) -> None:
        self._calls += 1
        interval = self.config.failure_every_n_calls
        if interval is not None and self._calls % interval == 0:
            raise RuntimeError("controlled simulated GPU failure")

    def _validate_generation(
        self,
        model_id: str,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
    ) -> ModelDefinition:
        model = self._require_loaded(model_id)
        if not prompts or any(not prompt.strip() for prompt in prompts):
            raise ValueError("at least one non-empty prompt is required")
        if max_new_tokens <= 0 or temperature < 0:
            raise ValueError("generation limits are invalid")
        request_sizes = [len(prompt.split()) + max_new_tokens for prompt in prompts]
        if any(size > model.max_context_tokens for size in request_sizes):
            raise ValueError("request exceeds the model context window")
        if sum(request_sizes) > model.max_batch_tokens:
            raise ValueError("batch exceeds the model token budget")
        return model

    def generate(
        self,
        model_id: str,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
    ) -> list[RuntimeResult]:
        self._validate_generation(model_id, prompts, max_new_tokens, temperature)
        self._before_generation()
        speedup = 1 + (len(prompts) - 1) * self.config.batching_efficiency
        generation_seconds = max_new_tokens * len(prompts) / (
            self.config.tokens_per_second * speedup
        )
        self._sleep(generation_seconds)
        return [self._result(model_id, prompt, max_new_tokens, temperature, generation_seconds)
                for prompt in prompts]

    def _result(
        self,
        model_id: str,
        prompt: str,
        output_tokens: int,
        temperature: float,
        generation_seconds: float,
    ) -> RuntimeResult:
        material = f"{self.config.seed}|{model_id}|{prompt}|{output_tokens}|{temperature}"
        digest = hashlib.sha256(material.encode()).digest()
        suffix = random.Random(digest).randrange(1_000_000)
        return RuntimeResult(
            text=f"[simulated:{suffix:06d}]",
            input_tokens=len(prompt.split()),
            output_tokens=output_tokens,
            tokenization_ms=0,
            generation_ms=generation_seconds * 1000,
            decoding_ms=0,
        )

    def stream(
        self,
        model_id: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> Iterable[RuntimeStreamChunk]:
        self._validate_generation(model_id, [prompt], max_new_tokens, temperature)
        self._before_generation()
        for token_number in range(1, max_new_tokens + 1):
            self._sleep(1 / self.config.tokens_per_second)
            yield RuntimeStreamChunk(f"sim-{token_number} ", token_number)

    def capacity(self) -> RuntimeCapacity:
        used = sum(self.models[model_id].estimated_memory_bytes for model_id in self._loaded)
        return RuntimeCapacity(
            self.device_type,
            self.config.device_name,
            self.config.total_memory_bytes,
            self.config.total_memory_bytes - used,
        )
