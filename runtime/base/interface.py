from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from serving_platform.domain import DeviceType, ModelDefinition, RuntimeType


@dataclass(frozen=True)
class RuntimeCapacity:
    device_type: DeviceType
    device_name: str
    total_memory_bytes: int | None
    available_memory_bytes: int | None


@dataclass(frozen=True)
class RuntimeResult:
    text: str
    input_tokens: int
    output_tokens: int
    tokenization_ms: float
    generation_ms: float
    decoding_ms: float


@dataclass(frozen=True)
class RuntimeStreamChunk:
    text: str
    output_tokens: int


class ModelRuntime(Protocol):
    """Synchronous runtime contract; workers isolate calls from their event loop."""

    runtime_type: RuntimeType
    device_type: DeviceType

    def load_model(self, model: ModelDefinition) -> None: ...
    def unload_model(self, model_id: str) -> None: ...
    def warmup_model(self, model_id: str) -> None: ...
    def is_model_loaded(self, model_id: str) -> bool: ...
    def count_prompt_tokens(self, model_id: str, prompt: str) -> int: ...
    def generate(
        self,
        model_id: str,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
    ) -> list[RuntimeResult]: ...
    def stream(
        self,
        model_id: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> Iterable[RuntimeStreamChunk]: ...
    def capacity(self) -> RuntimeCapacity: ...
