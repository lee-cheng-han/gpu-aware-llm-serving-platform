from collections.abc import Iterable

from inference.worker import InferenceWorker
from runtime.base import RuntimeCapacity, RuntimeResult, RuntimeStreamChunk
from serving_platform.domain import DeviceType, ModelDefinition, RuntimeType


class HuggingFaceRuntime:
    """Compatibility adapter around the existing single-model CPU worker."""

    runtime_type = RuntimeType.HUGGINGFACE
    device_type = DeviceType.CPU

    def __init__(self, model: ModelDefinition, worker: InferenceWorker | None = None):
        self.model = model
        self.worker = worker or InferenceWorker(model.model_id, model.revision)

    def _require_model(self, model_id: str) -> None:
        if model_id != self.model.model_id:
            raise ValueError(f"model is not registered: {model_id}")

    def load_model(self, model: ModelDefinition) -> None:
        if model.model_id != self.model.model_id or model.revision != self.model.revision:
            raise ValueError("runtime only loads its registered model and revision")
        self.worker.warmup()

    def unload_model(self, model_id: str) -> None:
        self._require_model(model_id)
        self.worker.unload()

    def warmup_model(self, model_id: str) -> None:
        self._require_model(model_id)
        self.worker.warmup()

    def is_model_loaded(self, model_id: str) -> bool:
        self._require_model(model_id)
        return self.worker.is_ready

    def count_prompt_tokens(self, model_id: str, prompt: str) -> int:
        self._require_model(model_id)
        return self.worker.count_prompt_tokens(prompt)

    def generate(
        self,
        model_id: str,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
    ) -> list[RuntimeResult]:
        self._require_model(model_id)
        return [RuntimeResult(**vars(result)) for result in self.worker.generate_batch(
            prompts, max_new_tokens, temperature
        )]

    def stream(
        self,
        model_id: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> Iterable[RuntimeStreamChunk]:
        self._require_model(model_id)
        for chunk in self.worker.stream(prompt, max_new_tokens, temperature):
            yield RuntimeStreamChunk(chunk.text, chunk.output_tokens)

    def capacity(self) -> RuntimeCapacity:
        # CPU RAM is intentionally reported as unknown rather than pretending it
        # has CUDA-like allocation semantics. Phase 2 adds a worker capacity probe.
        return RuntimeCapacity(DeviceType.CPU, "cpu", None, None)
