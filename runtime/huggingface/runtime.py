from collections.abc import Iterable
from importlib import import_module

from inference.worker import InferenceWorker
from runtime.base import RuntimeCapacity, RuntimeResult, RuntimeStreamChunk
from serving_platform.domain import DeviceType, ModelDefinition, RuntimeType


class HuggingFaceRuntime:
    """Compatibility adapter around the existing single-model CPU worker."""

    runtime_type = RuntimeType.HUGGINGFACE
    def __init__(
        self,
        model: ModelDefinition,
        worker: InferenceWorker | None = None,
        device_type: DeviceType = DeviceType.CPU,
        cuda_device_index: int = 0,
    ):
        if device_type not in {DeviceType.CPU, DeviceType.CUDA}:
            raise ValueError("Hugging Face runtime requires a CPU or CUDA device")
        self.model = model
        self.device_type = device_type
        self.cuda_device_index = cuda_device_index
        device = "cpu" if device_type == DeviceType.CPU else f"cuda:{cuda_device_index}"
        self.worker = worker or InferenceWorker(model.model_id, model.revision, device)

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
        if self.device_type == DeviceType.CPU:
            # System RAM and CUDA VRAM have different allocation semantics.
            return RuntimeCapacity(DeviceType.CPU, "cpu", None, None)
        torch = import_module("torch")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA worker requested but CUDA is unavailable")
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.cuda_device_index)
        name = torch.cuda.get_device_name(self.cuda_device_index)
        allocated_bytes = torch.cuda.memory_allocated(self.cuda_device_index)
        reserved_bytes = torch.cuda.memory_reserved(self.cuda_device_index)
        return RuntimeCapacity(
            DeviceType.CUDA,
            name,
            total_bytes,
            free_bytes,
            allocated_bytes,
            reserved_bytes,
        )
