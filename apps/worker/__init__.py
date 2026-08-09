"""Worker application package; worker registration is introduced in Phase 2."""
from apps.worker.service import (
    CpuHuggingFaceWorker,
    CudaHuggingFaceWorker,
    ManagedWorker,
    Worker,
    create_huggingface_worker,
)

__all__ = [
    "CpuHuggingFaceWorker",
    "CudaHuggingFaceWorker",
    "ManagedWorker",
    "Worker",
    "create_huggingface_worker",
]
