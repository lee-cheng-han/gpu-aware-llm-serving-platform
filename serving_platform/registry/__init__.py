from serving_platform.registry.interfaces import ModelRegistry, WorkerRegistry

__all__ = ["ModelRegistry", "WorkerRegistry"]
from serving_platform.registry.in_memory import InMemoryWorkerRegistry

__all__ = ["InMemoryWorkerRegistry"]
