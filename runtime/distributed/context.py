from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from importlib import import_module
from typing import Any

from runtime.distributed.config import DistributedBackend, DistributedConfig


@dataclass(frozen=True)
class DistributedStatus:
    enabled: bool
    initialized: bool
    owns_process_group: bool
    backend: str
    world_size: int
    rank: int
    local_rank: int


class DistributedContext:
    """Strict lifecycle wrapper around a torch.distributed process group."""

    def __init__(
        self,
        config: DistributedConfig,
        module_loader: Callable[[str], Any] = import_module,
    ) -> None:
        self.config = config
        self._module_loader = module_loader
        self._torch: Any | None = None
        self._distributed: Any | None = None
        self._initialized = False
        self._owns_process_group = False

    def initialize(self) -> DistributedStatus:
        if self._initialized:
            return self.status()
        if not self.config.enabled:
            self._initialized = True
            return self.status()
        torch = self._module_loader("torch")
        distributed = torch.distributed
        if not distributed.is_available():
            raise RuntimeError("torch.distributed is unavailable")
        if self.config.backend is DistributedBackend.NCCL:
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL requires CUDA; CPU fallback is forbidden")
            if not distributed.is_nccl_available():
                raise RuntimeError("this PyTorch build does not provide NCCL")
            if self.config.local_rank >= torch.cuda.device_count():
                raise RuntimeError("LOCAL_RANK does not identify an available CUDA device")
            torch.cuda.set_device(self.config.local_rank)
        if distributed.is_initialized():
            if (
                distributed.get_world_size() != self.config.world_size
                or distributed.get_rank() != self.config.rank
                or distributed.get_backend() != self.config.backend.value
            ):
                raise RuntimeError("existing process group does not match configuration")
        else:
            distributed.init_process_group(
                backend=self.config.backend.value,
                init_method=self.config.init_method,
                rank=self.config.rank,
                world_size=self.config.world_size,
                timeout=timedelta(seconds=self.config.timeout_seconds),
            )
            self._owns_process_group = True
        self._torch = torch
        self._distributed = distributed
        self._initialized = True
        return self.status()

    def all_reduce_sum(self, tensor: Any) -> Any:
        self._require_initialized()
        if not self.config.enabled:
            return tensor
        distributed = self._distributed
        assert distributed is not None
        distributed.all_reduce(tensor, op=distributed.ReduceOp.SUM)
        return tensor

    def broadcast(self, tensor: Any, source_rank: int = 0) -> Any:
        self._require_initialized()
        if not 0 <= source_rank < self.config.world_size:
            raise ValueError("broadcast source rank is outside the process group")
        if self.config.enabled:
            distributed = self._distributed
            assert distributed is not None
            distributed.broadcast(tensor, src=source_rank)
        return tensor

    def barrier(self) -> None:
        self._require_initialized()
        if self.config.enabled:
            distributed = self._distributed
            assert distributed is not None
            distributed.barrier()

    def all_gather_objects(self, value: Any) -> tuple[Any, ...]:
        self._require_initialized()
        if not self.config.enabled:
            return (value,)
        distributed = self._distributed
        assert distributed is not None
        gathered: list[Any] = [None] * self.config.world_size
        distributed.all_gather_object(gathered, value)
        return tuple(gathered)

    def status(self) -> DistributedStatus:
        return DistributedStatus(
            self.config.enabled,
            self._initialized,
            self._owns_process_group,
            self.config.backend.value,
            self.config.world_size,
            self.config.rank,
            self.config.local_rank,
        )

    def shutdown(self) -> None:
        if self._owns_process_group and self._distributed is not None:
            self._distributed.destroy_process_group()
        self._initialized = False
        self._owns_process_group = False
        self._torch = None
        self._distributed = None

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("distributed context is not initialized")
