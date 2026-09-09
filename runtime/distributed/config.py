from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum


class DistributedBackend(StrEnum):
    NCCL = "nccl"
    GLOO = "gloo"


@dataclass(frozen=True)
class DistributedConfig:
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: DistributedBackend = DistributedBackend.NCCL
    init_method: str = "env://"
    timeout_seconds: float = 60

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("distributed world size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("distributed rank must be within world size")
        if self.local_rank < 0:
            raise ValueError("distributed local rank must be non-negative")
        if not self.init_method:
            raise ValueError("distributed init method is required")
        if self.timeout_seconds <= 0:
            raise ValueError("distributed timeout must be positive")

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @classmethod
    def from_environment(cls) -> DistributedConfig:
        return cls(
            world_size=int(os.getenv("WORLD_SIZE", "1")),
            rank=int(os.getenv("RANK", "0")),
            local_rank=int(os.getenv("LOCAL_RANK", "0")),
            backend=DistributedBackend(os.getenv("DISTRIBUTED_BACKEND", "nccl")),
            init_method=os.getenv("DISTRIBUTED_INIT_METHOD", "env://"),
            timeout_seconds=float(os.getenv("DISTRIBUTED_TIMEOUT_SECONDS", "60")),
        )


@dataclass(frozen=True)
class TensorShard:
    rank: int
    world_size: int
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


def balanced_shard(total_size: int, rank: int, world_size: int) -> TensorShard:
    if total_size <= 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("tensor shard dimensions are invalid")
    base, remainder = divmod(total_size, world_size)
    start = rank * base + min(rank, remainder)
    size = base + int(rank < remainder)
    return TensorShard(rank, world_size, start, start + size)
