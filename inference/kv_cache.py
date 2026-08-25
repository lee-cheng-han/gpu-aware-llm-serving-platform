from __future__ import annotations

import math
from dataclasses import dataclass
from threading import Lock


class KVCacheCapacityError(RuntimeError):
    pass


class KVCacheOwnershipError(RuntimeError):
    pass


@dataclass(frozen=True)
class KVCacheSnapshot:
    total_blocks: int
    used_blocks: int
    free_blocks: int
    block_size_tokens: int
    bytes_per_block: int
    bytes_used: int
    utilization: float
    allocation_failures: int
    pressure: bool


@dataclass(frozen=True)
class KVBlockTable:
    owner_id: str
    block_ids: tuple[int, ...]
    block_size_tokens: int

    @property
    def token_capacity(self) -> int:
        return len(self.block_ids) * self.block_size_tokens


class KVBlockAllocator:
    """Atomic fixed-block allocator with explicit ownership and reference counts."""

    def __init__(
        self,
        total_blocks: int,
        block_size_tokens: int,
        bytes_per_block: int = 1,
        high_watermark: float = 0.9,
        low_watermark: float = 0.7,
    ) -> None:
        if min(total_blocks, block_size_tokens, bytes_per_block) <= 0:
            raise ValueError("KV block dimensions must be positive")
        if not 0 < low_watermark < high_watermark <= 1:
            raise ValueError("KV watermarks must satisfy 0 < low < high <= 1")
        self.total_blocks = total_blocks
        self.block_size_tokens = block_size_tokens
        self.bytes_per_block = bytes_per_block
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        self._free = list(range(total_blocks - 1, -1, -1))
        self._owners: dict[str, list[int]] = {}
        self._references: dict[int, int] = {}
        self._allocation_failures = 0
        self._pressure = False
        self._lock = Lock()

    def blocks_for_tokens(self, tokens: int) -> int:
        if tokens < 0:
            raise ValueError("token capacity must be non-negative")
        return math.ceil(tokens / self.block_size_tokens)

    def allocate(self, owner_id: str, token_capacity: int) -> tuple[int, ...]:
        if not owner_id:
            raise ValueError("KV owner identifier is required")
        required = self.blocks_for_tokens(token_capacity)
        with self._lock:
            if owner_id in self._owners:
                raise KVCacheOwnershipError(f"KV owner already exists: {owner_id}")
            if required > len(self._free):
                self._allocation_failures += 1
                self._update_pressure()
                raise KVCacheCapacityError(
                    f"KV cache requires {required} blocks but only {len(self._free)} are free"
                )
            blocks = [self._free.pop() for _ in range(required)]
            self._owners[owner_id] = blocks
            for block_id in blocks:
                self._references[block_id] = 1
            self._update_pressure()
            return tuple(blocks)

    def retain(self, owner_id: str, block_ids: tuple[int, ...]) -> None:
        """Attach shared blocks to an owner; used by the later prefix-cache layer."""
        if not owner_id or not block_ids:
            raise ValueError("owner and shared KV blocks are required")
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("shared KV block identifiers must be unique")
        with self._lock:
            if owner_id in self._owners:
                raise KVCacheOwnershipError(f"KV owner already exists: {owner_id}")
            if any(block_id not in self._references for block_id in block_ids):
                raise KVCacheOwnershipError("cannot retain an unallocated KV block")
            self._owners[owner_id] = list(block_ids)
            for block_id in block_ids:
                self._references[block_id] += 1
            self._update_pressure()

    def resize(self, owner_id: str, token_capacity: int) -> tuple[int, ...]:
        required = self.blocks_for_tokens(token_capacity)
        with self._lock:
            try:
                blocks = self._owners[owner_id]
            except KeyError as exc:
                raise KVCacheOwnershipError(f"unknown KV owner: {owner_id}") from exc
            if any(self._references[block_id] != 1 for block_id in blocks):
                raise KVCacheOwnershipError("cannot resize a shared KV block table")
            additional = required - len(blocks)
            if additional > len(self._free):
                self._allocation_failures += 1
                self._update_pressure()
                raise KVCacheCapacityError(
                    f"KV cache requires {additional} more blocks but only "
                    f"{len(self._free)} are free"
                )
            if additional > 0:
                new_blocks = [self._free.pop() for _ in range(additional)]
                blocks.extend(new_blocks)
                for block_id in new_blocks:
                    self._references[block_id] = 1
            elif additional < 0:
                for block_id in blocks[required:]:
                    del self._references[block_id]
                    self._free.append(block_id)
                del blocks[required:]
            self._update_pressure()
            return tuple(blocks)

    def release(self, owner_id: str) -> bool:
        with self._lock:
            blocks = self._owners.pop(owner_id, None)
            if blocks is None:
                return False
            for block_id in blocks:
                references = self._references[block_id] - 1
                if references:
                    self._references[block_id] = references
                else:
                    del self._references[block_id]
                    self._free.append(block_id)
            self._update_pressure()
            return True

    def block_table(self, owner_id: str) -> tuple[int, ...]:
        with self._lock:
            try:
                return tuple(self._owners[owner_id])
            except KeyError as exc:
                raise KVCacheOwnershipError(f"unknown KV owner: {owner_id}") from exc

    def table(self, owner_id: str) -> KVBlockTable:
        return KVBlockTable(owner_id, self.block_table(owner_id), self.block_size_tokens)

    def assert_invariants(self) -> None:
        with self._lock:
            free = set(self._free)
            allocated = set(self._references)
            if len(free) != len(self._free) or free & allocated:
                raise AssertionError("KV free and allocated block sets overlap")
            if free | allocated != set(range(self.total_blocks)):
                raise AssertionError("KV allocator lost or invented a block")
            expected: dict[int, int] = {}
            for blocks in self._owners.values():
                if len(blocks) != len(set(blocks)):
                    raise AssertionError("one KV table references a block twice")
                for block_id in blocks:
                    expected[block_id] = expected.get(block_id, 0) + 1
            if expected != self._references:
                raise AssertionError("KV reference counts do not match block tables")

    def snapshot(self) -> KVCacheSnapshot:
        with self._lock:
            used = len(self._references)
            return KVCacheSnapshot(
                total_blocks=self.total_blocks,
                used_blocks=used,
                free_blocks=len(self._free),
                block_size_tokens=self.block_size_tokens,
                bytes_per_block=self.bytes_per_block,
                bytes_used=used * self.bytes_per_block,
                utilization=used / self.total_blocks,
                allocation_failures=self._allocation_failures,
                pressure=self._pressure,
            )

    def _update_pressure(self) -> None:
        utilization = len(self._references) / self.total_blocks
        if not self._pressure and utilization >= self.high_watermark:
            self._pressure = True
        elif self._pressure and utilization <= self.low_watermark:
            self._pressure = False


class KVCacheManager(KVBlockAllocator):
    """Semantic cache-management boundary over the fixed-block allocator."""
