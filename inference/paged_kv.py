from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from inference.kv_cache import KVBlockAllocator, KVBlockTable


@dataclass(frozen=True)
class PagedKVConfig:
    total_blocks: int
    block_size_tokens: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if min(
            self.total_blocks,
            self.block_size_tokens,
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
        ) <= 0:
            raise ValueError("paged KV dimensions must be positive")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("paged KV device must be cpu, cuda, or cuda:<index>")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("paged KV dtype must be float16, bfloat16, or float32")


@dataclass(frozen=True)
class PagedSequence:
    sequence_id: str
    block_table: KVBlockTable
    tokens: int


@dataclass(frozen=True)
class PhysicalPagedKVSnapshot:
    total_blocks: int
    used_blocks: int
    free_blocks: int
    active_sequences: int
    stored_tokens: int
    allocated_bytes: int
    device: str
    dtype: str


class TorchPagedKVCache:
    """Physical K/V tensor pages and a correctness-first paged attention path.

    The layout is [layer, block, KV head, token-in-block, head dimension].
    PyTorch is imported lazily so control-plane and download-free tests do not need it.
    """

    def __init__(self, config: PagedKVConfig) -> None:
        self.config = config
        self._torch = import_module("torch")
        if config.device.startswith("cuda") and not self._torch.cuda.is_available():
            raise RuntimeError("CUDA paged KV was requested but CUDA is unavailable")
        try:
            dtype = getattr(self._torch, config.dtype)
        except AttributeError as exc:
            raise RuntimeError(f"PyTorch does not expose dtype {config.dtype}") from exc
        shape = (
            config.num_layers,
            config.total_blocks,
            config.num_kv_heads,
            config.block_size_tokens,
            config.head_dim,
        )
        self.key_pages = self._torch.empty(shape, dtype=dtype, device=config.device)
        self.value_pages = self._torch.empty(shape, dtype=dtype, device=config.device)
        self.device = str(self.key_pages.device)
        bytes_per_block = (
            2
            * config.num_layers
            * config.num_kv_heads
            * config.block_size_tokens
            * config.head_dim
            * self.key_pages.element_size()
        )
        self.allocator = KVBlockAllocator(
            config.total_blocks,
            config.block_size_tokens,
            bytes_per_block=bytes_per_block,
        )
        self._lengths: dict[str, int] = {}

    def allocate(self, sequence_id: str, token_capacity: int) -> PagedSequence:
        if token_capacity <= 0:
            raise ValueError("paged KV token capacity must be positive")
        self.allocator.allocate(sequence_id, token_capacity)
        self._lengths[sequence_id] = 0
        return PagedSequence(sequence_id, self.allocator.table(sequence_id), 0)

    def append(
        self,
        sequence_id: str,
        keys: tuple[Any, ...],
        values: tuple[Any, ...],
    ) -> PagedSequence:
        if len(keys) != self.config.num_layers or len(values) != self.config.num_layers:
            raise ValueError("one key and value tensor are required per model layer")
        try:
            start = self._lengths[sequence_id]
        except KeyError as exc:
            raise KeyError(f"unknown paged KV sequence: {sequence_id}") from exc
        token_counts = {int(tensor.shape[1]) for tensor in (*keys, *values)}
        expected_shape = (self.config.num_kv_heads, self.config.head_dim)
        if len(token_counts) != 1:
            raise ValueError("all paged KV tensors must contain the same token count")
        token_count = token_counts.pop()
        if token_count <= 0:
            raise ValueError("paged KV append requires at least one token")
        for tensor in (*keys, *values):
            if int(tensor.shape[0]) != expected_shape[0] or int(tensor.shape[2]) != expected_shape[1]:
                raise ValueError("paged KV tensor shape must be [KV heads, tokens, head dim]")
            if str(tensor.device) != self.device:
                raise ValueError("paged KV tensors must be on the configured device")
            if tensor.dtype != self.key_pages.dtype:
                raise ValueError("paged KV tensors must use the configured dtype")
        table = self.allocator.table(sequence_id)
        end = start + token_count
        if end > table.token_capacity:
            raise ValueError("paged KV append exceeds the reserved sequence capacity")
        source_offset = 0
        while source_offset < token_count:
            logical_position = start + source_offset
            table_offset, page_offset = divmod(
                logical_position, self.config.block_size_tokens
            )
            page_id = table.block_ids[table_offset]
            count = min(
                token_count - source_offset,
                self.config.block_size_tokens - page_offset,
            )
            for layer in range(self.config.num_layers):
                self.key_pages[
                    layer,
                    page_id,
                    :,
                    page_offset : page_offset + count,
                    :,
                ].copy_(keys[layer][:, source_offset : source_offset + count, :])
                self.value_pages[
                    layer,
                    page_id,
                    :,
                    page_offset : page_offset + count,
                    :,
                ].copy_(values[layer][:, source_offset : source_offset + count, :])
            source_offset += count
        self._lengths[sequence_id] = end
        return PagedSequence(sequence_id, table, end)

    def materialize(
        self,
        layer: int,
        sequence_ids: tuple[str, ...],
    ) -> tuple[Any, Any, Any]:
        """Gather different-length block tables into one padded attention batch."""
        if not 0 <= layer < self.config.num_layers:
            raise ValueError("paged KV layer is out of range")
        if not sequence_ids:
            raise ValueError("at least one sequence is required")
        if len(sequence_ids) != len(set(sequence_ids)):
            raise ValueError("paged KV batch contains duplicate sequences")
        try:
            lengths = [self._lengths[sequence_id] for sequence_id in sequence_ids]
        except KeyError as exc:
            raise KeyError(f"unknown paged KV sequence: {exc.args[0]}") from exc
        max_tokens = max(lengths)
        if max_tokens <= 0:
            raise ValueError("paged attention requires stored KV tokens")
        shape = (
            len(sequence_ids),
            self.config.num_kv_heads,
            max_tokens,
            self.config.head_dim,
        )
        keys = self._torch.zeros(
            shape, dtype=self.key_pages.dtype, device=self.config.device
        )
        values = self._torch.zeros_like(keys)
        mask = self._torch.zeros(
            (len(sequence_ids), 1, 1, max_tokens),
            dtype=self._torch.bool,
            device=self.config.device,
        )
        for row, (sequence_id, length) in enumerate(zip(sequence_ids, lengths, strict=True)):
            table = self.allocator.table(sequence_id)
            destination = 0
            for page_id in table.block_ids:
                count = min(self.config.block_size_tokens, length - destination)
                if count <= 0:
                    break
                keys[row, :, destination : destination + count, :].copy_(
                    self.key_pages[layer, page_id, :, :count, :]
                )
                values[row, :, destination : destination + count, :].copy_(
                    self.value_pages[layer, page_id, :, :count, :]
                )
                destination += count
            mask[row, :, :, :length] = True
        return keys, values, mask

    def attention(
        self,
        layer: int,
        sequence_ids: tuple[str, ...],
        query: Any,
        *,
        scale: float | None = None,
    ) -> Any:
        """Reference paged attention using a block-table gather and PyTorch SDPA."""
        keys, values, mask = self.materialize(layer, sequence_ids)
        if int(query.shape[0]) != len(sequence_ids):
            raise ValueError("query batch size must match the paged sequence batch")
        if int(query.shape[1]) != self.config.num_kv_heads:
            raise ValueError("reference attention requires equal query and KV head counts")
        if int(query.shape[3]) != self.config.head_dim:
            raise ValueError("query head dimension does not match paged KV storage")
        if str(query.device) != self.device or query.dtype != self.key_pages.dtype:
            raise ValueError("query must use the paged KV device and dtype")
        return self._torch.nn.functional.scaled_dot_product_attention(
            query,
            keys,
            values,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )

    def release(self, sequence_id: str) -> bool:
        self._lengths.pop(sequence_id, None)
        return self.allocator.release(sequence_id)

    def sequence(self, sequence_id: str) -> PagedSequence:
        try:
            tokens = self._lengths[sequence_id]
        except KeyError as exc:
            raise KeyError(f"unknown paged KV sequence: {sequence_id}") from exc
        return PagedSequence(sequence_id, self.allocator.table(sequence_id), tokens)

    def assert_invariants(self) -> None:
        self.allocator.assert_invariants()
        for sequence_id, tokens in self._lengths.items():
            if tokens > self.allocator.table(sequence_id).token_capacity:
                raise AssertionError("paged KV sequence exceeds its block table")

    def snapshot(self) -> PhysicalPagedKVSnapshot:
        allocation = self.allocator.snapshot()
        return PhysicalPagedKVSnapshot(
            allocation.total_blocks,
            allocation.used_blocks,
            allocation.free_blocks,
            len(self._lengths),
            sum(self._lengths.values()),
            self.key_pages.numel() * self.key_pages.element_size()
            + self.value_pages.numel() * self.value_pages.element_size(),
            self.device,
            self.config.dtype,
        )
