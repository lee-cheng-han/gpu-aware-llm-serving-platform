# KV-Cache Ownership and Accounting

`KVBlockAllocator` divides configured capacity into fixed token blocks. Admission reserves
enough blocks for the validated prompt plus the request's maximum output before SSE headers
are returned. Every sequence owns a `KVBlockTable`; resize and allocation are atomic, release
is idempotent, and reference counts provide the boundary needed by prefix caching.

The allocator enforces these invariants:

- every block is exactly free or allocated;
- a block table contains no duplicate block;
- reference counts equal the number of owning tables;
- failed allocation or resize changes no ownership;
- cancellation, timeout, failure, stop, EOS, and length completion release all ownership.

High/low watermarks use hysteresis so pressure state does not flap. `/metrics` reports total,
used, and free blocks; estimated bytes; utilization; allocation failures; and pressure.
`KV_CACHE_BYTES_PER_BLOCK=737280` is the pinned SmolLM2 float32 estimate for 16 tokens. Other
models or dtypes must configure their measured/calculated value.

This allocator remains admission and ownership accounting. `TorchPagedKVCache` now maps the
same block-table concept to physical key/value pages and provides a correctness-first gather
and attention path. The Hugging Face backend still uses per-sequence `DynamicCache` tensors;
it must consume the physical pages directly before the project can claim end-to-end
PagedAttention or reduced tensor fragmentation.
