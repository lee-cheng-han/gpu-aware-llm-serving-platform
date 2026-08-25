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

This is real admission and ownership accounting, not physical paged tensor storage. The
Hugging Face backend still uses per-sequence `DynamicCache` tensors. A later backend must map
these block tables to physical key/value pages and implement attention over them before the
project can claim PagedAttention or reduced tensor fragmentation.
