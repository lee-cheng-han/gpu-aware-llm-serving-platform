# Prefix Cache

The streaming token engine has a bounded token-radix prefix cache. After a real prefill, the
Hugging Face backend copies the immutable `DynamicCache`, attention mask, cache position,
and next-token logits before decode mutates the live sequence. A later request restores the
longest cached token prefix into an independent sequence. An exact match starts decode
without another prefill pass; a partial match submits only the unmatched prompt suffix to a
real forward pass before decoding with request-specific sampling parameters.

This is real prefill reuse, not a simulated cache hit. Cache keys contain token IDs rather
than raw prompt text. A model backend instance has one model and revision, so entries cannot
cross model revisions.

Two limits bound retained state:

- `PREFIX_CACHE_MAX_ENTRIES` limits LRU entries (default `32`).
- `PREFIX_CACHE_MAX_TOKENS` limits the sum of cached prompt tokens (default `2048`).
- Setting both values to `0` disables the cache.

Entries larger than the token budget bypass caching. Least-recently-used entries are
evicted until both bounds hold. Shutdown drops every snapshot. Metrics expose entries,
tokens, hits, exact hits, partial hits, matched tokens, misses, hit rate, insertions,
evictions, and bypasses in JSON and Prometheus formats.

The cache currently clones framework-managed KV tensors. It does not share physical pages,
use copy-on-write blocks, or deduplicate tensor storage between snapshots. Those physical
optimizations depend on the unfinished paged-attention backend. Run the optional pinned-model
tests to verify exact-restore parity and that partial-prefix reuse matches a full greedy
prefill.
