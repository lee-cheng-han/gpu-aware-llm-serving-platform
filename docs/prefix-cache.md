# Prefix Cache

The streaming token engine has a bounded exact-prefix cache. After a real prefill, the
Hugging Face backend copies the immutable `DynamicCache`, attention mask, cache position,
and next-token logits before decode mutates the live sequence. A later request with the
same tokenized prompt restores an independent copy, samples using its own parameters, and
starts decode without another prefill forward pass.

This is real prefill reuse, not a simulated cache hit. It is deliberately narrower than a
radix-tree prefix cache: only complete token-sequence matches are reusable. Cache keys do
not retain raw prompt text. A model backend instance has one model and revision, so entries
cannot cross model revisions.

Two limits bound retained state:

- `PREFIX_CACHE_MAX_ENTRIES` limits LRU entries (default `32`).
- `PREFIX_CACHE_MAX_TOKENS` limits the sum of cached prompt tokens (default `2048`).
- Setting both values to `0` disables the cache.

Entries larger than the token budget bypass caching. Least-recently-used entries are
evicted until both bounds hold. Shutdown drops every snapshot. Metrics expose entries,
tokens, hits, misses, hit rate, insertions, evictions, and bypasses in JSON and Prometheus
formats.

The cache currently clones framework-managed KV tensors. It does not share physical pages,
perform longest-prefix matching, or use copy-on-write blocks. Those optimizations depend on
the unfinished paged-attention backend. Run the optional pinned-model test to verify that a
restored greedy sequence produces the same first and next token as the original sequence.
