# ADR 002: Central Iteration Scheduling and Separate KV Ownership

## Context

The first token engine placed the prefill/decode loop inside each SSE generator. A single
executor prevented unsafe overlap, but requests could not form shared iterations and HTTP
consumer lifetime owned model progress. Hugging Face `DynamicCache` also offers real KV state
without a physical block allocator.

## Decision

Move model progress into one central scheduler with explicit waiting/running sets and
per-sequence event queues. Backends expose real batch prefill/decode operations. The Hugging
Face backend combines variable-length prefill with left padding and combines decode only
when physical cache lengths match. Configurable workload policies select prefill or decode;
optional prompt chunks grow the real cache incrementally.

Add a separate fixed-block allocator that reserves maximum logical sequence capacity before
streaming, maintains ownership/reference-count invariants, exposes pressure metrics, and
releases on every terminal path.

## Consequences

Concurrent compatible requests now share real forward passes and can join or leave between
iterations. Scheduler admission is memory-aware and testable without a model download.

Logical block ownership does not change the layout of Hugging Face cache tensors. The system
therefore does not claim PagedAttention, arbitrary-length physical decode batching, or GPU
fragmentation improvements. Those require a backend that maps block tables to real tensor
pages and an attention implementation that consumes them.
