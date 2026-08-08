# Scheduling Policies

## Currently executable local policies

`no_batching` dequeues and completes exactly one request before taking another. It is the
serial baseline.

`dynamic_batch` opens a bounded wait window after selecting a seed. It admits compatible
requests until maximum batch size, token budget, or time is exhausted. Estimated batch
work is `prompt_tokens + max_new_tokens` per request. This is a conservative scheduling
estimate, not exact memory or execution cost.

Compatibility currently includes output length and temperature. The future worker-local
contract will add model revision, dtype, streaming mode, context constraints, and deadline
compatibility.

This is dynamic micro-batching, not continuous batching: requests cannot join a generation
call after decoding starts.

## Planned global policies

Round robin, least queue depth, residency-aware, memory-aware least-loaded, and estimated
completion time are contracts only in Phase 1. They become executable in Phase 3 and Phase
4 and must return structured candidate, rejection, and score data.
