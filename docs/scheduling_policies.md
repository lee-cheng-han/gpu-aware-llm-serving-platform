# Scheduling Policies

## Currently executable local policies

`no_batching` dequeues and completes exactly one request before taking another. It is the
serial baseline.

`dynamic_batch` opens a bounded wait window after selecting a seed. It admits compatible
requests until maximum batch size, token budget, or time is exhausted. Estimated batch
work is `prompt_tokens + max_new_tokens` per request. This is a conservative scheduling
estimate, not exact memory or execution cost.

The original gateway scheduler checks output length and temperature. Managed worker batches
also check model, output length, temperature, streaming mode, token budget, cancellation,
and deadline expiry immediately before execution. Model revision and dtype become explicit
batch keys when the model registry is connected.

This is dynamic micro-batching, not continuous batching: requests cannot join a generation
call after decoding starts.

## Executable global policies

`round_robin` rotates deterministically over eligible worker identifiers.

`least_queue_depth` selects the smallest reported local queue, breaking ties by worker ID.
Queue depth is a point-in-time heartbeat value, so it is a routing signal rather than an
exact prediction of completion time.

Both policies reject unhealthy, draining, concurrency-saturated, and non-resident workers
before scoring. Decisions report candidates, per-worker rejection reasons, scores, and the
estimated request token count. Residency-aware, memory-aware least-loaded, and estimated
completion policies are introduced with placement.
