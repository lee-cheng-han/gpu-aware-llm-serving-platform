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

Both baseline policies reject unhealthy, draining, concurrency-saturated, and non-resident
workers before scoring. Decisions report candidates, per-worker rejection reasons, scores,
scoring inputs, and the estimated request token count.

## Placement policies

`model_residency_aware` prefers a resident model, then queue depth. A cold worker remains a
candidate only when its runtime matches and known available memory can fit the registered
model after the configured safety reserve.

`memory_aware_least_loaded` combines projected memory utilization, active-concurrency
utilization, and queue depth. For a resident CPU model with unknown system memory, it uses a
neutral `0.5` memory-utilization input rather than inventing GPU-like RAM accounting. An
unknown-memory worker is never selected for a cold load.

`estimated_completion_time` uses this deliberately approximate model:

```text
queue delay       = queue depth * request tokens / (throughput * max concurrency)
load penalty      = 0 when resident, otherwise the registered load timeout
generation time   = request tokens / recent throughput
batching penalty  = generation time * configured ratio * concurrency utilization
completion score  = queue delay + load penalty + generation time + batching penalty
```

`request tokens` means prompt tokens plus maximum new tokens. Missing throughput uses a
configurable conservative fallback. A worker is rejected when the estimate exceeds the
request deadline. These values are scheduling heuristics, not exact latency predictions.
