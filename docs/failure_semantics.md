# Failure Semantics

## Current single-worker behavior

- Queue saturation rejects before insertion with a structured 503 response.
- A queued request that expires is removed before execution and marked timed out.
- Hugging Face generation cannot be preempted. A deadline or cancellation during generation
  is applied after the model call returns, and a cancelled request cannot become completed.
- A client disconnect cancels queued work. Active local compute may continue, but remaining
  output is discarded.
- A streaming model exception wakes the SSE consumer and emits a structured error event.
- Shutdown stops admission, rejects queued and deferred requests, and waits for active work
  up to the configured grace period.

## Worker-layer behavior

- Workers that exceed the configured heartbeat timeout are marked unhealthy and cannot
  resume by sending a heartbeat; they must explicitly re-register.
- Draining workers reject new requests and model loads. Shutdown fails queued work, unloads
  resident models, and unregisters the worker.
- Simulated GPU failures are deterministic when a failure interval is configured. They are
  test behavior and are never presented as real CUDA failures.
- CUDA construction fails clearly when PyTorch does not report an available CUDA device.
- A request whose deadline has passed before global assignment becomes `TIMED_OUT`.
- If a selected worker disappears or rejects handoff, the request returns to global
  `QUEUED` state with its attempt count retained; it is not silently duplicated.
- Runtime failure marks every request in the affected local batch failed. A malformed
  result count is treated as runtime failure.
- Heartbeat and execution loops are supervised together. A loop failure cancels its peer,
  waits for non-interruptible threaded generation to finish, then unregisters the worker.
- Concurrent requests for one cold model share one load attempt and receive the same load
  failure. Failed loads release memory reservations and never publish residency.
- LRU eviction never selects a model with queued or active requests. If no safe eviction can
  satisfy the memory estimate and safety reserve, loading fails before runtime allocation.
- Hugging Face loading is a synchronous library call and cannot be safely force-cancelled.
  The control plane rechecks the request deadline after loading and does not enqueue an
  expired request.
- A cancelled global or worker-queued request is removed before execution. An active request
  becomes terminal immediately; uninterruptible model compute may finish, but its output is
  discarded and the request cannot later become completed.
- Admission reservations are released idempotently at terminal handling. Tenant queue,
  concurrency, and token limits are changed under one lock so concurrent admissions cannot
  oversubscribe them.

## Not implemented yet

There is no crash reassignment, persistent request state, or control-plane recovery yet.
Those semantics must be implemented and tested before multi-worker mode is advertised.
Partially streamed requests will never be retried automatically.
