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
- A reliability supervisor periodically expires stale workers and removes their local
  dispatch handles. Assigned requests that have never entered `RUNNING` or `STREAMING` may
  return to the global fair queue while retaining their attempt count.
- Requests that started or streamed are never automatically retried after worker loss; they
  fail to avoid duplicate inference. Retry budget exhaustion fails the request, while an
  expired original deadline produces `TIMED_OUT`. Every recovery decision records a reason.
- The Redis-compatible request store persists lifecycle metadata across process restarts.
  Prompts are replaced with a redaction marker and generated output is never stored. A
  reconstructed record is marked as lacking executable payload and fails clearly instead
  of accidentally scheduling the marker as a prompt.
- Request lookup is tenant-scoped. Missing requests and requests owned by another tenant
  both return the same not-found response so the endpoint does not leak request existence.
- Local HTTP worker calls have bounded timeouts. Authentication failures, connection
  failures, invalid worker identity, and non-success responses become transport errors;
  rejected handoffs return the request to global `QUEUED` state under the same rules as an
  in-process worker rejection.

## Not implemented yet

The default FastAPI generation path does not yet run through the multi-worker control plane,
and its default request store remains in memory. Redis client construction, deployment,
high availability, and migrations are operator concerns; the repository supplies the
adapter rather than running Redis. Recovery updates are not a distributed transaction with
queue insertion, so a production deployment still needs atomic persistence/queue claiming
and leader coordination. Partially streamed requests are intentionally never retried.
