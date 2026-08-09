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

## Not implemented yet

There is no crash reassignment, model-load reservation, persistent request state, or
control-plane recovery yet. Those semantics must be implemented and tested before
multi-worker mode is advertised. Partially streamed requests will never be retried
automatically.
