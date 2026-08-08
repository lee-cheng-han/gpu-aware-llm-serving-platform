# Architecture

## Current executable architecture

Phase 1 deliberately preserves the proven single-worker path:

```text
Client
  |
  v
FastAPI gateway (`apps.gateway`)
  |
  +--> validation and bounded HTTP concurrency
  +--> single-request SSE path --------------------+
  |                                                |
  v                                                v
bounded asyncio queue                         shared model lock
  |                                                ^
  v                                                |
serial or dynamic local scheduler ---> Hugging Face CPU worker
  |
  v
bounded in-memory metrics
```

The dynamic scheduler batches requests with compatible output length and temperature,
honors batch-size and estimated-token limits, and uses a deferred deque to avoid repeated
queue insertion. Blocking model calls run outside the event loop. Streaming and scheduled
generation share one execution lock, so the one-worker experiment never overlaps model
calls.

This remains a modular monolith. There is no global scheduler or multi-worker routing in
the executable request path yet.

## Phase 1 module boundaries

```text
apps/
  gateway/             stable FastAPI entry point
  control_plane/       process boundary reserved for Phase 2+
  worker/              process boundary reserved for Phase 2+

serving_platform/       (`platform` would shadow Python's standard-library module)
  domain/              typed models and request state machine data
  admission/           admission contract
  lifecycle/           validated transition service
  registry/            worker and model registry contracts
  request_state/       request-state persistence contract
  routing/             routing policy and explanation contract
  scheduling/          local scheduler contract
  telemetry/           bounded-cardinality telemetry contract

runtime/
  base/                model runtime contract and result types
  huggingface/         adapter around the existing CPU worker
```

Interfaces point inward toward framework-independent domain types. FastAPI and Pydantic
remain API-boundary concerns. The Hugging Face adapter refuses model identifiers other
than its registered model before loading, preventing arbitrary user-controlled model paths.

## Target two-level request flow

Later phases will incrementally activate this path:

```text
gateway -> authentication -> admission -> global scheduler
                                      |-> model registry
                                      |-> worker registry
                                      |-> request state store
                                      v
                     selected worker local queue
                                      v
                           local dynamic batching
                                      v
                                model runtime
```

The global scheduler will select a worker; it will not form model batches. Each worker
retains its own bounded queue and local scheduler. This separation lets placement account
for health, memory, residency, and load without turning global routing into a token decoding
loop.

## State and process boundaries

Today all state is in one process and is lost on restart. Phase 1 defines storage and
registry contracts but does not claim persistence. Worker processes, heartbeats, CUDA
capacity, simulated GPUs, and routing policies begin in Phase 2 and Phase 3.

See [the Phase 1 audit and migration plan](phase1_migration_plan.md) for the reuse map and
activation sequence.
