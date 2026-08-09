# Architecture

## Current executable architecture

The gateway deliberately preserves the proven single-worker path while the worker layer is
introduced behind tested interfaces:

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

## Worker-ready module boundaries

```text
apps/
  gateway/             stable FastAPI entry point
  control_plane/       future global scheduler process boundary
  worker/              managed worker lifecycle and device-specific workers

serving_platform/       (`platform` would shadow Python's standard-library module)
  domain/              typed models and request state machine data
  admission/           admission contract
  lifecycle/           validated transition service
  registry/            worker registry contract and in-memory implementation
  request_state/       request-state persistence contract
  routing/             routing policy and explanation contract
  scheduling/          local scheduler contract
  telemetry/           bounded-cardinality telemetry contract

runtime/
  base/                model runtime contract and result types
  huggingface/         CPU/CUDA adapter around the existing worker
  simulated_gpu/       deterministic, explicitly simulated device runtime
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

Today all state is in one process and is lost on restart. Workers now register and publish
health, queue, active-batch, residency, throughput, drain, and capacity snapshots. Stale
heartbeats are marked unhealthy, and recovery requires re-registration. CUDA capacity uses
PyTorch device APIs when CUDA is available; the deterministic simulated runtime models
memory, load delay, throughput, batching efficiency, and controlled failures without
claiming real inference. Global routing remains the next activation step.

See [the Phase 1 audit and migration plan](phase1_migration_plan.md) for the reuse map and
activation sequence.
