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

This remains a modular monolith. The compatibility generation routes retain the proven
single-worker path. When `PLATFORM_API_ENABLED=true`, `/v1/platform/generate` executes the
complete gateway-to-control-plane path through deterministic in-process simulated workers.
The same control-plane interface can now address authenticated HTTP worker handles in
independent local processes; a process-level integration test proves cold loading, dispatch,
execution, and lifecycle synchronization across that boundary.

## Worker-ready module boundaries

```text
apps/
  gateway/             stable FastAPI entry point
  control_plane/       global assignment and worker-handle dispatch
  worker/              managed lifecycle, HTTP contract, and runtime-specific workers

serving_platform/       (`platform` would shadow Python's standard-library module)
  domain/              typed models and request state machine data
  admission/           admission contract
  lifecycle/           validated transition service
  registry/            worker registry contract and in-memory implementation
  request_state/       request-state contract plus in-memory and Redis-compatible stores
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

## Two-level control-plane flow

The platform layer now executes this flow through an opt-in API and deterministic integration
tests:

```text
gateway -> authentication -> admission -> global scheduler
                                      |-> model registry
                                      |-> worker registry
                                      |-> request state store
                                      v
                     selected worker handle
                         |          |
                  in-process    authenticated HTTP
                         |          |
                     selected worker local queue
                                      v
                           local dynamic batching
                                      v
                                model runtime
```

The global scheduler selects a worker; it does not form model batches. Each worker
retains its own bounded queue and local scheduler. This separation lets placement account
for health, memory, residency, and load without turning global routing into a token decoding
loop.

## State and process boundaries

The default application uses process-local state. A Redis-compatible adapter can durably
store request metadata, lifecycle transitions, assignment, attempts, and retry reasons
without persisting prompts or outputs. Workers register and publish health, queue,
active-batch, residency, throughput, drain, and capacity snapshots. A reliability
supervisor expires stale heartbeats and invokes recovery; recovery requires worker
re-registration. CUDA capacity uses
PyTorch device APIs when CUDA is available; the deterministic simulated runtime models
memory, load delay, throughput, batching efficiency, and controlled failures without
claiming real inference. Round-robin and least-queue global policies now filter worker
health, drain state, concurrency, and model residency and return structured explanations.
Worker applications supervise heartbeats and local execution together with `TaskGroup`.
Placement policies additionally use registered model memory and context requirements,
PyTorch-reported CUDA free/allocated/reserved memory, a configurable safety reserve,
residency, throughput, and deadline feasibility. Simulated memory values remain explicitly
synthetic.

Cold assignments flow through the in-memory model catalog into the selected worker's model
cache. The worker reserves estimated memory before releasing its state lock, coalesces
duplicate loads with a condition variable, warms the model, then publishes residency before
the request enters its local queue. Under pressure it evicts least-recently-used models that
have neither queued nor active requests. A supervised scan applies per-model idle timeouts.

Tenant identity is derived by the gateway authenticator, never from request JSON. The
control-plane admission controller reserves queue, concurrency, and estimated-token capacity
atomically. Admitted requests enter a weighted deficit round-robin queue; tenant weight
controls quantum, priority chooses within a tenant, and wait-time aging prevents old
low-priority work from starving. The in-memory request store uses isolated snapshots and
tracks the explicit lifecycle without logging prompts or generated text.

Recovery is deliberately at-most-once for work that may have started: only assigned work
with no running or streaming transition can return to the fair global queue. Started work
fails when its worker is lost because the control plane cannot prove that execution stopped.
Retries retain attempt counts, record reasons, and obey both a maximum attempt budget and
the original deadline. Tenant-authenticated request status and cancellation routes expose
metadata without exposing prompts, output, or another tenant's request existence.
Records reconstructed from Redis are metadata-only and cannot be resubmitted after a full
process restart unless a deployment supplies a separate encrypted payload store.

## Local worker transport

`HttpWorkerClient` implements the same worker protocol as `ManagedWorker`, so placement is
independent of process location. It sends authenticated strict JSON over loopback for
registration, heartbeat, health, capacity, model lifecycle, request handoff, batch
execution, cancellation, drain, and shutdown operations. Worker snapshots and request
lifecycle changes are merged back into the control-plane-owned objects after each call.

The server requires `X-Worker-Token` on every internal endpoint and compares it in constant
time. Model loads accept only the worker's pre-registered full model definition. This is a
development transport, not a public security boundary: it has bounded client timeouts but
does not yet add TLS, token rotation, asynchronous multiplexing, service discovery, or a
durable message broker.
