# Current Architecture Audit

## Verified baseline

This audit began after the shared-admission and recovery-lease work. That pre-upgrade
baseline passed Ruff, mypy across 48 source files, 129 non-model tests, and the deterministic
scheduling evaluation. Phase 1 retains those components and adds the token engine described
below; model tests remain opt-in because they download weights.

The current local check passes Ruff, mypy across 67 application source files, 165 non-model
tests, five pinned/compatibility model tests, compilation, and the deterministic evaluation
gate. CUDA validation remains pending because this host exposes no usable CUDA device.

## Executable paths

The FastAPI gateway owns authentication, validation, bounded HTTP concurrency, request
metadata, structured errors, JSON metrics, and SSE framing. `POST /v1/generate` uses a
bounded `asyncio.Queue` and either serial execution or short-window static batching. It
calls one local Hugging Face worker through `asyncio.to_thread`. Compatible requests must
have equal output length and temperature and execute in one complete `model.generate()`
call.

`POST /v1/generate_stream` reserves concurrency, sequence capacity, and logical KV blocks
before returning headers. A central scheduler maintains waiting/running sets and issues real
batched prefill or one-token decode through Hugging Face `DynamicCache`. A dedicated executor
serializes device operations, and compatible requests share forward passes.
Disconnect, DELETE, and deadlines share one cancellation signal. An in-flight forward pass
may finish, but no subsequent token is emitted and private sequence state is always released.

The opt-in `POST /v1/platform/generate` path exercises tenant admission, weighted fair
queueing, global placement, cold loading, worker-local batching, lifecycle persistence, and
deterministic simulated inference. It intentionally does not replace the compatibility
routes yet.

## State and scheduling

Two request models currently coexist. `scheduler.InferenceRequest` represents the original
gateway scheduler and uses uppercase `RequestStatus`; `serving_platform.RequestRecord`
represents the control plane and uses the validated `RequestState` transition graph. The
gateway mirrors compatibility requests into the platform store so tenant-scoped status and
cancellation APIs can operate without exposing prompts.

The compatibility scheduler is request-level static batching, not continuous batching.
The control plane separates global worker placement from bounded worker-local batch
formation. Routing policies cover round-robin, least queue depth, model residency, memory
pressure, and estimated completion time with deterministic explanations.

## Workers, models, and resources

`ManagedWorker` owns registration, heartbeats, bounded local queues, model loading and
warmup, memory reservation, coalesced cold loads, protected LRU eviction, drain, shutdown,
and failure propagation. Runtime adapters support local Hugging Face CPU/CUDA execution and
an explicitly synthetic deterministic GPU model. CUDA capacity comes from PyTorch APIs and
never silently falls back to CPU.

The control plane can hold in-process workers or authenticated `HttpWorkerClient` handles.
The local HTTP contract covers registration, heartbeat, health, capacity, model lifecycle,
request handoff, batch execution, cancellation, drain, and shutdown. A process integration
test dispatches through two real local worker processes.

## Admission, persistence, and recovery

Tenant identity comes from bearer API-key authentication rather than request JSON. The
default admission controller atomically reserves queue, concurrency, and token capacity
under a process-local lock. A Redis-compatible controller performs the same lifecycle with
atomic Lua scripts, detects configuration drift, and supports idempotent release.

Request stores have in-memory and Redis-compatible implementations. The Redis adapter saves
metadata and transitions but deliberately redacts prompts and never stores generated text,
so restored records are not executable. Recovery retries only work that provably never
started. A renewable owner-checked Redis lease can restrict heartbeat expiry and recovery
to one supervisor instance.

## Metrics, tests, and deployment

The JSON metrics endpoint reports bounded-sample latency, TTFT, queue delay, throughput,
outcomes, batch sizes, rejection reasons, and model execution counts. It is not yet a
Prometheus or OpenTelemetry endpoint.

Normal tests require neither model downloads nor GPUs. Optional model tests exercise real
Hugging Face inference. Deterministic simulations validate routing and fairness without
claiming hardware performance. CPU and CUDA Dockerfiles, local deployment profiles, and a
guarded single-replica kind deployment exist. CI is manual-only to avoid automatic runner
charges.

## Reusable boundaries and known gaps

The gateway contracts, bounded queues, explicit state machine, admission, request stores,
worker registry, placement policies, runtime protocol, simulated runtime, transport, and
failure rules are reusable. They should evolve rather than be replaced.

The remaining gaps include whole-request calls in the static scheduler, physical rather than
logical KV paging, arbitrary-length decode batches, non-preemptible individual forward
passes, duplicated request lifecycle types, JSON-only metrics, process-local worker inventory,
and binary healthy/unhealthy heartbeat handling. Continuous batching,
custom paged KV allocation, prefix caching, NCCL, topology-aware placement, Triton, and CUDA
Graphs are not implemented.
