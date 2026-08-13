# Phase 1 Audit and Migration Plan

## Baseline audit

The repository began Phase 1 as a working single-process, single-model server. FastAPI
owns validation and admission, an `asyncio.Queue` owns bounded pending work, one scheduler
loop chooses serial or dynamic micro-batching, and one lazy Hugging Face worker executes
CPU generation. SSE bypasses the queue but shares the worker execution lock. Metrics and
request state are process-local.

The pre-refactor validation baseline was:

```text
ruff check .                       passed
mypy                              passed
pytest -m "not model" -q          42 passed, 1 deselected
```

The deselected test is the explicit real-model download and execution test. The local
validation interpreter is Python 3.14, which produces dependency deprecation warnings;
the platform target and CI interpreter are Python 3.12.

## Reusable components

| Existing component | Reuse decision |
|---|---|
| Original `app.api` and `app.main` | Migrate into `apps.gateway` |
| `inference.worker.InferenceWorker` | Preserve behind `HuggingFaceRuntime` |
| `scheduler.queue.RequestQueue` | Preserve as the local worker queue |
| Serial and dynamic schedulers | Preserve as local scheduling policies |
| `scheduler.request.InferenceRequest` | Preserve internally; map to the platform request model later |
| `app.metrics.Metrics` | Preserve until the telemetry interface receives a Prometheus adapter |
| Fake-worker test suite | Preserve for deterministic local-scheduler tests |
| Docker and Kubernetes files | Preserve; restructure only after Compose works in a later phase |

## Coupling identified

- `apps.gateway.main` constructs the queue, worker, scheduler, limiter, and metrics directly.
- API validation calls the concrete Hugging Face worker.
- Scheduler result types come from `inference.worker`.
- Request lifecycle transitions are assignments to an internal enum, not a global state machine.
- There is no global worker registry, model registry, routing layer, or persistent state store.
- CPU capacity, CUDA capacity, and simulated capacity do not yet share an abstraction.

## Phase 1 changes

1. Add framework-independent domain models for workers, models, requests, and assignments.
2. Add a validated lifecycle whose terminal states cannot transition.
3. Add contracts for runtime, admission, routing, registries, state storage, telemetry, and
   local scheduling.
4. Wrap the existing Hugging Face worker with a registered-model-only runtime adapter.
   Model revision now flows through configuration and Hugging Face loading.
5. Add `apps.gateway` as a stable application entry point without changing old imports.
6. Retarget packaging, Docker, CI, Ruff, and mypy to Python 3.12.
7. Keep the existing single-worker construction active until later phases have tested
   replacements.

The specification's suggested top-level `platform/` name is adapted to
`serving_platform/`. Python commonly imports its standard-library `platform` module before
application collection, making `platform.domain` unreliable and import-order dependent.

## Migration sequence

### Phase 2 — workers

Implement CPU, CUDA, and deterministic simulated workers against `ModelRuntime`; add an
in-memory worker registry, registration, heartbeat, capacity, drain, and shutdown behavior.

### Phase 3 — global scheduling

Place a control-plane scheduler between gateway and local workers. Implement filtering,
round robin, least queue depth, and structured routing explanations. Connect the managed
worker queue to token-bounded local batching and supervise worker execution and heartbeat
tasks with structured concurrency before enabling the multi-worker gateway path.

Implemented: deterministic round-robin and least-queue policies, structured rejection and
score data, deadline-aware assignment, process-local dispatch with safe handoff failure,
compatible local batch execution, and supervised worker application lifecycle. The public
gateway stays on its compatibility path until authenticated admission is introduced.

### Phase 4 — placement

Add real CUDA memory probes, simulated GPU capacity, residency-aware placement,
memory-aware least-loaded routing, and estimated-completion routing.

Implemented: PyTorch CUDA free, total, allocated, and reserved memory probes; deterministic
simulated capacity; a configurable placement safety reserve; model-residency-aware and
memory-aware least-loaded policies; estimated-completion scoring; context, runtime, memory,
and deadline feasibility filters; and structured scoring inputs. Cold placement produces a
decision but is not dispatched successfully until model loading is connected in Phase 5.

### Phase 5 — model lifecycle

Add an in-memory model registry, load coalescing, memory reservations, warmup, cache metrics,
and LRU eviction. Only registry entries may be loaded.

Implemented: cold placement loads only catalog definitions; duplicate loads share one
attempt; reservations are released after success or failure; warmup completes before queue
admission; memory pressure uses LRU eviction; queued and active models are protected; idle
timeouts are scanned by the supervised worker application; and cache hits, misses, cold
starts, coalesced loads, load failures, load duration, reservations, and evictions are
tracked. A deadline is rechecked after an uninterruptible runtime load.

### Phase 6 — requests and tenants

Connect the platform state machine to API requests, add authenticated tenant context,
admission quotas, fairness, global cancellation, and deadline checks.

Implemented: optional API-key authentication derives tenant identity from credentials;
gateway requests are mirrored into the typed lifecycle store; priority and explicit
deadlines are accepted without allowing body-supplied tenants; atomic global/per-tenant
queue, concurrency, and token reservations have stable rejection codes; weighted deficit
round robin prevents cross-tenant starvation while priority aging protects old low-priority
work; cancellation covers global, worker-queued, and logically active requests; and runtime
output after active cancellation is discarded. The multi-worker admission path remains a
modular control-plane API until deployment wiring is introduced.

### Phase 7 — reliability and persistence

Add heartbeat expiry, failed-worker reassignment rules, persistent request metadata, worker
draining, and explicit recovery paths. Add an automatically supervised heartbeat monitor
and ensure background-task failures propagate to process health.

### Phase 8 — evaluation

Build heterogeneous workload definitions, simulated and real-device runners, reports, and
graphs. Results remain empty until measurements are actually run.

## Improvement backlog integration

The post-Phase-2 review is incorporated into the sequence rather than maintained as a
separate wishlist:

- Completed foundation/worker hardening: preserve unknown CPU memory instead of reporting
  zero capacity; keep slow warmup and unload calls outside worker state locks; enforce
  simulator context and batch limits; test CUDA selection primitives without GPU hardware;
  and align project identity with the GPU-aware serving platform.
- Phase 3: execute managed queues through local batching, expand execution/accounting tests,
  and supervise worker tasks as a unit. Worker-process deployment configuration moves with
  the later deployment restructuring because the current worker directory is process-local.
- Phase 4: add routing tests for unknown CPU capacity and mocked CUDA memory pressure.
- Phase 5: coalesce duplicate loads and add safe memory reservations so load/unload races do
  not depend only on local worker locking.
- Phase 7: run heartbeat expiry periodically, test shutdown during execution, and make task
  and process failure semantics explicit.

## Compatibility rules

- `/health`, `/ready`, `/v1/generate`, `/v1/generate_stream`, and `/metrics` remain valid.
- The legacy `app/` compatibility package was removed after the gateway implementation
  moved to `apps/gateway/`; run and container commands use `apps.gateway.main:app`.
- Existing scheduler configuration remains valid.
- No request is routed through an unfinished control-plane path.
- New interfaces must have deterministic tests before replacing existing wiring.
- Simulated GPU output and performance must always be labelled simulated.
