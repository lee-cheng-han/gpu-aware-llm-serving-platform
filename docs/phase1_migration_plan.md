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

This phase also incorporates the post-tenant reliability review:

Implemented in this phase: a Redis-compatible prompt-redacting request store; automatic
heartbeat expiry; safe recovery that retries only never-started work; deadline and attempt
budgets with recorded retry reasons; tenant-scoped request status and cancellation routes;
and in-memory API-key hashing with constant-time verification. Recovery, persistence, and
tenant-isolation behavior have deterministic tests. The remaining numbered items below are
the production-hardening backlog, including full gateway activation and distributed atomicity.

1. Activate an opt-in gateway path that exercises authentication, admission, the fair
   global queue, placement, cold loading, worker batching, and terminal responses end to
   end. Keep the compatibility path available until parity tests pass.
2. Add a Redis or PostgreSQL request-state implementation for metadata, lifecycle state,
   assignment, attempts, timestamps, and terminal outcome. Do not persist prompts or output
   by default.
3. Run heartbeat expiry automatically, stop routing to stale workers, define explicit
   re-registration, and reassign only queued requests that have never started or streamed.
4. Wire admission reservations through queued, running, and every terminal outcome so
   exceptions, disconnects, model-load failures, deadlines, cancellation, and shutdown all
   release capacity exactly once.
5. Harden API-key management with hashed verification, key identifiers, rotation,
   revocation, and file/secret-manager loading. Never log raw credentials.
6. Export Prometheus-compatible request, scheduler, worker, model-cache, fairness,
   authentication, and deadline metrics using bounded-cardinality labels.
7. Add structured correlation IDs and OpenTelemetry spans across gateway, admission,
   routing, loading, worker queueing, batching, execution, and streaming. Never record
   prompts, generated text, API keys, or unrestricted tenant identifiers.
8. Add failure and end-to-end simulated integration tests covering gateway-to-worker
   execution, active cancellation, cold-load failure, worker expiry, safe reassignment,
   persistence restart, and graceful draining.
9. Add tenant-authorized `GET /v1/requests/{request_id}` status and
   `DELETE /v1/requests/{request_id}` cancellation endpoints with stable response and error
   schemas.
10. Support scoped idempotency keys so safe client retries return the original request
    rather than scheduling duplicate inference work.
11. Add bounded retry budgets and recorded retry reasons. Reassign only requests that have
    never started or streamed, and fail clearly when the budget or deadline is exhausted.
12. Carry model revision and dtype through registration, routing, placement, residency,
    cache keys, batching compatibility, and execution rather than treating `model_id` alone
    as a complete model identity.
13. Reserve worker memory and concurrency atomically during placement so concurrent
    scheduler decisions cannot overcommit the same point-in-time capacity snapshot.
14. Separate external API schemas, persistent control-plane records, scheduler commands,
    and runtime execution objects with explicit, tested conversion functions.
15. Validate all startup configuration in one pass, report every invalid setting together,
    and document which settings require a restart.
16. Define degraded-mode behavior for persistence, metrics, logging, and tracing outages so
    noncritical telemetry failures do not silently stop otherwise healthy inference.

### Phase 8 — evaluation

Build heterogeneous workload definitions, simulated and real-device runners, reports, and
graphs. Results remain empty until measurements are actually run.

The final evaluation and packaging review adds:

1. Reorganize the growing suite into `unit/`, `integration/`, `scheduling/`, `failure/`, and
   `benchmarks/` without changing coverage or test semantics.
2. Provide documented local CPU, deterministic simulated GPU, and real CUDA deployment
   profiles with suitable health checks, configuration examples, and resource limits.
3. Standardize local development and CI on the declared Python 3.12 target so dependency
   warnings from unsupported interpreter combinations do not obscure real failures.
4. Add deterministic benchmark regression gates for scheduler decision time, fairness,
   cache-hit rate, deadline misses, throughput, and tail latency.
5. Keep simulated results visibly separate from measured real-device reports. Real GPU
   reports must record hardware, software versions, model revision, configuration, and seed.
6. Add property-based state-machine, reservation-accounting, fairness, cancellation, and
   retry tests. Prove terminal requests never resume and counters never become negative.
7. Fuzz malformed JSON, extreme numeric values, Unicode, oversized headers, authorization
   formats, idempotency races, and SSE disconnect timing at the API boundary.
8. Introduce reproducible dependency locking, automated update checks, vulnerability
   scanning, container image scanning, and a software bill of materials.
9. Harden containers and Kubernetes with a read-only root filesystem, dropped capabilities,
   non-root execution, resource requests and limits, network policies, disruption budgets,
   and termination hooks aligned with worker draining.
10. Record architectural decisions for modular-monolith boundaries, persistence backend,
    retry safety, fairness, simulation semantics, model identity, and the distinction from
    continuous batching.
11. Automate semantic versions, changelogs, tagged container images, provenance
    attestations, and release-quality validation.
12. Profile tokenization, state-lock contention, queue operations, scheduler decisions,
    thread utilization, model loading, and memory-accounting drift under controlled load.
13. Add overload and dependency-failure experiments that verify documented degraded modes
    and backpressure behavior.

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
- Phases 7–8: the reliability, security, observability, integration, repository-organization,
  deployment-profile, target-version, and benchmark-gate improvements are specified in the
  expanded phase sections above.

## Compatibility rules

- `/health`, `/ready`, `/v1/generate`, `/v1/generate_stream`, and `/metrics` remain valid.
- The legacy `app/` compatibility package was removed after the gateway implementation
  moved to `apps/gateway/`; run and container commands use `apps.gateway.main:app`.
- Existing scheduler configuration remains valid.
- No request is routed through an unfinished control-plane path.
- New interfaces must have deterministic tests before replacing existing wiring.
- Simulated GPU output and performance must always be labelled simulated.
