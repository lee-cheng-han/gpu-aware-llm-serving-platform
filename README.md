# GPU-Aware LLM Serving Platform

> **Zero-cost default:** the project performs inference locally and does not call a paid
> model API or provision cloud resources. Hosted CI is manual-only. Cloud runners, hosted
> GPUs, managed inference services, and managed Kubernetes are intentionally not required.

## Overview

A local FastAPI serving platform for studying request scheduling and explicit token-level
inference on a Hugging Face causal model. The compatibility endpoint retains static
micro-batching while streaming requests enter a shared iteration scheduler for real batched
prefill and autoregressive decode against framework-managed KV caches.

## What This Project Demonstrates

- FastAPI, a bounded `asyncio` queue, and explicit request lifecycle states
- Local CPU Hugging Face inference with pinned SmolLM2-135M-Instruct by default
- Real prefill plus one-token decode using Hugging Face `DynamicCache`
- Continuous waiting/running sequence sets with iteration-level admission and compatible
  real-model prefill/decode batches
- Fixed-block KV capacity reservation, ownership tables, reference counts, watermarks, and
  exhaustion handling
- Bounded exact-prompt KV snapshot reuse with LRU eviction and real prefill bypass
- Prefill-first, decode-first, balanced, and optional real chunked-prefill execution
- Model-aware context-window validation and optional startup warm-up
- No-batching and dynamic micro-batching schedulers
- Queue, concurrency, prompt, output, and deadline limits
- Cancellation-aware request futures and graceful queue shutdown
- Cancellation-aware interleaved SSE streaming with token IDs, indices, finish reason,
  usage, JSON/Prometheus metrics, W3C trace correlation, and structured request logs
- Repeatable benchmark clients, Docker, and one-pod kind manifests
- Typed CPU/CUDA worker abstractions, deterministic simulated GPU execution, and an
  in-memory worker registry with rich inventory and staged suspect/unhealthy health
- Round-robin and least-queue global routing with structured decision explanations
- Supervised worker heartbeats and token-bounded local worker execution
- Residency-aware, memory-safe, and estimated-completion placement policies
- Atomic local placement reservations and demand-aware replica decisions
- Coalesced model loading, warmup, memory reservations, and protected LRU idle eviction
- API-key tenant identity, atomic quotas, weighted fairness, priority aging, and cancellation
- Supervised stale-worker recovery with bounded, reason-coded retries for unstarted work
- Redis-compatible, prompt-redacting request metadata persistence and tenant-scoped status APIs
- Authenticated local HTTP worker transport with strict JSON contracts and two-process tests
- Redis-compatible atomic admission accounting and owner-leased recovery coordination
- Deterministic runtime fault injection and optional OpenTelemetry OTLP/HTTP export

## What This Project Is Not

- Not “mini-vLLM,” PagedAttention, or a vLLM/TGI/Triton replacement
- The default gateway execution path is not yet distributed or production-scale; simulated
  GPU results are never real-device measurements
- The block allocator governs admission and ownership, but physical model tensors still use
  Hugging Face `DynamicCache`; this is not a custom paged-attention kernel

This project is not a replacement for vLLM, TGI, or Triton. It does not implement
PagedAttention, GPU KV tensor paging, tensor parallelism, or production-scale distributed
serving. The streaming path continuously schedules multiple sequences but only combines
decode work with compatible physical cache lengths until a paged-attention backend exists.

## Architecture

```text
POST /v1/generate -> validate/admit -> bounded queue -> scheduler policy -> CPU model
                                                \------> JSON metrics
POST /v1/generate_stream -> KV reserve -> waiting/running scheduler -> batched iterations -> SSE
```

All device work runs off the event loop through one dedicated executor lane and shares the
compatibility worker's execution lock. See the [current architecture](docs/current-architecture.md),
[target architecture](docs/target-architecture.md), and
[backend ADR](docs/adr/001-huggingface-prefill-decode.md).

## Quick Start

Python 3.12 is the supported platform target.

```bash
make install
make run
curl http://localhost:8000/health
```

Configuration is read from the environment:

| Variable | Default |
|---|---:|
| `MODEL_NAME` | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| `MODEL_REVISION` | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| `MODEL_DEVICE` | `cpu` (`cuda` or `cuda:<index>` in the CUDA image) |
| `MODEL_DTYPE` | `float32` |
| `INFERENCE_BACKEND` | `token_loop` |
| `MAX_ACTIVE_SEQUENCES` / `TOKEN_MAX_BATCH_SIZE` | `8` / `8` |
| `TOKEN_SCHEDULING_POLICY` | `balanced` |
| `PREFILL_CHUNK_SIZE_TOKENS` | `0` (full prefill) |
| `KV_CACHE_TOTAL_BLOCKS` / `KV_CACHE_BLOCK_SIZE_TOKENS` | `1024` / `16` |
| `KV_CACHE_BYTES_PER_BLOCK` | `737280` (reference-model float32 estimate) |
| `KV_CACHE_HIGH_WATERMARK` / `KV_CACHE_LOW_WATERMARK` | `0.9` / `0.7` |
| `PREFIX_CACHE_MAX_ENTRIES` / `PREFIX_CACHE_MAX_TOKENS` | `32` / `2048` (`0` / `0` disables) |
| `SCHEDULER_POLICY` | `no_batching` |
| `MAX_PROMPT_TOKENS` / `MAX_NEW_TOKENS` | `1024` / `128` |
| `MAX_PROMPT_CHARACTERS` | `16384` |
| `MAX_QUEUE_SIZE` / `MAX_CONCURRENT_REQUESTS` | `128` / `16` |
| `REQUEST_TIMEOUT_SECONDS` | `60` |
| `MAX_BATCH_SIZE` / `MAX_WAIT_MS` | `8` / `25` |
| `MAX_TOTAL_BATCH_TOKENS` | `1024` |
| `MODEL_WARMUP_ON_START` | `false` |
| `SHUTDOWN_GRACE_SECONDS` | `30` |
| `METRICS_SAMPLE_LIMIT` | `10000` |
| `API_KEYS` | empty (authentication disabled) |
| `CORS_ALLOWED_ORIGINS` | empty (cross-origin access disabled) |
| `PLATFORM_API_ENABLED` | `false` (opt-in deterministic control-plane API) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | empty (export disabled) |
| `OTEL_SERVICE_NAME` | `llm-inference-scheduler` |
| `ALLOW_REMOTE_TELEMETRY` | `false` |

Optional local OTLP export uses a separate free dependency profile:

```bash
make install-observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces make run
```

## API Usage

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready

curl -X POST http://localhost:8000/v1/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"Explain cache memory simply.","max_new_tokens":32,"temperature":0.7}'

curl -N -X POST http://localhost:8000/v1/generate_stream \
  -H 'content-type: application/json' \
  -d '{"prompt":"Explain PCIe simply.","max_new_tokens":32,"temperature":0.7,"top_p":0.9}'

curl http://localhost:8000/metrics
curl -H 'accept: text/plain' http://localhost:8000/metrics

# Use the request_id returned by a generation call.
curl http://localhost:8000/v1/requests/<request_id>
curl -X DELETE http://localhost:8000/v1/requests/<request_id>
```

Set `PLATFORM_API_ENABLED=true` to expose `POST /v1/platform/generate`. This opt-in route
executes authentication, tenant admission, weighted fair queueing, placement, cold model
loading, worker-local batching, lifecycle persistence, and terminal response conversion
through two deterministic local simulated workers. It returns the same `GenerateResponse`
schema as `/v1/generate`, performs no remote inference, and does not replace the compatibility
route.

### Independent local workers

The control plane can also dispatch to workers running as separate local processes. Each
worker requires a shared development token and exposes only the internal lifecycle contract:

```bash
WORKER_ID=worker-a WORKER_AUTH_TOKEN=local-secret \
  uvicorn apps.worker.main:app --host 127.0.0.1 --port 8101

WORKER_ID=worker-b WORKER_AUTH_TOKEN=local-secret \
  uvicorn apps.worker.main:app --host 127.0.0.1 --port 8102
```

These processes use the deterministic simulated runtime, make no external API calls, and
incur no model or cloud charges. `HttpWorkerClient` supplies the matching synchronous
control-plane handle. The transport is intended for localhost development: keep it bound to
loopback because it does not provide TLS, service discovery, or internet-facing hardening.

For multi-control-plane experiments, `RedisAdmissionController` stores global queue and
per-tenant queue, concurrency, and token reservations in one Redis hash. Lua scripts make
reserve, queued-to-running, and idempotent release transitions atomic across processes.
`RedisLease` provides compare-and-renew ownership for the recovery supervisor so only one
instance expires workers at a time. Redis is an optional adapter: the default path remains
in-memory, and all repository tests use a local deterministic fake rather than a hosted
service.

Malformed or unsupported input returns 400, oversized prompt/context input returns 413,
tenant or concurrent admission returns 429, unavailable capacity returns 503, and expired
generation requests return 504. Configured authentication failures return 401.

Errors use a stable `{"error":{"code","message","details"}}` envelope. `/health`
reports process health; `/ready` returns 200 only after the model has loaded. Set
`MODEL_WARMUP_ON_START=true` to load the model before accepting traffic. Prompt plus
requested output tokens must fit the model's context window. Set `temperature` to `0`
for deterministic greedy decoding.

Set `API_KEYS` to comma-separated `tenant:key` entries to enable development
authentication, for example `API_KEYS=research:key-one,interactive:key-two`. Generation
requests must then send `Authorization: Bearer <key>`. Keys and full prompts are never
logged; configured key material is held as SHA-256 digests for constant-time verification.
Request JSON accepts `priority` (`0`–`100`) and optional positive
`deadline_seconds`; extra fields such as `tenant_id` are rejected.
Set `CORS_ALLOWED_ORIGINS` to a comma-separated allowlist when browser access is required;
wildcard origins and credentialed CORS are not enabled.

The SSE path reserves sequence and logical KV capacity before response headers. A central
scheduler forms real prefill/decode batches, removes completed sequences independently, and
allows new requests to enter between iterations. Set `PREFILL_CHUNK_SIZE_TOKENS` to bound
each real prefill operation and allow balanced policies to interleave decode work. Token
events retain the original `token` field and
add `token_id` and zero-based `index`; the terminal event adds `finish_reason` and final
token usage. `stop_token_ids` accepts at most 32 unique, non-negative in-vocabulary IDs.
Greedy sampling (`temperature=0`) ignores `top_p`; positive temperature uses top-p
multinomial sampling. EOS and configured stop tokens terminate without emitting the special
token.

Client disconnect and `DELETE /v1/requests/{id}` set the same cancellation signal. A forward
pass already on the device may finish, but no later token is emitted; cleanup then releases
private cache state, sequence capacity, admission, and stream tracking. The non-streaming
`/v1/generate` route intentionally retains its existing static batching and whole-request
`generate()` behavior.

## Scheduler Design

The streaming token scheduler maintains explicit waiting and running sets. Each iteration
selects prefill or decode work using `prefill_first`, `decode_first`, or `balanced`. Compatible
sequences execute in one real Hugging Face batch; sequences terminate, fail, or cancel
independently. Decode batches currently require equal physical cache lengths because
`DynamicCache` is not PagedAttention. See [token scheduling](docs/token-scheduler.md) and
[KV-cache accounting](docs/kv-cache.md).

The non-streaming compatibility scheduler remains separate:

`no_batching` takes exactly one request and makes one model call. `dynamic_batch` takes a
first request, waits up to `MAX_WAIT_MS`, and collects until batch-size or estimated total
token limits are reached. Prompts are left-padded, generated in one call, then split back
by request. Deadlines begin at queue insertion and include both collection and generation.
Generation cannot be preempted; an over-deadline result is marked timeout after it returns.
On shutdown, the server stops admission, rejects queued requests, wakes an idle scheduler,
and allows an active model call to finish so no request future is abandoned. If a client
disconnects, queued work is marked cancelled and skipped; running Hugging Face generation
still finishes because v1 cannot preempt it.

Only requests with matching output length and temperature share a model call, preserving
request semantics. The collector moves incompatible candidates into a scheduler-owned
deferred deque, reuses compatible deferred work before waiting for new arrivals, and scans
each deferred item at most once per collection. This avoids repeated queue reinsertion,
reduces head-of-line blocking, and prevents a busy loop. Queue task accounting remains tied
to completion of the shared model call. Controlled benchmarks should still use uniform
parameters.

## Metrics

`/metrics` returns request outcome counters, active/queued gauges, p50/p95 latency and
TTFT, queue-wait percentiles, queue high-water mark, model invocation count, batch-size
histogram, and average validation-tokenization, worker-tokenization, generation, decoding,
and batch-collection times. It reports both process-lifetime and rolling 60-second request
and token throughput. Cancelled requests remain separate from failures and timeouts.
Recent latency percentiles, rejection reasons, active model executions, and the configured
sample limit are also exposed. Timing samples use bounded deques, so metrics memory does
not grow without limit. Token scheduler metrics add waiting/running counts, iteration and
batch counts, selected policy, maximum observed batch size, KV blocks/bytes/utilization,
allocation failures, and pressure state. Metrics reset on restart.
Exact-prefix metrics add retained entries/tokens, hits, misses, hit rate, insertions,
evictions, and bypasses. Prometheus text is selected with `Accept: text/plain`; JSON remains
the default for compatibility. HTTP responses propagate W3C trace IDs and structured logs
exclude prompts, generated text, and credentials. See [observability](docs/observability.md).

## Benchmark Methodology

Compare whole-request `generate()` with the explicit token loop on the same pinned model:

```bash
python -m benchmark.compare_token_backends --warmups 1 --trials 5
```

For the static scheduler comparison, first check model scale:

```bash
python benchmark/sanity_model_benchmark.py \
  --models HuggingFaceTB/SmolLM2-135M-Instruct --requests 50 --concurrency 8 --max-new-tokens 32
```

Then restart the server for each policy and run:

```bash
SCHEDULER_POLICY=no_batching make run
python benchmark/compare_schedulers.py --policy no_batching

SCHEDULER_POLICY=dynamic_batch make run
python benchmark/compare_schedulers.py --policy dynamic_batch
```

The client refuses to label results with a policy different from the server response.
It reuses pooled HTTP connections, excludes configurable warm-up requests, runs repeated
trials from a seeded prompt set, records every outcome, and can retain raw JSON:

```bash
python benchmark/compare_schedulers.py --policy dynamic_batch \
  --trials 3 --warmup-requests 5 \
  --output benchmark/results/dynamic_batch.json
```

See [the full methodology](docs/benchmark_methodology.md).

Run a bounded measured load against the local gateway with:

```bash
make benchmark-sustained-load
```

The load client refuses remote targets without explicit opt-in, reducing the chance of
accidentally exercising a billed endpoint.

The deterministic control-plane evaluation exercises heterogeneous tenants, priorities,
model residency, worker speeds, deadlines, and weighted fairness without loading a model:

```bash
make evaluation
```

It writes JSON, Markdown, and SVG artifacts under `benchmark/results/simulated/`. These are
always labelled simulations and are regression signals, never hardware performance claims.

## Benchmark Results

The repository includes a small measured CPU proof comparing the two model loops. It is a
correctness-oriented measurement, not a throughput claim; see [results](docs/results.md) and
the [raw report](docs/evidence/token-backend-cpu.json). Deterministic simulated scheduler
results remain separately labelled.

| Model / hardware | Policy | p50 ms | p95 ms | req/s | tokens/s | avg batch |
|---|---|---:|---:|---:|---:|---:|
| _pending measured run_ | no_batching | — | — | — | — | 1 |
| _pending measured run_ | dynamic_batch | — | — | — | — | — |

## Docker

```bash
docker build -t gpu-aware-llm-serving-platform:local .
docker run --rm -p 8000:8000 gpu-aware-llm-serving-platform:local
```

See [deployment profiles](deploy/README.md) for local CPU, deterministic simulation, and
real CUDA examples. CUDA execution uses a separate image and never silently falls back to
the CPU dependency set.

## Kubernetes

The one-replica kind flow is documented in [k8s/README.md](k8s/README.md).

## Tradeoffs

Read [Tradeoffs and Simplifications](docs/tradeoffs.md), especially the timeout,
streaming, CPU-only, and single-worker semantics.

## Future Work

The remaining production work includes shared cross-controller placement transactions,
real GPU evaluation, physical paged KV tensors/PagedAttention, arbitrary-length decode
batching, partial-prefix page sharing, and multi-GPU execution.

## Development Checks

```bash
make install-dev
make lint
make typecheck
make test
make evaluation
```

Normal tests never download a model. The explicit optional model suite includes the pinned
SmolLM2 proof (and retains tiny-gpt2 as a fast compatibility smoke test):

```bash
make test-model
```

The manual GitHub Actions workflow runs Python 3.12 tests, Ruff, mypy, compilation,
evaluation gates, dependency checks, and a non-pushing Docker build. It does not run on
pushes or pull requests, preventing automatic hosted-runner consumption. The same checks
can be run locally without a paid inference API.
