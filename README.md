# GPU-Aware LLM Serving Platform

> **Zero-cost default:** the project performs inference locally and does not call a paid
> model API or provision cloud resources. Hosted CI is manual-only. Cloud runners, hosted
> GPUs, managed inference services, and managed Kubernetes are intentionally not required.

## Overview

A lean local FastAPI server for studying the systems tradeoff between serial request
execution and short-window dynamic micro-batching on a CPU Hugging Face causal model.
The deep axis is queueing and scheduling—not chatbot product features.

## What This Project Demonstrates

- FastAPI, a bounded `asyncio` queue, and explicit request lifecycle states
- Local CPU Hugging Face inference (`sshleifer/tiny-gpt2` by default)
- Model-aware context-window validation and optional startup warm-up
- No-batching and dynamic micro-batching schedulers
- Queue, concurrency, prompt, output, and deadline limits
- Cancellation-aware request futures and graceful queue shutdown
- Single-request SSE streaming and JSON metrics
- Repeatable benchmark clients, Docker, and one-pod kind manifests
- Typed CPU/CUDA worker abstractions, deterministic simulated GPU execution, and an
  in-memory heartbeat-aware worker registry
- Round-robin and least-queue global routing with structured decision explanations
- Supervised worker heartbeats and token-bounded local worker execution
- Residency-aware, memory-safe, and estimated-completion placement policies
- Coalesced model loading, warmup, memory reservations, and protected LRU idle eviction
- API-key tenant identity, atomic quotas, weighted fairness, priority aging, and cancellation
- Supervised stale-worker recovery with bounded, reason-coded retries for unstarted work
- Redis-compatible, prompt-redacting request metadata persistence and tenant-scoped status APIs

## What This Project Is Not

- Not “mini-vLLM,” PagedAttention, continuous batching, or a vLLM/TGI/Triton replacement
- The default gateway execution path is not yet distributed or production-scale; simulated
  GPU results are never real-device measurements
- Not batched interleaved streaming or a custom KV-cache implementation

This project is not a replacement for vLLM, TGI, or Triton. It does not implement
PagedAttention, GPU KV-cache paging, tensor parallelism, continuous batching, or
production-scale distributed serving. v1 focuses on local request scheduling and dynamic
micro-batching using a CPU Hugging Face model.

## Architecture

```text
POST /v1/generate -> validate/admit -> bounded queue -> scheduler policy -> CPU model
                                                \------> JSON metrics
POST /v1/generate_stream -> validate/admit -> unbatched CPU streamer -> SSE
```

Generation runs off the event loop in a thread; one scheduler owns one lazy-loaded model.
See [architecture details](docs/architecture.md).

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
| `MODEL_NAME` | `sshleifer/tiny-gpt2` |
| `MODEL_REVISION` | `main` |
| `MODEL_DEVICE` | `cpu` (`cuda` or `cuda:<index>` in the CUDA image) |
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

## API Usage

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready

curl -X POST http://localhost:8000/v1/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"Explain cache memory simply.","max_new_tokens":32,"temperature":0.7}'

curl -N -X POST http://localhost:8000/v1/generate_stream \
  -H 'content-type: application/json' \
  -d '{"prompt":"Explain PCIe simply.","max_new_tokens":32,"temperature":0.7}'

curl http://localhost:8000/metrics

# Use the request_id returned by a generation call.
curl http://localhost:8000/v1/requests/<request_id>
curl -X DELETE http://localhost:8000/v1/requests/<request_id>
```

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

The SSE path reserves concurrency before response headers are sent, counts generated model
tokens independently of decoded text fragments, and always releases admission on success,
failure, or disconnect. Background model exceptions produce a structured SSE error followed
by `done`. Disconnecting cannot preempt an already-running Hugging Face generation thread;
its output is discarded when that local call eventually returns.

## Scheduler Design

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
not grow without limit. Metrics reset on restart.

## Benchmark Methodology

First check model scale:

```bash
python benchmark/sanity_model_benchmark.py \
  --models sshleifer/tiny-gpt2 gpt2 --requests 50 --concurrency 8 --max-new-tokens 32
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

The deterministic control-plane evaluation exercises heterogeneous tenants, priorities,
model residency, worker speeds, deadlines, and weighted fairness without loading a model:

```bash
make evaluation
```

It writes JSON, Markdown, and SVG artifacts under `benchmark/results/simulated/`. These are
always labelled simulations and are regression signals, never hardware performance claims.

## Benchmark Results

No real-device measurements are committed yet. Deterministic simulated results are kept in
a separate directory and include their workload, seed, thresholds, and classification.

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

The remaining production work includes wiring the control plane into the default gateway
execution path, distributed admission accounting, Prometheus/OpenTelemetry export,
idempotency keys, atomic placement reservations, real GPU evaluation, continuous batching,
sampling-parameter bucketing, and batched streaming.

## Development Checks

```bash
make install-dev
make lint
make typecheck
make test
make evaluation
```

Normal tests never download a model. An explicit optional smoke test downloads and executes
`sshleifer/tiny-gpt2`:

```bash
make test-model
```

The manual GitHub Actions workflow runs Python 3.12 tests, Ruff, mypy, compilation,
evaluation gates, dependency checks, and a non-pushing Docker build. It does not run on
pushes or pull requests, preventing automatic hosted-runner consumption. The same checks
can be run locally without a paid inference API.
