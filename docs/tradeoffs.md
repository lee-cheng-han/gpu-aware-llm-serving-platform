# Tradeoffs and Simplifications

## Why this is not mini-vLLM

This project studies a much smaller problem. It is not a replacement for vLLM, TGI, or
Triton. It does not implement PagedAttention, GPU KV tensor paging, tensor parallelism, or
production-scale distributed serving.

## Why no PagedAttention

PagedAttention solves GPU KV-cache allocation and fragmentation. The non-streaming path uses
whole Hugging Face `generate()` calls. Streaming now uses real prefill/decode with the
framework's `DynamicCache`, which exposes token boundaries without claiming a custom
allocator. A fixed-block scheduler allocator now provides capacity, ownership, reference
counts, and pressure behavior. The physical tensors remain framework-managed, so the block
tables must not be described as a PagedAttention implementation.

## Why CPU-only local model

CPU execution is reproducible on a laptop and avoids CUDA-specific installation and
capacity concerns. The results must not be generalized to GPU serving.

## Why pinned SmolLM2 for the reference proof

SmolLM2-135M-Instruct is small enough for a zero-cost CPU proof while exercising a modern
cache API. The immutable revision makes the result reproducible. Tiny GPT-2 remains useful
only for fast compatibility tests; final reports state the exact model, revision, and
hardware.

## Why one worker

One worker isolates batching policy as the independent variable. Multiple workers add
routing, CPU oversubscription, and model-memory duplication.

## Why Redis is an adapter, not a required service

The default queue and metrics remain process-local so the original experiment runs without
infrastructure. The platform layer provides a Redis-compatible request metadata adapter for
restart persistence, but deliberately leaves client construction and Redis deployment to
the operator. Prompts and outputs are excluded by default to limit sensitive-data exposure.

## Why no full API authentication or per-user rate limits

v1 has a single trusted caller: the local benchmark script. Full API-key authentication,
per-user rate limits, and token-per-minute budgets are intentionally scoped out. v1 keeps
queue size, concurrency, prompt length, and output length limits because those directly
protect the local scheduler during benchmarks.

## Why timeout starts at queue entry

The deadline includes queue wait, batching-window wait, and generation. Expired queued
requests are skipped. Hugging Face generation is not safely preempted in v1: if a deadline
passes during `generate()`, work finishes and the request is marked `TIMEOUT` afterward.
This is honest deadline accounting, not compute cancellation.

Client cancellation follows the same compute limitation. A request cancelled while queued
is skipped. If generation is already running, its result is discarded after the blocking
model call returns. Graceful shutdown rejects queued work and waits for the active call so
all scheduler result handles reach a terminal state.

## Why continuous batches still require compatible cache lengths

The SSE endpoint uses a central continuous scheduler and real batched forward passes. Standard
`DynamicCache` can combine sequences whose physical cache tensors have equal lengths. A late
joiner with a shorter cache enters the running set immediately but uses a separate compatible
decode group until physical paging removes that constraint.

Streaming admission and backend capacity are reserved before headers and released on every
terminal path. Cancellation is checked around every device call and emission. PyTorch
cannot interrupt a forward pass already running, so cleanup waits for that pass, emits
nothing afterward, and releases the sequence's private cache.

All inference paths share one model execution lock. A long stream can therefore delay
queued non-streaming work; allowing overlap would confound the one-worker experiment and
can oversubscribe CPU threads. Production systems would use explicit worker pools and
tenant-aware routing rather than a process-local lock.

Shutdown stops queue and stream admission, rejects deferred or queued requests, and waits
up to `SHUTDOWN_GRACE_SECONDS`. If blocking PyTorch work outlives that grace period, request
handles are failed and the scheduler task is cancelled, but the underlying Python thread
cannot be safely killed. The process may still spend time terminating local compute.

## Why Kubernetes is deployment-only

One replica preserves the single in-memory queue and worker. The manifests demonstrate
local packaging, not elasticity or high availability.

## What would change in a production version

A production system still needs durable queue claims, shared placement transactions,
leader coordination, secret-manager-backed key rotation, a deployed telemetry collector,
and GPU-native continuous batching. Prometheus exposition, structured logs, optional OTLP
tracing, staged worker health, and deterministic fault injection exist locally; Grafana,
autoscaling, and careful overload policy still depend on measured requirements.
