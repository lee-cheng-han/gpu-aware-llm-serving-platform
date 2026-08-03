# Tradeoffs and Simplifications

## Why this is not mini-vLLM

This project studies a much smaller problem. It is not a replacement for vLLM, TGI, or
Triton. It does not implement PagedAttention, GPU KV-cache paging, tensor parallelism,
continuous batching, or production-scale distributed serving.

## Why no PagedAttention

PagedAttention solves GPU KV-cache allocation and fragmentation. This CPU v1 does whole
Hugging Face `generate()` calls, so a custom cache manager would obscure the scheduling
experiment.

## Why CPU-only local model

CPU execution is reproducible on a laptop and avoids CUDA-specific installation and
capacity concerns. The results must not be generalized to GPU serving.

## Why tiny-gpt2 for development and gpt2 for possible benchmarking

`sshleifer/tiny-gpt2` makes functional iteration quick, but framework, HTTP, and
tokenization costs can dominate it. The sanity script helps decide whether `gpt2` gives a
clearer compute-heavy comparison. Final reports must state the exact model and hardware.

## Why one worker

One worker isolates batching policy as the independent variable. Multiple workers add
routing, CPU oversubscription, and model-memory duplication.

## Why no Redis

The queue and metrics are process-local by design. Redis would only add deployment and
failure modes to a one-process experiment.

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

## Why streaming is single-request only

The SSE endpoint bypasses the batching queue and uses a Transformers streamer. Batched,
interleaved token streaming requires per-sequence stopping and substantially different
scheduling. Batching benchmarks therefore use `/v1/generate`.

Streaming admission is reserved before headers are returned, and released in generator
cleanup on completion, failure, or client cancellation. A wrapper around the background
generation thread explicitly ends the streamer on model failure so the consumer cannot
wait forever. As with non-streaming inference, disconnecting does not preempt PyTorch;
the daemon generation thread finishes locally and its remaining output is ignored.

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

A production system needs authentication, tenant-aware admission control, durable or
distributed telemetry, structured logging, model warm-up/readiness, cancellation-aware
generation, multi-worker routing, and GPU-native continuous batching. Prometheus,
Grafana, autoscaling, and careful overload behavior would follow measured requirements.
