# Target Architecture

Status labels are evidence-based: **implemented**, **in progress**, and **planned**.

```text
client -> FastAPI gateway -> admission -> global routing -> token scheduler
                                                       -> placement engine
                                                       -> worker transport
                                                            -> GPU worker group
                                                               -> inference backend
                                                               -> paged KV cache
                                                               -> CUDA/PyTorch/Triton
```

## Implemented

- Authenticated FastAPI gateway, structured errors, SSE, bounded admission, and cancellation
- Static local batching plus deterministic global and worker-local scheduling
- Tenant quotas, weighted fairness, request persistence, explicit retries, and recovery
- CPU/CUDA Hugging Face adapter and deterministic simulated GPU runtime
- Memory- and residency-aware placement, managed model lifecycle, and replicas
- Authenticated local multi-process worker transport and owner-leased recovery coordination
- Local JSON metrics, repeatable simulated evaluation, Docker, kind, and manual CI

## Implemented: real token engine (CPU validated)

- Pinned SmolLM2 Hugging Face backend with explicit prompt tokenization
- Real prefill forward pass creating a framework-managed dynamic KV cache
- Autoregressive one-token decode, separate sampling, streaming, and resource cleanup
- CPU correctness evidence and strict CUDA-gated test coverage

This step uses Hugging Face `DynamicCache`. It has a real inference effect but is not the
project's planned paged/block allocator.

## Implemented: continuous scheduling and prefill/decode policies

- Central waiting/running sequence sets instead of one model loop per HTTP generator
- Iteration-level admission and independent completion, failure, timeout, and cancellation
- Real batched Hugging Face prefill and decode for physically compatible cache lengths
- Prefill-first, decode-first, balanced, and bounded chunked-prefill execution
- Per-request event queues preserving the existing SSE contract

## Implemented: KV block ownership and admission accounting

- Atomic fixed-block allocation, resize, release, block tables, and reference counts
- Complete sequence-capacity reservation before SSE response headers
- High/low watermark hysteresis, allocation-failure handling, byte estimates, and metrics
- Seeded randomized invariant, exhaustion, reuse, fragmentation, and cancellation tests

This allocator is real scheduler state, but Hugging Face still owns the physical
`DynamicCache` tensors. Physical paged tensor storage and a paged-attention kernel are in
progress and are not claimed as implemented.

## Implemented: bounded exact-prefix reuse

- Immutable post-prefill backend snapshots keyed by exact prompt token IDs
- Independent restored sequence state with request-specific sampling
- Entry-count and token-count bounds with deterministic LRU eviction
- JSON and Prometheus cache hit, miss, occupancy, and eviction metrics

This avoids repeated prefill forward passes for exact prompt matches. It is not radix-tree
longest-prefix matching or physical page sharing; see [prefix cache](prefix-cache.md).

## Implemented: local observability foundations

- Backward-compatible JSON metrics plus Prometheus text content negotiation
- W3C trace-context propagation and response correlation headers
- Structured, secret-safe HTTP request logs
- Bounded local sustained-load runner with remote-target cost guard
- Optional OpenTelemetry SDK server spans and OTLP/HTTP batch export with shutdown flushing

No collector or paid telemetry service is required by the default configuration.

## Implemented: worker inventory and staged health

- GPU/runtime inventory, monotonic heartbeat sequences, and reason-coded failures
- `HEALTHY` to `SUSPECT` to `UNHEALTHY` expiry with recovery from the suspect state
- Routing exclusion before destructive recovery begins
- Runtime-agnostic deterministic delay/failure injection and multi-process transport proof

## Implemented: atomic local placement and demand-aware replication

- Selection plus reservation under one lock for concurrent local schedulers
- Pending queue and cold-model memory reflected in later placement decisions
- Release on successful enqueue and every failed handoff
- Demand threshold, desired replica count, and scale-out terms in routing explanations

The reservation boundary is process-local. A shared transactional adapter remains necessary
before multiple independent control-plane replicas can place concurrently.

## In progress: CUDA validation

- Run the gated proof on an identified CUDA device
- Verify input/cache device placement, real CUDA memory growth, CPU greedy parity, and strict
  no-fallback behavior
- Record the machine-readable hardware, driver, CUDA, runtime, workload, and result report

CUDA support is implemented but remains explicitly unverified until this report passes on
owned or free GPU hardware.

## Planned in dependency order

1. Physical paged KV tensors, arbitrary-length decode batches, and paged attention
2. Radix-tree partial-prefix sharing and measured TTFT comparison
3. Multi-GPU execution with `torch.distributed`/NCCL and correctness comparisons
4. Topology discovery and topology-aware worker-group placement
5. Profile-guided Triton kernel and CUDA Graph experiments with eager fallbacks
6. Sustained memory-leak, crash, overload, and production-qualification evidence

No planned item is represented as an existing capability in public documentation. GPU and
multi-GPU performance numbers will appear only after recorded runs on identified hardware.
