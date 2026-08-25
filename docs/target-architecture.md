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

## In progress: CUDA validation

- Run the gated proof on an identified CUDA device
- Verify input/cache device placement, real CUDA memory growth, CPU greedy parity, and strict
  no-fallback behavior
- Record the machine-readable hardware, driver, CUDA, runtime, workload, and result report

CUDA support is implemented but remains explicitly unverified until this report passes on
owned or free GPU hardware.

## Planned in dependency order

1. Physical paged KV tensors, arbitrary-length decode batches, and paged attention
2. Prometheus metrics, OpenTelemetry tracing, structured logs, and sustained-load benchmarks
3. Prefix-cache sharing, bounded eviction, and measured TTFT comparison
4. Rich worker/GPU inventory, suspect health state, fault injection, and multi-process demo
5. Atomic GPU placement reservations, replication decisions, and demand-aware placement
6. Multi-GPU execution with `torch.distributed`/NCCL and correctness comparisons
7. Topology discovery and topology-aware worker-group placement
8. Profile-guided Triton kernel and CUDA Graph experiments with eager fallbacks
9. Sustained load, memory-leak, crash, overload, and production-qualification evidence

No planned item is represented as an existing capability in public documentation. GPU and
multi-GPU performance numbers will appear only after recorded runs on identified hardware.
