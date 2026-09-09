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

This allocator is real scheduler state, but Hugging Face still owns the default serving
path's `DynamicCache` tensors. The independent physical page pool below is not yet wired into
that model path.

## In progress: physical paged KV and attention

- Real PyTorch K/V page tensors indexed through scheduler-compatible block tables
- Capacity-checked multi-layer append and deterministic release
- Different-length block-table gather with masks and reference scaled-dot-product attention
- Strict CUDA-gated device, allocation, and layout report

The physical store and reference attention are implemented independently, but the default
Hugging Face model path still uses `DynamicCache`. Direct model integration, token parity,
and a no-copy page-consuming kernel remain before end-to-end PagedAttention is implemented.

## Implemented: bounded radix prefix reuse

- Immutable post-prefill backend snapshots indexed by prompt token IDs
- Longest-prefix lookup with exact-hit and partial-hit accounting
- Independent restored sequence state and real suffix-only prefill for partial matches
- Entry-count and token-count bounds with deterministic LRU eviction
- JSON and Prometheus cache hit, miss, occupancy, and eviction metrics

This avoids repeated prefix work for exact and partially matching prompts. Snapshots still
clone framework caches; this is not physical page sharing or copy-on-write paging. See
[prefix cache](prefix-cache.md).

## Implemented: conservative topology-aware grouping

- Worker inventory fields for node, device, NUMA node, PCIe root, and fabric identity
- Deterministic discovery of CUDA-peer, shared-PCIe-root, host, fabric, and network links
- Weighted shortest-path group scoring with worker-load tie breaking
- Explicit unknown-topology rejection and no unmeasured NVLink or bandwidth claims

This is a placement/control-plane capability. Model-parallel execution across the selected
group is still planned.

## In progress: distributed execution

- Validated `torchrun` rank, rendezvous, backend, and timeout configuration
- Strict NCCL initialization with explicit device selection and no CPU fallback
- All-reduce, broadcast, object-gather, barrier, lifecycle, and status boundaries
- Deterministic uneven tensor-shard planning and a real two-process validation command

Model weight sharding, tensor-parallel forward passes, and recorded multi-GPU correctness
evidence remain pending.

## Implemented: bounded local qualification harness

- Seeded allocator churn with invariant checks and complete release verification
- Prefix-cache bound checks, overload rejection, and admission reservation cleanup
- Deterministic injected runtime failure followed by a successful recovery call
- A configurable Python heap-growth gate and machine-readable JSON result

The harness is local deterministic evidence, not a GPU soak test or production certification.

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
owned or free GPU hardware. This validation is a gate within the three remaining
hardware phases below, rather than a separate implementation phase.

## Planned in dependency order

1. Integrate physical pages directly into model attention and prove token parity
2. Shard model execution across the NCCL worker group and record correctness comparisons
3. Profile-guided Triton kernel and CUDA Graph experiments with eager fallbacks

No planned item is represented as an existing capability in public documentation. GPU and
multi-GPU performance numbers, plus real-device soak and crash evidence, will appear only
after recorded runs on identified hardware.
