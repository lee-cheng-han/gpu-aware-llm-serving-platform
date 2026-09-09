# Results

## Measured CPU token-loop proof

On 2026-08-23, the pinned SmolLM2-135M-Instruct revision was tested on an Intel Core Ultra 7
155H CPU under WSL2 with Python 3.12.14, PyTorch 2.7.0+cpu, Transformers 4.51.3, and
`float32`. The optional integration test proved that greedy manual-loop token IDs match
Hugging Face `generate()` and that the dynamic KV cache is created at prefill and grows at
decode. A second real-model integration test exercised the existing SSE URL end to end and
verified non-emission plus correct terminal usage for both EOS and configured stop tokens.

The correctness-oriented benchmark used one warmup, two measured trials, one fixed prompt,
and four output tokens. Whole-request `generate()` averaged 262.4 ms and 15.59 tokens/s;
the manual loop averaged 296.0 ms end to end, 107.5 ms TTFT, 62.8 ms per subsequent decode
step, and 13.53 tokens/s. The manual loop was about 13% slower in this very small sample; no
improvement is claimed. Legacy TTFT is unavailable because `generate()` returns only at the end.
The complete workload and raw trials are in
[`docs/evidence/token-backend-cpu.json`](evidence/token-backend-cpu.json).

No GPU hardware measurement has been run. GPU validation remains pending and no CPU result
should be generalized to CUDA.

## Continuous-scheduler correctness evidence

The optional pinned-model suite proves that two concurrent sequences share one real batched
prefill and one real batched decode operation, while matching Hugging Face greedy reference
tokens. It also proves that bounded prompt chunks grow a real cache and produce the same
first token as full prefill. Download-free tests cover dynamic joining, independent
completion/failure/cancellation, cache exhaustion before streaming, atomic rollback,
watermark hysteresis, shared references, and 500 seeded randomized allocator transitions.

These are correctness results, not a throughput improvement claim. A new measured
multi-concurrency benchmark is still required before claiming performance gains.

The radix prefix cache has download-free lifecycle, longest-prefix, exact/partial hit,
eviction, exact-prefill-bypass, and suffix-only-prefill coverage. Pinned-model parity tests
check exact restored state and partial-prefix output against full greedy prefill, but the new
partial-prefix test has not yet been run in the current environment. No prefix-cache TTFT
improvement is claimed. The local sustained-load client can produce measured HTTP latency
and throughput reports while refusing remote targets without explicit opt-in.

The download-free control-plane suite also covers staged suspect/unhealthy transitions,
heartbeat recovery, rich inventory serialization, deterministic injected runtime failures,
concurrent queue and cold-memory reservations, failed-handoff rollback, and demand-triggered
replica selection. The optional observability integration test successfully exported a
non-empty protobuf trace payload to a temporary localhost OTLP/HTTP collector; no hosted
collector was contacted.

The bounded qualification command performs seeded KV allocator churn, prefix-cache bound
checks, overload/release validation, and deterministic failure/recovery checks. Its heap gate
measures Python allocations only. Results from this command are local correctness evidence,
not GPU performance or production certification.

## Hardware-path foundation status

Download-free tests validate page-layout configuration, distributed rank validation,
deterministic uneven shard boundaries, strict NCCL no-fallback behavior through a fake
runtime, collective lifecycle, and owned shutdown. The physical PyTorch page/gather/attention
tests were collected but skipped on this lightweight environment because PyTorch is absent.
The CUDA page test and real multi-process NCCL runner are implemented but have not run here
because no CUDA devices are exposed. No paged-attention or multi-GPU performance result is
claimed.

The versioned deterministic baseline is available at
[`benchmark/results/simulated/heterogeneous/report.md`](../benchmark/results/simulated/heterogeneous/report.md).
It is a control-plane regression fixture, not a GPU-aware platform measurement.
