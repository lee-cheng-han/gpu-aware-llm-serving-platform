# Worker Reliability and Fault Injection

Worker health uses a staged lifecycle. A delayed heartbeat first changes `HEALTHY` to
`SUSPECT`; routing immediately excludes suspect workers, but recovery does not requeue their
requests yet. A valid heartbeat clears the failure counter and returns the worker to
`HEALTHY`. Only expiry beyond the full timeout changes it to `UNHEALTHY` and triggers the
existing owner-leased recovery rules.

Worker snapshots carry a monotonic heartbeat sequence, consecutive failure count, last
reason-coded error, GPU index, compute capability, driver/runtime versions, memory totals,
queue depth, active batches, throughput, resident/loading models, and drain state. These
fields round-trip through the authenticated worker transport. Real CUDA values are reported
only when PyTorch can query a CUDA device; simulated values remain explicitly labelled.

`FaultInjectingRuntime` decorates any runtime with deterministic per-operation rules. A rule
can delay an operation and fail exact call numbers with a stable error code. It supports
repeatable load, warmup, generation, stream, capacity, and unload drills without external
services or paid infrastructure. It is a test/qualification mechanism and is not enabled in
the default serving path.
