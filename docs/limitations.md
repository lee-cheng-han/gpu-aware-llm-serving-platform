# Limitations

- The public gateway still executes through one local CPU Hugging Face worker until global
  routing is connected.
- CPU, CUDA, and simulated workers can be globally assigned and dispatched in the modular
  control plane, but that path is not active in the public API yet.
- Heartbeat expiry is in-memory and explicitly invoked by the control plane; a periodic
  monitor is not wired yet.
- No tenant admission or persistent request store is active yet.
- Global routing currently requires model residency; cold-start placement and memory-aware
  routing are not implemented yet.
- Standard Hugging Face `generate()` is used.
- There is no PagedAttention, custom KV-cache paging, tensor parallelism, or true continuous
  batching.
- Active Hugging Face generation cannot be interrupted immediately.
- Metrics, queues, and request outcomes are process-local.
- Deployment is single-host and single-replica.
- No GPU benchmark result should be inferred from CPU measurements.
- Simulated throughput is a deterministic workload model, not measured hardware performance.
