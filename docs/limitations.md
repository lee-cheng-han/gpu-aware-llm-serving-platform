# Limitations

- The public gateway still executes through one local CPU Hugging Face worker until global
  routing is connected.
- CPU, CUDA, and simulated worker implementations exist, but multi-worker dispatch is not
  active in the public request path yet.
- Heartbeat expiry is in-memory and explicitly invoked by the control plane; a periodic
  monitor is not wired yet.
- No global scheduler, tenant admission, or persistent request store is active yet.
- Standard Hugging Face `generate()` is used.
- There is no PagedAttention, custom KV-cache paging, tensor parallelism, or true continuous
  batching.
- Active Hugging Face generation cannot be interrupted immediately.
- Metrics, queues, and request outcomes are process-local.
- Deployment is single-host and single-replica.
- No GPU benchmark result should be inferred from CPU measurements.
- Simulated throughput is a deterministic workload model, not measured hardware performance.
