# Limitations

- Phase 1 still executes through one local CPU Hugging Face worker.
- No real or simulated GPU worker is active yet.
- No global scheduler, worker heartbeat, tenant admission, or persistent request store is
  active yet.
- Standard Hugging Face `generate()` is used.
- There is no PagedAttention, custom KV-cache paging, tensor parallelism, or true continuous
  batching.
- Active Hugging Face generation cannot be interrupted immediately.
- Metrics, queues, and request outcomes are process-local.
- Deployment is single-host and single-replica.
- No GPU benchmark result should be inferred from CPU measurements.
