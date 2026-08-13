# Limitations

- The public gateway still executes through one local CPU Hugging Face worker until global
  routing is connected.
- CPU, CUDA, and simulated workers can be globally assigned and dispatched in the modular
  control plane, but that path is not active in the public API yet.
- Heartbeat expiry is in-memory and explicitly invoked by the control plane; a periodic
  monitor is not wired yet.
- Tenant admission and fairness are in-memory control-plane components; the public API still
  uses its compatibility limiter until the multi-worker deployment path is activated.
- The in-memory request store contains prompts for the process lifetime. Persistent storage
  is not active, and future persistence will omit prompts and generated text by default.
- Model lifecycle state and cache metrics are process-local and reset on worker restart.
- Registered model memory is an estimate; runtime allocation can still fail despite a
  reservation and safety reserve.
- Estimated completion time assumes queued requests resemble the incoming request and uses
  registered load timeout as the cold-start penalty; it is not a latency guarantee.
- Standard Hugging Face `generate()` is used.
- There is no PagedAttention, custom KV-cache paging, tensor parallelism, or true continuous
  batching.
- Active Hugging Face generation cannot be interrupted immediately.
- Metrics, queues, and request outcomes are process-local.
- Deployment is single-host and single-replica.
- No GPU benchmark result should be inferred from CPU measurements.
- Simulated throughput is a deterministic workload model, not measured hardware performance.
