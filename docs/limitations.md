# Limitations

- The compatibility API still executes through one local Hugging Face worker. The opt-in
  platform API exercises global routing with deterministic in-process simulated workers.
  Authenticated local HTTP worker handles are implemented and process-tested, but are not
  yet selected from gateway configuration or intended for untrusted networks.
- The worker HTTP transport has authentication and bounded timeouts, but no TLS, service
  discovery, token rotation, durable delivery, or asynchronous connection pooling.
- Heartbeat expiry and recovery are supervised, but registry state and leadership remain
  process-local.
- Tenant admission and fairness are in-memory control-plane components; the public API still
  uses its compatibility limiter until the multi-worker deployment path is activated.
- The default request store contains prompts for the process lifetime. The Redis-compatible
  adapter persists metadata only; reconstructed records cannot resume without a separate
  encrypted payload store.
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
- The CUDA image and profile are packaging examples; no real-device result has been recorded.
