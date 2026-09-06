# Limitations

- The compatibility API still executes through one local Hugging Face worker. The opt-in
  platform API exercises global routing with deterministic in-process simulated workers.
  Authenticated local HTTP worker handles are implemented and process-tested, but are not
  yet selected from gateway configuration or intended for untrusted networks.
- The worker HTTP transport has authentication and bounded timeouts, but no TLS, service
  discovery, token rotation, durable delivery, or asynchronous connection pooling.
- Heartbeat expiry and recovery can use a Redis-compatible owner lease, but worker registry
  state remains process-local and recovery is not transactionally coupled to queue insertion.
- Shared Redis-compatible admission accounting is available, while the default platform API
  still uses the in-memory controller and the compatibility API uses its local limiter.
- Redis deployment, authentication, persistence, high availability, and client construction
  remain operator concerns; no Redis server or client package is bundled.
- The default request store contains prompts for the process lifetime. The Redis-compatible
  adapter persists metadata only; reconstructed records cannot resume without a separate
  encrypted payload store.
- Model lifecycle state and cache metrics are process-local and reset on worker restart.
- Registered model memory is an estimate; runtime allocation can still fail despite a
  reservation and safety reserve.
- Estimated completion time assumes queued requests resemble the incoming request and uses
  registered load timeout as the cold-start penalty; it is not a latency guarantee.
- The non-streaming compatibility path still uses Hugging Face `generate()` and static
  request batching. Streaming uses explicit prefill/decode with `DynamicCache`.
- Streaming has a true iteration scheduler with dynamically changing waiting/running sets.
  Real decode batches currently contain only equal-length physical caches; different-length
  sequences share scheduler iterations but require separate model calls.
- The KV block allocator enforces logical capacity and ownership, but it does not store the
  model's physical key/value tensors. There is no PagedAttention, custom tensor paging, or
  tensor parallelism yet.
- An active PyTorch forward pass cannot be interrupted immediately. Cancellation prevents
  subsequent decode and emission, then waits for the active operation before cache cleanup.
- Metrics, queues, prefix snapshots, placement reservations, and request outcomes are
  process-local. Prometheus text, trace correlation, structured logs, and optional
  OpenTelemetry OTLP/HTTP export are implemented, but no collector is bundled.
- Worker health has a staged suspect state and rich inventory, while registry persistence
  and placement-reservation transactions remain process-local. Independent control-plane
  replicas therefore cannot yet share those decisions atomically.
- Prefix caching uses longest token-prefix matches, bypasses exact prefill work, and performs
  real suffix-only prefill for partial matches. It is bounded by LRU entry/token limits and
  clones `DynamicCache` state rather than sharing physical pages.
- Topology discovery uses explicit inventory and conservative categorical costs. It does not
  infer NVLink, measure bandwidth, launch NCCL collectives, or make the gateway multi-GPU.
- The local qualification harness checks deterministic allocator, cache, overload, recovery,
  and Python-heap conditions. It is not a substitute for sustained real-GPU soak testing.
- Deployment is single-host and single-replica.
- No GPU benchmark result should be inferred from CPU measurements.
- Simulated throughput is a deterministic workload model, not measured hardware performance.
- CPU prefill/decode correctness and cache growth have been validated on the pinned model.
  The CUDA image and gated tests exist, but no real-device report has been recorded; GPU
  support remains explicitly unverified.
