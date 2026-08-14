# Benchmark Methodology

Record the exact model (`sshleifer/tiny-gpt2` for development or `gpt2` for a more
compute-heavy run), CPU model/core count, RAM, OS, Python/PyTorch versions, and whether
the machine was otherwise idle. Do not compare runs made on different hardware.

Use the same fixed prompt set, request count, concurrency, `max_new_tokens`, temperature,
and server limits for both policies. Warm the model before measurement. Restart the
single-worker server with `SCHEDULER_POLICY=no_batching`, run the comparison script, then
restart with `dynamic_batch`; record `MAX_BATCH_SIZE`, `MAX_WAIT_MS`, and total token cap.
Run several trials and retain raw JSON, not only the best run.

The HTTP benchmark uses one pooled asynchronous client, a concurrency semaphore, seeded
prompt generation, warm-up requests excluded from measurements, and explicit counts for
completed requests, HTTP rejections, and transport failures. Use `--output` to retain raw
per-request latency samples and the server configuration returned by `/metrics`.

Measure client p50/p95 end-to-end latency, server queue wait, achieved requests and output
tokens per second, queue-depth high-water mark, model invocation count, batch-size
distribution, and average batch size. Keep validation tokenization, worker tokenization,
batch collection, model generation, and decoding timings separate so a throughput change
is not incorrectly attributed to model execution. TTFT is meaningful for the single-request
SSE path but is not directly comparable to non-streaming batched completion latency.

Dynamic batching can improve throughput by amortizing model work while increasing latency
through the batching window and padding. A throughput win with worse tail latency is a
tradeoff, not an unconditional improvement. Tiny GPT-2 results may be noisy because host
overhead dominates; use the sanity benchmark before selecting the final model.

## Deterministic control-plane evaluation

`python -m benchmark.run_simulated_evaluation --check` expands the versioned heterogeneous
workload, schedules it through the real weighted fair queue, models worker availability and
model-cache residency, and writes JSON, Markdown, and SVG output. Gates cover completed
requests, deadline misses, simulated throughput, p95 latency, scheduler decision overhead,
cache-hit rate, and weighted fairness.

The request order, service model, and outcomes are deterministic. Scheduler decision time
is a local CPU measurement and therefore uses a deliberately generous regression ceiling.
This evaluation does not load PyTorch, perform inference, measure CUDA, or predict real GPU
throughput. Its reports must retain the `deterministic_simulation_not_real_hardware` label.

Real CPU/GPU reports must instead use the HTTP or direct-model runner and capture hardware,
driver, CUDA, OS, Python, PyTorch, Transformers, model identifier and revision, scheduler
configuration, workload, random seed, warm-up, and raw trials.
