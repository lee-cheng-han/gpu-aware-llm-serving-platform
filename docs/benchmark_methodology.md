# Benchmark Methodology

Record the exact model (`sshleifer/tiny-gpt2` for development or `gpt2` for a more
compute-heavy run), CPU model/core count, RAM, OS, Python/PyTorch versions, and whether
the machine was otherwise idle. Do not compare runs made on different hardware.

Use the same fixed prompt set, request count, concurrency, `max_new_tokens`, temperature,
and server limits for both policies. Warm the model before measurement. Restart the
single-worker server with `SCHEDULER_POLICY=no_batching`, run the comparison script, then
restart with `dynamic_batch`; record `MAX_BATCH_SIZE`, `MAX_WAIT_MS`, and total token cap.
Run several trials and retain raw JSON, not only the best run.

Measure client p50/p95 end-to-end latency, server queue wait, achieved requests and output
tokens per second, and average batch size. TTFT is meaningful for the single-request SSE
path but is not directly comparable to non-streaming batched completion latency.

Dynamic batching can improve throughput by amortizing model work while increasing latency
through the batching window and padding. A throughput win with worse tail latency is a
tradeoff, not an unconditional improvement. Tiny GPT-2 results may be noisy because host
overhead dominates; use the sanity benchmark before selecting the final model.
