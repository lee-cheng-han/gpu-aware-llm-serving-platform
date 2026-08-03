# Architecture

One FastAPI process owns a bounded `asyncio.Queue`, one scheduler loop, one lazy-loaded
CPU Hugging Face model, and in-memory metrics.

```text
HTTP request -> validation/concurrency cap -> bounded queue -> scheduler -> CPU worker
                                                        \-> in-memory metrics
SSE request  -> validation/concurrency cap ----------------> CPU worker (unbatched)
```

`no_batching` dispatches one queue item per model call. `dynamic_batch` opens a short
window after the first item and dispatches a padded micro-batch. Blocking PyTorch work
runs in a thread so the event loop can continue serving health and metrics requests.
There is deliberately one model worker, because parallel model calls would confound the
scheduling comparison and can oversubscribe CPU threads.

The bounded queue owns admission state as well as storage. During shutdown it rejects
queued requests, inserts a sentinel to wake an idle scheduler, and lets an active blocking
generation return. Client cancellation is recorded on the request future so queued work
can be skipped without leaving unresolved API waiters.

Timing is captured at execution boundaries: API prompt-token validation, scheduler queue
wait, batch collection, worker tokenization, the blocking `model.generate()` call, and
per-result decoding. The no-batching loop awaits each complete worker call before taking
the next queue item, making it a genuinely serial reference rather than a concurrency
configuration that happens to use batches of one.

The inference worker also owns a process-local execution lock shared by single generation,
batched generation, and SSE generation. This prevents streaming from overlapping scheduler
model calls and keeps the one-worker CPU experiment honest. Incompatible batch candidates
are held in a scheduler-owned deferred deque and reconsidered as future batch seeds.
