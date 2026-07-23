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
