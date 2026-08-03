import asyncio
import time

from scheduler.no_batching import finish_timeout
from scheduler.request import RequestStatus


async def collect_batch(queue, first, settings, metrics, deferred=None):
    collection_started = time.monotonic()
    batch, total = [first], first.estimated_tokens + first.max_new_tokens
    deadline = time.monotonic() + settings.max_wait_ms / 1000
    while len(batch) < settings.max_batch_size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            candidate = await asyncio.wait_for(queue.get(), remaining)
        except TimeoutError:
            break
        if candidate is None:
            # Preserve the shutdown signal for the outer scheduler loop.
            queue.task_done()
            queue.put_nowait(None)
            break
        if candidate.timed_out(settings.request_timeout_seconds):
            finish_timeout(candidate, metrics)
            queue.task_done()
            continue
        if candidate.status == RequestStatus.CANCELLED:
            metrics.record(candidate)
            queue.task_done()
            continue
        # A single generate() call has one sampling configuration. Keep v1
        # batches semantically correct by batching only compatible requests.
        if (candidate.max_new_tokens != first.max_new_tokens or
                candidate.temperature != first.temperature):
            if deferred is None:
                queue.put_nowait(candidate)
                queue.task_done()
                break
            deferred.append(candidate)
            queue.task_done()
            continue
        cost = candidate.estimated_tokens + candidate.max_new_tokens
        if batch and total + cost > settings.max_total_batch_tokens:
            if deferred is None:
                queue.put_nowait(candidate)
                queue.task_done()
                break
            deferred.append(candidate)
            queue.task_done()
            continue
        batch.append(candidate)
        total += cost
    collection_ms = (time.monotonic() - collection_started) * 1000
    for request in batch:
        request.batch_collection_ms = collection_ms
    return batch


async def process_batch(batch, worker, settings, metrics):
    now = time.monotonic()
    active = []
    for request in batch:
        if request.status == RequestStatus.CANCELLED:
            metrics.record(request)
        elif request.timed_out(settings.request_timeout_seconds):
            finish_timeout(request, metrics)
        else:
            request.status, request.started_at = RequestStatus.RUNNING, now
            request.batch_size = 0  # Set once expired/cancelled entries are filtered.
            active.append(request)
    if not active:
        return
    for request in active:
        request.batch_size = len(active)
    metrics.record_model_invocation(
        batch_size=len(active),
        collection_ms=max(request.batch_collection_ms for request in active),
    )
    metrics.model_execution_started()
    try:
        results = await asyncio.to_thread(
            worker.generate_batch, [r.prompt for r in active],
            active[0].max_new_tokens, active[0].temperature,
        )
        completed = time.monotonic()
        for request, result in zip(active, results, strict=True):
            cancelled = request.status == RequestStatus.CANCELLED
            request.completed_at = request.completed_at or completed
            request.input_tokens, request.output_tokens = result.input_tokens, result.output_tokens
            request.result_text = result.text
            request.worker_tokenization_ms = result.tokenization_ms
            request.generation_ms = result.generation_ms
            request.decoding_ms = result.decoding_ms
            request.status = (
                RequestStatus.CANCELLED if cancelled else
                RequestStatus.TIMEOUT if request.timed_out(settings.request_timeout_seconds)
                else RequestStatus.COMPLETED
            )
            metrics.record(request)
            request.finish(request.status, request.error_message)
    except Exception as exc:
        for request in active:
            request.status, request.error_message = RequestStatus.FAILED, str(exc)
            request.completed_at = time.monotonic()
            metrics.record(request)
            request.finish(request.status, request.error_message)
    except asyncio.CancelledError:
        for request in active:
            request.finish(RequestStatus.FAILED, "shutdown grace period expired")
            metrics.record(request)
        raise
    finally:
        metrics.model_execution_finished()
