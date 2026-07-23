import asyncio
import time
from scheduler.no_batching import finish_timeout
from scheduler.request import RequestStatus


async def collect_batch(queue, first, settings, metrics):
    batch, total = [first], first.estimated_tokens + first.max_new_tokens
    deadline = time.monotonic() + settings.max_wait_ms / 1000
    while len(batch) < settings.max_batch_size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            candidate = await asyncio.wait_for(queue.get(), remaining)
        except asyncio.TimeoutError:
            break
        if candidate.timed_out(settings.request_timeout_seconds):
            finish_timeout(candidate, metrics)
            queue.task_done()
            continue
        # A single generate() call has one sampling configuration. Keep v1
        # batches semantically correct by batching only compatible requests.
        if (candidate.max_new_tokens != first.max_new_tokens or
                candidate.temperature != first.temperature):
            queue.put_nowait(candidate)
            queue.task_done()
            break
        cost = candidate.estimated_tokens + candidate.max_new_tokens
        if batch and total + cost > settings.max_total_batch_tokens:
            # FIFO is relaxed here: safely return the candidate to the queue.
            queue.put_nowait(candidate)
            queue.task_done()
            break
        batch.append(candidate)
        total += cost
    return batch


async def process_batch(batch, worker, settings, metrics):
    now = time.monotonic()
    active = []
    for request in batch:
        if request.timed_out(settings.request_timeout_seconds):
            finish_timeout(request, metrics)
        else:
            request.status, request.started_at = RequestStatus.RUNNING, now
            request.batch_size = len(batch)
            active.append(request)
    if not active:
        return
    try:
        results = await asyncio.to_thread(
            worker.generate_batch, [r.prompt for r in active],
            active[0].max_new_tokens, active[0].temperature,
        )
        completed = time.monotonic()
        for request, result in zip(active, results):
            request.completed_at = completed
            request.input_tokens, request.output_tokens = result.input_tokens, result.output_tokens
            request.result_text = result.text
            request.status = (RequestStatus.TIMEOUT if request.timed_out(
                settings.request_timeout_seconds) else RequestStatus.COMPLETED)
            metrics.record(request)
            if request.future and not request.future.done():
                request.future.set_result(request)
    except Exception as exc:
        for request in active:
            request.status, request.error_message = RequestStatus.FAILED, str(exc)
            request.completed_at = time.monotonic()
            metrics.record(request)
            if request.future and not request.future.done():
                request.future.set_result(request)
