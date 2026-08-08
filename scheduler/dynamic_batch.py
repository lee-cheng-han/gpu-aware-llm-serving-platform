import asyncio
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from scheduler.no_batching import finish_timeout
from scheduler.request import InferenceRequest, RequestStatus


@dataclass
class BatchCollection(Sequence[InferenceRequest]):
    requests: list[InferenceRequest]
    queued_items: int
    total_estimated_tokens: int

    def __len__(self) -> int:
        return len(self.requests)

    def __getitem__(self, index):
        return self.requests[index]

    def __iter__(self) -> Iterator[InferenceRequest]:
        return iter(self.requests)


def _compatible(candidate: InferenceRequest, first: InferenceRequest) -> bool:
    return (
        candidate.max_new_tokens == first.max_new_tokens
        and candidate.temperature == first.temperature
    )


async def collect_batch(queue, first, settings, metrics, deferred=None) -> BatchCollection:
    """Collect a compatible bounded batch while preserving queue accounting.

    Requests moved from asyncio.Queue into the returned batch remain unfinished until
    the scheduler completes the model call. Deferred requests have already transferred
    ownership away from asyncio.Queue and therefore need no later task_done().
    """
    collection_started = time.monotonic()
    requests = [first]
    total = first.estimated_tokens + first.max_new_tokens
    queued_items = 0
    deadline = collection_started + settings.max_wait_ms / 1000

    def consider(candidate: InferenceRequest, from_queue: bool) -> str:
        nonlocal total, queued_items
        if candidate.timed_out(settings.request_timeout_seconds):
            finish_timeout(candidate, metrics)
            if from_queue:
                queue.task_done()
            return "continue"
        if candidate.status == RequestStatus.CANCELLED:
            metrics.record(candidate)
            if from_queue:
                queue.task_done()
            return "continue"

        cost = candidate.estimated_tokens + candidate.max_new_tokens
        fits = total + cost <= settings.max_total_batch_tokens
        if _compatible(candidate, first) and fits:
            requests.append(candidate)
            total += cost
            queued_items += int(from_queue)
            return "continue"

        if deferred is None:
            # Compatibility mode for direct callers: restore the candidate and stop.
            queue.put_nowait(candidate)
            if from_queue:
                queue.task_done()
            return "stop"
        deferred.append(candidate)
        if from_queue:
            queue.task_done()
        return "continue"

    # Reuse compatible deferred work first. Scan each pre-existing deferred item once,
    # rotating incompatible work to the tail so collection cannot busy-loop.
    deferred_to_scan = len(deferred) if deferred is not None else 0
    while len(requests) < settings.max_batch_size and deferred_to_scan:
        candidate = deferred.popleft()
        deferred_to_scan -= 1
        if consider(candidate, from_queue=False) == "stop":
            break

    while len(requests) < settings.max_batch_size:
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
        if consider(candidate, from_queue=True) == "stop":
            break

    collection_ms = (time.monotonic() - collection_started) * 1000
    for request in requests:
        request.batch_collection_ms = collection_ms
    return BatchCollection(requests, queued_items, total)


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
