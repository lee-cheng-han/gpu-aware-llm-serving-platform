import asyncio
import time

from scheduler.request import RequestStatus


def finish_timeout(request, metrics):
    request.finish(RequestStatus.TIMEOUT)
    metrics.record(request)


async def process_one(request, worker, settings, metrics):
    if request.status == RequestStatus.CANCELLED:
        metrics.record(request)
        return
    if request.timed_out(settings.request_timeout_seconds):
        finish_timeout(request, metrics)
        return
    request.status, request.started_at, request.batch_size = RequestStatus.RUNNING, time.monotonic(), 1
    request.batch_collection_ms = 0
    metrics.record_model_invocation(batch_size=1, collection_ms=0)
    metrics.model_execution_started()
    try:
        result = await asyncio.to_thread(
            worker.generate_one, request.prompt, request.max_new_tokens, request.temperature
        )
        cancelled = request.status == RequestStatus.CANCELLED
        request.completed_at = request.completed_at or time.monotonic()
        request.input_tokens, request.output_tokens, request.result_text = (
            result.input_tokens, result.output_tokens, result.text
        )
        request.worker_tokenization_ms = result.tokenization_ms
        request.generation_ms = result.generation_ms
        request.decoding_ms = result.decoding_ms
        request.status = (
            RequestStatus.CANCELLED if cancelled else
            RequestStatus.TIMEOUT if request.timed_out(settings.request_timeout_seconds)
            else RequestStatus.COMPLETED
        )
    except Exception as exc:
        request.status, request.error_message = RequestStatus.FAILED, str(exc)
        request.completed_at = time.monotonic()
    except asyncio.CancelledError:
        request.finish(RequestStatus.FAILED, "shutdown grace period expired")
        metrics.record(request)
        raise
    finally:
        metrics.model_execution_finished()
    metrics.record(request)
    request.finish(request.status, request.error_message)
