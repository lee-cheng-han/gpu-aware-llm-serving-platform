import asyncio
import time
from scheduler.request import RequestStatus


def finish_timeout(request, metrics):
    request.status = RequestStatus.TIMEOUT
    request.completed_at = time.monotonic()
    metrics.record(request)
    if request.future and not request.future.done():
        request.future.set_result(request)


async def process_one(request, worker, settings, metrics):
    if request.timed_out(settings.request_timeout_seconds):
        finish_timeout(request, metrics)
        return
    request.status, request.started_at, request.batch_size = RequestStatus.RUNNING, time.monotonic(), 1
    try:
        result = await asyncio.to_thread(
            worker.generate_one, request.prompt, request.max_new_tokens, request.temperature
        )
        request.completed_at = time.monotonic()
        request.input_tokens, request.output_tokens, request.result_text = (
            result.input_tokens, result.output_tokens, result.text
        )
        request.status = RequestStatus.TIMEOUT if request.timed_out(
            settings.request_timeout_seconds) else RequestStatus.COMPLETED
    except Exception as exc:
        request.status, request.error_message = RequestStatus.FAILED, str(exc)
        request.completed_at = time.monotonic()
    metrics.record(request)
    if request.future and not request.future.done():
        request.future.set_result(request)
