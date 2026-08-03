import asyncio
import json
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.errors import APIError
from app.limits import validate_request
from app.schemas import GenerateRequest, GenerateResponse, ReadinessResponse
from scheduler.queue import QueueClosed
from scheduler.request import InferenceRequest, RequestStatus

router = APIRouter()


def _response(item: InferenceRequest) -> GenerateResponse:
    return GenerateResponse(
        request_id=item.request_id, text=item.result_text,
        input_tokens=item.input_tokens, output_tokens=item.output_tokens,
        latency_ms=(item.completed_at - item.queued_at) * 1000,
        queue_wait_ms=(item.started_at - item.queued_at) * 1000 if item.started_at else 0,
        scheduler_policy=item.scheduler_policy, batch_size=item.batch_size,
        status=item.status.value,
    )


async def _validated(body: GenerateRequest, request: Request) -> tuple[int, float]:
    if not body.prompt.strip():
        raise APIError(400, "empty_prompt", "prompt must not be empty")
    worker = request.app.state.worker
    # Both operations touch the same tokenizer/model. Keep them serial: the
    # context lookup is trivial after the token-count call has loaded the model.
    tokenization_started = time.perf_counter()
    if getattr(worker, "run_inline_for_tests", False):
        tokens = worker.count_prompt_tokens(body.prompt)
    else:
        tokens = await asyncio.to_thread(worker.count_prompt_tokens, body.prompt)
    tokenization_ms = (time.perf_counter() - tokenization_started) * 1000
    context_window = worker.context_window_tokens()
    validate_request(
        body.prompt,
        body.max_new_tokens,
        tokens,
        context_window,
        request.app.state.settings,
    )
    return tokens, tokenization_ms


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready", response_model=ReadinessResponse)
async def ready(request: Request):
    worker = request.app.state.worker
    if not worker.is_ready:
        raise APIError(
            503,
            "model_not_ready",
            "model has not loaded successfully",
            {"model_name": worker.model_name, "load_error": worker.load_error},
        )
    return ReadinessResponse(
        status="ready",
        model_name=worker.model_name,
        model_loaded=True,
        context_window_tokens=worker.context_window_tokens(),
    )


@router.post("/v1/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest, request: Request):
    state = request.app.state
    state.metrics.received()
    try:
        async with state.limiter.slot():
            tokens, validation_tokenization_ms = await _validated(body, request)
            item = InferenceRequest(
                body.prompt, body.max_new_tokens, body.temperature, tokens,
                state.settings.scheduler_policy,
            )
            item.validation_tokenization_ms = validation_tokenization_ms
            try:
                await state.request_queue.submit(item)
            except QueueClosed:
                state.metrics.rejected(reason="service_shutting_down")
                raise APIError(
                    503, "service_shutting_down", "server is shutting down"
                ) from None
            except asyncio.QueueFull:
                raise APIError(503, "queue_full", "scheduler queue is full") from None
            try:
                # Shield keeps client cancellation from cancelling the scheduler's
                # result handle. Queued cancellations can then be skipped safely.
                result = await asyncio.shield(item.future)
            except asyncio.CancelledError:
                item.cancel("client disconnected")
                raise
            if result.status == RequestStatus.TIMEOUT:
                raise APIError(504, "request_timeout", "request timed out")
            if result.status == RequestStatus.REJECTED:
                raise APIError(503, "service_shutting_down", result.error_message)
            if result.status == RequestStatus.FAILED:
                raise APIError(
                    500, "generation_failed", "model generation failed",
                    {"reason": result.error_message},
                )
            return _response(result)
    except HTTPException as exc:
        if 400 <= exc.status_code < 500 or exc.status_code == 503:
            code = exc.detail.get("code") if isinstance(exc.detail, dict) else ""
            if code != "service_shutting_down":
                state.metrics.rejected(
                    queue_full=code == "queue_full",
                    reason=code or "http_error",
                )
        raise


@router.post("/v1/generate_stream")
async def generate_stream(body: GenerateRequest, request: Request):
    """Single-request SSE; it intentionally bypasses the batching queue."""
    state = request.app.state
    state.metrics.received()
    try:
        await state.stream_tracker.start()
    except HTTPException as exc:
        code = exc.detail.get("code") if isinstance(exc.detail, dict) else "http_error"
        state.metrics.rejected(reason=code)
        raise
    try:
        # Reserve admission before returning SSE headers. Acquiring inside the
        # body iterator creates a race in which two streams can both pass a
        # pre-check and one fails only after the response has started.
        await state.limiter.acquire()
    except HTTPException as exc:
        await state.stream_tracker.finish()
        code = exc.detail.get("code") if isinstance(exc.detail, dict) else "http_error"
        state.metrics.rejected(reason=code)
        raise
    try:
        tokens, validation_tokenization_ms = await _validated(body, request)
    except BaseException as exc:
        await state.limiter.release()
        await state.stream_tracker.finish()
        if isinstance(exc, HTTPException):
            code = exc.detail.get("code") if isinstance(exc.detail, dict) else "http_error"
            state.metrics.rejected(reason=code)
        raise

    item = InferenceRequest(
        body.prompt, body.max_new_tokens, body.temperature, tokens, "single_stream"
    )
    item.queued_at = item.started_at = time.monotonic()
    item.validation_tokenization_ms = validation_tokenization_ms
    item.status = RequestStatus.RUNNING

    async def events():
        try:
            state.metrics.record_model_invocation(
                batch_size=1, collection_ms=0, batched_call=False
            )
            state.metrics.model_execution_started()
            iterator = state.worker.stream(body.prompt, body.max_new_tokens, body.temperature)
            while True:
                if getattr(state.worker, "stream_inline_for_tests", False):
                    chunk = next(iterator, None)
                else:
                    token = await asyncio.to_thread(next, iterator, None)
                    chunk = token
                if chunk is None:
                    break
                if not item.first_token_at:
                    item.first_token_at = time.monotonic()
                item.output_tokens = chunk.output_tokens
                yield f"data: {json.dumps({'request_id': item.request_id, 'token': chunk.text})}\n\n"
            item.completed_at = time.monotonic()
            item.status = (RequestStatus.TIMEOUT if item.timed_out(
                state.settings.request_timeout_seconds) else RequestStatus.COMPLETED)
            yield f"data: {json.dumps({'request_id': item.request_id, 'done': True})}\n\n"
        except asyncio.CancelledError:
            item.cancel("streaming client disconnected")
            raise
        except Exception:
            item.finish(RequestStatus.FAILED, "streaming model generation failed")
            error_event = {
                'request_id': item.request_id,
                'error': {
                    'code': 'generation_failed',
                    'message': 'streaming model generation failed',
                },
            }
            yield f"data: {json.dumps(error_event)}\n\n"
            yield f"data: {json.dumps({'request_id': item.request_id, 'done': True})}\n\n"
        finally:
            item.input_tokens = tokens
            state.metrics.record(item)
            state.metrics.model_execution_finished()
            await state.limiter.release()
            await state.stream_tracker.finish()

    return StreamingResponse(events(), media_type="text/event-stream")


@router.get("/metrics")
async def metrics(request: Request):
    state = request.app.state
    snapshot = state.metrics.snapshot(
        queued=state.request_queue.queue.qsize(),
        active=state.limiter.active,
        max_queue_depth=state.request_queue.max_observed_size,
    )
    snapshot.update({
        "model_name": state.settings.model_name,
        "scheduler_policy": state.settings.scheduler_policy,
        "max_batch_size": state.settings.max_batch_size,
        "max_wait_ms": state.settings.max_wait_ms,
        "max_total_batch_tokens": state.settings.max_total_batch_tokens,
    })
    return snapshot
