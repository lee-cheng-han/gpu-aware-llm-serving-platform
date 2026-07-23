import asyncio
import json
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.errors import APIError
from app.limits import validate_request
from app.schemas import GenerateRequest, GenerateResponse, ReadinessResponse
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


async def _validated(body: GenerateRequest, request: Request) -> int:
    if not body.prompt.strip():
        raise APIError(400, "empty_prompt", "prompt must not be empty")
    worker = request.app.state.worker
    # Both operations touch the same tokenizer/model. Keep them serial: the
    # context lookup is trivial after the token-count call has loaded the model.
    if getattr(worker, "run_inline_for_tests", False):
        tokens = worker.count_prompt_tokens(body.prompt)
    else:
        tokens = await asyncio.to_thread(worker.count_prompt_tokens, body.prompt)
    context_window = worker.context_window_tokens()
    validate_request(
        body.prompt,
        body.max_new_tokens,
        tokens,
        context_window,
        request.app.state.settings,
    )
    return tokens


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
            tokens = await _validated(body, request)
            item = InferenceRequest(
                body.prompt, body.max_new_tokens, body.temperature, tokens,
                state.settings.scheduler_policy,
            )
            try:
                await state.request_queue.submit(item)
            except asyncio.QueueFull:
                state.metrics.rejected(queue_full=True)
                raise APIError(503, "queue_full", "scheduler queue is full")
            result = await item.future
            if result.status == RequestStatus.TIMEOUT:
                raise APIError(504, "request_timeout", "request timed out")
            if result.status == RequestStatus.FAILED:
                raise APIError(
                    500, "generation_failed", "model generation failed",
                    {"reason": result.error_message},
                )
            return _response(result)
    except HTTPException as exc:
        if exc.status_code == 429:
            state.metrics.rejected()
        raise


@router.post("/v1/generate_stream")
async def generate_stream(body: GenerateRequest, request: Request):
    """Single-request SSE; it intentionally bypasses the batching queue."""
    state = request.app.state
    state.metrics.received()
    tokens = await _validated(body, request)
    if state.limiter.active >= state.limiter.maximum:
        state.metrics.rejected()
        raise APIError(
            429, "concurrency_limit_exceeded",
            "maximum concurrent requests exceeded",
        )
    item = InferenceRequest(
        body.prompt, body.max_new_tokens, body.temperature, tokens, "single_stream"
    )
    item.queued_at = item.started_at = time.monotonic()
    item.status = RequestStatus.RUNNING

    async def events():
        async with state.limiter.slot():
            try:
                iterator = state.worker.stream(body.prompt, body.max_new_tokens, body.temperature)
                while True:
                    token = await asyncio.to_thread(next, iterator, None)
                    if token is None:
                        break
                    if not item.first_token_at:
                        item.first_token_at = time.monotonic()
                    item.output_tokens += 1
                    yield f"data: {json.dumps({'request_id': item.request_id, 'token': token})}\n\n"
                item.completed_at = time.monotonic()
                item.status = (RequestStatus.TIMEOUT if item.timed_out(
                    state.settings.request_timeout_seconds) else RequestStatus.COMPLETED)
            except Exception as exc:
                item.status, item.error_message = RequestStatus.FAILED, str(exc)
                item.completed_at = time.monotonic()
                yield f"data: {json.dumps({'request_id': item.request_id, 'error': str(exc)})}\n\n"
            item.input_tokens = tokens
            state.metrics.record(item)
            yield f"data: {json.dumps({'request_id': item.request_id, 'done': True})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.get("/metrics")
async def metrics(request: Request):
    state = request.app.state
    return state.metrics.snapshot(state.request_queue.queue.qsize(), state.limiter.active)
