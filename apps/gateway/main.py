import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.control_plane import build_local_simulated_pipeline
from apps.gateway.api import router
from apps.gateway.auth import ApiKeyAuthenticator, parse_api_keys
from apps.gateway.config import Settings
from apps.gateway.limits import ConcurrencyLimiter, StreamTracker
from apps.gateway.metrics import Metrics
from apps.gateway.observability import observe_request
from apps.gateway.otel import configure_opentelemetry
from inference.backend import BackendBundle, BackendExecutor
from inference.kv_cache import KVBlockAllocator
from inference.prefix_cache import PrefixCache
from inference.token_engine import TokenGenerationEngine
from inference.worker import InferenceWorker
from runtime.huggingface import HuggingFaceTokenBackend
from scheduler.queue import RequestQueue
from scheduler.scheduler_loop import run_scheduler
from serving_platform.request_state import InMemoryRequestStateStore

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    worker=None,
    platform_pipeline=None,
    token_engine=None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.model_warmup_on_start:
            await asyncio.to_thread(app.state.worker.warmup)
        task = asyncio.create_task(run_scheduler(
            app.state.request_queue, app.state.worker, settings, app.state.metrics
        ))
        app.state.scheduler_task = task
        yield
        app.state.request_queue.close(app.state.metrics)
        streams_finished = await app.state.stream_tracker.close(
            settings.shutdown_grace_seconds
        )
        if not streams_finished:
            logger.warning("stream shutdown grace period expired")
        try:
            await asyncio.wait_for(
                asyncio.shield(task), timeout=settings.shutdown_grace_seconds
            )
        except TimeoutError:
            logger.warning("scheduler shutdown grace period expired")
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        if app.state.token_engine is not None:
            await app.state.token_engine.shutdown()
        if app.state.otel is not None:
            await asyncio.to_thread(app.state.otel.shutdown)

    app = FastAPI(title="GPU-Aware LLM Serving Platform", version="1.0.0", lifespan=lifespan)
    app.state.otel = configure_opentelemetry(
        settings.otel_service_name,
        settings.otel_exporter_otlp_endpoint,
    )

    @app.middleware("http")
    async def request_observability(request, call_next):
        tracer = app.state.otel.tracer if app.state.otel is not None else None
        return await observe_request(request, call_next, logger, tracer)
    origins = [
        origin.strip()
        for origin in settings.cors_allowed_origins.split(",")
        if origin.strip()
    ]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )
    app.state.settings = settings
    app.state.worker = worker or InferenceWorker(
        settings.model_name,
        settings.model_revision,
        settings.model_device,
        settings.model_dtype,
    )
    app.state.token_engine = token_engine
    if (
        token_engine is None
        and settings.inference_backend == "token_loop"
        and isinstance(app.state.worker, InferenceWorker)
    ):
        backend = HuggingFaceTokenBackend(app.state.worker)
        app.state.token_engine = TokenGenerationEngine(
            BackendBundle(backend, BackendExecutor()),
            settings.max_active_sequences,
            max_batch_size=settings.token_max_batch_size,
            scheduling_policy=settings.token_scheduling_policy,
            kv_allocator=KVBlockAllocator(
                settings.kv_cache_total_blocks,
                settings.kv_cache_block_size_tokens,
                settings.kv_cache_bytes_per_block,
                settings.kv_cache_high_watermark,
                settings.kv_cache_low_watermark,
            ),
            prefill_chunk_size_tokens=settings.prefill_chunk_size_tokens,
            prefix_cache=PrefixCache(
                settings.prefix_cache_max_entries,
                settings.prefix_cache_max_tokens,
            ),
        )
    app.state.metrics = Metrics(settings.metrics_sample_limit)
    app.state.request_queue = RequestQueue(settings.max_queue_size)
    app.state.limiter = ConcurrencyLimiter(settings.max_concurrent_requests)
    app.state.stream_tracker = StreamTracker()
    credentials = parse_api_keys(settings.api_keys)
    app.state.authenticator = ApiKeyAuthenticator(credentials)
    app.state.platform_requests = InMemoryRequestStateStore()
    app.state.active_gateway_requests = {}
    app.state.platform_pipeline = platform_pipeline
    if settings.platform_api_enabled and platform_pipeline is None:
        app.state.platform_pipeline = build_local_simulated_pipeline(
            app.state.platform_requests,
            set(credentials.values()) or {"default"},
            settings.max_queue_size,
            settings.max_concurrent_requests,
            settings.max_prompt_tokens + settings.max_new_tokens,
            settings.max_total_batch_tokens,
        )
    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, exc):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "request_validation_error",
                    "message": "request body validation failed",
                    "details": jsonable_encoder(exc.errors()),
                }
            },
        )

    @app.exception_handler(HTTPException)
    async def http_error(_request, exc):
        if isinstance(exc.detail, dict) and {"code", "message"} <= exc.detail.keys():
            error = exc.detail
        else:
            error = {
                "code": "http_error",
                "message": str(exc.detail),
                "details": None,
            }
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": jsonable_encoder(error)},
            headers=exc.headers,
        )

    return app


app = create_app()
