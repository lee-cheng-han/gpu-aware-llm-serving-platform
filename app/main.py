import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app.api import router
from app.config import Settings
from app.limits import ConcurrencyLimiter
from app.metrics import Metrics
from inference.worker import InferenceWorker
from scheduler.queue import RequestQueue
from scheduler.scheduler_loop import run_scheduler


def create_app(settings: Settings | None = None, worker=None) -> FastAPI:
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
        with suppress(asyncio.CancelledError):
            await task

    app = FastAPI(title="LLM Inference Scheduler", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.worker = worker or InferenceWorker(settings.model_name)
    app.state.metrics = Metrics()
    app.state.request_queue = RequestQueue(settings.max_queue_size)
    app.state.limiter = ConcurrencyLimiter(settings.max_concurrent_requests)
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
