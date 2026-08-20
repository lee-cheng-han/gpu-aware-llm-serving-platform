from __future__ import annotations

import hmac
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException

from apps.worker.serialization import (
    capacity_to_dict,
    model_from_dict,
    request_from_dict,
    request_to_dict,
    result_to_dict,
    worker_state_to_dict,
)
from apps.worker.service import ManagedWorker
from serving_platform.domain import ModelDefinition, RequestRecord


def create_worker_http_app(
    worker: ManagedWorker,
    models: list[ModelDefinition],
    auth_token: str,
) -> FastAPI:
    if not auth_token:
        raise ValueError("worker transport auth token is required")
    catalog = {model.model_id: model for model in models}
    requests: dict[str, RequestRecord] = {}

    async def authenticate(x_worker_token: Annotated[str, Header()] = "") -> None:
        if not hmac.compare_digest(x_worker_token, auth_token):
            raise HTTPException(status_code=401, detail="invalid worker credential")

    app = FastAPI(
        title="Local LLM Worker Transport",
        version="1.0.0",
        dependencies=[Depends(authenticate)],
    )
    app.state.worker = worker
    app.state.models = catalog
    app.state.requests = requests

    @app.post("/internal/register")
    async def register():
        return worker_state_to_dict(worker.register())

    @app.post("/internal/heartbeat")
    async def heartbeat():
        return worker_state_to_dict(worker.heartbeat())

    @app.get("/internal/health")
    async def health():
        return {"status": worker.health().value}

    @app.get("/internal/capacity")
    async def capacity():
        return capacity_to_dict(worker.capacity())

    @app.post("/internal/models/load")
    async def load_model(payload: dict[str, Any]):
        supplied = model_from_dict(payload)
        expected = catalog.get(supplied.model_id)
        if expected is None or supplied != expected:
            raise HTTPException(status_code=400, detail="model is not registered")
        worker.load_model(expected)
        return {"loaded": expected.model_id}

    @app.post("/internal/models/{model_id}/warmup")
    async def warmup_model(model_id: str):
        worker.warmup_model(model_id)
        return {"warmed": model_id}

    @app.delete("/internal/models/{model_id}")
    async def unload_model(model_id: str):
        worker.unload_model(model_id)
        return {"unloaded": model_id}

    @app.post("/internal/requests")
    async def enqueue_request(payload: dict[str, Any]):
        request = request_from_dict(payload)
        worker.enqueue_request(request)
        requests[request.request_id] = request
        return request_to_dict(request)

    @app.delete("/internal/requests/{request_id}")
    async def cancel_request(request_id: str):
        cancelled = worker.cancel_request(request_id)
        request = requests.get(request_id)
        return {
            "cancelled": cancelled,
            "request": request_to_dict(request) if request is not None else None,
        }

    @app.post("/internal/batches/execute")
    async def execute_batch():
        executions = worker.execute_batch()
        return {
            "executions": [
                {
                    "request_id": execution.request_id,
                    "result": result_to_dict(execution.result),
                    "request": request_to_dict(requests[execution.request_id]),
                }
                for execution in executions
            ]
        }

    @app.post("/internal/drain")
    async def drain():
        worker.drain()
        return {"draining": True}

    @app.post("/internal/shutdown")
    async def shutdown():
        worker.shutdown()
        return {"status": "stopped"}

    return app
