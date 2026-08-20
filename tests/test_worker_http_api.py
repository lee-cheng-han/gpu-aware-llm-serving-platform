import httpx
import pytest

from apps.worker.api import create_worker_http_app
from apps.worker.factory import create_local_simulated_worker
from apps.worker.serialization import model_to_dict


@pytest.mark.asyncio
async def test_worker_http_api_requires_authentication_and_manages_model_lifecycle():
    worker, model = create_local_simulated_worker("worker-api")
    app = create_worker_http_app(worker, [model], "test-secret")
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
        assert (await client.get("/internal/health")).status_code == 401
        headers = {"X-Worker-Token": "test-secret"}

        registered = await client.post("/internal/register", headers=headers)
        assert registered.status_code == 200
        assert registered.json()["worker_id"] == "worker-api"

        loaded = await client.post(
            "/internal/models/load", headers=headers, json=model_to_dict(model)
        )
        assert loaded.status_code == 200
        heartbeat = await client.post("/internal/heartbeat", headers=headers)
        assert heartbeat.json()["resident_models"] == [model.model_id]


@pytest.mark.asyncio
async def test_worker_http_api_rejects_unregistered_model_definition():
    worker, model = create_local_simulated_worker("worker-api")
    app = create_worker_http_app(worker, [model], "test-secret")
    transport = httpx.ASGITransport(app=app)
    payload = model_to_dict(model)
    payload["revision"] = "unexpected"

    async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
        headers = {"X-Worker-Token": "test-secret"}
        await client.post("/internal/register", headers=headers)
        response = await client.post("/internal/models/load", headers=headers, json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "model is not registered"
