import httpx
from conftest import FakeWorker

from app.main import create_app


async def test_health(settings):
    app = create_app(settings, FakeWorker())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readiness_distinguishes_unloaded_model(settings):
    app = create_app(settings, FakeWorker())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/ready")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_not_ready"


async def test_warmup_makes_model_ready(settings):
    from dataclasses import replace

    worker = FakeWorker()
    app = create_app(replace(settings, model_warmup_on_start=True), worker)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/ready")
    assert response.status_code == 200
    assert response.json()["context_window_tokens"] == 1024
