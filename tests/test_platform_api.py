import httpx
from conftest import FakeWorker

from apps.gateway.config import Settings
from apps.gateway.main import create_app
from serving_platform.domain import RequestState


async def test_opt_in_platform_api_executes_every_control_plane_layer():
    settings = Settings(
        platform_api_enabled=True,
        api_keys="tenant-a:key-a,tenant-b:key-b",
        request_timeout_seconds=2,
    )
    compatibility_worker = FakeWorker()
    app = create_app(settings, compatibility_worker)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/platform/generate",
                headers={"Authorization": "Bearer key-a"},
                json={
                    "prompt": "schedule this request locally",
                    "max_new_tokens": 4,
                    "temperature": 0,
                    "priority": 25,
                },
            )
            request_id = response.json()["request_id"]
            owner_status = await client.get(
                f"/v1/requests/{request_id}",
                headers={"Authorization": "Bearer key-a"},
            )
            other_status = await client.get(
                f"/v1/requests/{request_id}",
                headers={"Authorization": "Bearer key-b"},
            )

    assert response.status_code == 200
    assert response.json()["scheduler_policy"] == "model_residency_aware"
    assert response.json()["status"] == "COMPLETED"
    assert response.json()["batch_size"] == 1
    assert response.json()["text"].startswith("[simulated:")
    assert compatibility_worker.batch_calls == []
    assert owner_status.status_code == 200
    assert other_status.status_code == 404

    record = app.state.platform_requests.get(request_id)
    assert record.status == RequestState.COMPLETED
    assert set(record.transition_timestamps) == {
        RequestState.RECEIVED,
        RequestState.VALIDATED,
        RequestState.ADMITTED,
        RequestState.QUEUED,
        RequestState.ASSIGNED,
        RequestState.RUNNING,
        RequestState.COMPLETED,
    }
    assert record.assigned_worker_id == "local-sim-worker-1"
    assert app.state.platform_pipeline.admission.snapshot("tenant-a") == {
        "queued": 0,
        "running": 0,
        "reserved_tokens": 0,
    }
    selected = app.state.platform_pipeline.workers.get(record.assigned_worker_id)
    assert selected is not None
    assert selected.model_cache_metrics().cold_starts == 1


async def test_platform_api_is_disabled_by_default():
    app = create_app(Settings(), FakeWorker())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/platform/generate", json={"prompt": "hello"}
        )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "platform_api_disabled"
