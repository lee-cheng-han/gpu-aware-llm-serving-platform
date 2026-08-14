import httpx
from conftest import FakeWorker

from apps.gateway.config import Settings
from apps.gateway.main import create_app
from scheduler.request import InferenceRequest
from serving_platform.domain import RequestRecord, RequestState


async def test_status_endpoint_is_tenant_isolated():
    app = create_app(Settings(api_keys="tenant-a:key-a,tenant-b:key-b"), FakeWorker())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            generated = await client.post(
                "/v1/generate",
                headers={"Authorization": "Bearer key-a"},
                json={"prompt": "hello"},
            )
            request_id = generated.json()["request_id"]
            owner = await client.get(
                f"/v1/requests/{request_id}",
                headers={"Authorization": "Bearer key-a"},
            )
            other = await client.get(
                f"/v1/requests/{request_id}",
                headers={"Authorization": "Bearer key-b"},
            )
    assert owner.status_code == 200
    assert owner.json()["status"] == "completed"
    assert "prompt" not in owner.json()
    assert other.status_code == 404


async def test_cancel_endpoint_marks_queued_request_terminal():
    app = create_app(Settings(api_keys="tenant:key"), FakeWorker())
    record = RequestRecord(
        "request", "tenant", "model", "secret", 1, 1, 0, 100, False
    )
    record.transition(RequestState.VALIDATED)
    record.transition(RequestState.ADMITTED)
    record.transition(RequestState.QUEUED)
    app.state.platform_requests.create(record)
    item = InferenceRequest("secret", 1, 0, 1, "no_batching", request_id="request")
    app.state.active_gateway_requests[item.request_id] = item
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.delete(
            "/v1/requests/request", headers={"Authorization": "Bearer key"}
        )
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert app.state.platform_requests.get("request").status == RequestState.CANCELLED
