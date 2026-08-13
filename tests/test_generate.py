import httpx
from conftest import FakeWorker

from apps.gateway.main import create_app
from serving_platform.domain import RequestState


async def test_successful_greedy_generation(settings):
    worker = FakeWorker()
    app = create_app(settings, worker)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/v1/generate", json={
                "prompt": "hello world",
                "max_new_tokens": 2,
                "temperature": 0,
            })
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "COMPLETED"
    assert body["input_tokens"] == 2
    assert body["output_tokens"] == 2
    assert worker.temperatures == [0]
    platform_record = app.state.platform_requests.get(body["request_id"])
    assert platform_record.status == RequestState.COMPLETED
    assert platform_record.priority == 0
    assert (
        platform_record.transition_timestamps[RequestState.QUEUED]
        <= platform_record.transition_timestamps[RequestState.RUNNING]
        <= platform_record.transition_timestamps[RequestState.COMPLETED]
    )
