import json
import time

import httpx

from apps.gateway.main import create_app
from inference.kv_cache import KVCacheSnapshot
from inference.token_engine import TokenSchedulerSnapshot
from inference.token_types import FinishReason, GeneratedToken, GenerationComplete


class FakeTokenEngine:
    def __init__(self):
        self.request_id = None
        self.cancelled = False

    async def validate_sampling(self, sampling):
        return None

    def reserve(self, request_id, prompt, sampling, deadline, **kwargs):
        self.request_id = request_id
        assert deadline > time.monotonic()

    async def stream(self, request_id):
        yield GeneratedToken(42, " answer", 0, 3)
        yield GenerationComplete(FinishReason.LENGTH, 2, 1, 2.5, 1.5)

    def cancel(self, request_id):
        self.cancelled = True
        return True

    async def shutdown(self):
        return None

    def scheduler_snapshot(self):
        return TokenSchedulerSnapshot(
            0,
            0,
            3,
            1,
            2,
            2,
            "balanced",
            KVCacheSnapshot(16, 0, 16, 4, 128, 0, 0, 0, False),
        )


def events(response):
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


async def test_token_stream_adds_token_identity_and_terminal_usage(settings):
    from conftest import FakeWorker

    token_engine = FakeTokenEngine()
    app = create_app(settings, FakeWorker(), token_engine=token_engine)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/generate_stream",
            json={
                "prompt": "hello",
                "max_new_tokens": 1,
                "temperature": 0,
                "top_p": 1,
                "stop_token_ids": [],
            },
        )

    payloads = events(response)
    assert response.status_code == 200
    assert payloads[0]["token"] == " answer"
    assert payloads[0]["token_id"] == 42
    assert payloads[0]["index"] == 0
    assert payloads[-1]["finish_reason"] == "length"
    assert payloads[-1]["usage"] == {"input_tokens": 2, "output_tokens": 1}
    assert app.state.platform_requests.get(token_engine.request_id).status.value == "completed"


async def test_token_stream_failure_has_terminal_reason_and_usage(settings):
    from conftest import FakeWorker

    class FailingTokenEngine(FakeTokenEngine):
        async def stream(self, request_id):
            if False:
                yield
            raise RuntimeError("controlled failure")

    token_engine = FailingTokenEngine()
    app = create_app(settings, FakeWorker(), token_engine=token_engine)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/generate_stream",
            json={"prompt": "hello", "max_new_tokens": 1, "temperature": 0},
        )

    payloads = events(response)
    assert payloads[-1]["done"] is True
    assert payloads[-1]["finish_reason"] == "failed"
    assert payloads[-1]["usage"] == {"input_tokens": 1, "output_tokens": 0}


async def test_backend_load_failure_uses_error_envelope_before_sse_headers(settings):
    from conftest import FakeWorker

    class UnavailableTokenEngine(FakeTokenEngine):
        async def validate_sampling(self, sampling):
            raise RuntimeError("controlled load failure")

    app = create_app(settings, FakeWorker(), token_engine=UnavailableTokenEngine())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/generate_stream",
            json={"prompt": "hello", "max_new_tokens": 1, "temperature": 0},
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "token_backend_unavailable"
    assert app.state.limiter.active == 0


async def test_metrics_include_token_scheduler_and_kv_accounting(settings):
    from conftest import FakeWorker

    app = create_app(settings, FakeWorker(), token_engine=FakeTokenEngine())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/metrics")

    payload = response.json()
    assert payload["token_scheduler"]["iterations"] == 3
    assert payload["token_scheduler"]["max_observed_batch_size"] == 2
    assert payload["kv_cache"]["kv_blocks_total"] == 16
    assert payload["kv_cache"]["kv_cache_utilization"] == 0
