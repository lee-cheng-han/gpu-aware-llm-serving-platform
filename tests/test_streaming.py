import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from conftest import FakeWorker
from starlette.requests import Request

from apps.gateway.api import generate_stream
from apps.gateway.main import create_app
from apps.gateway.schemas import GenerateRequest
from inference.worker import StreamChunk


def events(response):
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


async def test_stream_emits_tokens_done_and_releases_slot(settings):
    app = create_app(settings, FakeWorker())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/generate_stream", json={
            "prompt": "hello", "max_new_tokens": 2, "temperature": 0,
        })
        metrics = (await client.get("/metrics")).json()

    payloads = events(response)
    assert response.status_code == 200
    assert payloads[0]["token"] == " generated"
    assert payloads[-1]["done"] is True
    assert app.state.limiter.active == 0
    assert metrics["completed_requests"] == 1
    assert metrics["model_invocations"] == 1
    assert metrics["batches"] == 0
    assert metrics["tokens_per_second"] > 0


async def test_stream_failure_is_structured_and_releases_slot(settings):
    class FailingWorker(FakeWorker):
        def stream(self, prompt, max_new_tokens, temperature):
            raise RuntimeError("private model failure")
            yield

    app = create_app(settings, FailingWorker())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/generate_stream", json={"prompt": "hello"})
        metrics = (await client.get("/metrics")).json()

    payloads = events(response)
    assert payloads[0]["error"]["code"] == "generation_failed"
    assert "private model failure" not in response.text
    assert payloads[-1]["done"] is True
    assert app.state.limiter.active == 0
    assert metrics["failed_requests"] == 1


async def test_stream_reserves_concurrency_before_headers(settings):
    app = create_app(
        replace(settings, max_concurrent_requests=1),
        FakeWorker(),
    )
    await app.state.limiter.acquire()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/v1/generate_stream", json={"prompt": "hello"})
    finally:
        await app.state.limiter.release()

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "concurrency_limit_exceeded"
    assert app.state.metrics.snapshot()["rejected_requests"] == 1


async def test_invalid_stream_releases_reserved_slot(settings):
    app = create_app(settings, FakeWorker())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/generate_stream", json={"prompt": " "})
    assert response.status_code == 400
    assert app.state.limiter.active == 0


async def test_stream_cancellation_records_disconnect_and_releases_slot(
    settings, monkeypatch
):
    class TwoChunkWorker(FakeWorker):
        def __init__(self):
            super().__init__()
            self.stream_inline_for_tests = False

        def stream(self, prompt, max_new_tokens, temperature):
            yield StreamChunk(" one", 1)
            yield StreamChunk(" two", 2)

    calls = 0
    never = asyncio.Event()

    async def controlled_to_thread(function, *args):
        nonlocal calls
        calls += 1
        if calls == 2:
            await never.wait()
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", controlled_to_thread)
    app = create_app(settings, TwoChunkWorker())
    request = Request({"type": "http", "method": "POST", "path": "/", "app": app})
    response = await generate_stream(
        GenerateRequest(prompt="hello", max_new_tokens=2, temperature=0),
        request,
    )
    iterator = response.body_iterator
    assert " one" in await anext(iterator)

    pending = asyncio.create_task(anext(iterator))
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert app.state.limiter.active == 0
    assert app.state.metrics.snapshot()["cancelled_requests"] == 1
