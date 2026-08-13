from dataclasses import replace

import httpx
import pytest
from conftest import FakeWorker

from apps.gateway.limits import ConcurrencyLimiter
from apps.gateway.main import create_app


async def post(app, body):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.post("/v1/generate", json=body)


async def test_empty_prompt_rejected(settings):
    response = await post(create_app(settings, FakeWorker()), {"prompt": " "})
    assert response.status_code == 400


async def test_prompt_too_long(settings):
    settings = replace(settings, max_prompt_tokens=2)
    response = await post(create_app(settings, FakeWorker()), {"prompt": "one two three"})
    assert response.status_code == 413


async def test_output_too_long(settings):
    response = await post(create_app(settings, FakeWorker()), {
        "prompt": "hello", "max_new_tokens": settings.max_new_tokens + 1
    })
    assert response.status_code == 400


async def test_model_context_window_rejected(settings):
    response = await post(
        create_app(settings, FakeWorker(context_window=4)),
        {"prompt": "one two three", "max_new_tokens": 2},
    )
    assert response.status_code == 413
    body = response.json()["error"]
    assert body["code"] == "context_window_exceeded"
    assert body["details"]["context_window_tokens"] == 4


async def test_prompt_character_limit_rejects_before_tokenization(settings):
    settings = replace(settings, max_prompt_characters=4)
    worker = FakeWorker()
    response = await post(create_app(settings, worker), {"prompt": "12345"})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "prompt_size_exceeded"
    assert worker.is_ready is False


async def test_schema_errors_are_structured(settings):
    response = await post(
        create_app(settings, FakeWorker()),
        {"prompt": "hello", "temperature": -1},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "request_validation_error"


async def test_concurrency_limiter_rejects():
    limiter = ConcurrencyLimiter(1)
    async with limiter.slot():
        with pytest.raises(Exception) as captured:
            async with limiter.slot():
                pass
        assert captured.value.status_code == 429
