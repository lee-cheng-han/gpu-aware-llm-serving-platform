import httpx
import pytest
from conftest import FakeWorker

from apps.gateway.config import Settings
from apps.gateway.main import create_app


async def test_api_key_authentication_derives_tenant_without_body_field():
    app = create_app(Settings(api_keys="tenant-a:secret"), FakeWorker())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            missing = await client.post("/v1/generate", json={"prompt": "hello"})
            invalid = await client.post(
                "/v1/generate",
                headers={"Authorization": "Bearer wrong"},
                json={"prompt": "hello"},
            )
            valid = await client.post(
                "/v1/generate",
                headers={"Authorization": "Bearer secret"},
                json={"prompt": "hello"},
            )
    assert missing.status_code == 401
    assert missing.json()["error"]["code"] == "authentication_required"
    assert invalid.status_code == 401
    assert invalid.json()["error"]["code"] == "invalid_api_key"
    assert valid.status_code == 200
    records = app.state.platform_requests.list()
    assert len(records) == 1
    assert records[0].tenant_id == "tenant-a"


async def test_request_body_cannot_supply_tenant_identity():
    app = create_app(Settings(api_keys="tenant-a:secret"), FakeWorker())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/generate",
                headers={"Authorization": "Bearer secret"},
                json={"prompt": "hello", "tenant_id": "tenant-b"},
            )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "request_validation_error"


async def test_cors_allows_only_configured_origin():
    app = create_app(Settings(cors_allowed_origins="https://console.example"), FakeWorker())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        allowed = await client.options(
            "/v1/generate",
            headers={
                "Origin": "https://console.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        denied = await client.options(
            "/v1/generate",
            headers={
                "Origin": "https://attacker.example",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert allowed.headers["access-control-allow-origin"] == "https://console.example"
    assert "access-control-allow-origin" not in denied.headers


def test_invalid_api_key_configuration_is_rejected():
    with pytest.raises(ValueError, match="tenant:key"):
        create_app(Settings(api_keys="malformed"), FakeWorker())
