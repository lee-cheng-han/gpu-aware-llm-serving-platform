import json
import logging

import httpx

from apps.gateway.main import create_app
from apps.gateway.observability import prometheus_text, trace_context
from apps.gateway.otel import configure_opentelemetry


def test_prometheus_renderer_flattens_numeric_snapshots():
    rendered = prometheus_text(
        {
            "completed_requests": 3,
            "kv_cache": {"utilization": 0.25, "pressure": False},
            "model_name": "ignored-label-cardinality",
        }
    )
    assert "llm_scheduler_completed_requests 3" in rendered
    assert "llm_scheduler_kv_cache_utilization 0.25" in rendered
    assert "llm_scheduler_kv_cache_pressure 0" in rendered
    assert "ignored-label-cardinality" not in rendered


def test_trace_context_preserves_valid_parent_trace_id():
    trace_id, span_id = trace_context(
        "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    )
    assert trace_id == "0123456789abcdef0123456789abcdef"
    assert len(span_id) == 16


def test_opentelemetry_is_dependency_free_when_export_is_disabled():
    assert configure_opentelemetry("test-service", "") is None


async def test_http_observability_headers_logs_and_prometheus(settings, caplog):
    from conftest import FakeWorker

    app = create_app(settings, FakeWorker())
    parent = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    with caplog.at_level(logging.INFO, logger="apps.gateway.main"):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            health = await client.get("/health", headers={"traceparent": parent})
            metrics = await client.get(
                "/metrics", headers={"accept": "text/plain; version=0.0.4"}
            )

    assert health.headers["x-request-id"] == "0123456789abcdef0123456789abcdef"
    assert health.headers["traceparent"].startswith(
        "00-0123456789abcdef0123456789abcdef-"
    )
    assert health.headers["server-timing"].startswith("app;dur=")
    assert metrics.headers["content-type"].startswith("text/plain")
    assert "llm_scheduler_total_requests" in metrics.text
    payloads = [json.loads(record.message) for record in caplog.records]
    assert any(payload["event"] == "http_response_started" for payload in payloads)
