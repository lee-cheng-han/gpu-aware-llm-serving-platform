import httpx
import pytest

from benchmark.run_sustained_load import require_explicit_remote, run_load


def test_remote_target_requires_explicit_opt_in():
    require_explicit_remote("http://localhost:8000", False)
    with pytest.raises(ValueError, match="--allow-remote"):
        require_explicit_remote("https://paid.example", False)


async def test_load_report_contains_measured_request_statistics():
    async def handler(_request):
        return httpx.Response(200, json={"output_tokens": 2})

    report = await run_load(
        "http://test",
        0.01,
        2,
        "hello",
        2,
        transport=httpx.MockTransport(handler),
    )

    assert report["kind"] == "measured_http_load"
    assert report["requests"] > 0
    assert report["successful_requests"] == report["requests"]
    assert report["tokens_per_second"] > 0
    assert report["latency_ms"]["p95"] >= 0
