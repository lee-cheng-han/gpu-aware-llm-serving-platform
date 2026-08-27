import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import httpx
import pytest

from apps.gateway.config import Settings
from apps.gateway.main import create_app

pytestmark = [
    pytest.mark.otel,
    pytest.mark.skipif(
        os.getenv("RUN_OTEL_TESTS") != "1",
        reason="set RUN_OTEL_TESTS=1 after installing optional observability dependencies",
    ),
]


async def test_otlp_http_export_reaches_local_collector():
    from conftest import FakeWorker

    received = []

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("content-length", "0"))
            received.append((self.path, self.headers.get("content-type"), self.rfile.read(length)))
            self.send_response(200)
            self.end_headers()

        def log_message(self, _format, *args):
            return None

    server = HTTPServer(("127.0.0.1", 0), Collector)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1/traces"
    app = create_app(Settings(otel_exporter_otlp_endpoint=endpoint), FakeWorker())
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/health")
                assert response.status_code == 200
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert received
    path, content_type, body = received[0]
    assert path == "/v1/traces"
    assert content_type == "application/x-protobuf"
    assert body
