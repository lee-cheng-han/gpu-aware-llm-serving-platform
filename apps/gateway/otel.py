from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class OpenTelemetryRuntime:
    tracer: Any
    provider: Any

    def shutdown(self) -> None:
        self.provider.force_flush()
        self.provider.shutdown()


def configure_opentelemetry(service_name: str, endpoint: str) -> OpenTelemetryRuntime | None:
    """Configure an optional OTLP/HTTP exporter without affecting zero-dependency defaults."""
    if not endpoint:
        return None
    try:
        trace = import_module("opentelemetry.trace")
        resources = import_module("opentelemetry.sdk.resources")
        tracing = import_module("opentelemetry.sdk.trace")
        export = import_module("opentelemetry.sdk.trace.export")
        otlp = import_module("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OTLP export requires requirements-observability.txt"
        ) from exc
    provider = tracing.TracerProvider(
        resource=resources.Resource.create({"service.name": service_name})
    )
    provider.add_span_processor(
        export.BatchSpanProcessor(otlp.OTLPSpanExporter(endpoint=endpoint))
    )
    trace.set_tracer_provider(provider)
    return OpenTelemetryRuntime(trace.get_tracer(service_name), provider)
