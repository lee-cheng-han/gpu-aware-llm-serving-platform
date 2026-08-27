from __future__ import annotations

import json
import logging
import re
import secrets
import time
from collections.abc import Mapping
from importlib import import_module
from typing import Any

from fastapi import Request, Response

TRACEPARENT = re.compile(
    r"^00-(?P<trace_id>[0-9a-f]{32})-(?P<span_id>[0-9a-f]{16})-(?P<flags>[0-9a-f]{2})$"
)


def trace_context(value: str | None = None) -> tuple[str, str]:
    match = TRACEPARENT.fullmatch((value or "").lower())
    trace_id = match.group("trace_id") if match else secrets.token_hex(16)
    return trace_id, secrets.token_hex(8)


def structured_http_log(
    logger: logging.Logger,
    request: Request,
    response: Response,
    trace_id: str,
    duration_ms: float,
) -> None:
    logger.info(
        json.dumps(
            {
                "event": "http_response_started",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 3),
                "trace_id": trace_id,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _metric_name(parts: tuple[str, ...]) -> str:
    cleaned = [re.sub(r"[^a-zA-Z0-9_]", "_", part).strip("_") for part in parts]
    return "llm_scheduler_" + "_".join(part.lower() for part in cleaned if part)


def prometheus_text(snapshot: Mapping[str, Any]) -> str:
    """Render the existing bounded snapshot as dependency-free Prometheus text."""
    samples: list[tuple[str, float]] = []

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for key, nested in sorted(value.items(), key=lambda item: str(item[0])):
                visit(nested, (*path, str(key)))
        elif isinstance(value, bool):
            samples.append((_metric_name(path), float(value)))
        elif isinstance(value, int | float):
            samples.append((_metric_name(path), float(value)))

    visit(snapshot, ())
    lines = [
        "# Dependency-free Prometheus exposition for the local inference gateway.",
    ]
    for name, value in samples:
        lines.extend((f"# TYPE {name} gauge", f"{name} {value:g}"))
    return "\n".join(lines) + "\n"


async def observe_request(
    request: Request,
    call_next,
    logger: logging.Logger,
    tracer: Any | None = None,
) -> Response:
    started = time.perf_counter()
    if tracer is not None:
        trace = import_module("opentelemetry.trace")
        propagate = import_module("opentelemetry.propagate")
        context = propagate.extract(dict(request.headers))
        with tracer.start_as_current_span(
            f"{request.method} {request.url.path}",
            context=context,
            kind=trace.SpanKind.SERVER,
            attributes={
                "http.request.method": request.method,
                "url.path": request.url.path,
            },
        ) as span:
            span_context = span.get_span_context()
            trace_id = f"{span_context.trace_id:032x}"
            span_id = f"{span_context.span_id:016x}"
            request.state.trace_id = trace_id
            response = await call_next(request)
            span.set_attribute("http.response.status_code", response.status_code)
    else:
        trace_id, span_id = trace_context(request.headers.get("traceparent"))
        request.state.trace_id = trace_id
        response = await call_next(request)
    duration_ms = (time.perf_counter() - started) * 1000
    response.headers["x-request-id"] = trace_id
    response.headers["traceparent"] = f"00-{trace_id}-{span_id}-01"
    response.headers["server-timing"] = f'app;dur={duration_ms:.3f}'
    structured_http_log(logger, request, response, trace_id, duration_ms)
    return response
