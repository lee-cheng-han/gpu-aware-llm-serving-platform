# Observability

The gateway keeps the backward-compatible JSON snapshot at `GET /metrics`. Prometheus
scrapers can request the same URL with `Accept: text/plain`; the gateway returns the 0.0.4
text exposition without adding a metrics service or paid dependency.

Every HTTP response includes:

- `x-request-id`, using the incoming W3C trace ID when valid or a generated 128-bit ID
- `traceparent`, with the propagated trace ID and a new server span ID
- `server-timing`, containing gateway response-start latency

The gateway also emits a bounded-cardinality JSON log record for method, path, status,
duration, and trace ID. Prompt text, authorization values, tenant secrets, and generated
text are never logged.

Optional OpenTelemetry SDK tracing and OTLP-over-HTTP export are enabled by installing
`requirements-observability.txt` and setting `OTEL_EXPORTER_OTLP_ENDPOINT`, normally to a
local collector such as `http://localhost:4318/v1/traces`. The exporter creates server spans,
continues incoming W3C context, records HTTP method/path/status attributes, batches exports,
and flushes on shutdown. It is disabled by default. Non-loopback collectors are rejected
unless `ALLOW_REMOTE_TELEMETRY=true`, preventing accidental transmission to a billed vendor.

For a local measured load report, start the gateway and run:

```bash
make benchmark-sustained-load
```

The harness reports request status counts, latency percentiles, requests/second, and
tokens/second as JSON. It refuses non-loopback targets unless `--allow-remote` is supplied,
which helps prevent accidental use of a billed hosted endpoint. Results are not committed
or described as improvements unless a real run is recorded with its environment.
