from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from typing import Any
from urllib.parse import urlparse

import httpx


def require_explicit_remote(base_url: str, allow_remote: bool) -> None:
    host = (urlparse(base_url).hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1", "test"} and not allow_remote:
        raise ValueError("remote load targets require --allow-remote")


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * fraction), len(ordered) - 1)]


async def run_load(
    base_url: str,
    duration_seconds: float,
    concurrency: int,
    prompt: str,
    max_new_tokens: int,
    *,
    api_key: str = "",
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    if duration_seconds <= 0 or concurrency <= 0 or max_new_tokens <= 0:
        raise ValueError("duration, concurrency, and token count must be positive")
    deadline = time.monotonic() + duration_seconds
    latencies: list[float] = []
    statuses: Counter[int] = Counter()
    output_tokens = 0
    lock = asyncio.Lock()
    headers = {"authorization": f"Bearer {api_key}"} if api_key else {}

    async with httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        transport=transport,
        timeout=120,
    ) as client:

        async def worker() -> None:
            nonlocal output_tokens
            while time.monotonic() < deadline:
                started = time.perf_counter()
                try:
                    response = await client.post(
                        "/v1/generate",
                        json={
                            "prompt": prompt,
                            "max_new_tokens": max_new_tokens,
                            "temperature": 0,
                        },
                    )
                    status = response.status_code
                    tokens = (
                        int(response.json().get("output_tokens", 0))
                        if response.is_success
                        else 0
                    )
                except httpx.HTTPError:
                    status, tokens = 0, 0
                latency_ms = (time.perf_counter() - started) * 1000
                async with lock:
                    statuses[status] += 1
                    latencies.append(latency_ms)
                    output_tokens += tokens

        await asyncio.gather(*(worker() for _ in range(concurrency)))

    elapsed = duration_seconds
    completed = sum(statuses.values())
    successes = sum(count for status, count in statuses.items() if 200 <= status < 300)
    return {
        "kind": "measured_http_load",
        "target": base_url,
        "duration_seconds": duration_seconds,
        "concurrency": concurrency,
        "requests": completed,
        "successful_requests": successes,
        "status_counts": {str(key): value for key, value in sorted(statuses.items())},
        "requests_per_second": successes / elapsed,
        "tokens_per_second": output_tokens / elapsed,
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded load against a local gateway")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--duration-seconds", type=float, default=60)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--prompt", default="Explain continuous batching in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--allow-remote", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_explicit_remote(args.base_url, args.allow_remote)
    result = asyncio.run(
        run_load(
            args.base_url,
            args.duration_seconds,
            args.concurrency,
            args.prompt,
            args.max_new_tokens,
            api_key=args.api_key,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
