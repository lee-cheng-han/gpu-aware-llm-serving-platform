#!/usr/bin/env python3
"""Load-test one running scheduler policy. Restart the server between policies."""
import argparse
import concurrent.futures
import json
import statistics
import time

import httpx


def pct(values, p):
    values = sorted(values)
    return values[min(round((len(values) - 1) * p), len(values) - 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--policy", choices=["no_batching", "dynamic_batch"], required=True)
    parser.add_argument("--max-batch-size", type=int, default=8, help="server setting, recorded for context")
    parser.add_argument("--max-wait-ms", type=int, default=25, help="server setting, recorded for context")
    args = parser.parse_args()
    latencies, queue_waits, batches, tokens, observed = [], [], [], 0, set()

    def call(i):
        started = time.perf_counter()
        response = httpx.post(
            f"{args.base_url}/v1/generate",
            json={"prompt": f"Explain CPU scheduling simply. Example {i}.",
                  "max_new_tokens": args.max_new_tokens, "temperature": 0.7},
            timeout=300,
        )
        response.raise_for_status()
        return (time.perf_counter() - started) * 1000, response.json()

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        for latency, data in pool.map(call, range(args.requests)):
            latencies.append(latency)
            queue_waits.append(data["queue_wait_ms"])
            batches.append(data["batch_size"])
            tokens += data["output_tokens"]
            observed.add(data["scheduler_policy"])
    elapsed = time.perf_counter() - started
    if observed != {args.policy}:
        raise SystemExit(f"server policy is {observed}; restart it with SCHEDULER_POLICY={args.policy}")
    print(json.dumps({
        "policy": args.policy, "requests": args.requests, "concurrency": args.concurrency,
        "max_new_tokens": args.max_new_tokens, "p50_latency_ms": pct(latencies, .5),
        "p95_latency_ms": pct(latencies, .95), "requests_per_second": args.requests / elapsed,
        "tokens_per_second": tokens / elapsed, "avg_queue_wait_ms": statistics.mean(queue_waits),
        "avg_batch_size": statistics.mean(batches),
    }, indent=2))


if __name__ == "__main__":
    main()
