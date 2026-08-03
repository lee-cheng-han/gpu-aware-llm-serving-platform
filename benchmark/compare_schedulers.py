#!/usr/bin/env python3
"""Reproducible asynchronous load test for one running scheduler policy."""
import argparse
import asyncio
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import httpx


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * p), len(ordered) - 1)] if ordered else 0


async def run_trial(client, args, prompts):
    semaphore = asyncio.Semaphore(args.concurrency)
    latencies, queue_waits, batches = [], [], []
    output_tokens, policies = 0, set()
    outcomes: Counter[str] = Counter()

    async def call(prompt):
        nonlocal output_tokens
        async with semaphore:
            started = time.perf_counter()
            try:
                response = await client.post("/v1/generate", json={
                    "prompt": prompt,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                })
            except httpx.HTTPError:
                outcomes["transport_error"] += 1
                return
            latency = (time.perf_counter() - started) * 1000
            if response.status_code != 200:
                outcomes[f"http_{response.status_code}"] += 1
                return
            data = response.json()
            outcomes["completed"] += 1
            latencies.append(latency)
            queue_waits.append(data["queue_wait_ms"])
            batches.append(data["batch_size"])
            output_tokens += data["output_tokens"]
            policies.add(data["scheduler_policy"])

    started = time.perf_counter()
    await asyncio.gather(*(call(prompt) for prompt in prompts))
    elapsed = time.perf_counter() - started
    if policies and policies != {args.policy}:
        raise RuntimeError(
            f"server policy is {policies}; restart with SCHEDULER_POLICY={args.policy}"
        )
    completed = outcomes["completed"]
    return {
        "elapsed_seconds": elapsed,
        "outcomes": dict(sorted(outcomes.items())),
        "p50_latency_ms": percentile(latencies, .50),
        "p95_latency_ms": percentile(latencies, .95),
        "requests_per_second": completed / elapsed,
        "tokens_per_second": output_tokens / elapsed,
        "avg_queue_wait_ms": statistics.mean(queue_waits) if queue_waits else 0,
        "avg_batch_size": statistics.mean(batches) if batches else 0,
        "raw_latency_ms": latencies,
    }


async def async_main(args):
    rng = random.Random(args.seed)
    prompts = [
        f"Explain CPU scheduling simply. Example {rng.randrange(1_000_000)}."
        for _ in range(args.requests)
    ]
    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )
    async with httpx.AsyncClient(
        base_url=args.base_url, timeout=args.timeout, limits=limits
    ) as client:
        for i in range(args.warmup_requests):
            response = await client.post("/v1/generate", json={
                "prompt": f"Warm up local inference {i}.",
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
            })
            response.raise_for_status()
        server_metrics = (await client.get("/metrics")).json()
        observed = server_metrics.get("scheduler_policy")
        if observed != args.policy:
            raise RuntimeError(
                f"server policy is {observed}; restart with SCHEDULER_POLICY={args.policy}"
            )
        trials = [await run_trial(client, args, prompts) for _ in range(args.trials)]

    result = {
        "configuration": {
            "base_url": args.base_url,
            "policy": args.policy,
            "model_name": server_metrics.get("model_name"),
            "requests": args.requests,
            "concurrency": args.concurrency,
            "warmup_requests": args.warmup_requests,
            "trials": args.trials,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "server": {
                key: server_metrics.get(key) for key in (
                    "max_batch_size", "max_wait_ms", "max_total_batch_tokens",
                    "metrics_sample_limit",
                )
            },
        },
        "trials": trials,
        "summary": {
            key: statistics.mean(trial[key] for trial in trials)
            for key in (
                "p50_latency_ms", "p95_latency_ms", "requests_per_second",
                "tokens_per_second", "avg_queue_wait_ms", "avg_batch_size",
            )
        },
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
    print(rendered)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--warmup-requests", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output")
    parser.add_argument("--policy", choices=["no_batching", "dynamic_batch"], required=True)
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.trials) <= 0:
        parser.error("requests, concurrency, and trials must be positive")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
