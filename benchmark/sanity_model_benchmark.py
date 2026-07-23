#!/usr/bin/env python3
"""Direct CPU sanity check; models are tested independently, never simultaneously."""
import argparse
import json
import statistics
import time

from inference.worker import InferenceWorker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["sshleifer/tiny-gpt2", "gpt2"])
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=8,
                        help="micro-batch size for this direct model sanity check")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    prompts = [f"Explain cache memory simply. Request {i}." for i in range(args.requests)]
    for model_name in args.models:
        worker = InferenceWorker(model_name)
        latencies, tokens = [], 0
        started = time.perf_counter()
        for offset in range(0, len(prompts), args.concurrency):
            batch = prompts[offset:offset + args.concurrency]
            call_started = time.perf_counter()
            results = worker.generate_batch(batch, args.max_new_tokens, .7)
            per_request = (time.perf_counter() - call_started) * 1000
            latencies.extend([per_request] * len(batch))
            tokens += sum(result.output_tokens for result in results)
        elapsed = time.perf_counter() - started
        ordered = sorted(latencies)
        print(json.dumps({
            "model": model_name, "requests": args.requests, "concurrency": args.concurrency,
            "max_new_tokens": args.max_new_tokens,
            "p50_latency_ms": statistics.median(ordered),
            "p95_latency_ms": ordered[round((len(ordered) - 1) * .95)],
            "requests_per_second": args.requests / elapsed, "tokens_per_second": tokens / elapsed,
            "notes": "Direct batched model sanity check; excludes HTTP and scheduler overhead.",
        }, indent=2))


if __name__ == "__main__":
    main()
