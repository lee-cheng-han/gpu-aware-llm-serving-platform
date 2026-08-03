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
    parser.add_argument("--batch-size", "--concurrency", dest="batch_size", type=int, default=8,
                        help="direct model batch size (legacy alias: --concurrency)")
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    prompts = [f"Explain cache memory simply. Request {i}." for i in range(args.requests)]
    for model_name in args.models:
        worker = InferenceWorker(model_name)
        for _ in range(args.warmup_batches):
            worker.generate_batch(["Warm up local inference."], args.max_new_tokens, 0)
        trial_results = []
        for _ in range(args.trials):
            latencies, tokens = [], 0
            started = time.perf_counter()
            for offset in range(0, len(prompts), args.batch_size):
                batch = prompts[offset:offset + args.batch_size]
                call_started = time.perf_counter()
                results = worker.generate_batch(batch, args.max_new_tokens, 0)
                per_request = (time.perf_counter() - call_started) * 1000
                latencies.extend([per_request] * len(batch))
                tokens += sum(result.output_tokens for result in results)
            elapsed = time.perf_counter() - started
            ordered = sorted(latencies)
            trial_results.append({
                "p50_latency_ms": statistics.median(ordered),
                "p95_latency_ms": ordered[round((len(ordered) - 1) * .95)],
                "requests_per_second": args.requests / elapsed,
                "tokens_per_second": tokens / elapsed,
            })
        print(json.dumps({
            "model": model_name, "requests": args.requests, "batch_size": args.batch_size,
            "trials": args.trials, "warmup_batches": args.warmup_batches,
            "max_new_tokens": args.max_new_tokens,
            **{
                key: statistics.mean(trial[key] for trial in trial_results)
                for key in trial_results[0]
            },
            "notes": "Direct batched model sanity check; excludes HTTP and scheduler overhead.",
        }, indent=2))


if __name__ == "__main__":
    main()
