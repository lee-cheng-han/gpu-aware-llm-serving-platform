#!/usr/bin/env python3
"""Measured comparison of legacy generate() and explicit prefill/decode."""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

from inference.backend import BackendExecutor
from inference.token_types import SamplingParameters
from inference.worker import InferenceWorker
from runtime.huggingface import HuggingFaceTokenBackend


def cpu_identity() -> str:
    """Return a useful CPU name even when platform.processor() is empty on Linux."""
    reported = platform.processor().strip()
    if reported:
        return reported
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return "unknown"


def timed(function):
    started = time.perf_counter()
    result = function()
    return result, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument(
        "--revision", default="12fd25f77366fa6b3b4b768ec3050bf629380bac"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--prompt", default="Explain cache memory in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()
    if min(args.max_new_tokens, args.trials) <= 0 or args.warmups < 0:
        parser.error("token count and trials must be positive; warmups must be non-negative")

    import torch
    import transformers

    worker = InferenceWorker(args.model, args.revision, args.device, args.dtype)
    backend = HuggingFaceTokenBackend(worker)
    executor = BackendExecutor("token-benchmark")
    parameters = SamplingParameters(args.max_new_tokens, temperature=0)
    legacy: list[dict[str, float]] = []
    manual: list[dict[str, float]] = []
    try:
        for _ in range(args.warmups):
            worker.generate_one(args.prompt, min(args.max_new_tokens, 2), 0)
            tokenized = executor.run(backend.tokenize, args.prompt)
            executor.run(backend.prefill, "warmup", tokenized, parameters)
            executor.run(backend.release_sequence, "warmup")

        for trial in range(args.trials):
            result, elapsed = timed(
                lambda: worker.generate_one(args.prompt, args.max_new_tokens, 0)
            )
            legacy.append(
                {
                    "end_to_end_ms": elapsed * 1000,
                    "first_token_upper_bound_ms": elapsed * 1000,
                    "tokens_per_second": result.output_tokens / elapsed,
                }
            )

            sequence_id = f"trial-{trial}"
            tokenized = executor.run(backend.tokenize, args.prompt)
            first, prefill_seconds = timed(
                lambda current_id=sequence_id, current_prompt=tokenized: executor.run(
                    backend.prefill, current_id, current_prompt, parameters
                )
            )
            output_tokens = 1
            decode_seconds = 0.0
            last = first
            while output_tokens < args.max_new_tokens:
                last, step_seconds = timed(
                    lambda index=output_tokens, current_id=sequence_id: executor.run(
                        backend.decode,
                        current_id,
                        parameters,
                        index,
                    )
                )
                decode_seconds += step_seconds
                output_tokens += 1
                if last.token_id in executor.run(backend.eos_token_ids):
                    break
            executor.run(backend.release_sequence, sequence_id)
            total = prefill_seconds + decode_seconds
            manual.append(
                {
                    "end_to_end_ms": total * 1000,
                    "ttft_ms": prefill_seconds * 1000,
                    "avg_decode_step_ms": (
                        decode_seconds * 1000 / max(output_tokens - 1, 1)
                    ),
                    "tokens_per_second": output_tokens / total,
                }
            )

        result = {
            "classification": "measured_hardware_result",
            "model": {"id": args.model, "revision": args.revision},
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "processor": cpu_identity(),
                "logical_cpu_count": os.cpu_count(),
                "pytorch": torch.__version__,
                "transformers": transformers.__version__,
                "device": args.device,
                "dtype": args.dtype,
            },
            "workload": {
                "prompt": args.prompt,
                "max_new_tokens": args.max_new_tokens,
                "warmups": args.warmups,
                "trials": args.trials,
                "temperature": 0,
            },
            "legacy_generate": {
                key: statistics.mean(trial[key] for trial in legacy)
                for key in legacy[0]
            },
            "manual_token_loop": {
                key: statistics.mean(trial[key] for trial in manual)
                for key in manual[0]
            },
            "raw_trials": {"legacy_generate": legacy, "manual_token_loop": manual},
            "notes": [
                "Legacy generate() exposes no true TTFT; its full latency is an upper bound.",
                "Results compare correctness paths, not continuous batching.",
            ],
        }
        rendered = json.dumps(result, indent=2)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n")
        print(rendered)
    finally:
        executor.run(backend.shutdown)
        executor.shutdown()


if __name__ == "__main__":
    main()
